"""Prepare the LR2 E+S learning curve (25 % / 50 % nested train subsets); never train.

The 100 % point is the existing R42-R44 seed replicates. Each subset point changes
only the train rows (whole composition groups, repeated so one epoch keeps the
full-train step count) of the adopted seed-42 staged LR2 E+S recipe.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.local_global_compare import training_command
from tools.eval.reference_controls_prepare import ADOPTED_EPOCH, source_paths as reference_sources
from tools.eval.spectral_balance_prepare import completed_spectral_control
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import training_record
from utils.paths import file_sha256, input_path
from utils.spectral_balance import load_balance_protocol
from utils.spectral_balance_experiment import validate_balance_config
from utils.train_subset import build_subsets, write_subset

SCHEMA = 'lr2-es-learning-curve-preparation-v1'
DESIGN = 'docs/design/design-2026-11-06-learning-curve.md'
POINTS = (('f25', 0.25), ('f50', 0.5))  # run order: the cheaper question first
EXTRA_SOURCES = ('tools/eval/learning_curve_prepare.py', 'tools/eval/learning_curve_analyze.py',
                 'tools/checks/check_learning_curve.py', 'utils/train_subset.py')


def source_paths():
    return sorted(set(reference_sources()) | {ROOT / name for name in EXTRA_SOURCES})


def prepare(control_preparation, reference_controls, out, tag_prefix):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'use a new output directory: {out}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag_prefix):
        raise ValueError('isolated alphanumeric tag prefix required')
    original = json.loads(input_path(control_preparation).read_text())
    reference = ExperimentConfig(**original['configuration'])
    data_dir = input_path(reference.data_dir)
    protocol = load_balance_protocol(reference.spectral_balance_config, reference.data_dir)
    control = completed_spectral_control(control_preparation, protocol, data_dir,
                                         shape_aux=True, local_route='LR2')
    if control['selected_epoch'] != ADOPTED_EPOCH:
        raise ValueError('the learning curve extends the adopted staged LR2 E+S ep29 recipe')
    controls = json.loads(input_path(reference_controls).read_text())
    if controls.get('schema') != 'lr2-es-reference-controls-preparation-v1':
        raise ValueError('reference controls preparation required for the 100 % seed replicates')
    subsets = build_subsets(data_dir)
    if not set(subsets[0.25]['material_ids']) <= set(subsets[0.5]['material_ids']):
        raise ValueError('subsets must be nested')
    protocol_path = out / 'protocol.json'
    protocol_bytes = input_path(reference.spectral_balance_config).read_bytes()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'subsets').mkdir()
    protocol_path.write_bytes(protocol_bytes)
    hashes, points = {}, []
    for name, fraction in POINTS:
        subset_path = out / 'subsets' / f'train_{name}.json'
        hashes[str(subset_path.resolve())] = write_subset(subsets[fraction], subset_path)
        cfg = replace(reference, skip_existing=False, train_subset=str(subset_path.resolve()),
                      tag=f'_sbal_{tag_prefix}_{name}_e35')
        validate_balance_config(cfg)
        cfg = replace(cfg, spectral_balance_config=str(protocol_path.resolve()))
        before, after = training_record(reference), training_record(cfg)
        changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        if changed != {'spectral_balance_config', 'train_subset'}:
            raise ValueError(f'{name} changes more than its train rows: {sorted(changed)}')
        run = ROOT / f'results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}'
        history = ROOT / f'results/reports/history_{cfg.model_name.lower()}{cfg.tag}.csv'
        if run.exists() or history.exists():
            raise FileExistsError(f'{name} requires a fresh tag without existing run/history')
        meta = {k: v for k, v in subsets[fraction].items() if k != 'material_ids'}
        points.append({'name': name, 'fraction': fraction, 'subset': meta, 'subset_path': str(subset_path.resolve()),
                       'changed_training_fields': sorted(changed), 'configuration': asdict(cfg),
                       'run_dir': str(run), 'history': str(history), 'command': training_command(cfg)})
    sources = source_paths()
    hashes.update(control['input_sha256'])
    hashes[str(input_path(reference_controls))] = file_sha256(input_path(reference_controls))
    hashes[str(data_dir / 'manifest.json')] = file_sha256(data_dir / 'manifest.json')
    for item, digest in {**protocol['train_sha256'], **protocol['valid_guard_sha256']}.items():
        hashes[str(data_dir / item)] = digest
    hashes.update({str(p): file_sha256(p) for p in sources})
    hashes[str(protocol_path.resolve())] = file_sha256(protocol_path)
    if any(file_sha256(p) != digest for p, digest in hashes.items()):
        raise ValueError('an input changed during preparation')
    for source in sources:
        target = out / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    report = {'schema': SCHEMA, 'training_started': False, 'tag_prefix': tag_prefix, 'design': DESIGN,
              'completed_control': control, 'reference_controls': str(input_path(reference_controls)),
              'reference': protocol['reference'], 'protocol_copy_identical': protocol_path.read_bytes() == protocol_bytes,
              'points': points, 'run_order': [p['name'] for p in points],
              'full_train_point': 'R42-R44 seed replicates of the reference controls (not rerun)',
              'input_sha256': hashes, 'not_a_new_selection_or_adoption': True}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-preparation', default='results/runs/spectral_residual_20261030/report.json')
    parser.add_argument('--reference-controls', default='results/runs/reference_controls_20261103/report.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--tag-prefix', required=True)
    args = parser.parse_args()
    report = prepare(input_path(args.control_preparation), input_path(args.reference_controls),
                     input_path(args.out), args.tag_prefix)
    for point in report['points']:
        print(point['name'], point['subset']['materials'], shlex.join(point['command']))


if __name__ == '__main__':
    main()
