"""Prepare the phDOS head depth study (design 2026-11-06); never train.

Two seed-42 arms on the adopted staged LR2 E+S recipe change only the depth of
the legacy convolutional phDOS readout: h1 (one Conv1d, as the eDOS head) and
h2 (one hidden layer). The reference is the existing R42 run (depth 6).
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
from utils.spectral_balance_experiment import PHDOS_HEAD_DESIGN, validate_balance_config

SCHEMA = 'lr2-es-phdos-head-preparation-v1'
ARMS = (('h1_s42', 1), ('h2_s42', 2))
EXTRA_SOURCES = ('tools/eval/phdos_head_prepare.py', 'tools/eval/phdos_head_analyze.py',
                 'tools/eval/structure_generalization_analyze.py', 'tools/eval/structure_use_audit.py',
                 'tools/checks/check_phdos_head.py', 'model/heads.py')


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
    protocol = load_balance_protocol(reference.spectral_balance_config, reference.data_dir)
    control = completed_spectral_control(control_preparation, protocol, input_path(reference.data_dir),
                                         shape_aux=True, local_route='LR2')
    if control['selected_epoch'] != ADOPTED_EPOCH:
        raise ValueError('the head study compares against the adopted staged LR2 E+S ep29 control')
    controls = json.loads(input_path(reference_controls).read_text())
    if controls.get('schema') != 'lr2-es-reference-controls-preparation-v1':
        raise ValueError('reference controls preparation required for the seed-range context')
    path = out / 'protocol.json'
    arms = []
    for name, depth in ARMS:
        cfg = replace(reference, phdos_head_layers=depth, skip_existing=False, tag=f'_sbal_{tag_prefix}_{name}_e35')
        validate_balance_config(cfg)
        cfg = replace(cfg, spectral_balance_config=str(path.resolve()))
        before, after = training_record(reference), training_record(cfg)
        changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        if changed != {'spectral_balance_config', 'phdos_head_layers'}:
            raise ValueError(f'{name} changes more than the head depth: {sorted(changed)}')
        run = ROOT / f'results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}'
        history = ROOT / f'results/reports/history_{cfg.model_name.lower()}{cfg.tag}.csv'
        if run.exists() or history.exists():
            raise FileExistsError(f'{name} requires a fresh tag without existing run/history')
        arms.append({'name': name, 'arm': f'H{depth}', 'phdos_head_layers': depth, 'seed': 42, 'step': 1,
                     'changed_training_fields': sorted(changed), 'configuration': asdict(cfg),
                     'run_dir': str(run), 'history': str(history), 'command': training_command(cfg)})
    protocol_bytes = input_path(reference.spectral_balance_config).read_bytes()
    sources = source_paths()
    hashes = {**control['input_sha256'],
              str(input_path(reference_controls)): file_sha256(input_path(reference_controls)),
              str(input_path(reference.data_dir) / 'manifest.json'):
                  file_sha256(input_path(reference.data_dir) / 'manifest.json')}
    for name, digest in {**protocol['train_sha256'], **protocol['valid_guard_sha256']}.items():
        hashes[str(input_path(reference.data_dir) / name)] = digest
    hashes.update({str(p): file_sha256(p) for p in sources})
    if any(file_sha256(p) != digest for p, digest in hashes.items()):
        raise ValueError('an input changed during preparation')
    out.mkdir(parents=True, exist_ok=False)
    path.write_bytes(protocol_bytes)
    hashes[str(path.resolve())] = file_sha256(path)
    for source in sources:
        target = out / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    report = {'schema': SCHEMA, 'training_started': False, 'tag_prefix': tag_prefix, 'design': PHDOS_HEAD_DESIGN,
              'completed_control': control, 'reference_controls': str(input_path(reference_controls)),
              'reference': protocol['reference'], 'protocol_copy_identical': path.read_bytes() == protocol_bytes,
              'arms': arms, 'step1_order': [a['name'] for a in arms], 'input_sha256': hashes,
              'not_a_new_selection_or_adoption': True}
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
    for arm in report['arms']:
        print(arm['name'], shlex.join(arm['command']))


if __name__ == '__main__':
    main()
