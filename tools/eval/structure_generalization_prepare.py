"""Prepare the LR2 E+S generalization arms (G strain augmentation, D local drop); never train.

Step 1 runs g_s42 and d_s42. Step-2 seeds of an arm run only after the frozen
screen in ``structure_generalization_analyze.py screen`` promotes that arm.
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
from utils.structure_regularization import LOCAL_DROP_PROB

SCHEMA = 'lr2-es-structure-generalization-preparation-v1'
DESIGN = 'docs/design/design-2026-11-05-structure-generalization.md'
# Name, arm, seed, data-order seed, step.
ARMS = (
    ('g_s42', 'G', 42, None, 1),
    ('d_s42', 'D', 42, None, 1),
    ('g_s43', 'G', 43, 43, 2),
    ('g_s44', 'G', 44, 44, 2),
    ('d_s43', 'D', 43, 43, 2),
    ('d_s44', 'D', 44, 44, 2),
)
EXTRA_SOURCES = ('tools/eval/structure_generalization_prepare.py', 'tools/eval/structure_generalization_analyze.py',
                 'tools/eval/structure_use_audit.py', 'tools/checks/check_structure_generalization.py')


def arm_config(reference: ExperimentConfig, name, arm, seed, order, tag_prefix):
    return replace(reference, seed=seed, data_order_seed=order, skip_existing=False,
                   strain_augment=arm == 'G', local_drop_prob=LOCAL_DROP_PROB if arm == 'D' else None,
                   tag=f'_sbal_{tag_prefix}_{name}_e35')


def expected_changes(arm, seed, order):
    changed = {'spectral_balance_config', 'strain_augment' if arm == 'G' else 'local_drop_prob'}
    if seed != 42:
        changed.add('seed')
    if order is not None:
        changed.add('data_order_seed')
    return changed


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
        raise ValueError('generalization arms compare against the adopted staged LR2 E+S ep29 control')
    controls = json.loads(input_path(reference_controls).read_text())
    if controls.get('schema') != 'lr2-es-reference-controls-preparation-v1':
        raise ValueError('reference controls preparation required for the seed-mean comparison')
    path = out / 'protocol.json'
    arms = []
    for name, arm, seed, order, step in ARMS:
        cfg = arm_config(reference, name, arm, seed, order, tag_prefix)
        validate_balance_config(cfg)
        cfg = replace(cfg, spectral_balance_config=str(path.resolve()))
        before, after = training_record(reference), training_record(cfg)
        changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        if changed != expected_changes(arm, seed, order):
            raise ValueError(f'{name} changes more than its declared arm: {sorted(changed)}')
        run = ROOT / f'results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}'
        history = ROOT / f'results/reports/history_{cfg.model_name.lower()}{cfg.tag}.csv'
        if run.exists() or history.exists():
            raise FileExistsError(f'{name} requires a fresh tag without existing run/history')
        arms.append({'name': name, 'arm': arm, 'seed': seed, 'data_order_seed': order, 'step': step,
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
    report = {'schema': SCHEMA, 'training_started': False, 'tag_prefix': tag_prefix, 'design': DESIGN,
              'completed_control': control, 'reference_controls': str(input_path(reference_controls)),
              'reference': protocol['reference'], 'protocol_copy_identical': path.read_bytes() == protocol_bytes,
              'arms': arms, 'step1_order': [a['name'] for a in arms if a['step'] == 1],
              'step2_order': {arm: [a['name'] for a in arms if a['step'] == 2 and a['arm'] == arm]
                              for arm in ('G', 'D')},
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
    for arm in report['arms']:
        print(arm['step'], arm['name'], shlex.join(arm['command']))


if __name__ == '__main__':
    main()
