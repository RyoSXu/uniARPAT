"""Prepare LR2 E+S seed replicates, structure-blind S0 and single-task controls; never train."""
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
from tools.eval.spectral_balance_prepare import completed_spectral_control
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import training_record
from utils.paths import file_sha256, input_path
from utils.spectral_balance import load_balance_protocol
from utils.spectral_balance_experiment import expected_selection_metric, validate_balance_config

SCHEMA = 'lr2-es-reference-controls-preparation-v1'
ADOPTED_EPOCH = 29
# Name, route, seed, data-order seed, supervised task. Cheap S0 arms run first.
ARMS = (
    ('s0_s42', 'S0', 42, None, None),
    ('s0_s43', 'S0', 43, 43, None),
    ('s0_s44', 'S0', 44, 44, None),
    ('lr2_s43', 'LR2', 43, 43, None),
    ('lr2_s44', 'LR2', 44, 44, None),
    ('edos_s42', 'LR2', 42, None, 'edos'),
    ('phdos_s42', 'LR2', 42, None, 'phdos'),
)
SOURCE_FILES = ('tools/train.py', 'tools/eval/reference_controls_prepare.py',
                'tools/eval/reference_controls_analyze.py', 'tools/eval/spectral_balance_prepare.py',
                'tools/eval/local_global_compare.py', 'tools/eval/lg_valid_export.py',
                'tools/checks/check_reference_controls.py')


def arm_config(reference: ExperimentConfig, name, route, seed, order, task, tag_prefix):
    cfg = replace(reference, local_route=route, local_depth=0 if route == 'S0' else reference.local_depth,
                  seed=seed, data_order_seed=order, single_task=task, skip_existing=False,
                  tag=f'_sbal_{tag_prefix}_{name}_e35')
    return replace(cfg, selection_metric=expected_selection_metric(cfg))


def expected_changes(route, seed, order, task):
    changed = {'spectral_balance_config'}
    if route == 'S0':
        changed |= {'local_route', 'local_depth', 'selection_metric'}
    if seed != 42:
        changed.add('seed')
    if order is not None:
        changed.add('data_order_seed')
    if task is not None:
        changed |= {'single_task', 'selection_metric'}
    return changed


def source_paths():
    sources = [p for directory in ('model', 'utils') for p in (ROOT / directory).rglob('*.py')]
    return sorted(sources + [ROOT / name for name in SOURCE_FILES])


def prepare(control_preparation, out, tag_prefix):
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
        raise ValueError('reference controls require the adopted staged LR2 E+S ep29 control')
    path = out / 'protocol.json'
    arms = []
    for name, route, seed, order, task in ARMS:
        cfg = arm_config(reference, name, route, seed, order, task, tag_prefix)
        validate_balance_config(cfg)
        cfg = replace(cfg, spectral_balance_config=str(path.resolve()))
        before, after = training_record(reference), training_record(cfg)
        changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        if changed != expected_changes(route, seed, order, task):
            raise ValueError(f'{name} changes more than its declared control: {sorted(changed)}')
        run = ROOT / f'results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}'
        history = ROOT / f'results/reports/history_{cfg.model_name.lower()}{cfg.tag}.csv'
        if run.exists() or history.exists():
            raise FileExistsError(f'{name} requires a fresh tag without existing run/history')
        arms.append({'name': name, 'route': route, 'seed': seed, 'data_order_seed': order,
                     'single_task': task, 'selection_metric': cfg.selection_metric,
                     'changed_training_fields': sorted(changed), 'configuration': asdict(cfg),
                     'run_dir': str(run), 'history': str(history), 'command': training_command(cfg)})
    protocol_bytes = input_path(reference.spectral_balance_config).read_bytes()
    sources = source_paths()
    hashes = {**control['input_sha256'],
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
    report = {'schema': SCHEMA, 'training_started': False, 'tag_prefix': tag_prefix,
              'design': 'docs/design/design-2026-11-03-variance-and-reference-controls.md',
              'completed_control': control, 'reference': protocol['reference'],
              'protocol_copy_identical': path.read_bytes() == protocol_bytes,
              'arms': arms, 'run_order': [arm['name'] for arm in arms],
              'input_sha256': hashes, 'not_a_new_selection_or_adoption': True}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-preparation', default='results/runs/spectral_residual_20261030/report.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--tag-prefix', required=True)
    args = parser.parse_args()
    report = prepare(input_path(args.control_preparation), input_path(args.out), args.tag_prefix)
    for arm in report['arms']:
        print(arm['name'], shlex.join(arm['command']))


if __name__ == '__main__':
    main()
