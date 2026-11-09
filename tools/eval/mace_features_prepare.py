"""Prepare the MACE frozen-feature reference M1 (design 2026-11-08); never train.

One seed-42 arm adds frozen MACE invariant per-atom features to the working
point h2 (staged LR2 E+S with the two-layer phDOS head). The reference is the
completed h2_s42 run of the phDOS head study; its preparation, export and train
inference are frozen by hash here so the analysis compares against them.
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
from tools.eval.phdos_head_prepare import SCHEMA as HEAD_SCHEMA, source_paths as head_sources
from utils.experiment_config import ExperimentConfig
from utils.external_atom_features import feature_contract
from utils.local_global_experiment import training_record
from utils.paths import file_sha256, input_path
from utils.spectral_balance_experiment import EXTERNAL_FEATURE_DESIGN, validate_balance_config

SCHEMA = 'lr2-es-mace-features-preparation-v1'
ARMS = ('m1_s42',)
REFERENCE_ARM = 'h2_s42'
H2_EPOCH = 24
EXTRA_SOURCES = ('tools/eval/mace_features_prepare.py', 'tools/eval/mace_features_analyze.py',
                 'tools/checks/check_mace_features.py', 'tools/data/extract_mace_descriptors.py',
                 'utils/external_atom_features.py', 'utils/crystal.py', 'utils/dataset.py', 'utils/bucket_batch.py',
                 'model/model.py', 'model/transformer.py')


def source_paths():
    return sorted(set(head_sources()) | {ROOT / name for name in EXTRA_SOURCES})


def reference_control(head_preparation):
    """The completed h2 arm: configuration, selected epoch and frozen evidence paths."""
    head = json.loads(Path(head_preparation).read_text())
    if head.get('schema') != HEAD_SCHEMA:
        raise ValueError('phDOS head preparation required for the h2 reference')
    arm = next(a for a in head['arms'] if a['name'] == REFERENCE_ARM)
    base = Path(head_preparation).parent
    exports = base / 'analysis/exports'
    summary = json.loads((exports / f'{REFERENCE_ARM}_summary.json').read_text())
    if summary['best_epoch'] != H2_EPOCH or summary['checkpoint_epoch'] != H2_EPOCH:
        raise ValueError('the h2 reference is the adopted working point best ep24')
    preflight = json.loads((base / 'preflight/report.json').read_text())
    init = next(a for a in preflight['arms'] if a['name'] == REFERENCE_ARM)['initialization']
    train = base / 'analysis/train_inference' / f"{Path(arm['run_dir']).name}_train_samples.csv"
    evidence = {'samples': exports / f'{REFERENCE_ARM}_samples.csv',
                'spectra': exports / f'{REFERENCE_ARM}_spectra.npz',
                'summary': exports / f'{REFERENCE_ARM}_summary.json', 'train_samples': train,
                'best_checkpoint': Path(arm['run_dir']) / 'checkpoint_best.pth',
                'history': Path(arm['history'])}
    return {'label': REFERENCE_ARM, 'run_dir': arm['run_dir'], 'selected_epoch': H2_EPOCH,
            'configuration': arm['configuration'], 'head_preparation': str(head_preparation),
            'initialization_state_sha256': init['state_sha256'], 'best_score': summary['best_score'],
            **{key: str(path) for key, path in evidence.items()},
            'input_sha256': {str(path): file_sha256(path) for path in evidence.values()}}


def prepare(head_preparation, features, out, tag_prefix):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'use a new output directory: {out}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag_prefix):
        raise ValueError('isolated alphanumeric tag prefix required')
    control = reference_control(head_preparation)
    reference = ExperimentConfig(**control['configuration'])
    contract = feature_contract(input_path(features), input_path(reference.data_dir))
    path = out / 'protocol.json'
    arms = []
    for name in ARMS:
        cfg = replace(reference, external_atom_features=str(input_path(features)), skip_existing=False,
                      tag=f'_sbal_{tag_prefix}_{name}_e35')
        validate_balance_config(cfg)
        cfg = replace(cfg, spectral_balance_config=str(path.resolve()))
        before, after = training_record(reference), training_record(cfg)
        changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        if changed != {'spectral_balance_config', 'external_atom_features'}:
            raise ValueError(f'{name} changes more than the external features: {sorted(changed)}')
        run = ROOT / f'results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}'
        history = ROOT / f'results/reports/history_{cfg.model_name.lower()}{cfg.tag}.csv'
        if run.exists() or history.exists():
            raise FileExistsError(f'{name} requires a fresh tag without existing run/history')
        arms.append({'name': name, 'arm': 'M1', 'seed': 42, 'step': 1, 'external_atom_features': contract,
                     'changed_training_fields': sorted(changed), 'configuration': asdict(cfg),
                     'run_dir': str(run), 'history': str(history), 'command': training_command(cfg)})
    protocol_bytes = input_path(reference.spectral_balance_config).read_bytes()
    head = json.loads(Path(head_preparation).read_text())
    feature_root = input_path(features)
    # The head study's source hashes describe its own code revision; this preparation
    # re-hashes the current sources below and keeps only its data/protocol inputs.
    head_inputs = {p: d for p, d in head['input_sha256'].items() if not p.endswith('.py')}
    hashes = {**head_inputs, **control['input_sha256'],
              str(head_preparation): file_sha256(head_preparation),
              str(feature_root / 'manifest.json'): file_sha256(feature_root / 'manifest.json'),
              **{str(feature_root / f'{split}.npy'): digest for split, digest in contract['split_sha256'].items()}}
    sources = source_paths()
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
              'design': EXTERNAL_FEATURE_DESIGN, 'completed_control': control,
              'protocol_copy_identical': path.read_bytes() == protocol_bytes,
              'arms': arms, 'step1_order': [a['name'] for a in arms], 'input_sha256': hashes,
              'not_a_new_selection_or_adoption': True}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head-preparation', default='results/runs/phdos_head_20261106/report.json')
    parser.add_argument('--features', default='data/external/mace_mp0b3_medium_v1')
    parser.add_argument('--out', required=True)
    parser.add_argument('--tag-prefix', required=True)
    args = parser.parse_args()
    report = prepare(input_path(args.head_preparation), input_path(args.features).resolve(), input_path(args.out),
                     args.tag_prefix)
    for arm in report['arms']:
        print(arm['name'], shlex.join(arm['command']))


if __name__ == '__main__':
    main()
