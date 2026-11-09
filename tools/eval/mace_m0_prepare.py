"""Prepare the element-only MACE attribution arm M0 (design 2026-11-09); never train.

M0 is M1 with every atom's MACE descriptor replaced by its element's train mean
(`tools/data/build_element_mean_features.py`). Everything else equals h2 and M1:
same seed, initial tensors, data order and zero-initialized projection. The
references are h2_s42 (working point) and the completed M1 run, both frozen by hash.
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
from tools.eval.mace_features_prepare import SCHEMA as M1_SCHEMA, reference_control, source_paths as m1_sources
from utils.experiment_config import ExperimentConfig
from utils.external_atom_features import feature_contract, load_manifest
from utils.local_global_experiment import training_record
from utils.paths import file_sha256, input_path
from utils.spectral_balance_experiment import validate_balance_config

SCHEMA = 'lr2-es-mace-m0-preparation-v1'
DESIGN = 'docs/design/design-2026-11-09-mace-m0.md'
ARMS = ('m0_s42',)
M1_ARM = 'm1_s42'
EXTRA_SOURCES = ('tools/eval/mace_m0_prepare.py', 'tools/eval/mace_m0_analyze.py',
                 'tools/data/build_element_mean_features.py')


def source_paths():
    return sorted(set(m1_sources()) | {ROOT / name for name in EXTRA_SOURCES})


def m1_control(m1_preparation):
    """The completed M1 arm: its features, selected checkpoint and analysis exports."""
    m1 = json.loads(Path(m1_preparation).read_text())
    if m1.get('schema') != M1_SCHEMA:
        raise ValueError('MACE feature (M1) preparation required')
    arm = next(a for a in m1['arms'] if a['name'] == M1_ARM)
    analysis = Path(m1_preparation).parent / 'analysis'
    report = json.loads((analysis / 'report.json').read_text())
    entry = report['arms'][M1_ARM]
    if entry['verdict'] != 'effective' or not entry['export']['exported']:
        raise ValueError('M0 attributes an effective, exported M1')
    exports = analysis / 'exports'
    evidence = {'samples': exports / f'{M1_ARM}_samples.csv', 'spectra': exports / f'{M1_ARM}_spectra.npz',
                'summary': exports / f'{M1_ARM}_summary.json', 'analysis': analysis / 'report.json',
                'train_samples': analysis / 'train_inference' / f"{Path(arm['run_dir']).name}_train_samples.csv",
                'best_checkpoint': Path(arm['run_dir']) / 'checkpoint_best.pth', 'history': Path(arm['history'])}
    return {'label': M1_ARM, 'run_dir': arm['run_dir'], 'm1_preparation': str(m1_preparation),
            'selected_epoch': entry['export']['best_epoch'], 'best_score': entry['export']['best_score'],
            'external_atom_features': arm['external_atom_features'], 'summary_values': entry['summary'],
            **{key: str(path) for key, path in evidence.items()},
            'input_sha256': {str(path): file_sha256(path) for path in evidence.values()}}


def prepare(m1_preparation, features, out, tag_prefix):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'use a new output directory: {out}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag_prefix):
        raise ValueError('isolated alphanumeric tag prefix required')
    m1 = m1_control(m1_preparation)
    m1_prep = json.loads(Path(m1_preparation).read_text())
    head_preparation = m1_prep['completed_control']['head_preparation']
    control = reference_control(head_preparation)
    reference = ExperimentConfig(**control['configuration'])
    contract = feature_contract(input_path(features), input_path(reference.data_dir))
    manifest = load_manifest(input_path(features))
    source = manifest.get('derived_from', {})
    if source.get('manifest_sha256') != m1['external_atom_features']['manifest_sha256']:
        raise ValueError('M0 features must be derived from the M1 features')
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
        arms.append({'name': name, 'arm': 'M0', 'seed': 42, 'step': 1, 'external_atom_features': contract,
                     'changed_training_fields': sorted(changed), 'configuration': asdict(cfg),
                     'run_dir': str(run), 'history': str(history), 'command': training_command(cfg)})
    protocol_bytes = input_path(reference.spectral_balance_config).read_bytes()
    feature_root = input_path(features)
    m1_inputs = {p: d for p, d in m1_prep['input_sha256'].items() if not p.endswith('.py')}
    hashes = {**m1_inputs, **control['input_sha256'], **m1['input_sha256'],
              str(m1_preparation): file_sha256(m1_preparation),
              str(feature_root / 'manifest.json'): file_sha256(feature_root / 'manifest.json'),
              **{str(feature_root / f'{split}.npy'): digest for split, digest in contract['split_sha256'].items()}}
    sources = source_paths()
    hashes.update({str(p): file_sha256(p) for p in sources})
    if any(file_sha256(p) != digest for p, digest in hashes.items()):
        raise ValueError('an input changed during preparation')
    out.mkdir(parents=True, exist_ok=False)
    path.write_bytes(protocol_bytes)
    hashes[str(path.resolve())] = file_sha256(path)
    for source_file in sources:
        target = out / 'source_snapshot' / source_file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_file.read_bytes())
    report = {'schema': SCHEMA, 'training_started': False, 'tag_prefix': tag_prefix, 'design': DESIGN,
              'completed_control': control, 'completed_m1': m1,
              'protocol_copy_identical': path.read_bytes() == protocol_bytes,
              'arms': arms, 'step1_order': [a['name'] for a in arms], 'input_sha256': hashes,
              'not_a_new_selection_or_adoption': True}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m1-preparation', default='results/runs/mace_features_20261108/report.json')
    parser.add_argument('--features', default='data/external/mace_mp0b3_medium_elemmean_v1')
    parser.add_argument('--out', required=True)
    parser.add_argument('--tag-prefix', required=True)
    args = parser.parse_args()
    report = prepare(input_path(args.m1_preparation), input_path(args.features).resolve(), input_path(args.out),
                     args.tag_prefix)
    for arm in report['arms']:
        print(arm['name'], shlex.join(arm['command']))


if __name__ == '__main__':
    main()
