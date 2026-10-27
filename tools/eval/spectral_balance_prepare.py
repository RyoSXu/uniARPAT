"""Freeze train normalizers and A64 valid guards; print, never launch, training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from tools.eval.local_global_compare import training_command
from utils.experiment_config import ExperimentConfig
from utils.local_structure_experiment import FROZEN
from utils.paths import file_sha256, input_path
from utils.spectral_balance import (
    ENERGY_METRIC, load_balance_protocol, normalized_spectral_error, prepare_balance_protocol,
)
from utils.spectral_balance_experiment import balance_model_contract


def completed_energy_control(preparation, protocol, data_dir):
    """Bind the completed pure-energy control without retraining or reselecting it."""
    from tools.eval.lg_valid_export import load_run, select_best

    preparation = input_path(preparation)
    original = json.loads(preparation.read_text())
    cfg = ExperimentConfig(**original['configuration'])
    if cfg.spectral_shape_aux or input_path(cfg.data_dir) != data_dir.resolve():
        raise ValueError('shape comparison requires the completed pure-energy control on the same data')
    if load_balance_protocol(cfg.spectral_balance_config, cfg.data_dir) != protocol:
        raise ValueError('shape comparison must preserve all normalizers and checkpoint guards')
    run = ROOT / f'results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}'
    saved, params, contract = load_run(run)
    if contract != balance_model_contract(cfg, params):
        raise ValueError('completed control differs from its preparation contract')
    selected = select_best(run, cfg.selection_metric, saved)
    if selected['history'].epoch.tolist() != list(range(1, cfg.epochs + 1)):
        raise ValueError('pure-energy control must have completed its full frozen budget')
    exports = preparation.parent / 'analysis'
    summary_path = exports / 'candidate_valid_summary.json'
    samples_path = exports / 'candidate_valid_samples.csv'
    spectra_path = exports / 'candidate_valid_spectra.npz'
    summary = json.loads(summary_path.read_text())
    if (summary['best_epoch'] != int(selected['row'].epoch)
            or summary['checkpoint_epoch'] != int(selected['row'].epoch)
            or summary['samples'] != protocol['reference']['samples']):
        raise ValueError('control export must describe its selected complete-valid checkpoint')
    paths = [preparation, input_path(cfg.spectral_balance_config), run / 'config_used.yaml',
             run / 'checkpoint_best.pth', summary_path, samples_path, spectra_path,
             ROOT / f'results/reports/history_{cfg.model_name.lower()}{cfg.tag}.csv']
    return {'label': 'energy_only', 'run_dir': str(run), 'selected_epoch': int(selected['row'].epoch),
            'configuration': asdict(cfg),
            'summary': str(summary_path), 'samples': str(samples_path), 'spectra': str(spectra_path),
            'initialization': saved['runtime']['local_global_init'],
            'input_sha256': {str(p): file_sha256(p) for p in paths}}


def prepare(data_dir, analysis_dir, out, tag_prefix, *, shape_aux=False, control_preparation=None):
    out, data_dir, analysis_dir = Path(out), Path(data_dir), Path(analysis_dir)
    if out.exists():
        raise FileExistsError(f"use a new output directory: {out}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("isolated alphanumeric tag prefix required")
    if not isinstance(shape_aux, bool) or shape_aux != (control_preparation is not None):
        raise ValueError('shape auxiliary and its completed control preparation must be supplied together')
    protocol = prepare_balance_protocol(data_dir, analysis_dir)
    control = completed_energy_control(control_preparation, protocol, data_dir) if shape_aux else None
    hashes = {str(data_dir / 'manifest.json'): file_sha256(data_dir / 'manifest.json'),
              **{str(data_dir / name): digest for name, digest in protocol['train_sha256'].items()},
              **{str(data_dir / name): digest for name, digest in protocol['valid_guard_sha256'].items()}}
    for name in ('A64_valid_samples.csv', 'report.json'):
        hashes[str(analysis_dir / name)] = file_sha256(analysis_dir / name)
    if control is not None:
        hashes.update(control['input_sha256'])
        with np.load(control['spectra'], allow_pickle=False) as candidate, np.load(
                analysis_dir / 'A64_valid_spectra.npz', allow_pickle=False) as reference:
            if not np.array_equal(candidate['material_ids'], reference['material_ids']):
                raise ValueError('control and frozen A64 exports have different valid material IDs')
            for task in ('edos', 'phdos'):
                if not np.allclose(candidate[f'tgt_{task}'], reference[f'tgt_{task}'], rtol=0, atol=1e-7):
                    raise ValueError('control and frozen A64 exported targets differ')
    diagnostics = {}
    # These scores describe the existing, originally selected checkpoints only.
    # They cannot reselect a history whose other epoch spectra were not saved.
    for split in ('train', 'valid'):
        ids_path = data_dir / split / f'{split}_index.npy'
        atoms_path = data_dir / split / f'n_atoms_{split}.npy'
        for path in (ids_path, atoms_path):
            hashes[str(path)] = file_sha256(path)
        index = pd.Index(np.load(ids_path, allow_pickle=False))
        atoms = np.load(atoms_path, allow_pickle=False)
        for label in ('A64', 'LR2'):
            path = analysis_dir / f'{label}_{split}_spectra.npz'
            if not path.is_file():
                continue
            hashes[str(path)] = file_sha256(path)
            with np.load(path, allow_pickle=False) as archive:
                order = index.get_indexer(archive['material_ids'])
                if (order < 0).any():
                    raise ValueError('exported IDs do not belong to the requested split')
                row = {}
                for task in ('edos', 'phdos'):
                    error = normalized_spectral_error(
                        torch.from_numpy(archive[f'pred_blind_{task}']).double(),
                        torch.from_numpy(archive[f'tgt_{task}']).double(), atoms[order], protocol['tasks'][task])
                    row[task] = {'mean': float(error.mean()), 'median': float(torch.quantile(error, .5)),
                                 'p90': float(torch.quantile(error, .9))}
                row['balanced_blind_energy'] = .5 * (row['edos']['mean'] + row['phdos']['mean'])
                diagnostics[f'{label}_{split}'] = row
    source_paths = ('utils/spectral_balance.py', 'utils/spectral_balance_experiment.py',
                    'utils/checkpoint_selection.py', 'model/model.py', 'model/transformer.py',
                    'tools/train.py', 'tools/eval/spectral_balance_prepare.py', 'tools/eval/lg_valid_export.py',
                    'model/losses.py', 'utils/experiment_config.py', 'utils/local_global_experiment.py')
    for name in source_paths:
        hashes[str(ROOT / name)] = file_sha256(ROOT / name)
    if any(file_sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError('an input changed while preparing the protocol')
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'protocol.json'
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    cfg = ExperimentConfig(**{**FROZEN, 'local_route': 'I3', 'scale_mode': 'per_atom',
        'selection_metric': ENERGY_METRIC, 'spectral_balance_config': str(path.resolve()),
        'spectral_shape_aux': shape_aux, 'data_dir': str(data_dir.resolve()),
        'tag': f'_sbal_{tag_prefix}_{"shape_" if shape_aux else ""}a64_s42_e35'})
    report = {'schema': 'joint-spectral-balance-preparation-v1', 'training_started': False,
              'normalizers_fit_on': 'effective train only', 'reference': protocol['reference'],
              'existing_selected_checkpoint_diagnostics': diagnostics,
              'not_a_new_selection_or_adoption': True, 'input_sha256': hashes,
              'configuration': asdict(cfg), 'command': training_command(cfg)}
    if control is not None:
        original_cfg = ExperimentConfig(**control['configuration'])
        from utils.local_global_experiment import training_record
        original_training, candidate_training = training_record(original_cfg), training_record(cfg)
        changed = {key for key in original_training.keys() | candidate_training.keys()
                   if original_training.get(key) != candidate_training.get(key)}
        if changed != {'spectral_balance_config', 'spectral_shape_aux'}:
            raise ValueError(f'shape candidate changes more than the auxiliary and protocol path: {changed}')
        report['completed_control'] = control
    for name in source_paths:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/crystal_q1_v1')
    parser.add_argument('--analysis-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tag-prefix', required=True)
    parser.add_argument('--shape-aux', action='store_true')
    parser.add_argument('--control-preparation', help='Completed pure-energy preparation report.json')
    args = parser.parse_args()
    result = prepare(input_path(args.data_dir), input_path(args.analysis_dir), input_path(args.out), args.tag_prefix,
                     shape_aux=args.shape_aux, control_preparation=args.control_preparation)
    print(shlex.join(result['command']))


if __name__ == '__main__':
    main()
