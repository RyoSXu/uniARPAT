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
    ENERGY_METRIC, normalized_spectral_error, prepare_balance_protocol,
)


def prepare(data_dir, analysis_dir, out, tag_prefix):
    out, data_dir, analysis_dir = Path(out), Path(data_dir), Path(analysis_dir)
    if out.exists():
        raise FileExistsError(f"use a new output directory: {out}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("isolated alphanumeric tag prefix required")
    protocol = prepare_balance_protocol(data_dir, analysis_dir)
    hashes = {str(data_dir / 'manifest.json'): file_sha256(data_dir / 'manifest.json'),
              **{str(data_dir / name): digest for name, digest in protocol['train_sha256'].items()},
              **{str(data_dir / name): digest for name, digest in protocol['valid_guard_sha256'].items()}}
    for name in ('A64_valid_samples.csv', 'report.json'):
        hashes[str(analysis_dir / name)] = file_sha256(analysis_dir / name)
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
                    'tools/train.py', 'tools/eval/spectral_balance_prepare.py', 'tools/eval/lg_valid_export.py')
    for name in source_paths:
        hashes[str(ROOT / name)] = file_sha256(ROOT / name)
    if any(file_sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError('an input changed while preparing the protocol')
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'protocol.json'
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    cfg = ExperimentConfig(**{**FROZEN, 'local_route': 'I3', 'scale_mode': 'per_atom',
        'selection_metric': ENERGY_METRIC, 'spectral_balance_config': str(path.resolve()),
        'data_dir': str(data_dir.resolve()), 'tag': f'_sbal_{tag_prefix}_a64_s42_e35'})
    report = {'schema': 'joint-spectral-balance-preparation-v1', 'training_started': False,
              'normalizers_fit_on': 'effective train only', 'reference': protocol['reference'],
              'existing_selected_checkpoint_diagnostics': diagnostics,
              'not_a_new_selection_or_adoption': True, 'input_sha256': hashes,
              'configuration': asdict(cfg), 'command': training_command(cfg)}
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
    args = parser.parse_args()
    result = prepare(input_path(args.data_dir), input_path(args.analysis_dir), input_path(args.out), args.tag_prefix)
    print(shlex.join(result['command']))


if __name__ == '__main__':
    main()
