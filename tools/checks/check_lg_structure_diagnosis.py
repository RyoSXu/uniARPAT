"""Independently verify saved diagnosis exports and tighten structure matching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from tools.eval.lg_structure_diagnosis import RUNS
from tools.eval.lg_valid_export import load_run
from utils.builder import ConfigBuilder
from utils.paths import file_sha256

REFERENCES = {
    'baseline35': 'lg_baseline_q1v1_20261008/exports/baseline_samples.csv',
    'bp35': 'lg_global_depth_20261008/exports/bp_staged_g6_samples.csv',
    'baseline100': 'lg_long_budget_20261008/exports/baseline_e100_samples.csv',
    'bp100': 'lg_long_budget_20261008/exports/deep_g6_e100_samples.csv',
}


def verify(directory, output):
    if output.exists():
        raise FileExistsError(f'refusing existing verification: {output}')
    report = json.loads((directory / 'report.json').read_text())
    if report['smoke'] or set(report['groups']) != set(RUNS):
        raise ValueError('verification requires the completed four-model full diagnosis')
    result = {'source_sha256': file_sha256(Path(__file__)),
              'report_sha256': file_sha256(directory / 'report.json'),
              'valid_parity': {}, 'pair_formula': {}, 'strict_matches': {}}
    for label, reference in REFERENCES.items():
        original = pd.read_csv(ROOT / 'results/runs' / reference).set_index('material_id').sort_index()
        current = pd.read_csv(directory / f'{label}_valid_samples.csv').set_index('material_id').sort_index()
        if list(original.index) != list(current.index):
            raise ValueError('valid material identities differ: ' + label)
        differences = {}
        for column in current:
            if column.startswith(('blind_', 'oracle_')):
                np.testing.assert_allclose(current[column], original[column], rtol=2e-4, atol=2e-5,
                                           err_msg=label + ':' + column)
                if '_r2_' in column:
                    np.testing.assert_array_equal(current[column] < 0, original[column] < 0)
                differences[column] = float(np.max(np.abs(current[column] - original[column])))
        result['valid_parity'][label] = {'n': len(current), 'rtol': 2e-4, 'atol': 2e-5,
                                         'failure_flags_identical': True,
                                         'max_abs_by_metric': differences}
        for split in ('train', 'valid'):
            with np.load(directory / f'{label}_{split}_spectra.npz') as spectra:
                pairs = pd.read_csv(directory / f'{label}_{split}_reduced_edos_pairs.csv')
                positions = {mid: i for i, mid in enumerate(spectra['material_ids'].tolist())}
                i = np.array([positions[mid] for mid in pairs.left_id])
                j = np.array([positions[mid] for mid in pairs.right_id])
                residual = spectra['pred_shape_edos'].astype(float) - spectra['tgt_shape_edos'].astype(float)
                error = .5 * np.abs(residual[i] - residual[j]).sum(-1)
                np.testing.assert_allclose(error, pairs.difference_error, rtol=1e-12, atol=1e-12)
                result['pair_formula'][label + '_' + split] = {
                    'pairs': len(pairs), 'max_abs': float(np.max(np.abs(error - pairs.difference_error)))}
    saved, _, _ = load_run(ROOT / 'results/runs' / RUNS['bp35'])
    builder = ConfigBuilder(**saved['config'])
    settings = dict(ltol=.01, stol=.05, angle_tol=.5, scale=False)
    matcher = StructureMatcher(**settings, primitive_cell=True, attempt_supercell=False)
    for split in ('train', 'valid'):
        dataset = builder.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
        positions = {str(mid): i for i, mid in enumerate(dataset.material_ids)}
        candidates = report['groups']['baseline35']['splits'][split]['structure_matches']['matches']
        checked = []
        for pair in candidates:
            structures = []
            for key in ('left_id', 'right_id'):
                z, geometry = dataset.crystals[int(dataset.structure_indices[positions[pair[key]]])]
                structures.append(Structure(geometry.cell.numpy(), z.numpy(), geometry.frac_coords.numpy()))
            left, right = structures
            rms = matcher.get_rms_dist(left, right)
            checked.append({**pair, 'formula': left.composition.reduced_formula,
                            'strict_match': rms is not None,
                            'normalized_rms': float(rms[0]) if rms is not None else None,
                            'volume_per_atom_ratio': float((left.volume / len(left)) / (right.volume / len(right)))})
        result['strict_matches'][split] = {
            'candidate_pairs': len(checked), 'strict_count': sum(row['strict_match'] for row in checked),
            'settings': settings, 'pairs': checked}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'verified': str(output), 'valid_materials_per_model': 2313,
                      'strict_matches': {split: value['strict_count']
                                         for split, value in result['strict_matches'].items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    verify(args.diagnosis_dir, args.out)
