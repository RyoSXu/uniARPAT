"""Compare uMLIP+phonopy phonon DOS with the learned models on the same valid materials (phase A baseline).

Reads one or more `tools/data/umlip_phonon_baseline.py` outputs and the frozen
valid exports of h2 (working point) and M1. On the materials every baseline
computed, it reports the phDOS blind E with the training formula (per input-cell
atom, bin width, train energy normalization), an oracle-scale E (prediction
rescaled to the target window total: shape only), median R2, paired bootstrap
differences against h2 and M1, coverage, failures, imaginary-mode counts and
cost. Valid only; never reads test; not used for any selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.eval.reference_controls_analyze import interval

DRAWS = 2000
SEED = 20261109
IMAGINARY_CM = -20.0
MODELS = {'h2': 'results/runs/phdos_head_20261106/analysis/exports/h2_s42_spectra.npz',
          'm1': 'results/runs/mace_features_20261108/analysis/exports/m1_s42_spectra.npz'}
PROTOCOL = 'results/runs/mace_features_20261108/protocol.json'


def errors(prediction, target, atoms, entry):
    prediction, target = prediction.astype(np.float64), target.astype(np.float64)
    energy = np.square((prediction - target) / atoms[:, None]).sum(-1) * entry['bin_width'] / entry['energy']
    residual = np.square(target - prediction).sum(-1)
    total = np.square(target - target.mean(-1, keepdims=True)).sum(-1)
    return energy, 1.0 - residual / (total + 1e-8)


def oracle_scale(prediction, target):
    total = prediction.sum(-1, keepdims=True)
    return np.where(total > 0, prediction * target.sum(-1, keepdims=True) / np.maximum(total, 1e-12), prediction)


def load_baseline(path, ids):
    path = Path(path)
    rows = [json.loads(line) for line in (path / 'records.jsonl').read_text().splitlines() if line.strip()]
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['data']['split'] != 'valid':
        raise ValueError('baseline must be computed on valid')
    position = {material: i for i, material in enumerate(ids)}
    dos = {}
    for row in rows:
        if row['position'] != position[row['material_id']]:
            raise ValueError(f"{row['material_id']}: position differs from the valid order")
        if row['ok']:
            dos[row['material_id']] = np.asarray(row['dos'], dtype=np.float64)
    return manifest, rows, dos


def compare(baselines, out):
    out = Path(out)
    if (out / 'report.json').exists():
        raise FileExistsError(f'refusing existing comparison: {out}')
    protocol = json.loads((ROOT / PROTOCOL).read_text())['tasks']['phdos']
    valid = ROOT / 'data/crystal_q1_v1/valid'
    ids = np.load(valid / 'valid_index.npy', allow_pickle=False).astype(str)
    atoms = np.load(valid / 'n_atoms_valid.npy').astype(np.float64)
    target = np.load(valid / 'phdos_tgtdos_valid.npy').astype(np.float64)
    models = {}
    for name, path in MODELS.items():
        with np.load(ROOT / path, allow_pickle=False) as data:
            if data['material_ids'].astype(str).tolist() != ids.tolist():
                raise ValueError(f'{name} export does not follow the valid order')
            if not np.allclose(data['tgt_phdos'], target, atol=1e-6):
                raise ValueError(f'{name} export labels differ from the dataset')
            models[name] = data['pred_blind_phdos'].astype(np.float64)
    loaded = {name: load_baseline(path, ids) for name, path in baselines.items()}
    common = sorted(set.intersection(*(set(dos) for _, _, dos in loaded.values())), key=list(ids).index)
    index = np.asarray([list(ids).index(m) for m in common])
    draws = np.random.default_rng(SEED).integers(0, len(index), size=(DRAWS, len(index)))
    t, a = target[index], atoms[index]
    report = {'schema': 'umlip-phonon-compare-v1', 'valid_materials': len(ids), 'common_materials': len(index),
              'scope': 'valid only; phDOS; learned models are blind (predicted window total); not a selection',
              'imaginary_threshold_cm': IMAGINARY_CM, 'methods': {}}
    values = {}
    for name, prediction in models.items():
        energy, r2 = errors(prediction[index], t, a, protocol)
        oracle, _ = errors(oracle_scale(prediction[index], t), t, a, protocol)
        values[name] = energy
        report['methods'][name] = {'kind': 'learned', 'phdos_mean_E': float(energy.mean()),
                                   'phdos_median_E': float(np.median(energy)), 'oracle_scale_mean_E': float(oracle.mean()),
                                   'median_R2': float(np.median(r2))}
    for name, (manifest, rows, dos) in loaded.items():
        prediction = np.stack([dos[m] for m in common])
        energy, r2 = errors(prediction, t, a, protocol)
        oracle, _ = errors(oracle_scale(prediction, t), t, a, protocol)
        values[name] = energy
        ok = [r for r in rows if r['ok']]
        minimum = np.asarray([r['min_mesh_frequency_cm'] for r in ok])
        report['methods'][name] = {
            'kind': 'umlip+phonopy', 'settings': manifest['settings'], 'model_sha256': manifest['model']['sha256'],
            'computed': len(rows), 'ok': len(ok), 'failures': [r['error'] for r in rows if not r['ok']][:20],
            'imaginary_fraction': float((minimum < IMAGINARY_CM).mean()) if len(ok) else None,
            'relax_unconverged': int(sum(1 for r in ok if r.get('relax_converged') is False)),
            'seconds_per_material': float(np.mean([r['seconds'] for r in rows])),
            'phdos_mean_E': float(energy.mean()), 'phdos_median_E': float(np.median(energy)),
            'oracle_scale_mean_E': float(oracle.mean()), 'median_R2': float(np.median(r2)),
            'paired_vs_h2': interval(values['h2'], energy, draws), 'paired_vs_m1': interval(values['m1'], energy, draws),
            'wins_vs_h2_fraction': float((energy < values['h2']).mean())}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', action='append', required=True, help='name=directory (repeatable)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    baselines = dict(item.split('=', 1) for item in args.baseline)
    report = compare(baselines, args.out)
    print('common materials', report['common_materials'])
    for name, entry in report['methods'].items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()
                     if k in ('phdos_mean_E', 'phdos_median_E', 'oracle_scale_mean_E', 'median_R2', 'ok',
                              'imaginary_fraction', 'seconds_per_material', 'wins_vs_h2_fraction')})


if __name__ == '__main__':
    main()
