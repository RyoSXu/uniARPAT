"""Physical quantities derived from predicted and target DOS on valid (phase A evaluation).

phDOS -> vibrational heat capacity C_v and entropy S at a temperature (default
300 K), harmonic formulas on the positive-frequency part, each spectrum
normalized to 3 modes per input-cell atom (label window totals are not 3N, so
the thermodynamics compare spectral shapes on a common mode count).
Reported per mole of atoms (J/K/mol-atom).

eDOS (E - E_F in eV, window [-6, 6]) -> DOS at the Fermi level per atom (mean
of the two bins adjacent to E = 0), a DOS-based gap (width of the run of bins
below a threshold that starts in a bin adjacent to E_F; MP puts E_F at the
valence-band top of most semiconductors) and the non-metal call (gap >= 0.2 eV).
Thresholds are fixed here, not tuned on these results' outcome; several
are reported to show sensitivity. These are DOS-derived diagnostics, not
independent band-gap labels.

Inputs are the frozen valid exports (blind spectra) and optionally the
uMLIP phonon baselines (phDOS only). Valid only; never reads test.
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
from tools.eval.umlip_phonon_compare import load_baseline

DRAWS = 2000
SEED = 20261109
R_GAS = 8.314462618  # J/K/mol
HC_OVER_K = 1.438776877  # cm K
PH_WINDOW, PH_BINS = (-280.0, 980.0), 64
E_WINDOW, E_BINS = (-6.0, 6.0), 128
GAP_THRESHOLDS = (0.01, 0.05)  # states/eV/atom
NONMETAL_GAP = 0.2  # eV; a DOS gap of at least about two bins calls a non-metal
REFERENCE = 'h2'


def centers(window, bins):
    edges = np.linspace(window[0], window[1], bins + 1)
    return .5 * (edges[1:] + edges[:-1]), edges[1] - edges[0]


def thermo(dos, atoms, temperature):
    """Harmonic C_v and S per mole of atoms from binned mode densities (per input cell)."""
    omega, width = centers(PH_WINDOW, PH_BINS)
    positive = omega > 0
    g = np.clip(dos[:, positive], 0, None)
    modes = g.sum(-1) * width
    valid = modes > 0
    weight = np.where(valid[:, None], g * width / np.maximum(modes, 1e-12)[:, None] * 3.0, 0.0)  # 3 modes per atom
    x = HC_OVER_K * omega[positive] / temperature
    cv_mode = x ** 2 * np.exp(x) / np.expm1(x) ** 2
    s_mode = x / np.expm1(x) - np.log1p(-np.exp(-x))
    return {'cv': (weight * cv_mode).sum(-1) * R_GAS, 's': (weight * s_mode).sum(-1) * R_GAS, 'valid': valid}


def electronic(dos, atoms):
    energy, _ = centers(E_WINDOW, E_BINS)
    per_atom = np.clip(dos, 0, None) / atoms[:, None]
    below, above = int(np.searchsorted(energy, 0.0)) - 1, int(np.searchsorted(energy, 0.0))
    at_fermi = .5 * (per_atom[:, below] + per_atom[:, above])
    result = {'dos_at_fermi': at_fermi}
    width = energy[1] - energy[0]
    for threshold in GAP_THRESHOLDS:
        gaps = np.zeros(len(dos))
        for i, row in enumerate(per_atom):
            # MP places E_F at the valence-band top for most semiconductors, so the gap usually
            # starts in the first bin above E_F; a mid-gap E_F leaves both adjacent bins empty.
            start = above if row[above] < threshold else (below if row[below] < threshold else None)
            if start is None:
                continue
            low, high = start, start
            while low - 1 >= 0 and row[low - 1] < threshold:
                low -= 1
            while high + 1 < E_BINS and row[high + 1] < threshold:
                high += 1
            gaps[i] = (high - low + 1) * width
        result[f'gap_{threshold}'] = gaps
    return result


def classification(predicted_gap, target_gap):
    pred, true = predicted_gap >= NONMETAL_GAP, target_gap >= NONMETAL_GAP
    tp, tn = int((pred & true).sum()), int((~pred & ~true).sum())
    fp, fn = int((pred & ~true).sum()), int((~pred & true).sum())
    return {'accuracy': (tp + tn) / len(true), 'nonmetal_precision': tp / max(tp + fp, 1),
            'nonmetal_recall': tp / max(tp + fn, 1), 'target_nonmetal_fraction': float(true.mean())}


def evaluate(exports, baselines, out, temperature):
    out = Path(out)
    if (out / 'report.json').exists():
        raise FileExistsError(f'refusing existing report: {out}')
    valid = ROOT / 'data/crystal_q1_v1/valid'
    ids = np.load(valid / 'valid_index.npy', allow_pickle=False).astype(str)
    atoms = np.load(valid / 'n_atoms_valid.npy').astype(np.float64)
    target = {'edos': np.load(valid / 'edos_tgtdos_valid.npy').astype(np.float64),
              'phdos': np.load(valid / 'phdos_tgtdos_valid.npy').astype(np.float64)}
    predictions = {}
    for name, path in exports.items():
        with np.load(path, allow_pickle=False) as data:
            if data['material_ids'].astype(str).tolist() != ids.tolist():
                raise ValueError(f'{name} export does not follow the valid order')
            for task in ('edos', 'phdos'):
                if not np.allclose(data[f'tgt_{task}'], target[task], atol=1e-6):
                    raise ValueError(f'{name} labels differ from the dataset')
            predictions[name] = {task: data[f'pred_blind_{task}'].astype(np.float64) for task in ('edos', 'phdos')}
    if REFERENCE not in predictions:
        raise ValueError('the h2 export is the paired reference')
    draws_all = np.random.default_rng(SEED).integers(0, len(ids), size=(DRAWS, len(ids)))
    report = {'schema': 'physical-quantities-v1', 'temperature_K': temperature, 'valid_materials': len(ids),
              'units': {'cv': 'J/K/mol-atom', 's': 'J/K/mol-atom', 'dos_at_fermi': 'states/eV/atom', 'gap': 'eV'},
              'gap_thresholds': list(GAP_THRESHOLDS), 'phdos': {}, 'edos': {}}

    t_thermo = thermo(target['phdos'], atoms, temperature)
    usable = t_thermo['valid']
    report['phdos_usable_materials'] = int(usable.sum())
    errors = {}
    for name, prediction in predictions.items():
        p = thermo(prediction['phdos'], atoms, temperature)
        entry, errors[name] = {}, {}
        for key in ('cv', 's'):
            err = np.abs(p[key] - t_thermo[key])[usable]
            errors[name][key] = err
            entry[f'{key}_mae'] = float(err.mean())
            entry[f'{key}_median_abs'] = float(np.median(err))
        report['phdos'][name] = entry
    draws = np.random.default_rng(SEED).integers(0, int(usable.sum()), size=(DRAWS, int(usable.sum())))
    for name in predictions:
        if name != REFERENCE:
            for key in ('cv', 's'):
                report['phdos'][name][f'{key}_paired_vs_{REFERENCE}'] = interval(errors[REFERENCE][key], errors[name][key], draws)
    for name, path in baselines.items():
        _, rows, dos = load_baseline(path, ids)
        index = np.asarray([i for i, m in enumerate(ids) if m in dos and usable[i]])
        p = thermo(np.stack([dos[ids[i]] for i in index]), atoms[index], temperature)
        sub = np.random.default_rng(SEED).integers(0, len(index), size=(DRAWS, len(index)))
        entry = {'materials': int(len(index))}
        for key in ('cv', 's'):
            err = np.abs(p[key] - t_thermo[key][index])
            ref = np.abs(thermo(predictions[REFERENCE]['phdos'][index], atoms[index], temperature)[key] - t_thermo[key][index])
            entry.update({f'{key}_mae': float(err.mean()), f'{key}_mae_h2_same_materials': float(ref.mean()),
                          f'{key}_paired_vs_{REFERENCE}': interval(ref, err, sub)})
        report['phdos'][f'umlip_{name}'] = entry

    t_e = electronic(target['edos'], atoms)
    for name, prediction in predictions.items():
        p = electronic(prediction['edos'], atoms)
        err = np.abs(p['dos_at_fermi'] - t_e['dos_at_fermi'])
        entry = {'dos_at_fermi_mae': float(err.mean()),
                 'dos_at_fermi_spearman': float(np.corrcoef(np.argsort(np.argsort(p['dos_at_fermi'])),
                                                            np.argsort(np.argsort(t_e['dos_at_fermi'])))[0, 1])}
        for threshold in GAP_THRESHOLDS:
            key = f'gap_{threshold}'
            entry[f'gap_mae_{threshold}'] = float(np.abs(p[key] - t_e[key]).mean())
            entry[f'metal_call_{threshold}'] = classification(p[key], t_e[key])
        if name != REFERENCE:
            ref = np.abs(electronic(predictions[REFERENCE]['edos'], atoms)['dos_at_fermi'] - t_e['dos_at_fermi'])
            entry[f'dos_at_fermi_paired_vs_{REFERENCE}'] = interval(ref, err, draws_all)
        report['edos'][name] = entry
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export', action='append', required=True, help='name=spectra.npz (valid; include h2)')
    parser.add_argument('--baseline', action='append', default=[], help='name=uMLIP phonon directory (optional)')
    parser.add_argument('--temperature', type=float, default=300.0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = evaluate(dict(e.split('=', 1) for e in args.export), dict(b.split('=', 1) for b in args.baseline),
                      args.out, args.temperature)
    print(json.dumps({k: report[k] for k in ('phdos', 'edos')}, indent=1, default=float)[:4000])


if __name__ == '__main__':
    main()
