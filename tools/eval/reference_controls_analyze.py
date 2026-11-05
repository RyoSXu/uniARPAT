"""Analyze the LR2 E+S reference controls with the rules fixed in the 2026-11-03 design.

Valid only. Exports every completed arm through ``lg_valid_export.export`` (the
adopted R42 export is reused), recomputes per-material blind errors from the
exported physical spectra and reports seed spread, same-recipe ensembling,
structure contribution and joint versus single-task supervision.
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
import pandas as pd

from tools.eval.reference_controls_prepare import SCHEMA as PREPARATION_SCHEMA
from utils.paths import file_sha256, input_path

TASKS = ('edos', 'phdos')
DRAWS = 2000
SEED = 20261103
LR2_ADOPTION_GAP = 0.00604192  # LR2 minus A64 E+S joint mean E, 2026-10-30
REPLICATES = ('lr2_s42', 'lr2_s43', 'lr2_s44')
BLIND = ('s0_s42', 's0_s43', 's0_s44')


def per_material(spectra, atoms, protocol):
    """Blind E and R2 per material from physical spectra; same formulas as training."""
    result = {}
    for task in TASKS:
        entry = protocol['tasks'][task]
        prediction = spectra[f'pred_blind_{task}'].astype(np.float64)
        target = spectra[f'tgt_{task}'].astype(np.float64)
        error = (np.square((prediction - target) / atoms[:, None]).sum(-1)
                 * entry['bin_width'] / entry['energy'])
        residual = np.square(target - prediction).sum(-1)
        total = np.square(target - target.mean(-1, keepdims=True)).sum(-1)
        result[f'energy_{task}'] = error
        result[f'r2_{task}'] = 1.0 - residual / (total + 1e-8)
    result['energy_joint'] = .5 * (result['energy_edos'] + result['energy_phdos'])
    return result


def summarize(values):
    row = {'joint_mean_E': float(values['energy_joint'].mean())}
    for task in TASKS:
        energy, r2 = values[f'energy_{task}'], values[f'r2_{task}']
        row.update({f'{task}_mean_E': float(energy.mean()), f'{task}_median_E': float(np.median(energy)),
                    f'{task}_p90_E': float(np.quantile(energy, .9)), f'{task}_median_R2': float(np.median(r2)),
                    f'{task}_p10_R2': float(np.quantile(r2, .1)), f'{task}_failures': int((r2 < 0).sum())})
    return row


def interval(left, right, draws):
    """Mean(right - left) with a paired material bootstrap on a shared index matrix."""
    delta = np.asarray(right, np.float64) - np.asarray(left, np.float64)
    boot = delta[draws].mean(1)
    low, high = np.quantile(boot, [.025, .975])
    return {'delta': float(delta.mean()), 'ci95': [float(low), float(high)],
            'excludes_zero': bool(low > 0 or high < 0)}


def spread(rows, key):
    values = np.asarray([row[key] for row in rows])
    return {'values': values.tolist(), 'mean': float(values.mean()), 'range': float(np.ptp(values)),
            'sd': float(values.std(ddof=1)) if len(values) > 1 else None}


def same_composition(spectra, ids, keys):
    from tools.eval.lg_structure_diagnosis import pair_errors

    report = {}
    for grouping, labels in keys.items():
        for task in TASKS:
            frame = pair_errors(spectra[f'pred_shape_{task}'], spectra[f'tgt_shape_{task}'], labels, ids)
            grouped = frame.groupby('composition', sort=True)
            error, target = grouped.difference_error.mean(), grouped.target_tv.mean()
            report[f'{grouping}_{task}'] = {
                'pairs': int(len(frame)), 'groups': int(len(error)),
                'group_mean_error': float(error.mean()), 'zero_response_error': float(target.mean()),
                'skill_vs_zero_response': float(1 - error.mean() / target.mean()),
                'predicted_tv_median': float(frame.predicted_tv.median())}
    return report


def history_optimum(path):
    history = pd.read_csv(path)
    row = {}
    for name, column in (('joint', 'balanced_blind_energy'), ('edos', 'blind_energy_edos_mean'),
                         ('phdos', 'blind_energy_phdos_mean')):
        best = history.sort_values([column, 'epoch'], kind='stable').iloc[0]
        row[f'unguarded_{name}_min'] = {'epoch': int(best.epoch), 'value': float(best[column])}
    return row


def analyze(preparation_path, out):
    from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position
    from tools.eval.lg_valid_export import export, load_run
    from utils.builder import ConfigBuilder

    out = Path(out)
    if (out / 'report.json').exists():
        raise FileExistsError(f'refusing existing analysis: {out}')
    preparation = json.loads(Path(preparation_path).read_text())
    if preparation.get('schema') != PREPARATION_SCHEMA:
        raise ValueError('not a reference-controls preparation report')
    protocol = json.loads((Path(preparation_path).parent / 'protocol.json').read_text())
    control = preparation['completed_control']
    for path, digest in control['input_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'adopted control input changed: {path}')
    saved, _, _ = load_run(Path(control['run_dir']))
    dataset = ConfigBuilder(**saved['config']).get_dataset(split='valid', dos_minmax=True, dos_sumnorm=True)
    ids = [str(value) for value in dataset.material_ids]
    atoms = np.asarray([len(dataset.crystals[int(dataset.structure_indices[i])][0]) for i in range(len(ids))],
                       dtype=np.float64)
    if not np.array_equal(atoms, np.load(ROOT / 'data/crystal_q1_v1/valid/n_atoms_valid.npy').astype(np.float64)):
        raise ValueError('valid atom counts differ from the frozen sidecar')
    keys = composition_keys_by_position(dataset, range(len(ids)))

    exports = out / 'exports'
    sources = {'lr2_s42': (Path(control['samples']), Path(control['spectra']))}
    for arm in preparation['arms']:
        samples, summary, spectra = (exports / f"{arm['name']}_{suffix}"
                                     for suffix in ('samples.csv', 'summary.json', 'spectra.npz'))
        if not spectra.exists():
            export(Path(arm['run_dir']), samples, summary, spectra)
        sources[arm['name']] = (samples, spectra)

    values, arrays, checks = {}, {}, {}
    for name, (samples_path, spectra_path) in sources.items():
        with np.load(spectra_path, allow_pickle=False) as data:
            spectra = {key: data[key] for key in data.files}
        if spectra['material_ids'].tolist() != ids:
            raise ValueError(f'{name} export does not follow the valid dataset order')
        frame = pd.read_csv(samples_path).set_index('material_id').loc[ids]
        computed = per_material(spectra, atoms, protocol)
        task_error = max(float(np.max(np.abs(computed[f'energy_{t}'] - frame[f'blind_energy_{t}'].to_numpy())
                                      / np.maximum(1e-6, np.abs(frame[f'blind_energy_{t}'].to_numpy()))))
                         for t in TASKS)
        if task_error > 1e-3:
            raise ValueError(f'{name}: recomputed blind energy differs from its export ({task_error:.2e})')
        checks[name] = {'max_relative_energy_difference': task_error}
        values[name], arrays[name] = computed, spectra
    targets = arrays['lr2_s42']
    for name, spectra in arrays.items():
        for key in ('tgt_edos', 'tgt_phdos'):
            if not np.allclose(spectra[key], targets[key], rtol=0, atol=1e-7):
                raise ValueError(f'{name} labels differ from the adopted control')

    draws = np.random.default_rng(SEED).integers(0, len(ids), size=(DRAWS, len(ids)))
    summaries = {name: summarize(value) for name, value in values.items()}
    report = {'schema': 'lr2-es-reference-controls-analysis-v1', 'valid_materials': len(ids),
              'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'material, shared index matrix'},
              'export_checks': checks, 'arms': summaries}

    # 1. Training randomness of the adopted recipe and of S0.
    replicate_rows = [summaries[name] for name in REPLICATES]
    keys_of_interest = ['joint_mean_E'] + [f'{t}_{m}' for t in TASKS for m in ('mean_E', 'median_R2', 'failures')]
    report['seed_spread'] = {
        'lr2': {key: spread(replicate_rows, key) for key in keys_of_interest},
        's0': {key: spread([summaries[name] for name in BLIND], key) for key in keys_of_interest}}
    joint_range = report['seed_spread']['lr2']['joint_mean_E']['range']
    report['seed_rule'] = {
        'lr2_minus_a64_gap': -LR2_ADOPTION_GAP, 'lr2_joint_range': joint_range,
        'adoption_established_across_seeds': joint_range < LR2_ADOPTION_GAP,
        'action': ('keep adoption; later candidates must exceed the same-recipe seed range (user to confirm)'
                   if joint_range < LR2_ADOPTION_GAP else
                   'propose A64 E+S seeds 43/44 as phase 2 (user decision); do not run automatically')}

    # 2. Same-recipe ensemble of the three LR2 replicates.
    ensemble = {key: np.mean([arrays[name][key].astype(np.float64) for name in REPLICATES], 0)
                for key in ('pred_blind_edos', 'pred_blind_phdos')}
    ensemble.update({key: targets[key] for key in ('tgt_edos', 'tgt_phdos')})
    ensemble_values = per_material(ensemble, atoms, protocol)
    member_mean = {key: np.mean([values[name][key] for name in REPLICATES], 0) for key in ensemble_values}
    best = min(REPLICATES, key=lambda name: summaries[name]['joint_mean_E'])
    report['ensemble'] = {
        'members': list(REPLICATES), 'summary': summarize(ensemble_values), 'best_member': best,
        'vs_member_mean': {f'{t}_E': interval(member_mean[f'energy_{t}'], ensemble_values[f'energy_{t}'], draws)
                           for t in ('joint', *TASKS)},
        'vs_best_member': {f'{t}_E': interval(values[best][f'energy_{t}'], ensemble_values[f'energy_{t}'], draws)
                           for t in ('joint', *TASKS)}}

    # 3. Structure contribution: S0 minus LR2, seed-averaged per material.
    structure = {}
    for task in ('joint', *TASKS):
        r = np.mean([values[name][f'energy_{task}'] for name in REPLICATES], 0)
        s = np.mean([values[name][f'energy_{task}'] for name in BLIND], 0)
        key = 'joint_mean_E' if task == 'joint' else f'{task}_mean_E'
        pairwise = [[summaries[b][key] - summaries[a][key] for a in REPLICATES] for b in BLIND]
        separated = min(summaries[b][key] for b in BLIND) > max(summaries[a][key] for a in REPLICATES)
        result = interval(r, s, draws)
        structure[task] = {**result, 'pairwise_s0_minus_lr2': pairwise, 'all_s0_worse_than_all_lr2': separated,
                           'verdict': ('structure contributes' if result['excludes_zero'] and separated
                                       and result['delta'] > 0 else
                                       'directional only' if result['excludes_zero'] or separated
                                       else 'not distinguished')}
    report['structure_contribution'] = structure

    # 4. Joint versus single-task supervision (single seed; seed range is the yardstick).
    single = {}
    for name, task in (('edos_s42', 'edos'), ('phdos_s42', 'phdos')):
        r = np.mean([values[arm][f'energy_{task}'] for arm in REPLICATES], 0)
        result = interval(r, values[name][f'energy_{task}'], draws)
        seed_range = report['seed_spread']['lr2'][f'{task}_mean_E']
        lo, hi = min(seed_range['values']), max(seed_range['values'])
        inside = lo <= summaries[name][f'{task}_mean_E'] <= hi
        single[task] = {**result, 'single_task_mean_E': summaries[name][f'{task}_mean_E'],
                        'joint_seed_range': [lo, hi], 'inside_joint_seed_range': inside,
                        'verdict': ('not distinguished' if inside or not result['excludes_zero'] else
                                    'joint better' if result['delta'] > 0 else 'single task better')}
    report['joint_vs_single_task'] = single

    report['history_unguarded_optima'] = {
        arm['name']: history_optimum(arm['history']) for arm in preparation['arms']}
    report['history_unguarded_optima']['lr2_s42'] = history_optimum(
        ROOT / f"results/reports/history_m1{control['configuration']['tag']}.csv")
    report['same_composition'] = {name: same_composition(arrays[name], ids, keys) for name in arrays}
    # Ensemble shape = normalized averaged blind spectrum (the exported shapes are per member).
    ensemble_shapes = {f'pred_shape_{t}': ensemble[f'pred_blind_{t}']
                       / np.maximum(ensemble[f'pred_blind_{t}'].sum(-1, keepdims=True), 1e-30) for t in TASKS}
    ensemble_shapes.update({f'tgt_shape_{t}': targets[f'tgt_shape_{t}'] for t in TASKS})
    report['same_composition']['lr2_ensemble'] = same_composition(ensemble_shapes, ids, keys)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = analyze(input_path(args.preparation), input_path(args.out))
    print(json.dumps({key: report[key] for key in ('seed_rule', 'structure_contribution', 'joint_vs_single_task')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
