"""Screen and analyze the LR2 E+S generalization arms with the rules fixed in the 2026-11-05 design.

screen: export the step-1 runs (seed 42) and promote an arm when its joint blind
        mean E beats the R42-R44 seed mean by more than the same-recipe seed range
        and its selected checkpoint passes the fixed two-task guards.
final:  for promoted arms, compare the three-seed mean with R42-R44 (paired
        material bootstrap), the three-seed ensembles, and report diagnostics.
Valid only; train inference is used for the train/valid diagnostic, never for selection.
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

from tools.eval.reference_controls_analyze import TASKS, interval, per_material, same_composition, summarize
from tools.eval.structure_generalization_prepare import SCHEMA as PREPARATION_SCHEMA
from utils.paths import file_sha256, input_path

DRAWS = 2000
SEED = 20261105
SEED_RANGE = 0.001319          # LR2 E+S joint mean E range over seeds 42/43/44 (2026-11-04)
REFERENCE_MEAN = 0.2719253      # LR2 E+S joint mean E, mean of seeds 42/43/44
REFERENCE_ENSEMBLE = 0.2569714  # three-seed LR2 E+S ensemble joint mean E
REPLICATES = ('lr2_s42', 'lr2_s43', 'lr2_s44')
BLIND = ('s0_s42', 's0_s43', 's0_s44')
GEOMETRY = ROOT / 'results/runs/structure_use_audit_20261105'


def verified_preparation(path):
    preparation = json.loads(Path(path).read_text())
    if preparation.get('schema') != PREPARATION_SCHEMA:
        raise ValueError('not a structure-generalization preparation report')
    for item, digest in preparation['input_sha256'].items():
        if file_sha256(item) != digest:
            raise ValueError(f'frozen input changed since preparation: {item}')
    return preparation


def context(preparation):
    from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position
    from tools.eval.lg_valid_export import load_run
    from utils.builder import ConfigBuilder

    control = preparation['completed_control']
    saved, _, _ = load_run(Path(control['run_dir']))
    dataset = ConfigBuilder(**saved['config']).get_dataset(split='valid', dos_minmax=True, dos_sumnorm=True)
    ids = [str(value) for value in dataset.material_ids]
    atoms = np.asarray([len(dataset.crystals[int(dataset.structure_indices[i])][0]) for i in range(len(ids))],
                       dtype=np.float64)
    if not np.array_equal(atoms, np.load(ROOT / 'data/crystal_q1_v1/valid/n_atoms_valid.npy').astype(np.float64)):
        raise ValueError('valid atom counts differ from the frozen sidecar')
    references = Path(preparation['reference_controls']).parent / 'analysis/exports'
    sources = {'lr2_s42': (Path(control['samples']), Path(control['spectra']))}
    sources.update({name: (references / f'{name}_samples.csv', references / f'{name}_spectra.npz')
                    for name in (*REPLICATES[1:], *BLIND)})
    return ids, atoms, composition_keys_by_position(dataset, range(len(ids))), sources


def load(sources, ids, atoms, protocol):
    values, arrays, checks = {}, {}, {}
    for name, (samples_path, spectra_path) in sources.items():
        with np.load(spectra_path, allow_pickle=False) as data:
            spectra = {key: data[key] for key in data.files}
        if spectra['material_ids'].tolist() != ids:
            raise ValueError(f'{name} export does not follow the valid dataset order')
        frame = pd.read_csv(samples_path).set_index('material_id').loc[ids]
        computed = per_material(spectra, atoms, protocol)
        error = max(float(np.max(np.abs(computed[f'energy_{t}'] - frame[f'blind_energy_{t}'].to_numpy())
                                 / np.maximum(1e-6, np.abs(frame[f'blind_energy_{t}'].to_numpy())))) for t in TASKS)
        if error > 1e-3:
            raise ValueError(f'{name}: recomputed blind energy differs from its export ({error:.2e})')
        checks[name] = error
        values[name], arrays[name] = computed, spectra
    for name, spectra in arrays.items():
        for key in ('tgt_edos', 'tgt_phdos'):
            if not np.allclose(spectra[key], arrays['lr2_s42'][key], rtol=0, atol=1e-7):
                raise ValueError(f'{name} labels differ from the adopted control')
    return values, arrays, checks


def export_arms(preparation, names, out):
    from tools.eval.lg_valid_export import export

    exports, status = {}, {}
    for arm in preparation['arms']:
        if arm['name'] not in names:
            continue
        samples, summary, spectra = (out / 'exports' / f"{arm['name']}_{suffix}"
                                     for suffix in ('samples.csv', 'summary.json', 'spectra.npz'))
        history = pd.read_csv(arm['history']) if Path(arm['history']).exists() else None
        if history is None or history.epoch.tolist() != list(range(1, 36)):
            raise ValueError(f"{arm['name']} has not completed its 35-epoch budget")
        if not spectra.exists():
            try:
                export(Path(arm['run_dir']), samples, summary, spectra)
            except (FileNotFoundError, ValueError) as error:
                status[arm['name']] = {'exported': False, 'reason': str(error)}
                continue
        exports[arm['name']] = (samples, spectra)
        status[arm['name']] = {'exported': True, **{k: v for k, v in json.loads(summary.read_text()).items()
                                                    if k in ('best_epoch', 'best_score', 'checkpoint_epoch')}}
    return exports, status


def screen(preparation_path, out):
    preparation = verified_preparation(preparation_path)
    out = Path(out)
    if (out / 'screen.json').exists():
        raise FileExistsError(f'refusing existing screen: {out}')
    protocol = json.loads((Path(preparation_path).parent / 'protocol.json').read_text())
    ids, atoms, _, sources = context(preparation)
    exports, status = export_arms(preparation, set(preparation['step1_order']), out)
    values, _, checks = load({**{k: sources[k] for k in REPLICATES}, **exports}, ids, atoms, protocol)
    reference = float(np.mean([values[name]['energy_joint'].mean() for name in REPLICATES]))
    if abs(reference - REFERENCE_MEAN) > 5e-7:
        raise ValueError(f'R42-R44 seed mean changed: {reference}')
    result = {'schema': 'lr2-es-structure-generalization-screen-v1', 'reference_seed_mean': reference,
              'seed_range': SEED_RANGE, 'export_checks': checks, 'arms': {}, 'promoted': []}
    for name in preparation['step1_order']:
        arm = next(a for a in preparation['arms'] if a['name'] == name)
        entry = {'export': status[name]}
        if status[name]['exported']:
            joint = float(values[name]['energy_joint'].mean())
            summary = summarize(values[name])
            improvement = reference - joint
            entry.update(summary=summary, improvement_vs_seed_mean=improvement,
                         guards_passed=True,  # export re-verifies the fixed guards for the selected checkpoint
                         promoted=improvement > SEED_RANGE)
        else:
            entry.update(guards_passed=False, promoted=False)
        if entry['promoted']:
            result['promoted'].append(arm['arm'])
        result['arms'][name] = entry
    result['step2_runs'] = [n for arm in result['promoted'] for n in preparation['step2_order'][arm]]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'screen.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def geometry_benefit(ids, gain, task):
    directory = GEOMETRY / ('geometry' if task == 'edos' else 'geometry_phdos') / 'geometry_by_material.csv'
    if not directory.exists():
        return None
    table = pd.read_csv(directory)
    table = table[(table.task == task) & table.gate_class.isin(['exact', 'clipped']) & (table.split == 'valid')]
    frame = pd.DataFrame({'material_id': ids, 'gain': gain}).merge(table[['material_id', 'calc_volume_ratio']],
                                                                     on='material_id')
    frame['deviation'] = np.abs(np.log(frame.calc_volume_ratio))
    quintile = pd.qcut(frame.deviation, 5, labels=False, duplicates='drop')
    return {'materials': int(len(frame)), 'gain_by_deviation_quintile': frame.groupby(quintile).gain.mean().tolist()}


def final(preparation_path, out, train_inference=True):
    preparation = verified_preparation(preparation_path)
    out = Path(out)
    if (out / 'report.json').exists():
        raise FileExistsError(f'refusing existing analysis: {out}')
    screened = json.loads((out / 'screen.json').read_text())
    protocol = json.loads((Path(preparation_path).parent / 'protocol.json').read_text())
    ids, atoms, keys, sources = context(preparation)
    promoted = screened['promoted']
    names = set(preparation['step1_order']) | {n for arm in promoted for n in preparation['step2_order'][arm]}
    exports, status = export_arms(preparation, names, out)
    values, arrays, checks = load({**sources, **exports}, ids, atoms, protocol)
    draws = np.random.default_rng(SEED).integers(0, len(ids), size=(DRAWS, len(ids)))
    summaries = {name: summarize(value) for name, value in values.items()}
    report = {'schema': 'lr2-es-structure-generalization-analysis-v1', 'valid_materials': len(ids),
              'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'material, shared index matrix'},
              'screen': screened, 'export_status': status, 'export_checks': checks, 'arms': summaries,
              'comparisons': {}}

    def ensemble(members):
        spectra = {key: np.mean([arrays[name][key].astype(np.float64) for name in members], 0)
                   for key in ('pred_blind_edos', 'pred_blind_phdos')}
        spectra.update({key: arrays['lr2_s42'][key] for key in ('tgt_edos', 'tgt_phdos', 'tgt_shape_edos',
                                                                  'tgt_shape_phdos')})
        spectra.update({f'pred_shape_{t}': spectra[f'pred_blind_{t}']
                        / np.maximum(spectra[f'pred_blind_{t}'].sum(-1, keepdims=True), 1e-30) for t in TASKS})
        return spectra

    reference_ensemble = ensemble(REPLICATES)
    reference_values = per_material(reference_ensemble, atoms, protocol)
    if abs(reference_values['energy_joint'].mean() - REFERENCE_ENSEMBLE) > 5e-7:
        raise ValueError('reference ensemble changed')
    r_mean = {t: np.mean([values[n][f'energy_{t}'] for n in REPLICATES], 0) for t in ('joint', *TASKS)}
    s_mean = {t: np.mean([values[n][f'energy_{t}'] for n in BLIND], 0) for t in ('joint', *TASKS)}
    for arm in promoted:
        members = [preparation['step1_order'][0 if arm == 'G' else 1]] + preparation['step2_order'][arm]
        if any(name not in values for name in members):
            raise ValueError(f'{arm}: not every seed produced an eligible checkpoint')
        a_mean = {t: np.mean([values[n][f'energy_{t}'] for n in members], 0) for t in ('joint', *TASKS)}
        seed_means = [summaries[n]['joint_mean_E'] for n in members]
        main = interval(r_mean['joint'], a_mean['joint'], draws)
        improvement = REFERENCE_MEAN - float(np.mean(seed_means))
        arm_ensemble = ensemble(members)
        arm_values = per_material(arm_ensemble, atoms, protocol)
        entry = {
            'members': members, 'seed_joint_mean_E': seed_means, 'seed_mean': float(np.mean(seed_means)),
            'seed_range': float(np.ptp(seed_means)), 'improvement_vs_reference_seed_mean': improvement,
            'paired_seed_averaged': {t: interval(r_mean[t], a_mean[t], draws) for t in ('joint', *TASKS)},
            'verdict': ('improved' if main['ci95'][1] < 0 and improvement > SEED_RANGE else 'not distinguished'),
            'ensemble': {'summary': summarize(arm_values),
                         'vs_reference_ensemble': {t: interval(reference_values[f'energy_{t}'],
                                                               arm_values[f'energy_{t}'], draws)
                                                   for t in ('joint', *TASKS)}},
            'structure_contribution': {t: {'arm_minus_s0': float((a_mean[t] - s_mean[t]).mean()),
                                           'lr2_minus_s0': float((r_mean[t] - s_mean[t]).mean())}
                                       for t in ('joint', *TASKS)},
            'same_composition_ensemble': same_composition(arm_ensemble, ids, keys),
            'geometry_benefit': {t: geometry_benefit(ids, r_mean[t] - a_mean[t], t) for t in TASKS}}
        report['comparisons'][arm] = entry
    report['reference_same_composition_ensemble'] = same_composition(reference_ensemble, ids, keys)
    if train_inference:
        from tools.eval.structure_use_audit import infer
        runs = [Path(a['run_dir']) for a in preparation['arms'] if a['name'] in exports]
        target = out / 'train_inference'
        pending = [run for run in runs if not (target / f'{run.name}_train_samples.csv').exists()]
        if pending:
            infer(pending, 'train', target)
        report['train_valid'] = {}
        for arm in preparation['arms']:
            path = target / f"{Path(arm['run_dir']).name}_train_samples.csv"
            if path.exists():
                frame = pd.read_csv(path)
                report['train_valid'][arm['name']] = {
                    f'{t}_train_over_valid': float(frame[f'blind_energy_{t}'].mean()
                                                   / summaries[arm['name']][f'{t}_mean_E']) for t in TASKS}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False,
                                                default=float) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('screen', 'final'))
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--no-train-inference', action='store_true')
    args = parser.parse_args()
    if args.command == 'screen':
        result = screen(input_path(args.preparation), input_path(args.out))
        print(json.dumps({k: result[k] for k in ('reference_seed_mean', 'promoted', 'step2_runs')}, indent=2))
    else:
        report = final(input_path(args.preparation), input_path(args.out), not args.no_train_inference)
        print(json.dumps({arm: {k: v[k] for k in ('seed_mean', 'improvement_vs_reference_seed_mean', 'verdict')}
                          for arm, v in report['comparisons'].items()}, indent=2))


if __name__ == '__main__':
    main()
