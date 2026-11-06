"""Export and read the LR2 E+S learning curve with the rule fixed in the 2026-11-06 design.

Points: 25 % and 50 % train (seed 42, one run each) and 100 % = R42-R44 seed
replicates (mean, with their range). The decision reads the 50 % -> 100 % drop of
the joint blind mean E against the same-recipe seed range (0.001319) and about
three times that range (0.004); each task is read against its own range.
Valid only; never used for selection or adoption.
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

from tools.eval.reference_controls_analyze import summarize
from tools.eval.structure_generalization_analyze import REFERENCE_MEAN, REPLICATES, context, load
from utils.paths import file_sha256, input_path

PREPARATION_SCHEMA = 'lr2-es-learning-curve-preparation-v1'
SEED_RANGE = {'joint': 0.001319, 'edos': 0.000651, 'phdos': 0.003289}
EFFECT_FACTOR = 3.  # "clear" drop: more than about three seed ranges (decisions.md)


def verified_preparation(path):
    preparation = json.loads(Path(path).read_text())
    if preparation.get('schema') != PREPARATION_SCHEMA:
        raise ValueError('not a learning-curve preparation report')
    for item, digest in preparation['input_sha256'].items():
        if file_sha256(item) != digest:
            raise ValueError(f'frozen input changed since preparation: {item}')
    return preparation


def export_points(preparation, out):
    from tools.eval.lg_valid_export import export

    exports, status = {}, {}
    for point in preparation['points']:
        name = point['name']
        samples, summary, spectra = (out / 'exports' / f'{name}_{suffix}'
                                     for suffix in ('samples.csv', 'summary.json', 'spectra.npz'))
        history = pd.read_csv(point['history']) if Path(point['history']).exists() else None
        if history is None or history.epoch.tolist() != list(range(1, 36)):
            raise ValueError(f'{name} has not completed its 35-epoch budget')
        if not spectra.exists():
            export(Path(point['run_dir']), samples, summary, spectra)
        exports[name] = (samples, spectra)
        status[name] = {k: v for k, v in json.loads(summary.read_text()).items()
                        if k in ('best_epoch', 'best_score', 'checkpoint_epoch')}
    return exports, status


def verdict(drop, task):
    if drop > EFFECT_FACTOR * SEED_RANGE[task]:
        return 'not_saturated'
    if drop > SEED_RANGE[task]:
        return 'limited_gain'
    return 'saturated_within_noise'


def analyze(preparation_path, out):
    preparation = verified_preparation(preparation_path)
    out = Path(out)
    if (out / 'curve.json').exists():
        raise FileExistsError(f'refusing existing analysis: {out}')
    protocol = json.loads((Path(preparation_path).parent / 'protocol.json').read_text())
    ids, atoms, _, sources = context(preparation)
    exports, status = export_points(preparation, out)
    values, _, checks = load({**{k: sources[k] for k in REPLICATES}, **exports}, ids, atoms, protocol)
    full = float(np.mean([values[name]['energy_joint'].mean() for name in REPLICATES]))
    if abs(full - REFERENCE_MEAN) > 5e-7:
        raise ValueError(f'R42-R44 seed mean changed: {full}')
    tasks = {'joint': 'energy_joint', 'edos': 'energy_edos', 'phdos': 'energy_phdos'}
    curve = {}
    for point in preparation['points']:
        curve[point['name']] = {'fraction': point['fraction'], 'materials': point['subset']['materials'],
                                'seeds': [42], 'selected': status[point['name']],
                                'mean_E': {t: float(values[point['name']][k].mean()) for t, k in tasks.items()},
                                'summary': summarize(values[point['name']])}
    replicate_means = {t: [float(values[n][k].mean()) for n in REPLICATES] for t, k in tasks.items()}
    curve['f100'] = {'fraction': 1., 'materials': 18704, 'seeds': [42, 43, 44],
                     'mean_E': {t: float(np.mean(v)) for t, v in replicate_means.items()},
                     'seed_range': {t: float(np.ptp(v)) for t, v in replicate_means.items()},
                     'seed42_E': {t: v[0] for t, v in replicate_means.items()},
                     'summary_seed42': summarize(values['lr2_s42'])}
    reading = {}
    for task in tasks:
        e25, e50, e100 = (curve[n]['mean_E'][task] for n in ('f25', 'f50', 'f100'))
        drop = e50 - e100
        reading[task] = {
            'drop_25_to_50': e25 - e50, 'drop_50_to_100': drop, 'seed_range': SEED_RANGE[task],
            'clear_threshold': EFFECT_FACTOR * SEED_RANGE[task], 'verdict': verdict(drop, task),
            # Rough guide only: one more doubling at the last observed log-log slope.
            'loglog_slope_50_to_100': float(np.log(e50 / e100) / np.log(2.)),
            'next_doubling_drop_if_slope_holds': float(e100 - e100 * (e100 / e50)),
        }
    result = {'schema': 'lr2-es-learning-curve-analysis-v1', 'preparation_sha256': file_sha256(preparation_path),
              'export_checks': checks, 'curve': curve, 'reading': reading,
              'scope': 'valid only, blind; subset points are single seed-42 runs, 100 % is the R42-R44 mean; '
                       'repaired Q1, 35 epochs of full-train step count; not a selection or adoption'}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'curve.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = analyze(input_path(args.preparation), input_path(args.out))
    for task, entry in result['reading'].items():
        print(task, {n: round(result['curve'][n]['mean_E'][task], 6) for n in ('f25', 'f50', 'f100')},
              round(entry['drop_50_to_100'], 6), entry['verdict'])


if __name__ == '__main__':
    main()
