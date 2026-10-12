"""Paired comparison of the baseline arm against the local/global arms.

Pairs per material ID on the same valid set, then reports the arm minus
baseline difference for eDOS/phDOS x blind/oracle median R2, median MAE and the
R2<0 failure rate, each with a paired bootstrap 95% interval. Time and VRAM are
listed separately. The contrast is the whole encoder swap at seed 42 and 35
epochs; it is not an attribution to local aggregation alone.
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

METRICS = {
    'r2_median': lambda values: float(np.median(values)),
    'mae_median': lambda values: float(np.median(values)),
    'fail_rate_pct': lambda values: float((values < 0).mean() * 100),
}
SCOPE = 'whole encoder swap at seed 42, 35 epochs; not an attribution to local aggregation'


def load(summary_path: Path, samples_path: Path):
    summary = json.loads(summary_path.read_text())
    frame = pd.read_csv(samples_path)
    if frame['material_id'].duplicated().any():
        raise ValueError(f'duplicate material IDs: {samples_path}')
    return summary, frame.set_index('material_id').sort_index()


def bootstrap(base, other, column, draws, rng):
    """Paired bootstrap interval of the arm minus baseline statistic."""
    left = base[column].to_numpy()
    right = other[column].to_numpy()
    if left.shape != right.shape:
        raise ValueError(f'paired columns differ in shape: {column}')
    n = len(left)
    differences = {}
    for name, statistic in METRICS.items():
        samples = np.empty(draws)
        for draw in range(draws):
            index = rng.integers(0, n, n)
            samples[draw] = statistic(right[index]) - statistic(left[index])
        differences[name] = {
            'arm': statistic(right), 'baseline': statistic(left),
            'difference': statistic(right) - statistic(left),
            'ci95': [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            'ci95_contains_zero': bool(np.percentile(samples, 2.5) <= 0 <= np.percentile(samples, 97.5)),
        }
    return differences


def compare(baseline_summary, baseline_frame, arms, draws, seed, scope=SCOPE):
    rng = np.random.default_rng(seed)
    report = {'pairing': None, 'bootstrap': {'draws': draws, 'seed': seed,
                                             'unit': 'material', 'paired': True},
              'scope': scope,
              'runs': {'baseline': run_line(baseline_summary)}, 'comparisons': {}}
    for label, summary, frame in arms:
        if list(frame.index) != list(baseline_frame.index):
            raise ValueError(f'material IDs do not pair for {label}')
        report['runs'][label] = run_line(summary)
        entry = {}
        for mode in ('blind', 'oracle'):
            for task in ('edos', 'phdos'):
                from_r2 = bootstrap(baseline_frame, frame, f'{mode}_r2_{task}', draws, rng)
                from_mae = bootstrap(baseline_frame, frame, f'{mode}_mae_{task}', draws, rng)
                entry[f'{mode}_{task}_r2_median'] = from_r2['r2_median']
                entry[f'{mode}_{task}_fail_rate_pct'] = from_r2['fail_rate_pct']
                entry[f'{mode}_{task}_mae_median'] = from_mae['mae_median']
        report['comparisons'][label] = entry
    report['pairing'] = {'materials': int(len(baseline_frame)),
                         'ids_identical': True,
                         'first': str(baseline_frame.index[0])}
    return report


def run_line(summary):
    return {'tag': summary['tag'], 'best_epoch': summary['best_epoch'],
            'best_score': summary['best_score'], 'epochs': summary['epochs'],
            'contract_schema': summary['contract_schema'],
            'init_protocol': summary['init_protocol'],
            'epoch_time_s_median': summary['epoch_time_s_median'],
            'epoch_time_s_sum': summary['epoch_time_s_sum'],
            'peak_vram_mb_median': summary['peak_vram_mb_median'],
            'peak_vram_mb_max': summary['peak_vram_mb_max']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', nargs=2, type=Path, required=True,
                        metavar=('SUMMARY', 'SAMPLES'))
    parser.add_argument('--arm', nargs=3, action='append', required=True,
                        metavar=('LABEL', 'SUMMARY', 'SAMPLES'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--scope', default=SCOPE,
                        help='What the contrast covers; recorded in the report')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'refusing existing report: {args.out}')
    baseline_summary, baseline_frame = load(*args.baseline)
    arms = []
    for label, summary_path, samples_path in args.arm:
        summary, frame = load(Path(summary_path), Path(samples_path))
        arms.append((str(label), summary, frame))
    report = compare(baseline_summary, baseline_frame, arms, args.draws, args.seed, args.scope)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    main()
