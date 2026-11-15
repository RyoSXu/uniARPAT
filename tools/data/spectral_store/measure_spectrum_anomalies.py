"""Read-only per-spectrum features for an anomalous-spectrum screening rule (label contract v2, step 2).

For every stored spectrum of a non-test material (legacy test_q1 materials are skipped; for
PhononDB / JARVIS entries that are not tied to a material, all entries are read, since they
are raw source records), compute physical sanity features:

eDOS (staging/edos, staging/edos_legacy; MP tasks, spin-summed DOS, x = E - E_F):
  occ_ratio   = integral_{x<=0} DOS / task valence electrons (should be ~1: the occupied
                states hold the valence electrons)
  win_area    = integral_{-6}^{6} DOS / N (states per atom in the label window)
  peak        = max DOS / N inside the window
  energy_128  = sum((exact_bin/N)^2) * dx on E0 (the per-spectrum term of C)
phDOS (staging/phdos MP, staging/phdos_phonondb, staging/phdos_jarvis; x in cm^-1):
  area_atom   = integral DOS / N (3 for a normalised phonon DOS; JARVIS raw intensity is
                unnormalised and the legacy rule rescales it, so only its shape features count)
  neg_frac    = share of |DOS| area below -20 cm^-1 (imaginary modes)
  peak_rel    = max DOS / (mean DOS over the populated range) (spikiness)
  energy_64   = sum((exact_bin/N)^2) * dx on P0 after the legacy area normalisation to 3N

Output: results/runs/spectrum_anomaly_20261111/{edos,phdos}_features.parquet. Nothing in
the store is written.
"""
from __future__ import annotations

import argparse
import json
import os

from multiprocessing import Pool

from common import ROOT, STORE, exact_bin

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = ROOT / 'results/runs/spectrum_anomaly_20261111'
STAGING = STORE / 'staging'
E0 = np.linspace(-6.0, 6.0, 129)
P0 = np.linspace(-280.0, 980.0, 65)


def _sorted(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    order = np.argsort(x, kind='stable')
    return x[order], y[order]


def _integral(x, y, lo, hi):
    """Integral of the piecewise-linear interpolant over [lo, hi] (clipped to the data range)."""
    if len(x) < 2:
        return 0.0
    grid = np.concatenate([[max(lo, x[0])], x[(x > lo) & (x < hi)], [min(hi, x[-1])]])
    if grid[-1] <= grid[0]:
        return 0.0
    return float(np.trapz(np.interp(grid, x, y), grid))


def edos_features(job):
    path, skip = job
    out = []
    cols = ['material_id', 'source_task_id', 'is_canonical', 'status', 'x', 'y', 'task_nval', 'calc_numbers']
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=cols):
        for r in batch.to_pylist():
            if r['material_id'] in skip or r['status'] != 'ok':
                continue
            x, y = _sorted(r['x'], r['y'])
            n = len(r['calc_numbers'] or []) or 1
            win = (x >= -6) & (x <= 6)
            out.append({'material_id': r['material_id'], 'source_task_id': r['source_task_id'],
                        'is_canonical': r['is_canonical'], 'file': path.parent.name, 'n': n,
                        'task_nval': r['task_nval'], 'n_points': len(x),
                        'occ_ratio': _integral(x, y, -np.inf, 0.0) / r['task_nval'] if r['task_nval'] else np.nan,
                        'win_area': _integral(x, y, -6.0, 6.0) / n,
                        'peak': float(y[win].max() / n) if win.any() else 0.0,
                        'min_y': float(y.min()),
                        'energy_128': float(np.square(exact_bin(x, y, E0)[0].astype(np.float64) / n).sum()
                                            * (E0[1] - E0[0]))})
    return out


def phdos_features(job):
    path, skip = job
    out = []
    schema = pq.read_schema(path).names
    cols = [c for c in ('material_id', 'source_task_id', 'source_id', 'source', 'status', 'x', 'y', 'calc_numbers',
                        'primitive_cell', 'phonon_modes') if c in schema]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=cols):
        for r in batch.to_pylist():
            if r.get('material_id') in skip or r['status'] != 'ok':
                continue
            x, y = _sorted(r['x'], r['y'])
            if len(x) < 2:
                continue
            if r.get('calc_numbers'):
                n = len(r['calc_numbers'])
            elif r.get('primitive_cell'):
                n = len(json.loads(r['primitive_cell']).get('points') or []) or np.nan
            elif r.get('phonon_modes'):  # JARVIS: 3N Gamma modes
                n = len(r['phonon_modes']) / 3 or np.nan
            else:
                n = np.nan
            total = float(np.trapz(y, x))
            absolute = float(np.trapz(np.abs(y), x))
            neg = _integral(x, np.abs(y), -np.inf, -20.0)
            populated = y > 1e-6 * max(y.max(), 1e-30)
            span = x[populated].max() - x[populated].min() if populated.sum() > 1 else np.nan
            mean = total / span if span and span > 0 else np.nan
            scaled = y * (3 * n / total) if total > 0 and np.isfinite(n) else y * np.nan
            out.append({'material_id': r.get('material_id'), 'source_id': r.get('source_task_id') or r.get('source_id'),
                        'file': path.parent.name, 'n': n, 'area_atom': total / n if np.isfinite(n) else np.nan,
                        'neg_frac': neg / absolute if absolute > 0 else np.nan,
                        'peak_rel': float(y.max() / mean) if mean and mean > 0 else np.nan,
                        'min_x': float(x[populated].min()) if populated.any() else np.nan,
                        'max_x': float(x[populated].max()) if populated.any() else np.nan,
                        'energy_64': float(np.square(exact_bin(x, scaled, P0)[0].astype(np.float64) / n).sum()
                                           * (P0[1] - P0[0])) if np.isfinite(n) and n > 0 else np.nan})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('edos_features.parquet', 'phdos_features.parquet'):
        if (OUT / name).exists():
            raise FileExistsError(OUT / name)
    materials = pq.read_table(STAGING / 'materials.parquet', columns=['material_id', 'legacy_split']).to_pylist()
    skip = {m['material_id'] for m in materials if m['legacy_split'] == 'test'}
    os.nice(19)
    edos_parts = sorted((STAGING / 'edos').glob('part-*.parquet')) + sorted((STAGING / 'edos_legacy').glob('part-*.parquet'))
    phdos_parts = [STAGING / 'phdos/part-00.parquet'] + sorted((STAGING / 'phdos_phonondb').glob('part-*.parquet')) \
        + sorted((STAGING / 'phdos_jarvis').glob('part-*.parquet'))
    with Pool(args.workers) as pool:
        ph = [r for part in pool.imap_unordered(phdos_features, [(p, skip) for p in phdos_parts]) for r in part]
        pq.write_table(pa.Table.from_pylist(ph), OUT / 'phdos_features.parquet', compression='zstd')
        print('phdos', len(ph), flush=True)
        ed = [r for part in pool.imap_unordered(edos_features, [(p, skip) for p in edos_parts]) for r in part]
        pq.write_table(pa.Table.from_pylist(ed), OUT / 'edos_features.parquet', compression='zstd')
        print('edos', len(ed), flush=True)


if __name__ == '__main__':
    main()
