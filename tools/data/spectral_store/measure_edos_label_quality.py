"""Read-only eDOS label-quality measurement under label contract v2 (design 2026-11-11, step 3, eDOS part).

Questions (user-confirmed scope, 2026-11-11):
1. Label noise: E between two NSCF Uniform calculations of the same material (MP's canonical
   task vs another owned NSCF Uniform task, chosen by calculation volume closest to the
   input structure), at 128 and 256 bins on [-6, 6] eV, both sides common.exact_bin.
2. Static vs canonical: canonical task vs the owned Static task (closest volume), same
   binning, by Static functional (GGA / GGA+U) x canonical-spectrum class (metal / gapped).
3. Model reference: h2 valid blind eDOS E per material (legacy contract, box_average labels,
   C 5.5427), grouped by the same metal / gapped class; plus noise and Static re-measured
   under the legacy contract (box_average, 128 bins, same C) so that h2 is compared within
   one contract.

E = dx * sum((a/Na - b/Nb)^2) / C, N = atoms of each task's calculation cell. C under v2 is
fitted like utils.spectral_balance.fit_task_normalizer (mean over spectra of
sum((DOS/N)^2) * dx) on the canonical spectra of legacy train materials, separately for 128
and 256 bins ('C_own'); E at 256 bins is also given with the 128-bin C ('C_128') to show the
effect of the choice. Legacy test (test_q1) materials are skipped entirely; nothing in the
store is written. Output: results/runs/edos_label_quality_20261111/{report.json, pairs.parquet}.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from multiprocessing import Pool

from common import ROOT, STORE, box_average, choose_edos_task, exact_bin, load_calc_types, strip_mp

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = ROOT / 'results/runs/edos_label_quality_20261111'
H2 = ROOT / 'results/runs/phdos_head_20261106/analysis/exports/h2_s42_samples.csv'
GRID = {128: np.linspace(-6.0, 6.0, 129), 256: np.linspace(-6.0, 6.0, 257)}
DX = {n: 12.0 / n for n in GRID}
C_LEGACY = 5.542719504890618  # results/runs/spectral_residual_20261030/protocol.json (box_average labels)
METAL_MAX = 0.15  # eV; same canonical-spectrum class as verify_store.fallback_v2


def _gap(x, y, threshold=0.01):
    """Width (eV) of the empty region containing E=0 (0 if DOS at 0 is above threshold)."""
    if len(x) < 3:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    empty = y <= threshold
    i = min(max(int(np.searchsorted(x, 0.0)), 1), len(x) - 1)
    if not (empty[i - 1] and empty[i]):
        return 0.0
    lo, hi = i - 1, i
    while lo > 0 and empty[lo - 1]:
        lo -= 1
    while hi < len(x) - 1 and empty[hi + 1]:
        hi += 1
    return float(x[hi] - x[lo])


def _bin(job):
    path, wanted = job
    out = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['source_task_id', 'x', 'y']):
        ids = batch.column('source_task_id').to_pylist()
        take = [i for i, t in enumerate(ids) if t in wanted]
        if not take:
            continue
        for r in batch.take(pa.array(take)).to_pylist():
            x, y = np.asarray(r['x'], np.float64), np.asarray(r['y'], np.float64)
            near = (x > -1.0) & (x < 1.0)
            out[r['source_task_id']] = {
                'v2_128': exact_bin(x, y, GRID[128])[0].astype(np.float64),
                'v2_256': exact_bin(x, y, GRID[256])[0].astype(np.float64),
                'v1_128': box_average(x, y, GRID[128])[0].astype(np.float64),
                'gap': _gap(x[near], y[near])}
    return out


def stats(v):
    v = np.asarray(v, np.float64)
    if not len(v):
        return {'n': 0}
    return {'n': int(len(v)), 'mean': float(v.mean()), 'median': float(np.median(v)),
            'p90': float(np.percentile(v, 90)), 'p99': float(np.percentile(v, 99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if (OUT / 'report.json').exists():
        raise FileExistsError(OUT / 'report.json')
    staging = STORE / 'staging'
    calc_types = load_calc_types()
    ctype = lambda r: calc_types.get(strip_mp(r['source_task_id']), 'missing')  # noqa: E731
    parts = sorted((staging / 'edos').glob('part-*.parquet'))
    cols = ['material_id', 'source_task_id', 'functional', 'status', 'is_canonical', 'calc_volume_per_atom',
            'calc_numbers']
    rows = pa.concat_tables([pq.read_table(f, columns=cols) for f in parts]).to_pylist()
    materials = {m['material_id']: m for m in pq.read_table(
        staging / 'materials.parquet', columns=['material_id', 'legacy_split', 'cell', 'n_atoms']).to_pylist()}
    volume = {k: abs(float(np.linalg.det(np.asarray(m['cell'], np.float64).reshape(3, 3)))) / m['n_atoms']
              for k, m in materials.items() if m['cell'] and m['legacy_split'] != 'test'}
    by = {}
    for r in rows:
        if r['material_id'] in volume:  # legacy test materials never enter
            r['n'] = len(r['calc_numbers'] or []) or 1
            by.setdefault(r['material_id'], []).append(r)

    canon, noise_pairs, static_pairs, wanted = {}, [], [], set()
    for material_id, rs in by.items():
        c = [r for r in rs if r['is_canonical'] and r['status'] == 'ok']
        if not c:
            continue
        canon[material_id] = c[0]
        wanted.add(c[0]['source_task_id'])
        others = [r for r in rs if not r['is_canonical']]
        for kind, pairs in (('NSCF Uniform', noise_pairs), ('Static', static_pairs)):
            pool = [r for r in others if ctype(r).endswith(kind)]
            chosen, _ = choose_edos_task(pool, volume[material_id], calc_types, (kind,))
            if chosen is not None:
                pairs.append((material_id, c[0], chosen))
                wanted.add(chosen['source_task_id'])
    os.nice(19)
    binned = {}
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(_bin, [(p, wanted) for p in parts]):
            binned.update(part)

    train = [m for m in canon if materials[m]['legacy_split'] == 'train']
    C = {}
    for n in (128, 256):
        values = [(binned[canon[m]['source_task_id']][f'v2_{n}'] / canon[m]['n']) for m in train]
        C[n] = float(np.mean([np.square(v).sum() for v in values]) * DX[n])

    def e(a, b, key):
        va, vb = binned[a['source_task_id']], binned[b['source_task_id']]
        diff2 = lambda k: float(np.square(va[k] / a['n'] - vb[k] / b['n']).sum())  # noqa: E731
        return {'v2_128': DX[128] * diff2('v2_128') / C[128],
                'v2_256_C_own': DX[256] * diff2('v2_256') / C[256],
                'v2_256_C_128': DX[256] * diff2('v2_256') / C[128],
                'v1_128': DX[128] * diff2('v1_128') / C_LEGACY}[key]

    keys = ('v2_128', 'v2_256_C_own', 'v2_256_C_128', 'v1_128')
    klass = lambda m: 'metal' if (binned[canon[m]['source_task_id']]['gap'] or 0) < METAL_MAX else 'gapped'  # noqa: E731

    def grouped(pairs, group_of):
        groups = {}
        for material_id, a, b in pairs:
            per_pair.append({'set': 'noise' if pairs is noise_pairs else 'static', 'material_id': material_id,
                             'legacy_split': materials[material_id]['legacy_split'], 'class': klass(material_id),
                             'task_a': a['source_task_id'], 'task_b': b['source_task_id'], 'type_b': ctype(b),
                             'functional_a': a['functional'], 'functional_b': b['functional'],
                             'n_a': a['n'], 'n_b': b['n'],
                             'volume_ratio': (b['calc_volume_per_atom'] or np.nan) / (a['calc_volume_per_atom'] or np.nan),
                             **{k: e(a, b, k) for k in keys}})
            for g in group_of(material_id, a, b):
                groups.setdefault(g, []).append((a, b))
        return {g: {k: stats([e(a, b, k) for a, b in v]) for k in keys} for g, v in sorted(groups.items())}

    per_pair = []
    noise = grouped(noise_pairs, lambda m, a, b: (
        'all', f'all/{klass(m)}', f"functional:{a['functional']}->{b['functional']}",
        f"same_functional:{a['functional'] == b['functional']}/{klass(m)}"))
    static = grouped(static_pairs, lambda m, a, b: (
        f"{ctype(b).replace(' Static', '')}/{klass(m)}", f"{ctype(b).replace(' Static', '')}/all", 'all'))

    import csv
    with open(H2) as stream:
        h2 = {r['material_id']: float(r['blind_energy_edos']) for r in csv.DictReader(stream)}
    h2_groups = {'all': list(h2.values())}
    for m, value in h2.items():
        if m in canon:
            h2_groups.setdefault(klass(m), []).append(value)
    h2_groups['no_canonical_in_store'] = [v for m, v in h2.items() if m not in canon]

    report = {
        'contract': {'grid': '[-6, 6] eV', 'binning_v2': 'common.exact_bin', 'binning_v1': 'common.box_average',
                     'C_v2_fit': 'mean over canonical eDOS of legacy train materials of sum((DOS/N)^2)*dx',
                     'C_v2': {str(n): C[n] for n in C}, 'C_v2_train_materials': len(train), 'C_v1': C_LEGACY,
                     'metal_class': f'canonical-spectrum gap at E=0 < {METAL_MAX} eV',
                     'pair_rule': 'one partner task per material: closest calculation volume to the input structure',
                     'excluded': 'legacy test_q1 materials (no spectra read)'},
        'pairs': {'noise_nscf_uniform': len(noise_pairs), 'static': len(static_pairs)},
        'noise': noise, 'static': static,
        'h2_valid_blind_v1_128': {g: stats(v) for g, v in h2_groups.items()},
        'note': 'h2 numbers are legacy contract only (box_average labels, C_v1); compare them with the v1_128 '
                'columns. No model has been trained on v2 labels.'}
    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(per_pair), OUT / 'pairs.parquet', compression='zstd')
    (OUT / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'C_v2': report['contract']['C_v2'], 'pairs': report['pairs']}, indent=1))


if __name__ == '__main__':
    main()
