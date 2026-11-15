"""Read-only phDOS label noise and cross-source differences (design splits-v2, item B; step 3 phonon).

Only materials whose formula group goes to train under splits_v2 (no legacy valid/test side,
no new dev/test) are used. Records are attached to materials by the linking rule of
link_records (hint passes default + same crystal system, else a unique alternative).
Per-atom spectra: MP y / calculation atoms; PhononDB y / primitive atoms; JARVIS y x 3 /
full area (legacy rule). v2 binning (exact_bin) on P0 with 64 and 128 bins; E = dx *
sum((a - b)^2) / C with the v2 C of the same resolution (q1_v2 / q1_v2_x2 manifests).
Pairs per material: every pair of records, labelled by source/method kinds
(e.g. mp_pheasy|phonondb). Output: results/runs/phonon_sources_20261112/{pairs.parquet, report.json}.
"""
from __future__ import annotations

import json
from itertools import combinations

from common import ROOT, STORE, exact_bin
from splits_v2 import assign

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = ROOT / 'results/runs/phonon_sources_20261112'
STAGING = STORE / 'staging'
GRIDS = {64: np.linspace(-280, 980, 65), 128: np.linspace(-280, 980, 129)}


def linked(features: pd.DataFrame) -> pd.Series:
    """record_id -> material_id under the linking rule (None when unlinked)."""
    out = {}
    for r in features.itertuples():
        if r.hint_in_store and bool(r.default) and r.cs_hint == r.cs_record:
            out[r.record_id] = r.hint
        else:
            alt = json.loads(r.alt_matches) if isinstance(r.alt_matches, str) else []
            out[r.record_id] = alt[0][0] if len(alt) == 1 else None
    return pd.Series(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(STAGING / 'identity/record_features.parquet')
    feats = feats[feats.source != 'mp_edos']
    link = linked(feats)
    meta = pd.read_parquet(ROOT / 'results/runs/split_design_20261111/materials_meta.parquet',
                           columns=['material_id', 'formula', 'elements', 'pool'])
    lv, lt = set(meta[meta.pool == 'valid_q1'].formula), set(meta[meta.pool == 'test_q1'].formula)
    side = {r.material_id: assign(r.formula, r.elements, lv, lt, set()) for r in meta.itertuples()}
    pool = dict(zip(meta.material_id, meta.pool))  # train groups only; C3 holdouts and legacy eval sets excluded
    train_materials = {m for m, s in side.items() if s == 'train' and pool[m] in ('new', 'train_q1')}
    keep = {rid: m for rid, m in link.items() if m in train_materials}
    spectra = {}
    ph = pq.read_table(STAGING / 'phdos/part-00.parquet', columns=['source_task_id', 'method', 'x', 'y', 'calc_numbers',
                                                                     'status']).to_pylist()
    for r in ph:
        rid = f"mp_phonon:{r['source_task_id']}"
        if rid in keep:
            spectra[rid] = (f"mp_{r['method']}", np.asarray(r['x']), np.asarray(r['y']) / max(len(r['calc_numbers']), 1))
    for part in sorted((STAGING / 'phdos_phonondb').glob('part-*.parquet')):
        for r in pq.read_table(part, columns=['source_id', 'x', 'y', 'primitive_cell']).to_pylist():
            rid = f"phonondb:{r['source_id']}"
            if rid in keep:
                n = len(json.loads(r['primitive_cell'])['points'])
                spectra[rid] = ('phonondb', np.asarray(r['x']), np.asarray(r['y']) / n)
    for r in pq.read_table(STAGING / 'phdos_jarvis/part-0000.parquet', columns=['source_id', 'x', 'y']).to_pylist():
        rid = f"jarvis:{r['source_id']}"
        if rid in keep and r['x']:
            x, y = np.asarray(r['x'], float), np.asarray(r['y'], float)
            order = np.argsort(x)
            area = float(np.trapz(y[order], x[order]))
            if area > 0:
                spectra[rid] = ('jarvis', x, y * 3 / area)
    manifests = {64: 'q1_v2', 128: 'q1_v2_x2'}
    C = {n: json.loads((STORE / 'views' / v / 'manifest.json').read_text())['normalizer_train']['phdos']['energy']
         for n, v in manifests.items()}
    binned = {rid: {n: exact_bin(x, y, GRIDS[n])[0].astype(np.float64) for n in GRIDS} for rid, (_, x, y) in spectra.items()}
    by_material = {}
    for rid in spectra:
        by_material.setdefault(keep[rid], []).append(rid)
    rows = []
    for material_id, rids in by_material.items():
        for a, b in combinations(sorted(rids), 2):
            kind = '|'.join(sorted((spectra[a][0], spectra[b][0])))
            rows.append({'material_id': material_id, 'a': a, 'b': b, 'kind': kind,
                         **{f'E{n}': float((GRIDS[n][1] - GRIDS[n][0]) * np.square(binned[a][n] - binned[b][n]).sum() / C[n])
                            for n in GRIDS}})
    pairs = pd.DataFrame(rows)
    pairs.to_parquet(OUT / 'pairs.parquet')

    def stats(v):
        v = np.asarray(v)
        return {'n': int(len(v)), 'mean': float(v.mean()), 'median': float(np.median(v)),
                'p90': float(np.quantile(v, .9)), 'p99': float(np.quantile(v, .99))}
    report = {'C': C, 'records': len(spectra), 'materials': len(by_material),
              'kinds': {k: {f'E{n}': stats(g[f'E{n}']) for n in GRIDS} for k, g in pairs.groupby('kind') if len(g) >= 20}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    for k, v in report['kinds'].items():
        print(f"{k:28s} n={v['E64']['n']:5d} | 64: {v['E64']['mean']:.3f}/{v['E64']['median']:.3f}/{v['E64']['p90']:.3f}"
              f" | 128: {v['E128']['mean']:.3f}/{v['E128']['median']:.3f}/{v['E128']['p90']:.3f}")


if __name__ == '__main__':
    main()
