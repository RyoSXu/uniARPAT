"""Final tables of spectral-store-v1 before any new split (P1a; design 2026-11-10 P0).

User decision 2026-11-10 (relayed by the review session): new splits are deferred until
the store is complete and verified; they must exist before any model trains on new data.
Until then only the legacy splits are frozen:

- legacy_splits/{train_q1,valid_q1,test_q1}.txt with SHA-256 (crystal_q1_v1 as is);
- test_q1 (the final dual-label test, sealed until finalization) has every spectrum it
  owns (legacy labels and native MP spectra) written to sealed/, never to the public
  tables;
- materials.parquet gets `pool` (train_q1 / valid_q1 / test_q1 / new) and
  `eval_composition_excluded`: a new material whose absolute or reduced composition
  equals that of a valid_q1 or test_q1 material never enters training.

Outputs (data/spectral_store_v1/): materials.parquet, spectra_{edos,phdos,legacy}.parquet,
sealed/spectra_*.parquet, geometries.parquet, legacy_splits/, finalize_report.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from common import STORE, sha256_text

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

STAGING = STORE / 'staging'
SEALED_POOLS = ('test_q1',)


def read_parts(directory: Path, columns=None):
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in sorted(directory.glob('part-*.parquet'))])


def write_spectra(table: pa.Table, sealed_ids: set, name: str, out: Path):
    is_sealed = pc.fill_null(pc.is_in(table.column('material_id'), value_set=pa.array(sorted(sealed_ids))), False)
    pq.write_table(table.filter(pc.invert(is_sealed)), out / f'{name}.parquet', compression='zstd')
    pq.write_table(table.filter(is_sealed), out / 'sealed' / f'{name}.parquet', compression='zstd')
    return {'public': table.num_rows - int(pc.sum(is_sealed).as_py() or 0), 'sealed': int(pc.sum(is_sealed).as_py() or 0)}


def geometries(materials: list, spectra: dict) -> pa.Table:
    rows, volume = [], {}
    for m in materials:
        if not m.get('cell'):
            continue
        cell = np.asarray(m['cell'], dtype=np.float64).reshape(3, 3)
        v = abs(float(np.linalg.det(cell))) / m['n_atoms']
        volume[m['material_id']] = v
        rows.append({'material_id': m['material_id'], 'role': 'input', 'source_task_id': None,
                     'volume_per_atom': v, 'volume_ratio_to_input': 1.0, 'n_atoms': m['n_atoms']})
    for role, table in spectra.items():
        cols = table.select(['material_id', 'source_task_id', 'calc_volume_per_atom', 'calc_numbers'])
        for r in cols.to_pylist():
            if not r['material_id'] or r['calc_volume_per_atom'] is None:
                continue
            base = volume.get(r['material_id'])
            rows.append({'material_id': r['material_id'], 'role': role, 'source_task_id': r['source_task_id'],
                         'volume_per_atom': r['calc_volume_per_atom'],
                         'volume_ratio_to_input': r['calc_volume_per_atom'] / base if base else None,
                         'n_atoms': len(r['calc_numbers'])})
    return pa.Table.from_pylist(rows)


def mark_pools(materials: list) -> Counter:
    excluded_abs, excluded_red = set(), set()
    for m in materials:
        m['pool'] = f"{m['legacy_split']}_q1" if m['legacy_split'] else 'new'
        if m['legacy_split'] in ('valid', 'test'):
            excluded_abs.add(m['composition'])
            excluded_red.add(m['reduced_composition'])
    counts = Counter()
    for m in materials:
        m['eval_composition_excluded'] = bool(
            m['pool'] == 'new' and m.get('composition') is not None
            and (m['composition'] in excluded_abs or m['reduced_composition'] in excluded_red))
        counts[(m['pool'], m['eval_composition_excluded'])] += 1
    return counts


def main():
    out = STORE
    for name in ('legacy_splits', 'sealed', 'materials.parquet'):
        if (out / name).exists():
            raise FileExistsError(f'refusing existing output: {out / name}')
    materials = pq.read_table(STAGING / 'materials.parquet').to_pylist()
    pools = mark_pools(materials)
    (out / 'legacy_splits').mkdir()
    (out / 'sealed').mkdir()
    hashes = {}
    for pool in ('train_q1', 'valid_q1', 'test_q1'):
        ids = sorted(m['material_id'] for m in materials if m['pool'] == pool)
        text = '\n'.join(ids) + '\n'
        (out / 'legacy_splits' / f'{pool}.txt').write_text(text)
        hashes[pool] = {'n': len(ids), 'sha256': sha256_text(text)}
    sealed = {m['material_id'] for m in materials if m['pool'] in SEALED_POOLS}
    edos, phdos = read_parts(STAGING / 'edos'), read_parts(STAGING / 'phdos')
    report = {'pools': {f'{p}{"/eval_excluded" if e else ""}': n for (p, e), n in sorted(pools.items())},
              'legacy_split_hashes': hashes, 'sealed_pools': list(SEALED_POOLS),
              'new_splits': 'deferred (user decision 2026-11-10); required before training on new data',
              'rows': {'spectra_edos': write_spectra(edos, sealed, 'spectra_edos', out),
                       'spectra_phdos': write_spectra(phdos, sealed, 'spectra_phdos', out),
                       'spectra_legacy': write_spectra(pq.read_table(STAGING / 'legacy_spectra.parquet'), sealed,
                                                       'spectra_legacy', out)}}
    (out / 'legacy_splits' / 'hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    pq.write_table(pa.Table.from_pylist(materials), out / 'materials.parquet', compression='zstd')
    pq.write_table(geometries(materials, {'edos_calc': edos, 'phdos_calc': phdos}), out / 'geometries.parquet',
                   compression='zstd')
    (out / 'finalize_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
