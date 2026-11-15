"""Read-only statistics for the step-4 split design (scale, source mapping, formula groups, strata).

No spectrum values of legacy test materials are read (only which materials own which
sources). Label-derived quantities (W, metal/gapped) are used for counts of usable
labels only, never to place a material. Output: results/runs/split_design_20261111/stats.json
and materials_meta.parquet (one row per material: pool, reduced formula, n_atoms, elements,
crystal system, label/source flags, formula-group hash bucket).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from multiprocessing import Pool

from common import ROOT, STORE

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = ROOT / 'results/runs/split_design_20261111'
STAGING = STORE / 'staging'
SALT = 'spectral-store-v1/split/2026-11'  # illustration only; the real salt is fixed when the user approves
MAX_ATOMS = 160


def alpha(i: int) -> str:
    from emmet.core.mpid import AlphaID
    s = str(AlphaID(int(i), prefix='mp'))
    return 'mp-' + s[3:].rjust(8, 'a')


def bucket(formula: str) -> float:
    return int(hashlib.sha256(f'{SALT}:{formula}'.encode()).hexdigest()[:12], 16) / 16 ** 12


def crystal_system(job):
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    out = []
    for material_id, cell, numbers, frac in job:
        try:
            s = Structure(Lattice(np.asarray(cell).reshape(3, 3)), numbers, np.asarray(frac).reshape(-1, 3))
            out.append((material_id, SpacegroupAnalyzer(s, symprec=0.1).get_crystal_system()))
        except Exception:
            out.append((material_id, 'unknown'))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pq.read_table(STORE / 'materials.parquet', columns=[
        'material_id', 'pool', 'n_atoms', 'numbers', 'cell', 'frac', 'reduced_composition', 'eval_composition_excluded',
        'deprecated']).to_pandas()
    from pymatgen.core import Composition, Element
    m['formula'] = [Composition({Element.from_Z(z).symbol: list(n).count(z) for z in set(n)}).reduced_formula
                    for n in m.numbers]
    m['elements'] = [sorted({Element.from_Z(int(z)).symbol for z in n}) for n in m.numbers]
    ids = set(m.material_id)
    # eDOS: canonical, usable (status ok, W <= 3), non-test only (test spectra are not read)
    f = pd.read_parquet(ROOT / 'results/runs/spectrum_anomaly_20261111/edos_features.parquet',
                        columns=['material_id', 'is_canonical', 'win_area', 'n', 'task_nval', 'file'])
    f = f[f.is_canonical & (f.file == 'edos')]
    f['W'] = f.win_area * f.n / f.task_nval
    m['edos_canonical'] = m.material_id.isin(set(f.material_id))
    m['edos_anomalous'] = m.material_id.isin(set(f[f.W > 3].material_id))
    # phDOS sources per material
    mp = pq.read_table(STAGING / 'phdos/part-00.parquet', columns=['material_id', 'method', 'status']).to_pandas()
    mp = mp[mp.status == 'ok']
    m['ph_mp'] = m.material_id.isin(set(mp.material_id))
    m['ph_mp_methods'] = m.material_id.map(mp.groupby('material_id').method.apply(lambda s: ','.join(sorted(set(s)))))
    index = json.loads((STORE / 'raw_phonondb/index.json').read_text())
    pdb = pd.DataFrame(index)
    pdb['material_id'] = pdb.mp_int.map(alpha)
    pdb['in_store'] = pdb.material_id.isin(ids)
    ing = pd.concat([pq.read_table(p, columns=['source_id', 'status', 'reduced_formula']).to_pandas()
                     for p in sorted((STAGING / 'phdos_phonondb').glob('part-*.parquet'))])
    pdb = pdb.merge(ing.rename(columns={'source_id': 'serial'}), on='serial', how='left')
    jar = pq.read_table(STAGING / 'phdos_jarvis/part-0000.parquet', columns=['source_id', 'mp_reference', 'status',
                                                                              'reduced_formula']).to_pandas()
    jar = jar[jar.status == 'ok'].copy()
    jar['material_id'] = jar.mp_reference.map(lambda r: alpha(r[3:]) if isinstance(r, str) and r.startswith('mp-')
                                              and r[3:].isdigit() else None)
    jar['in_store'] = jar.material_id.isin(ids)
    formula_of = dict(zip(m.material_id, m.formula))
    for d in (pdb, jar):
        d['formula_agrees'] = [formula_of.get(i) == r for i, r in zip(d.material_id, d.reduced_formula)]
    m['ph_phonondb_index'] = m.material_id.isin(set(pdb[pdb.in_store].material_id))
    m['ph_phonondb_ok'] = m.material_id.isin(set(pdb[pdb.in_store & (pdb.status == 'ok')].material_id))
    m['ph_jarvis_ok'] = m.material_id.isin(set(jar[jar.in_store].material_id))
    # legacy labels recovered (v2 view membership)
    kept = set()
    for split in ('train', 'valid'):
        kept |= set(np.load(STORE / f'views/q1_v2/{split}/{split}_index.npy'))
    kept |= set(np.load(STORE / 'sealed/views/q1_v2/test/test_index.npy'))
    m['legacy_v2_kept'] = m.material_id.isin(kept)
    m['group_bucket'] = m.formula.map(bucket)
    # crystal system (structure metadata only)
    rows = list(zip(m.material_id, m.cell, m.numbers, m.frac))
    chunks = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]
    with Pool(3) as pool:
        cs = dict(x for part in pool.imap_unordered(crystal_system, chunks) for x in part)
    m['crystal_system'] = m.material_id.map(cs)
    meta = m.drop(columns=['cell', 'frac', 'numbers'])
    meta.to_parquet(OUT / 'materials_meta.parquet')
    stats = {
        'phonondb_index': {'entries': len(pdb), 'mp_id_in_store': int(pdb.in_store.sum()),
                           'ingested_ok': int((pdb.status == 'ok').sum()),
                           'ingested_ok_in_store': int((pdb.in_store & (pdb.status == 'ok')).sum()),
                           'formula_agrees_among_ingested_in_store': int((pdb.in_store & (pdb.status == 'ok')
                                                                          & pdb.formula_agrees).sum())},
        'jarvis_ok_pages': {'n': len(jar), 'mp_reference_in_store': int(jar.in_store.sum()),
                            'formula_agrees_in_store': int((jar.in_store & jar.formula_agrees).sum()),
                            'no_mp_reference': int(jar.material_id.isna().sum())},
        'pools': Counter(m.pool),
    }
    (OUT / 'stats.json').write_text(json.dumps(stats, indent=1, default=int) + '\n')
    print(json.dumps(stats, indent=1, default=int))


if __name__ == '__main__':
    main()
