"""Read-only structure identity check for the step-4 design (proposal A).

For every non-test material, compare the store's input structure with the structure behind
each spectrum source:
- MP canonical eDOS task: calculation cell (spectra_edos calc_*);
- MP phonon (preferred pheasy, else dfpt): calculation cell (staging/phdos calc_*);
- PhononDB: the archive's unit cell (staging/phdos_phonondb unit_cell), mapped by MP id;
- JARVIS: the dft_3d (12-12-2022) structure of the JVASP id, mapped by mp_reference.

Two matchers (pymatgen StructureMatcher):
- strict  (ltol 0.01, stol 0.05, angle 0.5 deg, no volume scaling) -- the 11-05 audit rule;
- default (ltol 0.2, stol 0.3, angle 5 deg, volume scaled) -- same prototype/phase.
Also the per-atom volume ratio (source / input). Output:
results/runs/split_design_20261111/structure_identity.parquet (one row per material x source).
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from multiprocessing import Pool

from common import ROOT, STORE

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = ROOT / 'results/runs/split_design_20261111'
STAGING = STORE / 'staging'
STRICT = dict(ltol=.01, stol=.05, angle_tol=.5, scale=False)
DEFAULT = dict(ltol=.2, stol=.3, angle_tol=5, scale=True)


def alpha(i):
    from emmet.core.mpid import AlphaID
    s = str(AlphaID(int(i), prefix='mp'))
    return 'mp-' + s[3:].rjust(8, 'a')


def _struct(lattice, species, coords, cartesian=False):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice(np.asarray(lattice, np.float64).reshape(3, 3)), species,
                     np.asarray(coords, np.float64).reshape(-1, 3), coords_are_cartesian=cartesian)


def _match(job):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    strict, default = StructureMatcher(**STRICT), StructureMatcher(**DEFAULT)
    out = []
    for material_id, source, left, right in job:
        try:
            a, b = _struct(*left), _struct(*right)
            ratio = (b.volume / len(b)) / (a.volume / len(a))
            out.append({'material_id': material_id, 'source': source, 'n_input': len(a), 'n_source': len(b),
                        'volume_ratio': float(ratio), 'strict': bool(strict.fit(a, b)),
                        'default': bool(default.fit(a, b)), 'error': None})
        except Exception as exc:
            out.append({'material_id': material_id, 'source': source, 'error': f'{type(exc).__name__}'})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    path = OUT / 'structure_identity.parquet'
    if path.exists():
        raise FileExistsError(path)
    m = pq.read_table(STORE / 'materials.parquet', columns=['material_id', 'pool', 'cell', 'numbers', 'frac']).to_pandas()
    m = m[m.pool != 'test_q1']
    inp = {r.material_id: (r.cell, [int(z) for z in r.numbers], r.frac) for r in m.itertuples()}
    jobs = []
    e = pq.read_table(STORE / 'spectra_edos.parquet', columns=['material_id', 'is_canonical', 'status', 'calc_lattice',
                                                                'calc_numbers', 'calc_frac'],
                      filters=[('is_canonical', '=', True), ('status', '=', 'ok')]).to_pandas()
    for r in e.drop_duplicates('material_id').itertuples():
        if r.material_id in inp:
            jobs.append((r.material_id, 'mp_edos', inp[r.material_id],
                         (r.calc_lattice, [int(z) for z in r.calc_numbers], r.calc_frac)))
    ph = pq.read_table(STAGING / 'phdos/part-00.parquet', columns=['material_id', 'method', 'status', 'calc_lattice',
                                                                     'calc_numbers', 'calc_frac']).to_pandas()
    ph = ph[ph.status == 'ok'].assign(rank=lambda d: d.method.map({'pheasy': 0, 'dfpt': 1}).fillna(9))
    for r in ph.sort_values('rank').drop_duplicates('material_id').itertuples():
        if r.material_id in inp:
            jobs.append((r.material_id, 'mp_phonon', inp[r.material_id],
                         (r.calc_lattice, [int(z) for z in r.calc_numbers], r.calc_frac)))
    index = {d['serial']: alpha(d['mp_int']) for d in json.loads((STORE / 'raw_phonondb/index.json').read_text())}
    for part in sorted((STAGING / 'phdos_phonondb').glob('part-*.parquet')):
        for r in pq.read_table(part, columns=['source_id', 'status', 'unit_cell']).to_pylist():
            mid = index.get(r['source_id'])
            if r['status'] == 'ok' and mid in inp and r['unit_cell']:
                cell = json.loads(r['unit_cell'])
                jobs.append((mid, 'phonondb', inp[mid], (cell['lattice'], [p['symbol'] for p in cell['points']],
                                                          [p['coordinates'] for p in cell['points']])))
    jar = pq.read_table(STAGING / 'phdos_jarvis/part-0000.parquet', columns=['source_id', 'mp_reference', 'status']).to_pandas()
    jar = jar[(jar.status == 'ok') & jar.mp_reference.fillna('').str.match(r'^mp-\d+$')]
    want = dict(zip(jar.source_id, jar.mp_reference.map(lambda r: alpha(r[3:]))))
    with zipfile.ZipFile(STORE / 'raw_jarvis/jdft_3d-12-12-2022.json.zip') as z:
        records = json.loads(z.read(z.namelist()[0]))
    for d in records:
        mid = want.get(d['jid'])
        if mid in inp:
            atoms = d['atoms']
            jobs.append((mid, 'jarvis', inp[mid], (atoms['lattice_mat'], atoms['elements'], atoms['coords'],
                                                   bool(atoms.get('cartesian')))))
    del records
    os.nice(19)
    chunks = [jobs[i:i + 500] for i in range(0, len(jobs), 500)]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(_match, chunks) for r in part]
    frame = pd.DataFrame(rows)
    frame.to_parquet(path)
    print(frame.groupby('source').agg(n=('material_id', 'size'), strict=('strict', 'mean'), default=('default', 'mean'),
                                      errors=('error', lambda s: s.notna().sum())).to_string())


if __name__ == '__main__':
    main()
