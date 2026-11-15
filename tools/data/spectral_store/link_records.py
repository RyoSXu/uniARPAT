"""Record-level structure identity (design splits-v2, 2026-11-12 rules 1-4). Read-only features.

A record is one calculation with its own id and its own structure: an MP phonon task, an MP
canonical eDOS task, a PhononDB entry, a JARVIS page. Its MP id (owner, mp_int or
mp_reference) is only a hint. For each record:
- against the hinted MP material: default StructureMatcher (ltol 0.2, stol 0.3, 5 deg, volume
  scaled), a tight one (0.75x tolerances), crystal system of both (symprec 0.1), normalised
  RMS displacement and per-atom volume ratio;
- if that fails (or there is no hint): every other store material with the same reduced
  formula, keeping those that pass default + same crystal system (with their RMS).
Only structures are read (legacy test records included: structure only, no spectra).
Output: STORE/staging/identity/record_features.parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from multiprocessing import Pool

from common import STORE

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STAGING = STORE / 'staging'
OUT = STAGING / 'identity'
DEFAULT = dict(ltol=.2, stol=.3, angle_tol=5, scale=True)
TIGHT = dict(ltol=.15, stol=.225, angle_tol=3.75, scale=True)
MATERIALS = {}  # material_id -> (lattice, species, frac, crystal system, formula); filled before fork


def alpha(i):
    from emmet.core.mpid import AlphaID
    s = str(AlphaID(int(i), prefix='mp'))
    return 'mp-' + s[3:].rjust(8, 'a')


def _struct(lattice, species, coords, cartesian=False):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice(np.asarray(lattice, np.float64).reshape(3, 3)), list(species),
                     np.asarray(coords, np.float64).reshape(-1, 3), coords_are_cartesian=cartesian)


def _crystal_system(s):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        return SpacegroupAnalyzer(s, symprec=0.1).get_crystal_system()
    except Exception:
        return 'unknown'


def _compare(job):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    default, tight = StructureMatcher(**DEFAULT), StructureMatcher(**TIGHT)
    by_formula = _compare.by_formula
    out = []
    for record_id, source, hint, formula, structure in job:
        row = {'record_id': record_id, 'source': source, 'hint': hint, 'formula': formula}
        try:
            r = _struct(*structure)
        except Exception as exc:
            out.append(dict(row, error=f'structure:{type(exc).__name__}'))
            continue
        row['n_record'], row['cs_record'] = len(r), _crystal_system(r)
        hint_ok = False
        if hint in MATERIALS:
            lat, sp, frac, cs, _ = MATERIALS[hint]
            m = _struct(lat, sp, frac)
            rms = default.get_rms_dist(m, r)
            row.update(hint_in_store=True, cs_hint=cs, volume_ratio=(r.volume / len(r)) / (m.volume / len(m)),
                       default=rms is not None, tight=bool(tight.fit(m, r)),
                       rms=None if rms is None else float(rms[0]))
            hint_ok = row['default'] and cs == row['cs_record']
        else:
            row['hint_in_store'] = False
        if not hint_ok:
            matches = []
            for other in by_formula.get(formula, []):
                if other == hint:
                    continue
                lat, sp, frac, cs, _ = MATERIALS[other]
                if cs != row['cs_record']:
                    continue
                rms = default.get_rms_dist(_struct(lat, sp, frac), r)
                if rms is not None:
                    matches.append((other, float(rms[0])))
            row['alt_matches'] = json.dumps(sorted(matches, key=lambda x: x[1]))
        out.append(row)
    return out


def records():
    out = []
    ph = pq.read_table(STAGING / 'phdos/part-00.parquet', columns=[
        'source_task_id', 'material_id', 'status', 'calc_lattice', 'calc_numbers', 'calc_frac']).to_pylist()
    for r in ph:
        if r['status'] == 'ok' and r['calc_numbers']:
            out.append((f"mp_phonon:{r['source_task_id']}", 'mp_phonon', r['material_id'],
                        (r['calc_lattice'], [int(z) for z in r['calc_numbers']], r['calc_frac'])))
    for path in (STORE / 'spectra_edos.parquet', STORE / 'sealed/spectra_edos.parquet'):
        e = pq.read_table(path, columns=['source_task_id', 'material_id', 'calc_lattice', 'calc_numbers', 'calc_frac'],
                          filters=[('is_canonical', '=', True), ('status', '=', 'ok')]).to_pylist()
        for r in e:
            out.append((f"mp_edos:{r['source_task_id']}", 'mp_edos', r['material_id'],
                        (r['calc_lattice'], [int(z) for z in r['calc_numbers']], r['calc_frac'])))
    index = {d['serial']: alpha(d['mp_int']) for d in json.loads((STORE / 'raw_phonondb/index.json').read_text())}
    for part in sorted((STAGING / 'phdos_phonondb').glob('part-*.parquet')):
        for r in pq.read_table(part, columns=['source_id', 'status', 'unit_cell']).to_pylist():
            if r['status'] == 'ok' and r['unit_cell']:
                cell = json.loads(r['unit_cell'])
                out.append((f"phonondb:{r['source_id']}", 'phonondb', index.get(r['source_id']),
                            (cell['lattice'], [p['symbol'] for p in cell['points']],
                             [p['coordinates'] for p in cell['points']])))
    jar = pq.read_table(STAGING / 'phdos_jarvis/part-0000.parquet', columns=['source_id', 'mp_reference', 'status']).to_pandas()
    jar = jar[jar.status == 'ok']
    hint = {j: (alpha(r[3:]) if isinstance(r, str) and r[3:].isdigit() else None)
            for j, r in zip(jar.source_id, jar.mp_reference)}
    with zipfile.ZipFile(STORE / 'raw_jarvis/jdft_3d-12-12-2022.json.zip') as z:
        data = json.loads(z.read(z.namelist()[0]))
    for d in data:
        if d['jid'] in hint:
            a = d['atoms']
            out.append((f"jarvis:{d['jid']}", 'jarvis', hint[d['jid']],
                        (a['lattice_mat'], a['elements'], a['coords'], bool(a.get('cartesian')))))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    path = OUT / 'record_features.parquet'
    if path.exists():
        raise FileExistsError(path)
    from pymatgen.core import Composition, Element
    meta = pd.read_parquet(STORE.parent.parent / 'results/runs/split_design_20261111/materials_meta.parquet',
                           columns=['material_id', 'crystal_system'])
    cs = dict(zip(meta.material_id, meta.crystal_system))
    m = pq.read_table(STORE / 'materials.parquet', columns=['material_id', 'cell', 'numbers', 'frac']).to_pylist()
    by_formula = {}
    for r in m:
        numbers = [int(z) for z in r['numbers']]
        formula = Composition({Element.from_Z(z).symbol: numbers.count(z) for z in set(numbers)}).reduced_formula
        MATERIALS[r['material_id']] = (r['cell'], numbers, r['frac'], cs.get(r['material_id'], 'unknown'), formula)
        by_formula.setdefault(formula, []).append(r['material_id'])
    _compare.by_formula = by_formula
    jobs = []
    for record_id, source, hint, structure in records():
        species = structure[1]
        symbols = [Element.from_Z(s).symbol if isinstance(s, int) else s for s in species]
        formula = Composition({e: symbols.count(e) for e in set(symbols)}).reduced_formula
        jobs.append((record_id, source, hint, formula, structure))
    print('records', len(jobs), flush=True)
    os.nice(10)
    chunks = [jobs[i:i + 400] for i in range(0, len(jobs), 400)]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(_compare, chunks) for r in part]
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(path)
    print(frame.groupby('source').agg(n=('record_id', 'size'), default=('default', 'mean'),
                                      tight=('tight', 'mean')).to_string())


if __name__ == '__main__':
    main()
