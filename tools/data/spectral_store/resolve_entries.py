"""Resolve records that link_records/build_splits_v2 left unattached (user decision 2026-11-12).

1. Ambiguous records (pass the default matcher and crystal system against more than one
   same-formula MP material): re-checked with the tight matcher; exactly one tight match -> the
   record attaches to that material; otherwise it stays an independent entry.
2. Independent entries (still ambiguous, or no match): within a reduced formula, entries are
   grouped when they pass the default matcher with the same crystal system against every member
   of the group (complete linkage; greedy in priority order). A group is one unit whose id and
   input structure are those of its first member: MP task > PhononDB > JARVIS, then record id.
Structures are those of records_v11 (no spectra are read).
Output: STORE/staging/identity/entry_resolution.parquet with record_id, material_id (attached MP
material or None), unit_id (group representative for entries, else None), resolution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from common import STORE

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from link_records import DEFAULT, TIGHT, _struct

OUT = STORE / 'staging' / 'identity' / 'entry_resolution.parquet'
ORDER = {'mp_edos': 0, 'mp_phonon': 0, 'phonondb': 1, 'jarvis': 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    if OUT.exists():
        raise FileExistsError(OUT)
    from pymatgen.analysis.structure_matcher import StructureMatcher
    default, tight = StructureMatcher(**DEFAULT), StructureMatcher(**TIGHT)
    links = pd.read_parquet(STORE / 'records_v11/record_links.parquet')
    features = pd.read_parquet(STORE / 'staging/identity/record_features.parquet',
                               columns=['record_id', 'cs_record', 'alt_matches']).set_index('record_id')
    structures = pd.read_parquet(STORE / 'records_v11/record_structures.parquet').set_index('record_id')
    materials = {m['material_id']: m for m in pq.read_table(
        STORE / 'materials.parquet', columns=['material_id', 'cell', 'numbers', 'frac']).to_pylist()}

    def record_structure(rid):
        s = structures.loc[rid]
        return _struct(s.lattice, s.species, s.frac)

    rows, stats = [], Counter()
    entries = []
    for r in links[links.link.isin(['ambiguous', 'no_match'])].itertuples():
        if r.link == 'ambiguous':
            structure = record_structure(r.record_id)
            candidates = [mid for mid, _ in json.loads(features.loc[r.record_id, 'alt_matches'])]
            passing = [mid for mid in candidates
                       if tight.fit(_struct(materials[mid]['cell'], [int(z) for z in materials[mid]['numbers']],
                                            materials[mid]['frac']), structure)]
            if len(passing) == 1:
                rows.append({'record_id': r.record_id, 'material_id': passing[0], 'unit_id': None,
                             'resolution': 'tight_unique'})
                stats['ambiguous_attached'] += 1
                continue
            stats['ambiguous_kept'] += 1
        entries.append(r)
    entries.sort(key=lambda r: (r.formula, ORDER[r.source], r.record_id))
    groups = []  # (formula, [(rid, structure, crystal system)])
    for r in entries:
        structure, cs = record_structure(r.record_id), features.loc[r.record_id, 'cs_record']
        for formula, members in groups:
            if formula == r.formula and all(cs == c and default.fit(s, structure) for _, s, c in members):
                members.append((r.record_id, structure, cs))
                break
        else:
            groups.append((r.formula, [(r.record_id, structure, cs)]))
    for _, members in groups:
        representative = members[0][0]
        stats['entry_units'] += 1
        stats[f'entry_units_of_{len(members) if len(members) < 4 else "4+"}'] += 1
        for rid, _, _ in members:
            rows.append({'record_id': rid, 'material_id': None, 'unit_id': representative,
                         'resolution': 'entry_representative' if rid == representative else 'entry_member'})
    frame = pd.DataFrame(rows)
    if frame.record_id.duplicated().any() or len(frame) != stats['ambiguous_attached'] + len(entries):
        raise ValueError('every unattached record must be resolved exactly once')
    frame.to_parquet(OUT)
    sources = frame.merge(links[['record_id', 'source']]).groupby(['resolution', 'source']).size()
    print(json.dumps({**stats, 'records': len(frame)}, indent=1))
    print(sources.to_string())
    cross = frame[frame.unit_id.notna()].merge(links[['record_id', 'source']]).groupby('unit_id').source.nunique()
    print('units with records from more than one source', int((cross > 1).sum()), int(np.sum(cross > 0)))


if __name__ == '__main__':
    main()
