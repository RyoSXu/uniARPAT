"""Recover the native eDOS behind every legacy eDOS label (design 2026-11-11 step 1b).

Two kinds of targets:
- known: label tasks matched by the 2026-11-05 audit that the store lacks because MP now
  assigns the task to another material (train/valid). They are stored as they are.
- search: legacy labels without a matched task (59 train/valid; all of test_q1, which the
  audit did not read). Every Delta total-dos task whose structure has the same reduced
  formula is re-binned with the legacy rule (box_average, E0) and compared with the frozen
  label (common.classify_label, at most 4 spike-clipped bins). The match is the exact one,
  else the one with fewest clipped bins, then the smallest task id.

Outputs (refuse to overwrite):
- staging/edos_legacy/part-*.parquet: native rows (EDOS_SCHEMA; material_id = the legacy
  material, is_canonical False) for train/valid targets;
- staging/edos_legacy/links.parquet: material, task, link kind, gate class, MP owner, number
  of matching tasks (train/valid);
- sealed/edos_legacy_test.parquet and sealed/edos_legacy_links_test.parquet: the same for
  test_q1. Only counts are printed; nothing about test labels or spectra is shown.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from multiprocessing import Pool

from build_store import EDOS_SCHEMA, STAGING, delta_files, edos_record, load_index
from common import STORE, box_average, classify_label

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

E0 = np.linspace(-6.0, 6.0, 129)
MAX_CLIPPED = 4
OUT = STAGING / 'edos_legacy'


def reduced_formula(structure: dict):
    from pymatgen.core import Composition
    counts = Counter(site['species'][0]['element'] for site in structure['sites'])
    return Composition(counts).reduced_formula


def _scan(job):
    path, run_type, known, by_formula = job
    found, gate = [], []
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=256, columns=[
            'identifier', 'structure', 'energies', 'spin_up_densities', 'spin_down_densities', 'efermi']):
        for row in batch.to_pylist():
            task = row['identifier']
            if task in known:
                found.append(edos_record(row, run_type, known[task], False))
            try:
                formula = reduced_formula(row['structure'])
            except Exception:
                continue
            targets = by_formula.get(formula)
            if not targets:
                continue
            record = edos_record(row, run_type, None, False)
            if record['status'].startswith('unreadable'):
                continue
            binned, _ = box_average(record['x'], record['y'], E0)
            for material_id, values, mask in targets:
                kind, off = classify_label(binned, values, mask, MAX_CLIPPED)
                if kind != 'differs':
                    gate.append((material_id, task, kind, off, dict(record, material_id=material_id)))
    return found, gate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=7)
    args = parser.parse_args()
    for path in (OUT, STORE / 'sealed' / 'edos_legacy_test.parquet'):
        if path.exists():
            raise FileExistsError(path)
    index = load_index()
    owner = index['task_owner']
    stored = set(pa.concat_tables([pq.read_table(f, columns=['source_task_id'])
                                   for f in sorted((STAGING / 'edos').glob('part-*.parquet'))]).column(0).to_pylist())
    labels = pq.read_table(STAGING / 'legacy_spectra.parquet').to_pylist()
    materials = {m['material_id']: m for m in pq.read_table(
        STAGING / 'materials.parquet', columns=['material_id', 'legacy_split', 'numbers']).to_pylist()}
    known, search = {}, {}
    for r in labels:
        if r['type'] != 'edos':
            continue
        if r['matched_task'] and r['matched_task'] not in stored:
            known[r['matched_task']] = r['material_id']
        elif not r['matched_task']:
            search[r['material_id']] = (np.asarray(r['values'], np.float32), np.asarray(r['mask'], bool))
    from pymatgen.core import Composition, Element
    by_formula = {}
    for material_id, (values, mask) in search.items():
        numbers = materials[material_id]['numbers']
        formula = Composition({Element.from_Z(z).symbol: numbers.count(z) for z in set(numbers)}).reduced_formula
        by_formula.setdefault(formula, []).append((material_id, values, mask))
    jobs = [(path, run_type, known, by_formula) for path, run_type in delta_files(
        'core/electronic-structure/total-dos', 'run_type')]
    jobs.sort(key=lambda job: -job[0].stat().st_size)
    found, gate = [], []
    with Pool(min(args.workers, len(jobs))) as pool:
        for f, g in pool.imap_unordered(_scan, jobs):
            found.extend(f)
            gate.extend(g)
    best = {}
    matches = Counter(m for m, *_ in gate)
    for material_id, task, kind, off, record in gate:
        key = (kind != 'exact', off, task)
        if material_id not in best or key < best[material_id][0]:
            best[material_id] = (key, task, kind, record)
    split = {m: materials[m]['legacy_split'] for m in materials}
    rows = {'public': [], 'sealed': []}
    links = {'public': [], 'sealed': []}
    for record in found:
        m = record['material_id']
        rows['public'].append(record)
        links['public'].append({'material_id': m, 'source_task_id': record['source_task_id'],
                                'link': 'audit_matched_other_owner', 'gate_class': 'audit',
                                'mp_owner': owner.get(record['source_task_id']), 'n_matching_tasks': None})
    for material_id, (_, task, kind, record) in best.items():
        side = 'sealed' if split[material_id] == 'test' else 'public'
        rows[side].append(record)
        links[side].append({'material_id': material_id, 'source_task_id': task, 'link': 'gate_matched',
                            'gate_class': kind, 'mp_owner': owner.get(task), 'n_matching_tasks': matches[material_id]})
    OUT.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows['public'], schema=EDOS_SCHEMA), OUT / 'part-00.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pylist(links['public']), OUT / 'links.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pylist(rows['sealed'], schema=EDOS_SCHEMA), STORE / 'sealed' / 'edos_legacy_test.parquet',
                   compression='zstd')
    pq.write_table(pa.Table.from_pylist(links['sealed']), STORE / 'sealed' / 'edos_legacy_links_test.parquet',
                   compression='zstd')
    count = lambda s: dict(Counter(f"{split[m]}/{'matched' if m in best else 'unmatched'}" for m in s))  # noqa: E731
    report = {'known_targets': len(known), 'known_found': len(found),
              'search_targets': count(search),
              'search_gate_class': dict(Counter(f'{split[m]}/{best[m][2]}' for m in best)),
              'search_ambiguous_materials': sum(matches[m] > 1 for m in best)}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
