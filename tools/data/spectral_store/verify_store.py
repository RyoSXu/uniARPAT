"""P1 verification of spectral-store-v1 (design 2026-11-10 P0, "P1 核验").

Subcommands (each writes one JSON report under data/spectral_store_v1/verify/):

- compat:  the q1_compat view equals crystal_q1_v1 train/valid array by array.
- legacy:  for legacy materials, the new pipeline re-bins (a) the label task matched by
           the 2026-11-05 audit and (b) MP's canonical task / preferred MP phonon, and
           compares with the frozen labels (exact / differs only on spike-clipped bins /
           differs). Train and valid only; test labels are not read.
- distribution: n_atoms, element counts and element frequency per pool (train_q1,
           valid_q1, test_q1, new, new_eval_excluded) from input structures only.
- outliers: legacy materials whose label-task calculation volume differs from the
           input structure by more than 30% (from the 2026-11-05 audit table).
- fallback: eDOS fallback-task rule (common.choose_edos_task) checked where MP's canonical
           task exists: hide it, pick a fallback among the other owned tasks, compare with
           the canonical spectrum (E on the E0 grid with the project constants, per-atom by
           each task's own cell; shape TV). Also calc-type and functional distributions of
           canonical and candidate tasks. Reads staging tables; test_q1 materials skipped.
- fallback_v2: the same pairs under label contract v2 (common.exact_bin on both sides),
           by fallback calc type x functional x metal/gapped (from the canonical spectrum
           and from the fallback spectrum itself); plus the area kept by box_average
           relative to exact_bin on canonical eDOS and preferred MP phDOS (no test_q1).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from multiprocessing import Pool

from common import (LEGACY, ROOT, STORE, box_average, choose_edos_task, classify_label, exact_bin, load_calc_types,
                    strip_mp)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = STORE / 'verify'
GRIDS = {'edos': np.linspace(-6.0, 6.0, 129), 'phdos': np.linspace(-280.0, 980.0, 65)}
AUDIT = ROOT / 'results/runs/structure_use_audit_20261105'


def write(name, report):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1)[:3000])


def compat(view: Path):
    report = {}
    for split in ('train', 'valid'):
        entry = {}
        for name in ('cell', 'atomic_numbers', 'frac_coords', 'n_atoms', 'atom_offsets', 'nvalence',
                     'edos_tgtdos', 'phdos_tgtdos', 'edos_mask', 'phdos_mask'):
            a = np.load(view / split / f'{name}_{split}.npy')
            b = np.load(LEGACY / split / f'{name}_{split}.npy')
            entry[name] = bool(a.shape == b.shape and np.array_equal(a, b.astype(a.dtype)))
        a = np.load(view / split / f'{split}_index.npy').astype(str)
        b = np.load(LEGACY / split / f'{split}_index.npy').astype(str)
        entry['index'] = bool(np.array_equal(a, b))
        report[split] = entry
    report['all_equal'] = all(all(v.values()) for v in report.values() if isinstance(v, dict))
    write('compat', report)


classify = classify_label


def legacy():
    materials = pq.read_table(STORE / 'materials.parquet',
                              columns=['material_id', 'legacy_split', 'cell', 'n_atoms']).to_pylist()
    wanted = {m['material_id'] for m in materials if m['legacy_split'] in ('train', 'valid')}
    volume = {m['material_id']: abs(float(np.linalg.det(np.asarray(m['cell'], np.float64).reshape(3, 3)))) / m['n_atoms']
              for m in materials if m['material_id'] in wanted}
    calc_types = load_calc_types()
    labels = pq.read_table(STORE / 'spectra_legacy.parquet').to_pylist()
    labels = {(r['type'], r['material_id']): r for r in labels if r['material_id'] in wanted}
    report = {}
    for task in ('edos', 'phdos'):
        table = pq.read_table(STORE / f'spectra_{task}.parquet')
        table = table.filter(pc.is_in(table.column('material_id'), value_set=pa.array(sorted(wanted))))
        by_task, preferred, owned, rule_kind = {}, {}, {}, Counter()
        rank = {'pheasy': 0, 'dfpt': 1}
        for r in table.to_pylist():
            owned.setdefault(r['material_id'], []).append(r)
            if r['status'] not in ('ok', 'truncated'):
                continue
            by_task[r['source_task_id']] = r
            key = r['material_id']
            if task == 'phdos':
                cur = preferred.get(key)
                if cur is None or rank.get(r['method'], 9) < rank.get(cur['method'], 9):
                    preferred[key] = r
        if task == 'edos':  # the store rule: canonical, else the fallback task
            for material_id, rows in owned.items():
                chosen, kind = choose_edos_task(rows, volume[material_id], calc_types, FALLBACK_VARIANTS['type_then_volume'])
                if chosen is not None:
                    preferred[material_id] = chosen
                    rule_kind[kind] += 1
        counts = {'matched_task': Counter(), 'preferred_task': Counter(), 'preferred_fallback_only': Counter()}
        same_task = Counter()
        for (t, material_id), label in labels.items():
            if t != task:
                continue
            old, mask = np.asarray(label['values'], np.float32), np.asarray(label['mask'], bool)
            for kind, row in (('matched_task', by_task.get(label['matched_task'])),
                              ('preferred_task', preferred.get(material_id))):
                if row is None:
                    counts[kind]['no_spectrum'] += 1
                    continue
                new, _ = box_average(row['x'], row['y'], GRIDS[task])
                result = classify(new, old, mask)[0]
                counts[kind][result] += 1
                if kind == 'preferred_task' and task == 'edos' and not row['is_canonical']:
                    counts['preferred_fallback_only'][result] += 1
            if label['matched_task'] and material_id in preferred:
                same_task[preferred[material_id]['source_task_id'] == label['matched_task']] += 1
        report[task] = {k: dict(v) for k, v in counts.items()}
        report[task]['preferred_equals_matched_task'] = {str(k): v for k, v in same_task.items()}
        report[task]['preferred_rule_kind'] = dict(rule_kind)
    write('legacy', report)


def distribution():
    materials = pq.read_table(STORE / 'materials.parquet', columns=[
        'material_id', 'pool', 'eval_composition_excluded', 'n_atoms', 'numbers', 'structure_status']).to_pylist()
    groups = {}
    for m in materials:
        if not m['numbers']:
            continue
        name = m['pool'] if m['pool'] != 'new' else ('new_eval_excluded' if m['eval_composition_excluded'] else 'new')
        groups.setdefault(name, []).append(m)
    train = Counter(z for m in groups.get('train_q1', []) for z in set(m['numbers']))
    report = {}
    for name, rows in groups.items():  # structures only; no labels are read here
        atoms = np.asarray([m['n_atoms'] for m in rows])
        n_elements = np.asarray([len(set(m['numbers'])) for m in rows])
        elements = Counter(z for m in rows for z in set(m['numbers']))
        report[name] = {
            'materials': len(rows),
            'n_atoms_quantiles': {q: float(np.percentile(atoms, q)) for q in (10, 25, 50, 75, 90)},
            'fraction_1_to_4_atoms': float((atoms <= 4).mean()),
            'fraction_over_160_atoms': float((atoms > 160).mean()),
            'n_elements_distribution': {int(k): v / len(rows) for k, v in sorted(Counter(n_elements).items())},
            'top_elements_fraction': {int(z): round(c / len(rows), 4) for z, c in elements.most_common(15)},
            'fraction_with_element_unseen_in_train_q1': sum(any(train[z] == 0 for z in set(m['numbers']))
                                                            for m in rows) / len(rows)}
    write('distribution', report)


def outliers():
    rows = []
    for name, task in (('geometry', 'edos'), ('geometry_phdos', 'phdos')):
        with open(AUDIT / name / 'geometry_by_material.csv') as stream:
            for r in csv.DictReader(stream):
                if r['task'] == task and r['volume_dev'] and float(r['volume_dev']) > 0.30:
                    rows.append({'material_id': r['material_id'], 'task': task, 'split': r['split'],
                                 'formula': r['formula'], 'n_atoms': int(r['n_atoms']),
                                 'calc_n_atoms': float(r['calc_n_atoms']) if r['calc_n_atoms'] else None,
                                 'calc_volume_ratio': float(r['calc_volume_ratio']),
                                 'gate_class': r['gate_class'], 'dos_task': r['dos_task']})
    report = {'count': len(rows), 'by_task': dict(Counter(r['task'] for r in rows)),
              'by_split': dict(Counter(r['split'] for r in rows)),
              'atom_count_differs': sum(r['calc_n_atoms'] is not None and int(r['calc_n_atoms']) != r['n_atoms']
                                        for r in rows),
              'rows': rows}
    write('outliers', report)


E_CONST = {'edos': (0.09375, 5.542719504890618)}  # results/runs/spectral_residual_20261030/protocol.json
FALLBACK_VARIANTS = {'volume': (), 'type_then_volume': ('NSCF Uniform', 'Static')}


def _bin_tasks(job):
    path, wanted = job
    out = {}
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=256, columns=['source_task_id', 'x', 'y']):
        ids = batch.column('source_task_id').to_pylist()
        take = [i for i, t in enumerate(ids) if t in wanted]
        if take:
            for r in batch.take(pa.array(take)).to_pylist():
                out[r['source_task_id']] = box_average(r['x'], r['y'], GRIDS['edos'])[0]
    return out


def _shape(v):
    v = np.clip(np.asarray(v, np.float64), 0, None)
    total = v.sum()
    return v / total if total > 0 else np.zeros_like(v)


def fallback(workers: int):
    staging = STORE / 'staging'
    calc_types = load_calc_types()
    parts = sorted((staging / 'edos').glob('part-*.parquet'))
    cols = ['material_id', 'source_task_id', 'functional', 'status', 'is_canonical', 'calc_volume_per_atom',
            'calc_numbers']
    rows = pa.concat_tables([pq.read_table(f, columns=cols) for f in parts]).to_pylist()
    materials = {m['material_id']: m for m in pq.read_table(
        staging / 'materials.parquet', columns=['material_id', 'legacy_split', 'cell', 'n_atoms']).to_pylist()}
    volume = {}
    for material_id, m in materials.items():
        if m['cell'] and m['legacy_split'] != 'test':
            volume[material_id] = abs(float(np.linalg.det(np.asarray(m['cell'], np.float64).reshape(3, 3)))) / m['n_atoms']
    by = {}
    for r in rows:
        if r['material_id'] in volume:
            r['n_calc'] = len(r['calc_numbers'] or [])
            by.setdefault(r['material_id'], []).append(r)
    ctype = lambda r: calc_types.get(strip_mp(r['source_task_id']), 'missing')  # noqa: E731
    report = {'calc_types_known': len(calc_types)}
    canonical_rows = [r for rs in by.values() for r in rs if r['is_canonical'] and r['status'] == 'ok']
    report['canonical_calc_types'] = dict(Counter(ctype(r) for r in canonical_rows).most_common())
    no_canonical = [m for m, rs in by.items() if not any(r['is_canonical'] and r['status'] == 'ok' for r in rs)]
    report['no_canonical_materials'] = len(no_canonical)
    report['no_canonical_candidate_calc_types'] = dict(Counter(
        ctype(r) for m in no_canonical for r in by[m] if r['status'] == 'ok').most_common(15))
    # Hidden-canonical experiment
    trials, wanted = {}, set()
    for material_id, rs in by.items():
        canon = [r for r in rs if r['is_canonical'] and r['status'] == 'ok']
        others = [r for r in rs if not r['is_canonical']]
        if not canon:
            continue
        for name, order in FALLBACK_VARIANTS.items():
            chosen, kind = choose_edos_task(others, volume[material_id], calc_types, order)
            if chosen is not None:
                trials.setdefault(name, []).append((material_id, canon[0], chosen))
                wanted |= {canon[0]['source_task_id'], chosen['source_task_id']}
    with Pool(workers) as pool:
        binned = {}
        for part in pool.imap_unordered(_bin_tasks, [(p, wanted) for p in parts]):
            binned.update(part)
    width, energy = E_CONST['edos']
    for name, items in trials.items():
        e, tv, by_type, atoms_differ = [], [], {}, 0
        for material_id, canon, chosen in items:
            a, b = binned[canon['source_task_id']], binned[chosen['source_task_id']]
            na, nb = canon['n_calc'] or 1, chosen['n_calc'] or 1
            atoms_differ += na != nb
            value = float(width * (((a / na - b / nb) ** 2).sum()) / energy)
            t = float(.5 * np.abs(_shape(a) - _shape(b)).sum())
            e.append(value)
            tv.append(t)
            key = f"{ctype(chosen)} / same_functional={chosen['functional'] == canon['functional']}"
            by_type.setdefault(key, []).append((value, t))
        q = lambda v: {str(p): float(np.percentile(v, p)) for p in (10, 25, 50, 75, 90)}  # noqa: E731
        report[name] = {'materials': len(items), 'calc_atoms_differ': int(atoms_differ),
                        'E_quantiles': q(e), 'E_mean': float(np.mean(e)), 'TV_quantiles': q(tv),
                        'fraction_E_below_0.0526': float(np.mean(np.asarray(e) < 0.0526)),
                        'by_fallback_type': {k: {'n': len(v), 'E_median': float(np.median([x[0] for x in v])),
                                                 'TV_median': float(np.median([x[1] for x in v]))}
                                             for k, v in sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:12]}}
    # What the rule would pick where no canonical task exists
    for name, order in FALLBACK_VARIANTS.items():
        picked = [choose_edos_task(by[m], volume[m], calc_types, order)[0] for m in no_canonical]
        picked = [p for p in picked if p is not None]
        ratios = np.asarray([p['calc_volume_per_atom'] / volume[p['material_id']] for p in picked
                             if p['calc_volume_per_atom']])
        report[f'{name}_on_no_canonical'] = {
            'materials_with_pick': len(picked), 'calc_types': dict(Counter(ctype(p) for p in picked).most_common(8)),
            'functionals': dict(Counter(p['functional'] for p in picked)),
            'volume_ratio_quantiles': {str(p): float(np.percentile(ratios, p)) for p in (1, 10, 50, 90, 99)},
            'abs_volume_dev_gt_0.3': int((np.abs(ratios - 1) > 0.3).sum())}
    write('fallback', report)


def _bin_both(job):
    path, wanted, grid, x_scale = job
    out = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['source_task_id', 'x', 'y']):
        ids = batch.column('source_task_id').to_pylist()
        take = [i for i, t in enumerate(ids) if t in wanted]
        if take:
            for r in batch.take(pa.array(take)).to_pylist():
                x, y = np.asarray(r['x']), np.asarray(r['y'])
                exact = exact_bin(x, y, grid)[0].astype(np.float64)
                box = box_average(x, y, grid)[0].astype(np.float64)
                near = (x > -1.0) & (x < 1.0)
                out[r['source_task_id']] = (exact, box, _gap(x[near], y[near]) if x_scale else None)
    return out


def _gap(x, y, threshold=0.01):
    """Width (eV) of the empty region containing E=0 (0 if DOS at 0 is above threshold)."""
    if len(x) < 3:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    empty = y <= threshold  # states/eV per calculation cell
    i = int(np.searchsorted(x, 0.0))
    i = min(max(i, 1), len(x) - 1)
    if not (empty[i - 1] and empty[i]):
        return 0.0
    lo, hi = i - 1, i
    while lo > 0 and empty[lo - 1]:
        lo -= 1
    while hi < len(x) - 1 and empty[hi + 1]:
        hi += 1
    return float(x[hi] - x[lo])


def fallback_v2(workers: int):
    staging = STORE / 'staging'
    calc_types = load_calc_types()
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
        if r['material_id'] in volume:
            r['n_calc'] = len(r['calc_numbers'] or []) or 1
            by.setdefault(r['material_id'], []).append(r)
    ctype = lambda r: calc_types.get(strip_mp(r['source_task_id']), 'missing')  # noqa: E731
    pairs, wanted = [], set()
    for material_id, rs in by.items():
        canon = [r for r in rs if r['is_canonical'] and r['status'] == 'ok']
        if not canon:
            continue
        chosen, _ = choose_edos_task([r for r in rs if not r['is_canonical']], volume[material_id], calc_types,
                                     FALLBACK_VARIANTS['type_then_volume'])
        if chosen is not None:
            pairs.append((canon[0], chosen))
            wanted |= {canon[0]['source_task_id'], chosen['source_task_id']}
    with Pool(workers) as pool:
        binned = {}
        for part in pool.imap_unordered(_bin_both, [(p, wanted, GRIDS['edos'], True) for p in parts]):
            binned.update(part)
    width, energy = E_CONST['edos']
    e_of = lambda a, na, b, nb: float(width * (((a / na - b / nb) ** 2).sum()) / energy)  # noqa: E731
    groups, area = {}, []
    for canon, chosen in pairs:
        ca, cb, cgap = binned[canon['source_task_id']]
        fa, fb, fgap = binned[chosen['source_task_id']]
        if ca.sum() > 0:
            area.append(cb.sum() / ca.sum())
        kind = ctype(chosen).replace('GGA+U ', 'GGA+U|').replace('GGA ', 'GGA|')
        canon_class = 'metal' if (cgap or 0) < 0.15 else 'gapped'
        self_class = 'metal' if (fgap or 0) < 0.3 else 'gapped'  # coarse Static grids: wider threshold
        value = (e_of(ca, canon['n_calc'], fa, chosen['n_calc']),
                 e_of(cb, canon['n_calc'], fb, chosen['n_calc']),
                 float(.5 * np.abs(_shape(ca) - _shape(fa)).sum()))
        for key in (f'{kind} / canonical:{canon_class}', f'{kind} / self:{self_class}', f'{kind} / all'):
            groups.setdefault(key, []).append(value)

    def summary(v):
        v = np.asarray(v)
        return {'n': len(v), 'E_v2_median': float(np.median(v[:, 0])), 'E_v2_mean': float(v[:, 0].mean()),
                'E_v2_p90': float(np.percentile(v[:, 0], 90)), 'fraction_E_v2_le_0.15': float((v[:, 0] <= .15).mean()),
                'E_v1_median': float(np.median(v[:, 1])), 'TV_v2_median': float(np.median(v[:, 2]))}
    report = {'pairs': len(pairs), 'groups': {k: summary(v) for k, v in sorted(groups.items()) if len(v) >= 20},
              'edos_canonical_area_kept_by_box_average': {str(q): float(np.percentile(area, q)) for q in (10, 50, 90)}}
    # phDOS area kept (preferred MP phonon of non-test materials)
    ph = pq.read_table(staging / 'phdos' / 'part-00.parquet', columns=['material_id', 'status', 'x', 'y']).to_pylist()
    kept = []
    for r in ph:
        if r['status'] == 'ok' and r['material_id'] in volume:
            e = exact_bin(r['x'], r['y'], GRIDS['phdos'])[0].sum()
            if e > 0:
                kept.append(box_average(r['x'], r['y'], GRIDS['phdos'])[0].sum() / e)
    report['phdos_area_kept_by_box_average'] = {str(q): float(np.percentile(kept, q)) for q in (10, 50, 90)}
    write('fallback_v2', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check', choices=['compat', 'legacy', 'distribution', 'outliers', 'fallback', 'fallback_v2'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--view', default=str(STORE / 'views/q1_compat'))
    args = parser.parse_args()
    {'compat': lambda: compat(Path(args.view)), 'legacy': legacy, 'distribution': distribution,
     'outliers': outliers, 'fallback': lambda: fallback(args.workers),
     'fallback_v2': lambda: fallback_v2(args.workers)}[args.check]()


if __name__ == '__main__':
    main()
