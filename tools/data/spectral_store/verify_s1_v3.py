"""Read-only checks of the full re-split views s1_v3 / s1_v3_x2 (sealed test by digests and script only).

1. Every manifest payload matches its sha256; train and valid load through the training dataset.
2. Unlabelled targets are zero with an all-false mask; new labelled rows have a covered bin.
3. Every legacy row equals its q1_v2 row (from whichever old split it came): identical geometry;
   spectra of the labels still taken from the legacy export equal except bins clipped by either
   view's isolated-spike thresholds.
4. Leakage recomputed from the written lists: no reduced formula in more than one set; every
   list member is placed by splits_v3.assign; the lists cover every legacy material; every listed
   independent entry is the representative of its resolve_entries group.
5. Variants with label rules (v3_1): no row is labelled by an excluded source; a legacy label is
   used exactly when it is not in exclusions.parquet, and label presence matches its replacement.
6. When labels.parquet exists (v3_2): it lists exactly the labelled rows of the main view, and no
   record label breaks the volume / hint-deviation / ambiguous-identity rules.
Printed output is counts only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from common import ROOT, STORE
from verify_s1_v2 import check_dataset, check_payload, check_unlabelled, load

sys.path.insert(0, str(ROOT))

PUBLIC, SEALED = ('train', 'valid'), ('test',)
TASKS = ('edos', 'phdos')


def check_legacy(root, split, manifest, sources):
    """sources: (root, split, manifest) of q1_v2 train / valid / sealed test."""
    ids = load(root, split, 'index')
    code = manifest['source_codes']['legacy']
    is_legacy = {t: load(root, split, f'source_{t}') == code for t in TASKS}
    rows = np.flatnonzero(is_legacy['edos'] | is_legacy['phdos'])
    found = np.zeros(len(ids), int)
    entry = {'legacy_rows': int(len(rows))}
    offsets, numbers, frac = (load(root, split, k) for k in ('atom_offsets', 'atomic_numbers', 'frac_coords'))
    for legacy_root, legacy_split, legacy_manifest in sources:
        legacy_ids = load(legacy_root, legacy_split, 'index')
        position = {m: i for i, m in enumerate(legacy_ids)}
        mine = np.asarray([i for i in rows if ids[i] in position], int)
        if not len(mine):
            continue
        found[mine] += 1
        other = np.asarray([position[ids[i]] for i in mine])
        for key in ('cell', 'n_atoms', 'nvalence'):
            if not np.array_equal(load(root, split, key)[mine], load(legacy_root, legacy_split, key)[other]):
                raise ValueError(f'{split}: legacy {key} differs from q1_v2/{legacy_split}')
        lo, ln, lf = (load(legacy_root, legacy_split, k) for k in ('atom_offsets', 'atomic_numbers', 'frac_coords'))
        for i, j in zip(mine, other):
            a, b = slice(offsets[i], offsets[i + 1]), slice(lo[j], lo[j + 1])
            if not (np.array_equal(numbers[a], ln[b]) and np.array_equal(frac[a], lf[b])):
                raise ValueError(f'{split}: legacy atoms differ for {ids[i]}')
        part = {'rows': int(len(mine))}
        for task in TASKS:
            keep = is_legacy[task][mine]
            values = load(root, split, f'{task}_tgtdos')[mine[keep]]
            theirs = load(legacy_root, legacy_split, f'{task}_tgtdos')[other[keep]]
            if not np.array_equal(load(root, split, f'{task}_mask')[mine[keep]],
                                  load(legacy_root, legacy_split, f'{task}_mask')[other[keep]]):
                raise ValueError(f'{split}/{task}: legacy coverage differs')
            mine_threshold = np.asarray(manifest['spike_thresholds'][task], np.float32)
            their_threshold = np.asarray(legacy_manifest['spike_thresholds'][task], np.float32)
            differ = values != theirs
            if (differ & ~((values == mine_threshold) | (theirs == their_threshold))).any():
                raise ValueError(f'{split}/{task}: legacy spectra differ beyond spike clipping')
            part[f'{task}_clip_only_bins'] = int(differ.sum())
            part[f'{task}_legacy_rows'] = int(keep.sum())
        entry[f'from_q1_v2_{legacy_split}'] = part
    if (found[rows] != 1).any():
        raise ValueError(f'{split}: legacy rows not found exactly once in q1_v2')
    return entry


def check_rules(root, split, manifest, exclusions, variant_cfg):
    """Excluded sources never label a row; legacy labels are used exactly when not in the exclusions table."""
    ids = load(root, split, 'index')
    table = exclusions[exclusions.set == split].set_index(['unit_id', 'task'])
    out = {}
    for task in TASKS:
        source, has = load(root, split, f'source_{task}'), load(root, split, f'has_{task}')
        for kind in variant_cfg['exclude_kinds'].get(task, ()):
            if (source == manifest['source_codes'][kind]).any():
                raise ValueError(f'{split}/{task}: a {kind} label is present')
        listed = np.asarray([(m, task) in table.index for m in ids])
        if (listed & (source == manifest['source_codes']['legacy'])).any():
            raise ValueError(f'{split}/{task}: an excluded legacy label is still used')
        replaced = np.asarray([listed[i] and table.loc[(m, task), 'replacement'] is not None for i, m in enumerate(ids)])
        if (has != (~listed | replaced))[listed].any():
            raise ValueError(f'{split}/{task}: label presence disagrees with the exclusions table')
        out[task] = {'excluded_rows': int(listed.sum()), 'replaced_rows': int(replaced.sum())}
    return out


def check_provenance(root, split, labels, variant_cfg):
    """labels.parquet lists exactly the labelled rows; record labels obey the deviation rules."""
    import pandas as pd
    from build_splits_v3 import hint_deviation

    ids = load(root, split, 'index')
    table = labels[labels.set == split]
    out = {}
    for task in TASKS:
        rows = table[table.task == task]
        if sorted(rows.unit_id) != sorted(ids[load(root, split, f'has_{task}')]):
            raise ValueError(f'{split}/{task}: labels.parquet does not match the labelled rows')
        limit = variant_cfg['max_volume_deviation']
        records = rows[rows.link != 'legacy']
        if limit is not None:
            features = pd.read_parquet(STORE / 'staging/identity/record_features.parquet').set_index('record_id')
            resolution = pd.read_parquet(STORE / 'staging/identity/entry_resolution.parquet')
            tight = set(resolution.record_id[resolution.resolution == 'tight_unique'])
            bad = ((records.volume_ratio - 1).abs() > limit) | records.origin.isin(tight) | records.origin.map(
                lambda r: hint_deviation(features, r, limit) is not None)
            if bad.any():
                raise ValueError(f'{split}/{task}: {int(bad.sum())} record labels break the deviation rules')
        out[task] = {'labels': int(len(rows)), 'record_labels': int(len(records))}
    return out


def check_leakage(variant):
    from build_splits_v2 import formula_of
    from splits_v3 import assign

    lists = {s: [x for x in (STORE / f'splits_{variant}' / f'{s}.txt').read_text().split('\n') if x] for s in PUBLIC}
    lists['test'] = [x for x in (STORE / f'sealed/splits_{variant}/test.txt').read_text().split('\n') if x]
    materials = {m['material_id']: m for m in pq.read_table(
        STORE / 'materials.parquet', columns=['material_id', 'numbers', 'pool']).to_pylist()}
    entries = pq.read_table(STORE / 'records_v11/record_links.parquet', columns=['record_id', 'formula']).to_pydict()
    entry_formula = dict(zip(entries['record_id'], entries['formula']))
    groups, misplaced = defaultdict(set), 0
    for name, ids in lists.items():
        for unit in ids:
            if unit in materials:
                formula, elements = formula_of(materials[unit]['numbers'])
            else:
                formula = entry_formula[unit]
                from pymatgen.core import Composition
                elements = sorted(str(e) for e in Composition(formula).elements)
            groups[formula].add(name)
            misplaced += assign(formula, elements) != name
    resolution = pq.read_table(STORE / 'staging/identity/entry_resolution.parquet').to_pandas()
    representatives = set(resolution.loc[resolution.resolution == 'entry_representative', 'record_id'])
    listed_entries = {u for ids in lists.values() for u in ids if u not in materials}
    if listed_entries - representatives:
        raise ValueError(f'{len(listed_entries - representatives)} listed entries are not group representatives')
    shared = sum(len(v) > 1 for v in groups.values())
    legacy = {m for m, x in materials.items() if x['pool'] in ('train_q1', 'valid_q1', 'test_q1')}
    listed = set().union(*map(set, lists.values()))
    if shared or misplaced or legacy - listed or sum(map(len, lists.values())) != len(listed):
        raise ValueError(f'leakage: shared {shared}, misplaced {misplaced}, unlisted legacy {len(legacy - listed)}')
    return {'formula_groups_in_more_than_one_set': shared, 'misplaced_units': misplaced,
            'legacy_unlisted': 0, 'listed_entries_all_representatives': True,
            'list_sizes': {k: len(v) for k, v in lists.items()}}


def main():
    import argparse

    import pandas as pd
    from build_splits_v3 import VARIANTS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    variant = parser.parse_args().variant
    cfg = VARIANTS[variant]
    exclusions = pd.concat([pd.read_parquet(STORE / d / f'splits_{variant}' / 'exclusions.parquet') for d in ('', 'sealed')]
                           ) if variant != 'v3' else None
    label_files = [STORE / d / f'splits_{variant}' / 'labels.parquet' for d in ('', 'sealed')]
    labels = pd.concat([pd.read_parquet(p) for p in label_files]) if all(p.exists() for p in label_files) else None
    report = {}
    for name, legacy_name in zip(cfg['views'], ('q1_v2', 'q1_v2_x2')):
        root, sealed_root = STORE / 'views' / name, STORE / 'sealed/views' / name
        manifest = json.loads((root / 'manifest.json').read_text())
        legacy_manifest = json.loads((STORE / 'views' / legacy_name / 'manifest.json').read_text())
        sources = [(STORE / 'views' / legacy_name, 'train', legacy_manifest),
                   (STORE / 'views' / legacy_name, 'valid', legacy_manifest),
                   (STORE / 'sealed/views' / legacy_name, 'test', legacy_manifest)]
        sealed = json.loads((sealed_root / 'manifest.json').read_text())
        view = {'payload_files': check_payload(root, manifest),
                'sealed_payload_files': check_payload(sealed_root, sealed),
                'labels': {s: check_unlabelled(root, manifest, s) for s in PUBLIC},
                'sealed_labels': {s: check_unlabelled(sealed_root, manifest, s) for s in SEALED},
                'legacy': {s: check_legacy(root, s, manifest, sources) for s in PUBLIC}
                | {s: check_legacy(sealed_root, s, manifest, sources) for s in SEALED}}
        if exclusions is not None:
            view['rules'] = {s: check_rules(root, s, manifest, exclusions, cfg) for s in PUBLIC} | {
                s: check_rules(sealed_root, s, manifest, exclusions, cfg) for s in SEALED}
        if labels is not None and name == next(iter(cfg['views'])):
            view['provenance'] = {s: check_provenance(root, s, labels, cfg) for s in PUBLIC} | {
                s: check_provenance(sealed_root, s, labels, cfg) for s in SEALED}
        if name == next(iter(cfg['views'])):
            view['dataset'] = {s: check_dataset(root, s) for s in PUBLIC}
        report[name] = view
    report['leakage'] = check_leakage(variant)
    out = STORE / f'splits_{variant}' / 'verify_report.json'
    out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
