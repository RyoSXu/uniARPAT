"""Read-only checks of the step-4 views s1_v2 / s1_v2_x2 (public splits only; sealed sets by digests).

1. Every manifest payload matches its sha256; every public split loads through the training
   dataset (packed crystals, masks, valence counts, partial-label flags).
2. Unlabelled targets are zero with an all-false mask; new labelled rows have a covered bin.
3. Legacy rows equal the q1_v2 rows of the same materials: identical geometry; spectra equal
   except bins clipped by either view's own isolated-spike thresholds.
4. Leakage recomputed from the written split lists: no reduced formula in train and in any
   evaluation list; legacy valid/test lists unchanged; C3 holdouts absent from every view.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from common import ROOT, STORE, file_sha256

sys.path.insert(0, str(ROOT))

PUBLIC = ('train', 'valid_q1', 'valid_supp', 'dev', 'dev_indep')
SEALED = ('test_q1', 'test_supp', 'test', 'test_indep')
TASKS = ('edos', 'phdos')


def load(root, split, key):
    name = f'{split}_index.npy' if key == 'index' else f'{key}_{split}.npy'
    return np.load(root / split / name, allow_pickle=False)


def check_payload(root, manifest):
    bad = [name for name, entry in manifest['files'].items() if file_sha256(root / name) != entry['sha256']]
    if bad:
        raise ValueError(f'{root.name}: payload digests differ: {bad[:3]}')
    return len(manifest['files'])


def check_dataset(root, split):
    from utils.dataset import Dos_Dataset

    dataset = Dos_Dataset(data_dir=str(root), split='valid' if split != 'train' else 'train',
                          view_split=None if split == 'train' else split, dos_minmax=True, dos_sumnorm=True)
    item = dataset[0]
    if dataset.labels is None or len(item) != 19 or item[18].shape != (2,):
        raise ValueError(f'{split}: partial-label field missing')
    return {'rows': len(dataset), 'labels': dataset.labels.sum(0).tolist()}


def check_unlabelled(root, manifest, split):
    """Legacy rows may lack fully covered bins (inherited from q1_v2; the balanced loss ignores coverage)."""
    out = {}
    for task in TASKS:
        has = load(root, split, f'has_{task}')
        values, mask = load(root, split, f'{task}_tgtdos'), load(root, split, f'{task}_mask')
        if (values[~has] != 0).any() or mask[~has].any():
            raise ValueError(f'{split}/{task}: unlabelled rows carry targets or coverage')
        uncovered = has & ~mask.any(axis=1)
        legacy = load(root, split, f'source_{task}') == manifest['source_codes']['legacy']
        if (uncovered & ~legacy).any():
            raise ValueError(f'{split}/{task}: new labelled row without covered bins')
        out[task] = {'labelled': int(has.sum()), 'legacy_uncovered': int(uncovered.sum())}
    return out


def check_legacy(root, manifest, legacy_root, legacy_manifest, pairs):
    report = {}
    for split, legacy_split in pairs:
        ids, legacy_ids = load(root, split, 'index'), load(legacy_root, legacy_split, 'index')
        source = load(root, split, 'source_edos')
        rows = np.flatnonzero(source == manifest['source_codes']['legacy'])
        position = {m: i for i, m in enumerate(legacy_ids)}
        missing = [ids[i] for i in rows if ids[i] not in position]
        if missing:
            raise ValueError(f'{split}: legacy rows absent from q1_v2/{legacy_split}: {missing[:3]}')
        other = np.asarray([position[ids[i]] for i in rows])
        for key in ('cell', 'n_atoms', 'nvalence'):
            if not np.array_equal(load(root, split, key)[rows], load(legacy_root, legacy_split, key)[other]):
                raise ValueError(f'{split}: legacy {key} differs from q1_v2')
        offsets, legacy_offsets = load(root, split, 'atom_offsets'), load(legacy_root, legacy_split, 'atom_offsets')
        numbers, legacy_numbers = load(root, split, 'atomic_numbers'), load(legacy_root, legacy_split, 'atomic_numbers')
        frac, legacy_frac = load(root, split, 'frac_coords'), load(legacy_root, legacy_split, 'frac_coords')
        for i, j in zip(rows, other):
            a, b = slice(offsets[i], offsets[i + 1]), slice(legacy_offsets[j], legacy_offsets[j + 1])
            if not (np.array_equal(numbers[a], legacy_numbers[b]) and np.array_equal(frac[a], legacy_frac[b])):
                raise ValueError(f'{split}: legacy atoms differ for {ids[i]}')
        entry = {'legacy_rows': int(len(rows)), 'q1_v2_rows': int(len(legacy_ids))}
        for task in TASKS:
            mine = load(root, split, f'{task}_tgtdos')[rows]
            theirs = load(legacy_root, legacy_split, f'{task}_tgtdos')[other]
            if not np.array_equal(load(root, split, f'{task}_mask')[rows],
                                  load(legacy_root, legacy_split, f'{task}_mask')[other]):
                raise ValueError(f'{split}/{task}: legacy coverage differs')
            mine_threshold = np.asarray(manifest['spike_thresholds'][task], np.float32)
            their_threshold = np.asarray(legacy_manifest['spike_thresholds'][task], np.float32)
            differ = mine != theirs
            explained = differ & ((mine == mine_threshold) | (theirs == their_threshold))
            if (differ & ~explained).any():
                raise ValueError(f'{split}/{task}: legacy spectra differ beyond spike clipping')
            entry[f'{task}_clip_only_bins'] = int(differ.sum())
        report[split] = entry
    return report


def check_leakage():
    from build_splits_v2 import formula_of

    lists = {}
    for directory in (STORE / 'splits', STORE / 'sealed' / 'splits'):
        for path in directory.glob('*.txt'):
            lists[path.stem] = [line for line in path.read_text().split('\n') if line]
    materials = {m['material_id']: m for m in pq.read_table(
        STORE / 'materials.parquet', columns=['material_id', 'numbers', 'pool']).to_pylist()}
    entries = pq.read_table(STORE / 'records_v11/record_links.parquet', columns=['record_id', 'formula']).to_pydict()
    entry_formula = dict(zip(entries['record_id'], entries['formula']))
    groups = defaultdict(set)
    for name, ids in lists.items():
        for unit in ids:
            formula = formula_of(materials[unit]['numbers'])[0] if unit in materials else entry_formula[unit]
            groups[formula].add(name)
    evaluation = set(PUBLIC + SEALED) - {'train'}
    shared = sum('train' in v and bool(v & evaluation) for v in groups.values())
    legacy = {p: sorted(m for m, x in materials.items() if x['pool'] == p) for p in ('valid_q1', 'test_q1')}
    unchanged = {p: sorted(lists[p]) == legacy[p] for p in legacy}
    holdout = set(lists['holdout'])
    in_views = sum(len(holdout & set(load(STORE / 'views/s1_v2', s, 'index'))) for s in PUBLIC)
    if shared or not all(unchanged.values()) or in_views:
        raise ValueError(f'leakage: shared groups {shared}, legacy unchanged {unchanged}, holdouts in views {in_views}')
    return {'formula_groups_train_and_eval': shared, 'legacy_lists_unchanged': unchanged,
            'holdouts_in_public_views': in_views, 'list_sizes': {k: len(v) for k, v in sorted(lists.items())}}


def main():
    legacy_root = {'s1_v2': STORE / 'views/q1_v2', 's1_v2_x2': STORE / 'views/q1_v2_x2'}
    report = {}
    for name in ('s1_v2', 's1_v2_x2'):
        root = STORE / 'views' / name
        manifest = json.loads((root / 'manifest.json').read_text())
        legacy_manifest = json.loads((legacy_root[name] / 'manifest.json').read_text())
        sealed = json.loads((STORE / 'sealed/views' / name / 'manifest.json').read_text())
        view = {'payload_files': check_payload(root, manifest),
                'sealed_payload_files': check_payload(STORE / 'sealed/views' / name, sealed),
                'labels': {s: check_unlabelled(root, manifest, s) for s in PUBLIC},
                'legacy': check_legacy(root, manifest, legacy_root[name], legacy_manifest,
                                       (('train', 'train'), ('valid_q1', 'valid')))}
        if name == 's1_v2':
            view['dataset'] = {s: check_dataset(root, s) for s in PUBLIC}
        report[name] = view
    report['leakage'] = check_leakage()
    out = STORE / 'splits' / 'verify_report.json'
    out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
