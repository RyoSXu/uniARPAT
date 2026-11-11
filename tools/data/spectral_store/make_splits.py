"""New splits of spectral-store-v1 -- DEFERRED (design 2026-11-10 P0).

Status: user decision 2026-11-10 -- new splits are drawn only after the store is complete
and verified (and before any training on new data); the rules below are the first draft
and will be revised then (the old test_q1 is the final dual-label test; new evaluation
sets will be eDOS-only and phDOS-only). A first draw on 2026-11-10 gave only 4 dual-label
materials under the canonical-only eDOS rule and was discarded unused. Final tables
without new splits are written by finalize.py; main() refuses to run.

Draft rules:

Order (fixed; ID lists frozen with SHA-256):
1. Legacy splits stay as they are: train_q1, valid_q1 (development/selection), test_q1
   (secondary final result; it was used for diagnostics).
2. A new material is eligible when it is in current MP, not deprecated, BY-C, its input
   structure standardized, n_atoms <= 160, and it has at least one usable spectrum (eDOS:
   canonical task with status ok; phDOS: an MP phonon task with status ok).
3. Evaluation exclusion: reduced compositions of valid_q1 and test_q1.
4. New test sets are drawn as whole reduced-composition groups whose composition occurs
   in no legacy split:
   - test_dual: groups with dual-label members; ~1,000 dual materials (~800 if the dual
     pool after gates is below 5,000); members without both labels are reserved;
   - test_edos: groups none of whose members has a usable phDOS; ~2,000 materials.
   - dev_edos (report only, not for selection): same rule as test_edos, ~2,000.
5. Every remaining eligible material whose reduced composition is outside the exclusion
   set and the new evaluation groups becomes train_new.
6. Spectra of test_dual, test_edos and test_q1 are written to sealed/; the public tables
   never contain them.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from common import STORE

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STAGING = STORE / 'staging'
SEED = 20261110
MAX_ATOMS = 160
TARGETS = {'test_dual': 1000, 'test_dual_small': 800, 'test_edos': 2000, 'dev_edos': 2000}
DUAL_POOL_MIN = 5000
SEALED_SPLITS = ('test_dual', 'test_edos', 'test_q1')


def read_parts(directory: Path, columns=None):
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in sorted(directory.glob('part-*.parquet'))])


def usable(directory: Path, canonical_only: bool = False):
    columns = ['material_id', 'status'] + (['is_canonical'] if canonical_only else [])
    table = read_parts(directory, columns).to_pylist()
    return {r['material_id'] for r in table
            if r['material_id'] and r['status'] == 'ok' and (not canonical_only or r['is_canonical'])}


def draw_groups(groups: dict, members_of, target: int, rng) -> list:
    """Random whole groups until the target count of counted members is reached."""
    keys = sorted(groups)
    rng.shuffle(keys)
    chosen, count = [], 0
    for key in keys:
        if count >= target:
            break
        chosen.append(key)
        count += len(members_of(key))
    return chosen


def assign(materials: list, edos_ok: set, phdos_ok: set):
    legacy = {m['material_id']: m['legacy_split'] for m in materials if m['legacy_split']}
    legacy_reduced = defaultdict(set)
    for m in materials:
        if m['legacy_split']:
            legacy_reduced[m['legacy_split']].add(m['reduced_composition'])
    exclusion = legacy_reduced['valid'] | legacy_reduced['test']
    seen_legacy = exclusion | legacy_reduced['train']
    reasons, eligible = Counter(), {}
    for m in materials:
        if m['legacy_split']:
            continue
        if not m['in_current_mp'] or m['deprecated']:
            reasons['deprecated_or_missing'] += 1
        elif m['license'] != 'BY-C':
            reasons['license'] += 1
        elif m['structure_status'] != 'ok':
            reasons['structure'] += 1
        elif m['n_atoms'] > MAX_ATOMS:
            reasons['too_many_atoms'] += 1
        elif m['material_id'] not in edos_ok and m['material_id'] not in phdos_ok:
            reasons['no_usable_spectrum'] += 1
        elif m['reduced_composition'] in exclusion:
            reasons['valid_test_q1_composition'] += 1
        else:
            eligible[m['material_id']] = m
    groups = defaultdict(list)
    for material_id, m in eligible.items():
        groups[m['reduced_composition']].append(material_id)
    unseen = {k: v for k, v in groups.items() if k not in seen_legacy}
    dual = lambda ids: [i for i in ids if i in edos_ok and i in phdos_ok]  # noqa: E731
    dual_pool = sum(len(dual(v)) for v in unseen.values())
    rng = np.random.default_rng(SEED)
    target_dual = TARGETS['test_dual'] if dual_pool >= DUAL_POOL_MIN else TARGETS['test_dual_small']
    dual_groups = {k: v for k, v in unseen.items() if dual(v)}
    chosen_dual = draw_groups(dual_groups, lambda k: dual(unseen[k]), target_dual, rng)
    remaining = {k: v for k, v in unseen.items() if k not in set(chosen_dual)}
    edos_only = {k: v for k, v in remaining.items()
                 if not any(i in phdos_ok for i in v) and any(i in edos_ok for i in v)}
    chosen_test_edos = draw_groups(edos_only, lambda k: edos_only[k], TARGETS['test_edos'], rng)
    rest = {k: v for k, v in edos_only.items() if k not in set(chosen_test_edos)}
    chosen_dev = draw_groups(rest, lambda k: rest[k], TARGETS['dev_edos'], rng)
    splits = {'train_q1': [], 'valid_q1': [], 'test_q1': [], 'test_dual': [], 'test_edos': [],
              'dev_edos': [], 'train_new': [], 'reserved': []}
    for material_id, split in legacy.items():
        splits[f'{split}_q1'].append(material_id)
    eval_groups = set(chosen_dual) | set(chosen_test_edos) | set(chosen_dev)
    for key in chosen_dual:
        for i in unseen[key]:
            splits['test_dual' if i in edos_ok and i in phdos_ok else 'reserved'].append(i)
    for key in chosen_test_edos:
        splits['test_edos'].extend(i for i in edos_only[key] if i in edos_ok)
    for key in chosen_dev:
        splits['dev_edos'].extend(i for i in rest[key] if i in edos_ok)
    for key, ids in groups.items():
        if key not in eval_groups:
            splits['train_new'].extend(ids)
    splits = {k: sorted(v) for k, v in splits.items()}
    report = {'seed': SEED, 'eligibility_exclusions': dict(reasons), 'eligible_new': len(eligible),
              'dual_pool_unseen_compositions': dual_pool, 'test_dual_target': target_dual,
              'groups': {'test_dual': len(chosen_dual), 'test_edos': len(chosen_test_edos),
                         'dev_edos': len(chosen_dev)},
              'counts': {k: len(v) for k, v in splits.items()}}
    return splits, report


def check_disjoint(splits: dict, by_id: dict):
    sets = {k: set(v) for k, v in splits.items()}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f'material in two splits: {a}/{b}')
    evaluation = ('valid_q1', 'test_q1', 'test_dual', 'test_edos', 'dev_edos')
    train_reduced = {by_id[i]['reduced_composition'] for i in splits['train_new']}
    report = {}
    for name in evaluation:
        reduced = {by_id[i]['reduced_composition'] for i in splits[name]}
        absolute = {by_id[i]['composition'] for i in splits[name]}
        report[name] = {'train_new_reduced_overlap': len(reduced & train_reduced),
                        'train_new_absolute_overlap': len(absolute & {by_id[i]['composition'] for i in splits['train_new']})}
        if report[name]['train_new_reduced_overlap']:
            raise ValueError(f'train_new shares reduced compositions with {name}')
    for a in ('test_dual', 'test_edos', 'dev_edos'):
        for b in ('test_dual', 'test_edos', 'dev_edos', 'train_q1'):
            if a != b and {by_id[i]['reduced_composition'] for i in splits[a]} & \
                    {by_id[i]['reduced_composition'] for i in splits[b]}:
                raise ValueError(f'new evaluation split {a} shares reduced compositions with {b}')
    return report


def main():
    raise SystemExit('new splits are deferred (user decision 2026-11-10); see the module docstring')


if __name__ == '__main__':
    main()
