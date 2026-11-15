"""Full re-split (user decision 2026-11-12): one train / valid / test over every labelled unit.

Labels, record links, flags and view layout are those of build_splits_v2 (step-4 rules: legacy
materials keep their step-2 v2 labels; other units take the linked record by source priority and
volume; disputed / volume / link flags). Only the placement changes: splits_v3.assign on the
reduced formula, with no regard to the legacy train/valid/test membership. Independent entries are
placed like materials and carry is_entry. Test is written under sealed/.
Outputs: splits_v3/, sealed/splits_v3/, views/s1_v3 (128/64) and views/s1_v3_x2 (256/128) with
their sealed test under sealed/views/. Record links and structures are those of records_v11/.

Variant v3_1 (user decision 2026-11-13) applies one label rule to every unit: JARVIS phonon spectra
are never a label, and a legacy label whose calculation structure fails the structure check is not
used. An excluded legacy label is replaced by the unit's best remaining linked record (same rule as
new units) or left empty; units left without labels are dropped. Every excluded label is listed in
exclusions.parquet (public sets under splits_v3_1/, test under sealed/splits_v3_1/).

Variant v3_2 (user decision 2026-11-14) adds: a label is not used when its calculation structure
departs from the material it was computed for (the record's own MP hint, when that material is in
the store: default matcher fails, or per-atom volume differs by > 10%), when its volume differs by
> 10% from the input structure, or when its identity was only settled by the tight matcher among
several materials. Crystal-system-only differences do not count. The same holds for legacy labels.
An exclusions.parquet row is written whenever the label the unrestricted rule would pick is not
used; labels.parquet records the origin (record id or legacy origin) of every label in the views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

from build_splits_v2 import (DISPUTED_E, LINK_CODES, MAX_ATOMS, PRIORITY, RANGES, SOURCE_CODES, VOLUME_FLAG,
                             e_metric, formula_of, formula_z, grid, links, load_records, valence, volume_per_atom)
from common import ROOT, STORE, clip_isolated_spikes, exact_bin, file_sha256, load_calc_types, spike_thresholds, strip_mp
from make_view_v2 import W_MAX, build_split, edos_window_ratio
from splits_v3 import RARE_ELEMENTS, SALT, assign

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils.source_report import LEGACY_RAW, legacy_structure_flags
from utils.spectral_balance import fit_task_normalizer

VARIANTS = {
    'v3': {'views': {'s1_v3': {'edos': 128, 'phdos': 64}, 's1_v3_x2': {'edos': 256, 'phdos': 128}},
           'exclude_kinds': {}, 'exclude_legacy_mismatch': False, 'max_volume_deviation': None},
    'v3_1': {'views': {'s1_v3_1': {'edos': 128, 'phdos': 64}, 's1_v3_1_x2': {'edos': 256, 'phdos': 128}},
             'exclude_kinds': {'phdos': ('jarvis',)}, 'exclude_legacy_mismatch': True, 'max_volume_deviation': None},
    'v3_2': {'views': {'s1_v3_2': {'edos': 128, 'phdos': 64}, 's1_v3_2_x2': {'edos': 256, 'phdos': 128}},
             'exclude_kinds': {'phdos': ('jarvis',)}, 'exclude_legacy_mismatch': True, 'max_volume_deviation': 0.10},
}
V3_LINK_CODES = {**LINK_CODES, 'entry_match': 5}  # another record of the same independent entry
PUBLIC = ('train', 'valid')
SEALED = ('test',)
LEGACY_POOLS = ('train_q1', 'valid_q1', 'test_q1')


def legacy_exclusions(cfg):
    """(material, task) -> (reason, step-2 origin) for the legacy labels a variant does not use."""
    if not cfg['exclude_kinds'] and not cfg['exclude_legacy_mismatch']:
        return {}
    raw = pd.concat([pd.read_parquet(STORE / path, columns=['material_id', 'task', 'origin']) for path in LEGACY_RAW])
    checks = legacy_structure_flags(STORE, raw.material_id.unique()) if cfg['exclude_legacy_mismatch'] else {}
    out = {}
    for r in raw.itertuples():
        kind = r.origin.partition(':')[0]
        if kind in cfg['exclude_kinds'].get(r.task, ()):
            out[(r.material_id, r.task)] = (f'source_{kind}', r.origin)
        elif checks.get((r.material_id, r.task), (None,))[0] is True:
            out[(r.material_id, r.task)] = ('structure_mismatch', r.origin)
        elif cfg['max_volume_deviation'] is not None and checks.get((r.material_id, r.task), (None, None))[1] is True:
            out[(r.material_id, r.task)] = ('volume_deviation', r.origin)
    return out


def hint_deviation(features, record_id, limit):
    """Why a record departs from the store material it was computed for, or None (no such material, or none)."""
    if record_id not in features.index:
        return None
    f = features.loc[record_id]
    if not f.hint_in_store or pd.isna(f.default):
        return None
    if not f.default:
        return 'prototype_deviation'
    if abs(float(f.volume_ratio) - 1) > limit:
        return 'volume_deviation'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()
    cfg = VARIANTS[args.variant]
    RESOLUTIONS = cfg['views']
    main_view = next(iter(RESOLUTIONS))
    out_splits, sealed_splits = STORE / f'splits_{args.variant}', STORE / 'sealed' / f'splits_{args.variant}'
    for path in [out_splits, sealed_splits] + [STORE / r / 'views' / v for r in ('', 'sealed') for v in RESOLUTIONS]:
        if path.exists():
            raise FileExistsError(path)
    records_dir = STORE / 'records_v11'
    materials = {m['material_id']: m for m in pq.read_table(STORE / 'materials.parquet').to_pylist()}
    for m in materials.values():
        m['formula'], m['elements'] = formula_of(m['numbers'])
    link = links()
    saved = pd.read_parquet(records_dir / 'record_links.parquet')
    if not link.fillna('').astype(str).reset_index(drop=True).equals(saved.fillna('').astype(str).reset_index(drop=True)):
        raise ValueError('record links differ from records_v11')
    calc_types = load_calc_types()
    owned = pd.concat([pq.read_table(path, columns=['material_id', 'source_task_id', 'is_canonical', 'status'],
                                     filters=[('is_canonical', '=', False), ('status', '=', 'ok')]).to_pandas()
                       for path in (STORE / 'spectra_edos.parquet', STORE / 'sealed/spectra_edos.parquet')])
    owned = owned[owned.source_task_id.map(lambda t: calc_types.get(strip_mp(t), '').endswith('NSCF Uniform'))]
    other_nscf = owned.groupby('material_id').source_task_id.apply(list).to_dict()
    wanted_edos = {r[len('mp_edos:'):] for r in link[link.source == 'mp_edos'].record_id}
    rec = load_records(wanted_edos, {t for v in other_nscf.values() for t in v})
    print('records loaded', len(rec), flush=True)

    # ---------------------------------------------------------------- units and primary labels (as v2)
    units = {}
    for mid, m in materials.items():
        units[mid] = {'kind': 'material', 'pool': m['pool'], 'formula': m['formula'], 'elements': m['elements'],
                      'structure': (m['cell'], [int(z) for z in m['numbers']], m['frac']), 'n': m['n_atoms'],
                      'nvalence': m['nvalence'], 'candidates': {'edos': [], 'phdos': []}}
    resolution = pd.read_parquet(STORE / 'staging/identity/entry_resolution.parquet').set_index('record_id')
    for r in link.itertuples():
        if r.record_id not in rec:
            continue
        task = 'edos' if r.source == 'mp_edos' else 'phdos'
        if r.material_id is not None and r.material_id in units:
            units[r.material_id]['candidates'][task].append((r.record_id, r.link))
            continue
        resolved = resolution.loc[r.record_id]
        if resolved.material_id is not None:
            units[resolved.material_id]['candidates'][task].append((r.record_id, 'rehung'))
            continue
        uid = resolved.unit_id
        if uid not in units:
            k = rec[uid]
            formula, elements = formula_of(k['structure'][1])
            units[uid] = {'kind': 'entry', 'pool': 'new', 'formula': formula, 'elements': elements,
                          'structure': k['structure'], 'n': k['n'], 'nvalence': valence(k['structure'][1]),
                          'candidates': {'edos': [], 'phdos': []}}
        units[uid]['candidates'][task].append((r.record_id, 'own' if r.record_id == uid else 'entry_match'))
    flags = Counter()
    excluded = legacy_exclusions(cfg)
    limit = cfg['max_volume_deviation']
    features = pd.read_parquet(STORE / 'staging/identity/record_features.parquet').set_index('record_id')
    tight_only = set(resolution.index[resolution.resolution == 'tight_unique'])
    exclusion_rows = []
    for uid, u in units.items():
        vin = volume_per_atom(u['structure'])
        u['labels'] = {}
        for task in RANGES:
            legacy = u['pool'] in LEGACY_POOLS
            if legacy and (uid, task) not in excluded:
                continue
            cands = []
            for rid, kind in u['candidates'][task] if u['n'] <= MAX_ATOMS else []:
                k = rec[rid]
                if task == 'edos' and edos_window_ratio(k['x'], k['y'], k['nval']) > W_MAX:
                    flags['edos_anomalous_W>3'] += 1
                    continue
                ratio = volume_per_atom(k['structure']) / vin
                reason = None
                if k['kind'] in cfg['exclude_kinds'].get(task, ()):
                    reason = f"source_{k['kind']}"
                elif limit is not None:
                    reason = ('identity_ambiguous' if rid in tight_only else hint_deviation(features, rid, limit)
                              or ('volume_deviation' if abs(ratio - 1) > limit else None))
                cands.append(((PRIORITY.get(k['kind'], 0), abs(np.log(ratio)), rid), rid, kind, ratio, reason))
            cands.sort()
            usable = [c for c in cands if c[4] is None]
            if usable:
                _, rid, kind, ratio, _ = usable[0]
                u['labels'][task] = {'rid': rid, 'link': kind, 'volume_ratio': ratio,
                                     'others': [c[1] for c in usable[1:]]}
            replacement = u['labels'][task]['rid'] if task in u['labels'] else None
            if legacy:
                reason, origin = excluded[(uid, task)]
                exclusion_rows.append((uid, task, reason, origin, replacement))
            elif cands and cands[0][4] is not None:
                exclusion_rows.append((uid, task, cands[0][4], cands[0][1], replacement))

    # ---------------------------------------------------------------- sets: formula only
    sets = {uid: assign(u['formula'], u['elements']) for uid, u in units.items()
            if u['pool'] in LEGACY_POOLS or u['labels']}
    sides = {}
    for uid, s in sets.items():
        sides.setdefault(units[uid]['formula'], set()).add(s)
    shared = [f for f, v in sides.items() if len(v) > 1]
    if shared:
        raise ValueError(f'{len(shared)} formula groups in more than one set, e.g. {shared[:3]}')

    legacy_public = pq.read_table(STORE / 'spectra_legacy.parquet').to_pylist()
    legacy_sealed = pq.read_table(STORE / 'sealed/spectra_legacy.parquet').to_pylist()
    C_main = {t: json.loads((STORE / 'views/q1_v2/manifest.json').read_text())['normalizer_train'][t]['energy']
              for t in RANGES}
    disputed = {}
    for uid, u in units.items():
        for task, lab in u['labels'].items():
            k = rec[lab['rid']]
            bins = RESOLUTIONS[main_view][task]
            a = exact_bin(k['x'], k['y'], grid(task, bins))[0].astype(float)
            others = lab['others'] if task == 'phdos' else [f'mp_edos:{t}' for t in other_nscf.get(uid, [])]
            worst = 0.0
            for rid in others:
                o = rec.get(rid)
                if o is None or o['kind'] == 'jarvis':
                    continue
                b = exact_bin(o['x'], o['y'], grid(task, bins))[0].astype(float)
                worst = max(worst, e_metric(a, k['n'], b, o['n'], task, bins, C_main[task]))
            disputed[(uid, task)] = worst > DISPUTED_E

    # ---------------------------------------------------------------- views
    report = {'salt': SALT, 'rare_elements': sorted(RARE_ELEMENTS), 'disputed_E': DISPUTED_E,
              'volume_flag': VOLUME_FLAG, 'anomalous': dict(flags), 'views': {}}
    order = {s: sorted(u for u, x in sets.items() if x == s) for s in PUBLIC + SEALED}
    provenance = []
    for name, bins in RESOLUTIONS.items():
        legacy = {'train': build_split('train', bins, materials, [r for r in legacy_public if r['legacy_split'] == 'train']),
                  'valid': build_split('valid', bins, materials, [r for r in legacy_public if r['legacy_split'] == 'valid']),
                  'test': build_split('test', bins, materials, legacy_sealed)}
        legacy_rows, legacy_origin = {}, {}
        for b in legacy.values():
            for i, mid in enumerate(b['ids']):
                legacy_rows[mid] = {t: (b['arrays'][t][0][i], b['arrays'][t][1][i]) for t in RANGES}
            for r in b['raw']:
                key = (r['material_id'], r['task'])
                legacy_origin[key] = r['origin']
                if key in excluded and excluded[key][1] != r['origin']:
                    raise ValueError(f'{key}: legacy origin differs from the step-2 export')

        def labelled(uid, t):
            return (uid in legacy_rows and (uid, t) not in excluded) or t in units[uid]['labels']

        data = {}
        for s in PUBLIC + SEALED:
            ids = [u for u in order[s] if (units[u]['pool'] not in LEGACY_POOLS or u in legacy_rows)
                   and any(labelled(u, t) for t in RANGES)]
            n = len(ids)
            arr = {t: (np.zeros((n, bins[t]), np.float32), np.zeros((n, bins[t]), bool), np.zeros(n, bool),
                       np.zeros(n, np.int8), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.int8)) for t in RANGES}
            for i, uid in enumerate(ids):
                u = units[uid]
                for t in RANGES:
                    values, masks, has, source, dis, vol, lk = arr[t]
                    if uid in legacy_rows and (uid, t) not in excluded:
                        values[i], masks[i] = legacy_rows[uid][t]
                        has[i], source[i] = True, SOURCE_CODES['legacy']
                    elif t in u['labels']:
                        lab = u['labels'][t]
                        k = rec[lab['rid']]
                        v, cov = exact_bin(k['x'], np.asarray(k['y'], float) * (u['n'] / k['n']), grid(t, bins[t]))
                        values[i], masks[i], has[i] = v, cov >= 1 - 1e-6, True
                        source[i] = SOURCE_CODES['mp'] if k['kind'] in ('mp_edos', 'mp_pheasy') else SOURCE_CODES.get(
                            k['kind'], 0)
                        dis[i] = disputed[(uid, t)]
                        vol[i] = abs(lab['volume_ratio'] - 1) > VOLUME_FLAG
                        lk[i] = V3_LINK_CODES[lab['link']]
                    if name == main_view and has[i]:
                        lab = None if uid in legacy_rows and (uid, t) not in excluded else u['labels'][t]
                        provenance.append((uid, t, s, legacy_origin[(uid, t)] if lab is None else lab['rid'],
                                           'legacy' if lab is None else lab['link'],
                                           np.nan if lab is None else float(lab['volume_ratio'])))
            data[s] = (ids, arr)
        thresholds, C, clipped = {}, {}, {}
        for t in RANGES:
            ids, arr = data['train']
            values, masks, has = arr[t][:3]
            thresholds[t] = spike_thresholds(values[has], masks[has])
            atoms = np.asarray([units[u]['n'] for u in ids], float)[has]
            for s, (_, a) in data.items():
                if a[t][2].any():
                    new, clip = clip_isolated_spikes(a[t][0][a[t][2]], a[t][1][a[t][2]], thresholds[t])
                    a[t][0][a[t][2]] = new
                    clipped[f'{s}/{t}'] = int(clip.sum())
            C[t] = fit_task_normalizer(values[has].astype(np.float64), atoms,
                                       (RANGES[t][1] - RANGES[t][0]) / bins[t])
        legacy_manifest = json.loads((ROOT / 'data/crystal_q1_v1/manifest.json').read_text())
        manifest = {k: legacy_manifest[k] for k in ('schema', 'cell_convention', 'frac_convention', 'primitive_policy',
                                                     'geometry_precision')}
        manifest.update({'label_contract': 'v2', 'binning': 'exact_piecewise_linear', 'partial_labels': True,
                         'store': 'spectral-store-v1.1', 'grid': {t: {'range': list(RANGES[t]), 'bins': bins[t]}
                                                                 for t in RANGES},
                         'source_codes': SOURCE_CODES, 'link_codes': V3_LINK_CODES, 'split_rule': {
                             'salt': SALT, 'rare_elements': sorted(RARE_ELEMENTS),
                             'design': 'design-2026-11-11-splits-v2, full re-split 2026-11-12'},
                         'flags': {'disputed': f'another MP/PhononDB record (eDOS: NSCF Uniform task) differs by E > {DISPUTED_E}',
                                   'volume': f'|label calculation volume / input volume - 1| > {VOLUME_FLAG}'},
                         'spike_thresholds': {t: v.tolist() for t, v in thresholds.items()}, 'spike_clipped_bins': clipped,
                         'normalizer_train': C, 'training_exclusions': legacy_manifest['training_exclusions'],
                         'variant': args.variant, 'label_rules': {
                             'excluded_label_sources': {t: list(v) for t, v in cfg['exclude_kinds'].items()},
                             'legacy_structure_mismatch': 'excluded' if cfg['exclude_legacy_mismatch'] else 'kept',
                             'replacement': 'best remaining linked record by the new-unit rule, else unlabelled',
                             'max_volume_deviation': cfg['max_volume_deviation'],
                             'hint_deviation': 'excluded' if cfg['max_volume_deviation'] is not None else 'kept',
                             'exclusions_table': f'splits_{args.variant}/exclusions.parquet',
                             'labels_table': f'splits_{args.variant}/labels.parquet'},
                         'splits': {}, 'files': {}})
        sealed_manifest = {'label_contract': 'v2', 'view': name, 'splits': {}, 'files': {}}
        for s, (ids, arr) in data.items():
            root = (STORE / 'sealed' / 'views' / name) if s in SEALED else (STORE / 'views' / name)
            target = sealed_manifest if s in SEALED else manifest
            directory = root / s
            directory.mkdir(parents=True)
            us = [units[u] for u in ids]
            counts = np.asarray([u['n'] for u in us], np.int32)
            arrays = {'cell': np.asarray([np.asarray(u['structure'][0], np.float32).reshape(3, 3) for u in us],
                                         np.float32).reshape(-1, 3, 3),
                      'atomic_numbers': np.concatenate([np.asarray([formula_z(z) for z in u['structure'][1]], np.uint8)
                                                        for u in us]),
                      'frac_coords': np.concatenate([np.asarray(u['structure'][2], np.float32).reshape(-1, 3)
                                                     for u in us]),
                      'n_atoms': counts, 'atom_offsets': np.r_[0, np.cumsum(counts, dtype=np.int64)],
                      'nvalence': np.asarray([u['nvalence'] if u['nvalence'] is not None else np.nan for u in us],
                                             np.float32),
                      'is_entry': np.asarray([u['kind'] == 'entry' for u in us], bool), 'index': np.asarray(ids)}
            for t in RANGES:
                values, masks, has, source, dis, vol, lk = arr[t]
                arrays.update({f'{t}_tgtdos': values, f'{t}_mask': masks, f'has_{t}': has, f'source_{t}': source,
                               f'disputed_{t}': dis, f'volume_flag_{t}': vol, f'link_{t}': lk})
            for key, array in arrays.items():
                filename = f'{s}_index.npy' if key == 'index' else f'{key}_{s}.npy'
                np.save(directory / filename, array, allow_pickle=False)
                target['files'][f'{s}/{filename}'] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                                      'sha256': file_sha256(directory / filename)}
            target['splits'][s] = {'n': len(ids), 'has_edos': int(arr['edos'][2].sum()),
                                   'has_phdos': int(arr['phdos'][2].sum()),
                                   'dual': int((arr['edos'][2] & arr['phdos'][2]).sum())}
        (STORE / 'views' / name / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        (STORE / 'sealed' / 'views' / name / 'manifest.json').write_text(json.dumps(sealed_manifest, indent=2) + '\n')
        report['views'][name] = {'public': manifest['splits'], 'sealed_counts': sealed_manifest['splits'],
                                 'C': {t: C[t]['energy'] for t in RANGES}, 'clipped': clipped}

    # ---------------------------------------------------------------- split lists and report
    out_splits.mkdir(parents=True)
    sealed_splits.mkdir(parents=True)
    hashes = {}
    for s, ids in order.items():
        directory = sealed_splits if s in SEALED else out_splits
        (directory / f'{s}.txt').write_text('\n'.join(ids) + '\n')
        hashes[s] = {'n': len(ids), 'sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    (out_splits / 'report.json').write_text(json.dumps({'split_hashes': hashes, 'salt': SALT}, indent=1) + '\n')
    exclusions = pd.DataFrame(exclusion_rows, columns=['unit_id', 'task', 'reason', 'excluded', 'replacement'])
    exclusions['set'] = [assign(units[u]['formula'], units[u]['elements']) for u in exclusions.unit_id]
    exclusions['unit_listed'] = exclusions.unit_id.isin(list(sets))
    labels = pd.DataFrame(provenance, columns=['unit_id', 'task', 'set', 'origin', 'link', 'volume_ratio'])
    for directory, names in ((out_splits, PUBLIC), (sealed_splits, SEALED)):
        exclusions[exclusions.set.isin(names)].reset_index(drop=True).to_parquet(directory / 'exclusions.parquet')
        labels[labels.set.isin(names)].reset_index(drop=True).to_parquet(directory / 'labels.parquet')
    report['exclusions'] = {s: {f'{t}:{r}:{"replaced" if rep else "empty"}': n for (t, r, rep), n in Counter(
        zip(g.task, g.reason, g.replacement.notna())).items()} for s, g in exclusions.groupby('set')}
    report['leakage'] = {'formula_groups_in_more_than_one_set': len(shared),
                         'placement': 'reduced formula only; deprecated or alias ids cannot split a group'}
    report['sets'] = {s: len(v) for s, v in order.items()}
    report['legacy_pool_by_set'] = {s: dict(Counter(units[u]['pool'] for u in order[s])) for s in order}
    report['entries_by_set'] = {s: sum(units[u]['kind'] == 'entry' for u in order[s]) for s in order}
    report['labels_by_source'] = {s: dict(Counter(f"{t}:{rec[lab['rid']]['kind']}" for uid in order[s]
                                                  for t, lab in units[uid]['labels'].items())) for s in order}
    report['flags'] = {s: dict(Counter(f'{t}:disputed' for uid in order[s] for t in units[uid]['labels']
                                       if disputed.get((uid, t))) + Counter(
        f'{t}:volume>10%' for uid in order[s] for t, lab in units[uid]['labels'].items()
        if abs(lab['volume_ratio'] - 1) > VOLUME_FLAG) + Counter(
        f"{t}:link_{lab['link']}" for uid in order[s] for t, lab in units[uid]['labels'].items())) for s in order}
    (out_splits / 'build_report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('sets', 'legacy_pool_by_set', 'entries_by_set', 'leakage', 'anomalous',
                                             'exclusions')},
                     indent=1), flush=True)


if __name__ == '__main__':
    main()
