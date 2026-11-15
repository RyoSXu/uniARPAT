"""Step 4 execution (design 2026-11-11 splits-v2 with the 2026-11-12 record-identity rules).

1. Links: every record (MP phonon task, MP canonical eDOS task, PhononDB entry, JARVIS page)
   is attached to a store material when its structure matches the hinted MP material (default
   StructureMatcher + same crystal system; flagged 'edge' when only the default, not the tight,
   matcher passes), else to the unique same-formula material that matches ('rehung'). Ambiguous
   or unmatched records become independent entries with their own id and structure.
2. Primary labels: legacy materials keep the step-2 labels (legacy raw spectrum, q1_v2 rules).
   Other units: eDOS = linked MP canonical task with W <= 3; phDOS = linked record by MP pheasy >
   dfpt > PhononDB > JARVIS, then calculation volume closest to the input structure, then id.
3. Flags: disputed (another record of the same task on the same unit differs by E > 0.5 at the
   main resolution; eDOS: other NSCF Uniform tasks; phDOS: MP and PhononDB records), volume (> 10%
   per-atom volume difference between the label's calculation structure and the input structure),
   link kind of each label record.
4. Sets by splits_v2 (formula only): train; valid_q1 (legacy, unchanged); valid_supp (new
   materials of legacy valid groups); dev; test_q1, test_supp, test (sealed); independent entries
   of dev/test kept apart (dev_indep; test_indep sealed); legacy train of eval groups held out.
5. Views (partial labels; has_*, source codes, flags) at 128/64 (s1_v2) and 256/128 (s1_v2_x2);
   spike thresholds and C fitted on labelled train rows; sealed sets under sealed/views/.
   Raw sampled points of every label with its record id (structures in records_v11/).
Test-side spectra are processed by the script only; printed output is counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter

from common import (ROOT, STORE, clip_isolated_spikes, exact_bin, file_sha256, load_calc_types, spike_thresholds,
                    strip_mp)
from make_view_v2 import W_MAX, build_split, edos_window_ratio
from splits_v2 import RARE_ELEMENTS, SALT, assign

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.spectral_balance import fit_task_normalizer

STAGING = STORE / 'staging'
RANGES = {'edos': (-6.0, 6.0), 'phdos': (-280.0, 980.0)}
RESOLUTIONS = {'s1_v2': {'edos': 128, 'phdos': 64}, 's1_v2_x2': {'edos': 256, 'phdos': 128}}
MAX_ATOMS = 160
DISPUTED_E = 0.5
VOLUME_FLAG = 0.10
PRIORITY = {'mp_pheasy': 0, 'mp_dfpt': 1, 'phonondb': 2, 'jarvis': 3}
SOURCE_CODES = {'none': 0, 'legacy': 1, 'mp': 2, 'mp_dfpt': 4, 'phonondb': 5, 'jarvis': 6}
PUBLIC = ('train', 'valid_q1', 'valid_supp', 'dev', 'dev_indep')
SEALED = ('test_q1', 'test_supp', 'test', 'test_indep')
LINK_CODES = {'none': 0, 'hint': 1, 'hint_edge': 2, 'rehung': 3, 'own': 4}


def alpha(i):
    from emmet.core.mpid import AlphaID
    s = str(AlphaID(int(i), prefix='mp'))
    return 'mp-' + s[3:].rjust(8, 'a')


def grid(task, bins):
    lo, hi = RANGES[task]
    return np.linspace(lo, hi, bins + 1)


def formula_of(species):
    from pymatgen.core import Composition, Element
    symbols = [Element.from_Z(int(s)).symbol if not isinstance(s, str) else s for s in species]
    return Composition({e: symbols.count(e) for e in set(symbols)}).reduced_formula, sorted(set(symbols))


# ------------------------------------------------------------------ links
def links():
    f = pd.read_parquet(STAGING / 'identity/record_features.parquet')
    out = []
    for r in f.itertuples():
        alt = json.loads(r.alt_matches) if isinstance(r.alt_matches, str) else []
        hint_pass = bool(r.hint_in_store) and not pd.isna(r.default) and bool(r.default) and r.cs_hint == r.cs_record
        if hint_pass:
            kind, material = ('hint' if r.tight else 'hint_edge'), r.hint
        elif len(alt) == 1:
            kind, material = 'rehung', alt[0][0]
        else:
            kind, material = ('ambiguous' if alt else 'no_match'), None
        out.append({'record_id': r.record_id, 'source': r.source, 'hint': r.hint, 'formula': r.formula,
                    'material_id': material, 'link': kind})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ records (structure + spectrum)
def load_records(wanted_edos: set, other_edos: set):
    """record_id -> dict(kind, x, y (record cell), n, structure, nval)."""
    rec = {}
    ph = pq.read_table(STAGING / 'phdos/part-00.parquet', columns=[
        'source_task_id', 'method', 'status', 'x', 'y', 'calc_lattice', 'calc_numbers', 'calc_frac']).to_pylist()
    for r in ph:
        if r['status'] == 'ok' and r['calc_numbers']:
            rec[f"mp_phonon:{r['source_task_id']}"] = {
                'kind': f"mp_{r['method']}", 'x': np.asarray(r['x']), 'y': np.asarray(r['y']), 'n': len(r['calc_numbers']),
                'structure': (r['calc_lattice'], [int(z) for z in r['calc_numbers']], r['calc_frac'])}
    for part in sorted((STAGING / 'phdos_phonondb').glob('part-*.parquet')):
        for r in pq.read_table(part, columns=['source_id', 'status', 'x', 'y', 'primitive_cell']).to_pylist():
            if r['status'] == 'ok':
                cell = json.loads(r['primitive_cell'])
                rec[f"phonondb:{r['source_id']}"] = {
                    'kind': 'phonondb', 'x': np.asarray(r['x']), 'y': np.asarray(r['y']), 'n': len(cell['points']),
                    'structure': (cell['lattice'], [p['symbol'] for p in cell['points']],
                                  [p['coordinates'] for p in cell['points']])}
    jar = {r['source_id']: r for r in pq.read_table(STAGING / 'phdos_jarvis/part-0000.parquet',
                                                   columns=['source_id', 'status', 'x', 'y']).to_pylist()
           if r['status'] == 'ok' and r['x']}
    from pymatgen.core import Lattice
    with zipfile.ZipFile(STORE / 'raw_jarvis/jdft_3d-12-12-2022.json.zip') as z:
        for d in json.loads(z.read(z.namelist()[0])):
            if d['jid'] in jar:
                a, r = d['atoms'], jar[d['jid']]
                coords = np.asarray(a['coords'], float)
                if a.get('cartesian'):
                    coords = Lattice(a['lattice_mat']).get_fractional_coords(coords)
                x, y = np.asarray(r['x'], float), np.asarray(r['y'], float)
                order = np.argsort(x)
                area = float(np.trapz(y[order], x[order]))
                n = len(a['elements'])
                if area > 0:
                    rec[f"jarvis:{d['jid']}"] = {'kind': 'jarvis', 'x': x, 'y': y * (3 * n / area), 'n': n,
                                                 'structure': (a['lattice_mat'], a['elements'], coords.tolist())}
    cols = ['source_task_id', 'x', 'y', 'calc_lattice', 'calc_numbers', 'calc_frac', 'task_nval']
    for path in (STORE / 'spectra_edos.parquet', STORE / 'sealed/spectra_edos.parquet'):
        ids = sorted(wanted_edos | other_edos)
        for r in pq.read_table(path, columns=cols, filters=[('source_task_id', 'in', ids)]).to_pylist():
            key = f"mp_edos:{r['source_task_id']}"
            if key in rec:
                continue
            rec[key] = {'kind': 'mp_edos', 'x': np.asarray(r['x']), 'y': np.asarray(r['y']),
                        'n': len(r['calc_numbers']), 'nval': r['task_nval'],
                        'structure': (r['calc_lattice'], [int(z) for z in r['calc_numbers']], r['calc_frac'])}
    return rec


def volume_per_atom(structure):
    lattice = np.asarray(structure[0], float).reshape(3, 3)
    return abs(float(np.linalg.det(lattice))) / len(structure[1])


def e_metric(a, na, b, nb, task, bins, C):
    dx = (RANGES[task][1] - RANGES[task][0]) / bins
    return float(dx * np.square(a / na - b / nb).sum() / C)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    out_splits, sealed_splits = STORE / 'splits', STORE / 'sealed' / 'splits'
    for path in [out_splits, sealed_splits, STORE / 'records_v11'] + [STORE / 'views' / v for v in RESOLUTIONS]:
        if path.exists():
            raise FileExistsError(path)
    materials = {m['material_id']: m for m in pq.read_table(STORE / 'materials.parquet').to_pylist()}
    for m in materials.values():
        m['formula'], m['elements'] = formula_of(m['numbers'])
    legacy_formula = {p: {m['formula'] for m in materials.values() if m['pool'] == p}
                      for p in ('train_q1', 'valid_q1', 'test_q1')}
    side_of = lambda formula, elements: assign(formula, elements, legacy_formula['valid_q1'],  # noqa: E731
                                               legacy_formula['test_q1'], legacy_formula['train_q1'])
    link = links()
    calc_types = load_calc_types()
    # other NSCF Uniform tasks per material (for eDOS disputed flags)
    owned = pq.read_table(STORE / 'spectra_edos.parquet', columns=['material_id', 'source_task_id', 'is_canonical', 'status'],
                          filters=[('is_canonical', '=', False), ('status', '=', 'ok')]).to_pandas()
    owned_sealed = pq.read_table(STORE / 'sealed/spectra_edos.parquet', columns=['material_id', 'source_task_id', 'is_canonical',
                                                                                'status'],
                                 filters=[('is_canonical', '=', False), ('status', '=', 'ok')]).to_pandas()
    owned = pd.concat([owned, owned_sealed])
    owned = owned[owned.source_task_id.map(lambda t: calc_types.get(strip_mp(t), '').endswith('NSCF Uniform'))]
    other_nscf = owned.groupby('material_id').source_task_id.apply(list).to_dict()
    wanted_edos = {r[len('mp_edos:'):] for r in link[link.source == 'mp_edos'].record_id}
    rec = load_records(wanted_edos, {t for v in other_nscf.values() for t in v})
    print('records loaded', len(rec), flush=True)

    # ---------------------------------------------------------------- units and primary labels
    units = {}  # unit_id -> dict(input structure, formula, elements, legacy, labels {task: rid}, link kinds)
    for mid, m in materials.items():
        units[mid] = {'kind': 'material', 'pool': m['pool'], 'formula': m['formula'], 'elements': m['elements'],
                      'structure': (m['cell'], [int(z) for z in m['numbers']], m['frac']), 'n': m['n_atoms'],
                      'nvalence': m['nvalence'], 'candidates': {'edos': [], 'phdos': []}}
    for r in link.itertuples():
        if r.record_id not in rec:
            continue
        task = 'edos' if r.source == 'mp_edos' else 'phdos'
        if r.material_id is not None and r.material_id in units:
            units[r.material_id]['candidates'][task].append((r.record_id, r.link))
        else:
            k = rec[r.record_id]
            formula, elements = formula_of(k['structure'][1])
            units[r.record_id] = {'kind': 'entry', 'pool': 'new', 'formula': formula, 'elements': elements,
                                  'structure': k['structure'], 'n': k['n'], 'nvalence': valence(k['structure'][1]),
                                  'candidates': {'edos': [], 'phdos': []}, 'entry_link': r.link}
            units[r.record_id]['candidates'][task].append((r.record_id, 'own'))
    labels, flags = {}, Counter()
    for uid, u in units.items():
        vin = volume_per_atom(u['structure'])
        u['labels'] = {}
        if u['pool'] in ('train_q1', 'valid_q1', 'test_q1'):
            continue  # legacy labels come from the step-2 builder
        if u['n'] > MAX_ATOMS:
            continue
        for task in RANGES:
            cands = []
            for rid, kind in u['candidates'][task]:
                k = rec[rid]
                if task == 'edos' and edos_window_ratio(k['x'], k['y'], k['nval']) > W_MAX:
                    flags['edos_anomalous_W>3'] += 1
                    continue
                ratio = volume_per_atom(k['structure']) / vin
                cands.append(((PRIORITY.get(k['kind'], 0), abs(np.log(ratio)), rid), rid, kind, ratio))
            if cands:
                cands.sort()
                _, rid, kind, ratio = cands[0]
                u['labels'][task] = {'rid': rid, 'link': kind, 'volume_ratio': ratio,
                                     'others': [c[1] for c in cands[1:]]}
    # ---------------------------------------------------------------- sets
    sets = {}
    for uid, u in units.items():
        side = side_of(u['formula'], u['elements'])
        if u['pool'] == 'train_q1':
            sets[uid] = 'holdout' if side in ('valid_side', 'test_side') else 'train'
        elif u['pool'] in ('valid_q1', 'test_q1'):
            sets[uid] = u['pool']
        elif not u['labels']:
            continue
        elif side == 'valid_side':
            sets[uid] = 'valid_supp'
        elif side == 'test_side':
            sets[uid] = 'test_supp'
        elif u['kind'] == 'entry' and side in ('dev', 'test'):
            sets[uid] = f'{side}_indep'
        else:
            sets[uid] = side
    sides = {}
    for uid, s in sets.items():
        sides.setdefault(units[uid]['formula'], set()).add(s)
    shared = [f for f, v in sides.items() if 'train' in v and v - {'train', 'holdout'}]
    if shared:
        raise ValueError(f'{len(shared)} formula groups in train and an evaluation set, e.g. {shared[:3]}')
    # ---------------------------------------------------------------- legacy rows (unclipped)
    legacy_public = pq.read_table(STORE / 'spectra_legacy.parquet').to_pylist()
    legacy_sealed = pq.read_table(STORE / 'sealed/spectra_legacy.parquet').to_pylist()
    C_main = {t: json.loads((STORE / 'views/q1_v2/manifest.json').read_text())['normalizer_train'][t]['energy']
              for t in RANGES}
    # disputed flags at the main resolution
    disputed = {}
    for uid, u in units.items():
        for task, lab in u['labels'].items():
            k = rec[lab['rid']]
            bins = RESOLUTIONS['s1_v2'][task]
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
    report = {'salt': SALT, 'rare_elements': sorted(RARE_ELEMENTS), 'disputed_E': DISPUTED_E, 'volume_flag': VOLUME_FLAG,
              'links': link.groupby(['source', 'link']).size().unstack(fill_value=0).to_dict('index'),
              'anomalous': dict(flags), 'views': {}}
    order = {s: sorted([u for u, x in sets.items() if x == s]) for s in PUBLIC + SEALED + ('holdout',)}
    for name, bins in RESOLUTIONS.items():
        legacy = {'train': build_split('train', bins, materials, [r for r in legacy_public if r['legacy_split'] == 'train']),
                  'valid': build_split('valid', bins, materials, [r for r in legacy_public if r['legacy_split'] == 'valid']),
                  'test': build_split('test', bins, materials, legacy_sealed)}
        legacy_rows = {}
        for split, b in legacy.items():
            for i, mid in enumerate(b['ids']):
                legacy_rows[mid] = {t: (b['arrays'][t][0][i], b['arrays'][t][1][i]) for t in RANGES}
        data = {}
        for s in PUBLIC + SEALED:
            ids = [u for u in order[s] if units[u]['pool'] not in ('train_q1', 'valid_q1', 'test_q1') or u in legacy_rows]
            n = len(ids)
            arr = {t: (np.zeros((n, bins[t]), np.float32), np.zeros((n, bins[t]), bool), np.zeros(n, bool),
                       np.zeros(n, np.int8), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.int8)) for t in RANGES}
            for i, uid in enumerate(ids):
                u = units[uid]
                for t in RANGES:
                    values, masks, has, source, dis, vol, lk = arr[t]
                    if uid in legacy_rows:
                        values[i], masks[i] = legacy_rows[uid][t]
                        has[i], source[i] = True, SOURCE_CODES['legacy']
                    elif t in u['labels']:
                        lab = u['labels'][t]
                        k = rec[lab['rid']]
                        scale = u['n'] / k['n']
                        v, cov = exact_bin(k['x'], np.asarray(k['y'], float) * scale, grid(t, bins[t]))
                        values[i], masks[i], has[i] = v, cov >= 1 - 1e-6, True
                        source[i] = SOURCE_CODES['mp'] if k['kind'] in ('mp_edos', 'mp_pheasy') else SOURCE_CODES.get(
                            k['kind'], 0)
                        dis[i] = disputed[(uid, t)]
                        vol[i] = abs(lab['volume_ratio'] - 1) > VOLUME_FLAG
                        lk[i] = LINK_CODES[lab['link']]
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
                         'source_codes': SOURCE_CODES, 'link_codes': LINK_CODES, 'split_rule': {
                             'salt': SALT, 'rare_elements': sorted(RARE_ELEMENTS), 'design': 'design-2026-11-11-splits-v2'},
                         'flags': {'disputed': f'another MP/PhononDB record (eDOS: NSCF Uniform task) differs by E > {DISPUTED_E}',
                                   'volume': f'|label calculation volume / input volume - 1| > {VOLUME_FLAG}'},
                         'spike_thresholds': {t: v.tolist() for t, v in thresholds.items()}, 'spike_clipped_bins': clipped,
                         'normalizer_train': C, 'training_exclusions': legacy_manifest['training_exclusions'],
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
                                                        for u in us]) if us else np.zeros(0, np.uint8),
                      'frac_coords': np.concatenate([np.asarray(u['structure'][2], np.float32).reshape(-1, 3)
                                                     for u in us]) if us else np.zeros((0, 3), np.float32),
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
    # ---------------------------------------------------------------- split lists, records, leakage
    out_splits.mkdir(parents=True)
    sealed_splits.mkdir(parents=True)
    hashes = {}
    for s, ids in order.items():
        directory = sealed_splits if s in SEALED else out_splits
        (directory / f'{s}.txt').write_text('\n'.join(ids) + '\n')
        hashes[s] = {'n': len(ids), 'sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    (out_splits / 'report.json').write_text(json.dumps({'split_hashes': hashes, 'salt': SALT}, indent=1) + '\n')
    records_dir = STORE / 'records_v11'
    records_dir.mkdir()
    link.to_parquet(records_dir / 'record_links.parquet')
    pd.DataFrame([{'record_id': rid, 'kind': k['kind'], 'n_atoms': k['n'],
                   'lattice': np.asarray(k['structure'][0], float).reshape(-1).tolist(),
                   'species': [str(s) for s in k['structure'][1]],
                   'frac': np.asarray(k['structure'][2], float).reshape(-1).tolist()} for rid, k in rec.items()]
                 ).to_parquet(records_dir / 'record_structures.parquet')
    group = {}
    for uid, s in sets.items():
        group.setdefault(units[uid]['formula'], set()).add(s)
    train_like = {'train'}
    evals = set(PUBLIC + SEALED) - train_like
    report['leakage'] = {
        'formula_groups_train_and_eval': sum(bool(v & train_like) and bool(v & evals) for v in group.values()),
        'formula_groups_dev_and_test_new': sum({'dev', 'test'} <= v for v in group.values()),
        'legacy_materials_in_new_dev_test': sum(units[u]['pool'] != 'new' for u in order['dev'] + order['test']),
        'deprecated_or_alias_ids_split': 'none by construction (placement depends on the reduced formula only)',
        'holdout_legacy_train': len(order['holdout'])}
    report['sets'] = {s: len(v) for s, v in order.items()}
    report['labels_by_source'] = {}
    for s in PUBLIC + SEALED:
        c = Counter()
        for uid in order[s]:
            for t, lab in units[uid]['labels'].items():
                c[f"{t}:{rec[lab['rid']]['kind']}"] += 1
        report['labels_by_source'][s] = dict(c)
    report['flags'] = {s: dict(Counter(f'{t}:disputed' for uid in order[s] for t in units[uid]['labels']
                                       if disputed.get((uid, t))) + Counter(
        f'{t}:volume>10%' for uid in order[s] for t, lab in units[uid]['labels'].items()
        if abs(lab['volume_ratio'] - 1) > VOLUME_FLAG) + Counter(
        f"{t}:link_{lab['link']}" for uid in order[s] for t, lab in units[uid]['labels'].items())) for s in order}
    (out_splits / 'build_report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('sets', 'leakage', 'anomalous')}, indent=1), flush=True)


def valence(species):
    """Valence electrons of a cell from MP POTCAR ZVAL (same table as build_store.calc_structure)."""
    from build_store import ZVAL
    from pymatgen.core import Element
    return float(sum(ZVAL[Element.from_Z(int(z)).symbol if not isinstance(z, str) else z]['zval'] for z in species))


def formula_z(s):
    from pymatgen.core import Element
    return int(s) if not isinstance(s, str) else Element(s).Z


if __name__ == '__main__':
    main()
