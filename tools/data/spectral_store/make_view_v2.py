"""Label-contract v2 views of the legacy materials (design 2026-11-11, step 2).

Every legacy label is regenerated from the raw spectrum identified in step 1 (the one that
the legacy rule maps exactly onto the frozen label), with only the binning changed:

- eDOS raw spectrum: the audit-matched task (spectra_edos, or staging/edos_legacy when MP
  now assigns it to another material) for train/valid; the gate-matched task
  (sealed/edos_legacy_test) for test.
- phDOS raw spectrum: the matched MP phonon task (train/valid); for test, and for train/valid
  labels without a matched task, the material's own MP phonon spectrum that passes the legacy
  gate (spectra_phdos, sealed/spectra_phdos for test); otherwise the
  PhononDB / JARVIS record of the step-1 links (JARVIS intensity x 3N / full area, N =
  input atoms, as in the legacy pipeline).
- Binning: common.exact_bin on [-6, 6] eV / [-280, 980] cm^-1; densities of a calculation
  cell with another atom count are rescaled to the input cell; mask = bin fully inside the
  sampled range (not used by the working point).
- Anomalous eDOS (user decision 2026-11-11): W = states in [-6, 6] eV / valence electrons
  of the calculation cell > 3 -> label removed. phDOS is not screened; every row carries
  `phdos_modes_per_atom` (full-spectrum area / input atoms; 3 when complete; JARVIS is
  normalised by construction) as a flag.
- Isolated spikes: legacy rule (per-bin p99.9 on covered train values, isolated bins only)
  with thresholds refitted on the v2 train labels of the same resolution, applied to all
  splits.
- The view keeps dual-label rows only (the trainer has no partial-label path yet); every
  material left out is listed with its reason in the manifest.
- C (utils.spectral_balance.fit_task_normalizer) is fitted on the v2 train labels and
  written to the manifest.

Resolutions: --bins edos=128,phdos=64 (main benchmark, view q1_v2) or edos=256,phdos=128
(view q1_v2_x2). Test (sealed) is written only with --sealed, under sealed/views/<name>/,
and only shapes, finiteness and hashes are printed. --raw-points also writes the raw
sampled points behind every exported label (per input cell, after unit conversion and the
JARVIS scaling) to exports/<name>_raw/ (test: sealed/exports/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from common import ROOT, STORE, box_average, classify_label, clip_isolated_spikes, exact_bin, file_sha256, spike_thresholds

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from utils.spectral_balance import fit_task_normalizer

STAGING = STORE / 'staging'
RANGES = {'edos': (-6.0, 6.0), 'phdos': (-280.0, 980.0)}
LEGACY_BINS = {'edos': 128, 'phdos': 64}
W_MAX = 3.0
LEGACY_MANIFEST = ROOT / 'data/crystal_q1_v1/manifest.json'
SPLITS = {'train': 'train_q1', 'valid': 'valid_q1', 'test': 'test_q1'}


def grid(task, bins):
    lo, hi = RANGES[task]
    return np.linspace(lo, hi, bins + 1)


def _rows(path, key, ids, columns):
    table = pq.read_table(path, columns=columns, filters=[(key, 'in', sorted(ids))]) if ids else None
    return {} if table is None else {r[key]: r for r in table.to_pylist()}


def _parts(directory):
    return sorted(Path(directory).glob('part-*.parquet'))


def edos_window_ratio(x, y, task_nval):
    """W: states in [-6, 6] eV of the calculation cell / its valence electrons."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    order = np.argsort(x[keep], kind='stable')
    x, y = x[keep][order], y[keep][order]
    lo, hi = RANGES['edos']
    inner = (x > lo) & (x < hi)
    pts = np.concatenate([[max(lo, x[0])], x[inner], [min(hi, x[-1])]])
    area = float(np.trapz(np.interp(pts, x, y), pts)) if pts[-1] > pts[0] else 0.0
    return area / task_nval if task_nval else np.inf


def raw_sources(split: str, labels: dict, materials: dict):
    """(task, material) -> dict(x, y, n_calc, origin, source, nval) or a reason string."""
    sealed = split == 'test'
    out = {}
    ids = {m for (t, m) in labels}
    edos_cols = ['material_id', 'source_task_id', 'x', 'y', 'calc_numbers', 'task_nval', 'status']
    ph_cols = ['material_id', 'source_task_id', 'method', 'x', 'y', 'calc_numbers', 'status']
    # ---- eDOS
    if sealed:
        links = {r['material_id']: r for r in pq.read_table(STORE / 'sealed/edos_legacy_links_test.parquet').to_pylist()}
        rows = {}
        for r in pq.read_table(STORE / 'sealed/edos_legacy_test.parquet', columns=edos_cols).to_pylist():
            if r['material_id'] in links and r['source_task_id'] == links[r['material_id']]['source_task_id']:
                rows[r['material_id']] = r
        for m in ids:
            out[('edos', m)] = rows.get(m, 'unrecovered')
            if m in rows:
                out[('edos', m)] = dict(rows[m], origin=f"mp:{rows[m]['source_task_id']}", source='mp')
    else:
        matched = {m: labels[('edos', m)]['matched_task'] for m in ids if labels[('edos', m)]['matched_task']}
        owned = _rows(STORE / 'spectra_edos.parquet', 'source_task_id', set(matched.values()), edos_cols)
        extra = {}
        for r in pq.read_table(STAGING / 'edos_legacy/part-00.parquet', columns=edos_cols).to_pylist():
            extra.setdefault(r['material_id'], r)
        for m in ids:
            r = owned.get(matched.get(m)) or extra.get(m)
            out[('edos', m)] = dict(r, origin=f"mp:{r['source_task_id']}", source='mp') if r else 'unrecovered'
    # ---- phDOS
    if sealed:
        mp = {}
        for r in pq.read_table(STORE / 'sealed/spectra_phdos.parquet', columns=ph_cols).to_pylist():
            if r['status'] == 'ok':
                mp.setdefault(r['material_id'], []).append(r)
        links = {r['material_id']: r for r in pq.read_table(STORE / 'sealed/phdos_legacy_links_test.parquet').to_pylist()}
    else:
        matched = {m: labels[('phdos', m)]['matched_task'] for m in ids if labels[('phdos', m)]['matched_task']}
        mp_rows = _rows(STORE / 'spectra_phdos.parquet', 'source_task_id', set(matched.values()), ph_cols)
        mp = {m: [mp_rows[t]] for m, t in matched.items() if t in mp_rows}
        # labels without an audit-matched task: the material's own MP phonon spectra, through the gate
        unmatched = ids - set(matched)
        if unmatched:
            for r in pq.read_table(STORE / 'spectra_phdos.parquet', columns=ph_cols,
                                   filters=[('material_id', 'in', sorted(unmatched))]).to_pylist():
                if r['status'] == 'ok':
                    mp.setdefault(r['material_id'], []).append(r)
        links = {r['material_id']: r for r in pq.read_table(STAGING / 'phdos_legacy_links.parquet').to_pylist()}
    other = {}
    for name in ('phdos_phonondb', 'phdos_jarvis'):
        wanted = {r['source_id'] for r in links.values() if r['source'] == name.split('_')[1]}
        for part in _parts(STAGING / name):
            other.update(_rows(part, 'source_id', wanted, ['source', 'source_id', 'x', 'y', 'status']))
    p0 = grid('phdos', LEGACY_BINS['phdos'])
    for m in ids:
        label = labels[('phdos', m)]
        chosen = None
        for r in sorted(mp.get(m, []), key=lambda r: ({'pheasy': 0, 'dfpt': 1}.get(r['method'], 9), r['source_task_id'])):
            if (not sealed and m in matched) or classify_label(box_average(r['x'], r['y'], p0)[0], np.asarray(label['values'], np.float32),
                                            np.asarray(label['mask'], bool), 4)[0] != 'differs':
                chosen = dict(r, origin=f"mp:{r['source_task_id']}", source='mp')
                break
        if chosen is None and m in links and links[m]['source_id'] in other:
            r = other[links[m]['source_id']]
            n = materials[m]['n_atoms']
            y = np.asarray(r['y'], np.float64)
            if r['source'] == 'jarvis':  # legacy rule: 3N / native full-spectrum area
                area = float(np.trapz(y, r['x']))
                y = y * (3 * n / area) if area > 0 else y
            chosen = {'x': r['x'], 'y': y.tolist(), 'calc_numbers': None, 'origin': f"{r['source']}:{r['source_id']}",
                      'source': r['source']}
        out[('phdos', m)] = chosen or 'unrecovered'
    return out


def build_split(split, bins, materials, legacy_rows):
    labels = {(r['type'], r['material_id']): r for r in legacy_rows}
    ids = sorted({m for (_, m) in labels}, key=lambda m: materials[m]['legacy_row'])
    sources = raw_sources(split, labels, materials)
    kept, dropped, rescaled, sources_used = [], Counter(), Counter(), Counter()
    values = {t: [] for t in RANGES}
    masks = {t: [] for t in RANGES}
    modes, raw = [], []
    for m in ids:
        n_input = materials[m]['n_atoms']
        row, reason = {}, None
        for task in RANGES:
            src = sources[(task, m)]
            if isinstance(src, str):
                reason = f'{task}:{src}'
                break
            if task == 'edos' and edos_window_ratio(src['x'], src['y'], src['task_nval']) > W_MAX:
                reason = 'edos:anomalous_W>3'
                break
            y = np.asarray(src['y'], np.float64)
            n_calc = len(src['calc_numbers'] or []) or n_input
            if n_calc != n_input:
                y = y * (n_input / n_calc)
                rescaled[task] += 1
            v, cov = exact_bin(src['x'], y, grid(task, bins[task]))
            row[task] = (v, cov >= 1.0 - 1e-6, src, y)
        if reason:
            dropped[reason] += 1
            continue
        kept.append(m)
        for task in RANGES:
            v, mask, src, y = row[task]
            values[task].append(v)
            masks[task].append(mask)
            sources_used[f"{task}:{src['source']}"] += 1
            raw.append({'material_id': m, 'task': task, 'origin': src['origin'], 'x': np.asarray(src['x'], np.float64),
                        'y': y})
        ph_x, ph_y = np.asarray(row['phdos'][2]['x'], np.float64), row['phdos'][3]
        order = np.argsort(ph_x, kind='stable')
        modes.append(float(np.trapz(ph_y[order], ph_x[order])) / n_input)
    arrays = {t: (np.asarray(values[t], np.float32), np.asarray(masks[t], bool)) for t in RANGES}
    return {'ids': kept, 'arrays': arrays, 'modes': np.asarray(modes, np.float32), 'raw': raw,
            'report': {'legacy_rows': len(ids), 'kept': len(kept), 'dropped': dict(dropped),
                       'rescaled_to_input_cell': dict(rescaled), 'sources': dict(sources_used)},
            'dropped_ids': None}


def write_split(directory, split, ids, materials, arrays, modes, manifest, root):
    directory.mkdir(parents=True)
    ms = [materials[i] for i in ids]
    counts = np.asarray([m['n_atoms'] for m in ms], dtype=np.int32)
    out = {'cell': np.asarray([m['cell'] for m in ms], dtype=np.float32).reshape(-1, 3, 3),
           'atomic_numbers': np.concatenate([np.asarray(m['numbers'], dtype=np.uint8) for m in ms]),
           'frac_coords': np.concatenate([np.asarray(m['frac'], dtype=np.float32).reshape(-1, 3) for m in ms]),
           'n_atoms': counts, 'atom_offsets': np.r_[0, np.cumsum(counts, dtype=np.int64)],
           'nvalence': np.asarray([m['nvalence'] for m in ms], dtype=np.float32),
           'phdos_modes_per_atom': modes, 'index': np.asarray(ids)}
    for task, (v, mask) in arrays.items():
        out[f'{task}_tgtdos'], out[f'{task}_mask'] = v, mask
    for name, array in out.items():
        filename = f'{split}_index.npy' if name == 'index' else f'{name}_{split}.npy'
        path = directory / filename
        np.save(path, array, allow_pickle=False)
        manifest['files'][str(path.relative_to(root))] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                                          'sha256': file_sha256(path)}
    manifest['splits'][split] = {'n': len(ids), 'atoms': int(counts.sum()), 'max_atoms': int(counts.max())}


def write_raw(path, split, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(r, split=split, x=r['x'].tolist(), y=np.asarray(r['y']).tolist()) for r in raw])
    pq.write_table(table, path, compression='zstd')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True)
    parser.add_argument('--bins', default='edos=128,phdos=64')
    parser.add_argument('--sealed', action='store_true', help='also write the sealed test labels (counts printed only)')
    parser.add_argument('--raw-points', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='build and print the summary; write nothing')
    args = parser.parse_args()
    bins = {k: int(v) for k, v in (item.split('=') for item in args.bins.split(','))}
    out = STORE / 'views' / args.name
    sealed_out = STORE / 'sealed' / 'views' / args.name
    for path in ((out, sealed_out) if args.sealed else (out,)) if not args.dry_run else ():
        if path.exists():
            raise FileExistsError(path)
    materials = {m['material_id']: m for m in pq.read_table(STORE / 'materials.parquet').to_pylist()}
    legacy = pq.read_table(STORE / 'spectra_legacy.parquet').to_pylist()
    built = {}
    for split in ('train', 'valid'):
        built[split] = build_split(split, bins, materials, [r for r in legacy if r['legacy_split'] == split])
    if args.sealed:
        built['test'] = build_split('test', bins, materials, pq.read_table(STORE / 'sealed/spectra_legacy.parquet').to_pylist())
    # spikes: thresholds on v2 train labels (covered bins), applied to all splits
    thresholds, clipped = {}, {}
    for task in RANGES:
        v, mask = built['train']['arrays'][task]
        thresholds[task] = spike_thresholds(v, mask)
        for split, b in built.items():
            values, clip = clip_isolated_spikes(*b['arrays'][task], thresholds[task])
            b['arrays'][task] = (values, b['arrays'][task][1])
            clipped[f'{split}/{task}'] = int(clip.sum())
    atoms = np.asarray([materials[m]['n_atoms'] for m in built['train']['ids']], np.float64)
    normalizer = {t: fit_task_normalizer(built['train']['arrays'][t][0].astype(np.float64), atoms,
                                         float(np.diff(RANGES[t])[0] / bins[t])) for t in RANGES}
    legacy_manifest = json.loads(LEGACY_MANIFEST.read_text())
    manifest = {k: legacy_manifest[k] for k in ('schema', 'cell_convention', 'frac_convention', 'primitive_policy',
                                                 'geometry_precision')}
    manifest.update({
        'label_contract': 'v2', 'binning': 'exact_piecewise_linear', 'store': 'spectral-store-v1',
        'label_pool': f'spectral-store-v1 view {args.name} (legacy materials, v2 labels)',
        'grid': {t: {'range': list(RANGES[t]), 'bins': bins[t]} for t in RANGES},
        'anomaly_rule': {'edos': f'W = states in [-6, 6] eV / calculation-cell valence electrons > {W_MAX}: removed',
                         'phdos': 'not screened; phdos_modes_per_atom recorded'},
        'dual_label_only': True, 'spike_thresholds': {t: v.tolist() for t, v in thresholds.items()},
        'spike_clipped_bins': clipped, 'normalizer_train': normalizer,
        'inputs': {name: file_sha256(STORE / name) for name in (
            'materials.parquet', 'spectra_legacy.parquet', 'staging/edos_legacy/links.parquet',
            'staging/phdos_legacy_links.parquet')},
        'training_exclusions': legacy_manifest['training_exclusions'],
        'reports': {s: b['report'] for s, b in built.items() if s != 'test'}, 'splits': {}, 'files': {}})
    if args.dry_run:
        print(json.dumps({'reports': manifest['reports'], 'clipped': clipped,
                          'C': {t: normalizer[t]['energy'] for t in RANGES},
                          'test_report': built['test']['report'] if args.sealed else None}, indent=1))
        return
    out.mkdir(parents=True)
    for split in ('train', 'valid'):
        b = built[split]
        write_split(out / split, split, b['ids'], materials, b['arrays'], b['modes'], manifest, out)
        if args.raw_points:
            write_raw(STORE / 'exports' / f'{args.name}_raw' / f'{split}.parquet', split, b['raw'])
    manifest['split_material_sha256'] = {s: hashlib.sha256('\n'.join(built[s]['ids']).encode()).hexdigest()
                                         for s in ('train', 'valid')}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    summary = {'splits': manifest['splits'], 'reports': manifest['reports'], 'spike_clipped_bins': clipped,
               'C': {t: normalizer[t]['energy'] for t in RANGES}}
    if args.sealed:
        b = built['test']
        sealed_manifest = {'label_contract': 'v2', 'view': args.name, 'splits': {}, 'files': {}, 'report': b['report']}
        write_split(sealed_out / 'test', 'test', b['ids'], materials, b['arrays'], b['modes'], sealed_manifest, sealed_out)
        sealed_manifest['finite'] = {t: bool(np.isfinite(b['arrays'][t][0]).all()) for t in RANGES}
        (sealed_out / 'manifest.json').write_text(json.dumps(sealed_manifest, indent=2) + '\n')
        if args.raw_points:
            write_raw(STORE / 'sealed' / 'exports' / f'{args.name}_raw' / 'test.parquet', 'test', b['raw'])
        summary['test_counts_only'] = {'report': b['report'], 'finite': sealed_manifest['finite'],
                                       'shapes': {k: v['shape'] for k, v in sealed_manifest['files'].items()},
                                       'manifest_sha256': file_sha256(sealed_out / 'manifest.json')}
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
