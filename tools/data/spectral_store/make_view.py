"""Training views of spectral-store-v1 (P1c; design 2026-11-10 P0).

A view is generated from a YAML config and written in the packed crystal format
read by utils/crystal_cache.py (same 11 arrays per split and a crystal-q1-v1
manifest), plus per-material task presence (has_edos, has_phdos) and source codes
(source_edos, source_phdos) for partial-label training.

- Legacy materials (train_q1/valid_q1) keep their frozen structures and labels.
- New materials are binned from native spectra with the dataset rule on E0/P0;
  eDOS: the MP canonical task, or with `edos_rule: canonical_or_fallback` also the
  fallback task of common.choose_edos_task (source code 3); phDOS: MP pheasy, else MP
  dfpt (legacy priority). A calculation cell with a different atom count than the input
  cell is rescaled to the input cell (densities are extensive; counted in the manifest).
- Isolated spikes of new spectra are clipped with per-bin p99.9 thresholds fitted on
  the view's train rows only (legacy labels are already clipped and stay frozen).
- A task a material lacks is stored as zeros with mask False and has_<task> False.
- Sealed splits (test_q1, test_dual, test_edos) are refused unless --unseal-final is
  given, which is logged in the view manifest. Do not use it before finalization.

Example config:

    name: u3
    splits:
      train: [train_q1, train_new]
      valid: [valid_q1]
      dev_edos: [dev_edos]
    tasks: {edos: true, phdos: true}     # which new labels enter the train split
    new_edos_limit: null                 # or {n: 20000} nested random subset of new eDOS-only materials
    max_atoms: 160
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import (ROOT, STORE, box_average, choose_edos_task, clip_isolated_spikes, file_sha256, load_calc_types,
                    spike_thresholds)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

GRIDS = {'edos': np.linspace(-6.0, 6.0, 129), 'phdos': np.linspace(-280.0, 980.0, 65)}
SEALED = {'test_q1', 'test_dual', 'test_edos'}
LEGACY_SPLITS = {'train_q1', 'valid_q1', 'test_q1'}
SOURCE_CODES = {'none': 0, 'legacy': 1, 'mp': 2, 'mp_fallback': 3}
EDOS_RULES = ('canonical', 'canonical_or_fallback')
FALLBACK_TYPE_ORDER = ('NSCF Uniform', 'Static')
NESTED_SEED = 20261110
LEGACY_MANIFEST = ROOT / 'data/crystal_q1_v1/manifest.json'


def load_table(name: str, sealed_ok: bool, columns=None):
    public = pq.read_table(STORE / f'{name}.parquet', columns=columns)
    if not sealed_ok:
        return public
    return pa.concat_tables([public, pq.read_table(STORE / 'sealed' / f'{name}.parquet', columns=columns)])


def native_spectra(task: str, wanted: set, sealed_ok: bool, materials: dict, edos_rule: str = 'canonical'):
    """material_id -> (x, y, source code, calc atom count) for the preferred usable MP spectrum.

    eDOS: common.choose_edos_task -- MP's canonical task; with edos_rule
    'canonical_or_fallback' also the fallback task (source code mp_fallback).
    phDOS: MP pheasy, else MP dfpt (legacy priority).
    """
    if not wanted:
        return {}
    table = load_table(f'spectra_{task}', sealed_ok)
    table = table.filter(pc.and_(pc.equal(table.column('status'), 'ok'),
                                 pc.is_in(table.column('material_id'),
                                          value_set=pa.array(sorted(wanted), type=pa.string()))))
    rows = {}
    for row in table.to_pylist():
        rows.setdefault(row['material_id'], []).append(row)
    chosen = {}
    if task == 'edos':
        if edos_rule not in EDOS_RULES:
            raise ValueError(f'unknown edos_rule {edos_rule}')
        calc_types = load_calc_types() if edos_rule == 'canonical_or_fallback' else {}
        for material_id, candidates in rows.items():
            m = materials[material_id]
            volume = abs(float(np.linalg.det(np.asarray(m['cell'], np.float64).reshape(3, 3)))) / m['n_atoms']
            row, kind = choose_edos_task(candidates, volume, calc_types, FALLBACK_TYPE_ORDER,
                                         allow_fallback=edos_rule == 'canonical_or_fallback')
            if row is not None:
                chosen[material_id] = (row['x'], row['y'], SOURCE_CODES['mp' if kind == 'canonical' else 'mp_fallback'],
                                       len(row['calc_numbers'] or []))
        return chosen
    rank = {'pheasy': 0, 'dfpt': 1}
    for material_id, candidates in rows.items():
        row = min(candidates, key=lambda r: (rank.get(r['method'], 9), r['source_task_id']))
        chosen[material_id] = (row['x'], row['y'], SOURCE_CODES['mp'], len(row['calc_numbers'] or []))
    return chosen


def split_hashes(requested: set) -> dict:
    hashes = json.loads((STORE / 'legacy_splits' / 'hashes.json').read_text())
    new = STORE / 'splits' / 'report.json'
    if new.exists():
        hashes.update(json.loads(new.read_text())['split_hashes'])
    return {name: hashes[name] for name in sorted(requested)}


def nested_subset(ids: list, n: int) -> set:
    order = sorted(ids, key=lambda i: hashlib.sha256(f'{NESTED_SEED}:{i}'.encode()).hexdigest())
    return set(order[:n])


def build(config: dict, out: Path, unseal: bool):
    if out.exists():
        raise FileExistsError(f'refusing existing view: {out}')
    requested = {s for parts in config['splits'].values() for s in parts}
    if requested & SEALED and not unseal:
        raise PermissionError(f'sealed splits requested without --unseal-final: {sorted(requested & SEALED)}')
    split_ids = {name: (STORE / ('legacy_splits' if name in LEGACY_SPLITS else 'splits') / f'{name}.txt')
                 .read_text().split() for name in requested}
    materials = {m['material_id']: m for m in pq.read_table(STORE / 'materials.parquet').to_pylist()}
    legacy = {(r['type'], r['material_id']): r for r in load_table('spectra_legacy', unseal).to_pylist()}
    new_ids = {i for name in requested - LEGACY_SPLITS for i in split_ids[name]}
    # Task toggles choose which new labels enter the train split; evaluation splits always
    # carry every label they have.
    tasks = config.get('tasks', {'edos': True, 'phdos': True})
    edos_rule = config.get('edos_rule', 'canonical')
    native = {t: native_spectra(t, new_ids, unseal, materials, edos_rule) for t in GRIDS}

    def label(task, material_id, view_split):
        return material_id in native[task] and (view_split != 'train' or bool(tasks.get(task)))

    limit = config.get('new_edos_limit')
    if limit:
        # Nested random subset of the new train materials whose eDOS label is added; with
        # phDOS enabled, materials that bring a new phDOS label keep their eDOS too.
        train_new = set(split_ids.get('train_new', []))
        protected = set(native['phdos']) if tasks.get('phdos') else set()
        pool = sorted(i for i in train_new if i in native['edos'] and i not in protected)
        keep = nested_subset(pool, int(limit['n']))
        native['edos'] = {k: v for k, v in native['edos'].items()
                          if k not in train_new or k in keep or k in protected}
    max_atoms = int(config.get('max_atoms', 160))
    rows = {}
    for view_split, parts in config['splits'].items():
        selected = []
        for part in parts:
            for material_id in split_ids[part]:
                m = materials[material_id]
                if part in LEGACY_SPLITS:
                    selected.append((material_id, 'legacy'))
                elif m['n_atoms'] <= max_atoms and (label('edos', material_id, view_split)
                                                    or label('phdos', material_id, view_split)):
                    selected.append((material_id, 'new'))
        rows[view_split] = selected
    binned, rescaled = {}, {}
    for view_split, selected in rows.items():
        for task in GRIDS:
            n_bins = len(GRIDS[task]) - 1
            values = np.zeros((len(selected), n_bins), dtype=np.float32)
            masks = np.zeros((len(selected), n_bins), dtype=bool)
            has = np.zeros(len(selected), dtype=bool)
            source = np.zeros(len(selected), dtype=np.int8)
            new_row = np.zeros(len(selected), dtype=bool)
            for i, (material_id, kind) in enumerate(selected):
                if kind == 'legacy':
                    r = legacy[(task, material_id)]
                    values[i], masks[i], has[i], source[i] = r['values'], r['mask'], True, SOURCE_CODES['legacy']
                elif label(task, material_id, view_split):
                    x, y, code, n_calc = native[task][material_id]
                    v, c = box_average(x, y, GRIDS[task])
                    n_input = materials[material_id]['n_atoms']
                    if n_calc and n_calc != n_input:  # densities are extensive: express per input cell
                        v = v * (n_input / n_calc)
                        rescaled[task] = rescaled.get(task, 0) + 1
                    values[i], masks[i], has[i], source[i] = v, c.astype(bool), True, code
                    new_row[i] = True
            binned[(view_split, task)] = [values, masks, has, source, new_row]
    thresholds, clipped = {}, {}
    for task in GRIDS:
        values, masks, has = binned[('train', task)][:3] if ('train', task) in binned else (None, None, None)
        if values is None or not has.any():
            continue
        thresholds[task] = spike_thresholds(values[has], masks[has])
        for view_split in rows:
            entry = binned[(view_split, task)]
            if entry[4].any():
                new_values, clip = clip_isolated_spikes(entry[0][entry[4]], entry[1][entry[4]], thresholds[task])
                entry[0][entry[4]] = new_values
                clipped[f'{view_split}/{task}'] = int(clip.sum())
    legacy_manifest = json.loads(LEGACY_MANIFEST.read_text())
    manifest = {key: legacy_manifest[key] for key in ('schema', 'grid', 'cell_convention', 'frac_convention',
                                                       'primitive_policy', 'geometry_precision')}
    manifest.update({'label_pool': f"spectral-store-v1 view {config['name']}", 'atom_cap': max_atoms,
                     'store': 'spectral-store-v1', 'view_config': config,
                     'view_config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                     'partial_labels': True, 'source_codes': SOURCE_CODES, 'unsealed': bool(unseal),
                     'spike_thresholds': {t: v.tolist() for t, v in thresholds.items()}, 'spike_clipped_bins': clipped,
                     'edos_rule': edos_rule, 'rescaled_to_input_cell': rescaled,
                     'training_exclusions': legacy_manifest['training_exclusions'],
                     'quality_review': legacy_manifest['quality_review'],
                     'split_hashes': split_hashes(requested),
                     'splits': {}, 'files': {}})
    out.mkdir(parents=True)
    for view_split, selected in rows.items():
        directory = out / view_split
        directory.mkdir()
        ms = [materials[i] for i, _ in selected]
        counts = np.asarray([m['n_atoms'] for m in ms], dtype=np.int32)
        arrays = {
            'cell': np.asarray([m['cell'] for m in ms], dtype=np.float32).reshape(-1, 3, 3),
            'atomic_numbers': np.concatenate([np.asarray(m['numbers'], dtype=np.uint8) for m in ms]),
            'frac_coords': np.concatenate([np.asarray(m['frac'], dtype=np.float32).reshape(-1, 3) for m in ms]),
            'n_atoms': counts, 'atom_offsets': np.r_[0, np.cumsum(counts, dtype=np.int64)],
            'nvalence': np.asarray([m['nvalence'] for m in ms], dtype=np.float32)}
        for task in GRIDS:
            values, masks, has, source, _ = binned[(view_split, task)]
            arrays.update({f'{task}_tgtdos': values, f'{task}_mask': masks, f'has_{task}': has,
                           f'source_{task}': source})
        arrays['index'] = np.asarray([i for i, _ in selected])
        for name, array in arrays.items():
            filename = f'{view_split}_index.npy' if name == 'index' else f'{name}_{view_split}.npy'
            path = directory / filename
            np.save(path, array, allow_pickle=False)
            manifest['files'][f'{view_split}/{filename}'] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                                            'sha256': file_sha256(path)}
        manifest['splits'][view_split] = {
            'n': len(selected), 'atoms': int(counts.sum()), 'max_atoms': int(counts.max()),
            'legacy': sum(k == 'legacy' for _, k in selected), 'new': sum(k == 'new' for _, k in selected),
            'has_edos': int(binned[(view_split, 'edos')][2].sum()), 'has_phdos': int(binned[(view_split, 'phdos')][2].sum()),
            'training_n': len(selected) - sum(i in legacy_manifest['training_exclusions'] for i, _ in selected)}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest['splits'], indent=1))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--out', default=None)
    parser.add_argument('--unseal-final', action='store_true', help='only for the final, logged evaluation')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out) if args.out else STORE / 'views' / config['name']
    build(config, out, args.unseal_final)


if __name__ == '__main__':
    main()
