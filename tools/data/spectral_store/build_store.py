"""Build the staging tables of spectral-store-v1 (P1a; design 2026-11-10 P0).

Stages (each refuses to overwrite its own output):

- index:     MP metadata pages -> material index (license, deprecation, canonical DOS
             task, task -> material map, phonon task metadata).
- edos:      every DOS task of the local public Delta total-dos table that MP assigns to
             a material (materials/core task_ids), MP's canonical DOS task flagged
             (is_canonical); native spectrum (energies - E_F, spin-summed density,
             float64), task-cell valence, Q1 truncation ratio, calculation structure.
- phdos:     every MP phonon task in the local Delta table mapped to its material;
             native spectrum in cm^-1 (frequency x 33.35641, density / 33.35641),
             calculation structure, sum-rule breaking.
- materials: input structure per material. Legacy materials keep their frozen
             crystal_q1_v1 arrays; new ones follow the same rule (MP summary
             structure, pymatgen primitive standard, symprec 0.1, FP32 lattice
             parameters -> build_cell_from_lattice). Element-table valence count.
- legacy:    frozen crystal_q1_v1 labels as legacy_binned spectra (E0/P0 values and
             masks), with the label task matched by the 2026-11-05 audit when known.

Outputs go to data/spectral_store_v1/staging/. Splits and sealing are done by
make_splits.py; training arrays by make_view.py. Never reads test labels for
anything but copying them into the store; nothing here selects models.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from common import (DELTA, LEGACY, SPLITS_LEGACY, STORE, SYMPREC, THZ2CM, TRUNC_MIN, ROOT,
                    composition_key, load_pages, reduced_key, strip_mp)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STAGING = STORE / 'staging'
RAW = STORE / 'raw_mp'
RAW_CORE = STORE / 'raw_mp_core'
ZVAL = json.loads((ROOT / 'data/reference/valence_by_element.json').read_text())
AUDIT = ROOT / 'results/runs/structure_use_audit_20261105'
CHUNK = 500
_CALC = [('calc_lattice', pa.list_(pa.float64())), ('calc_numbers', pa.list_(pa.int16())),
         ('calc_frac', pa.list_(pa.float64())), ('calc_volume_per_atom', pa.float64())]
_COMMON = [('spectrum_id', pa.string()), ('material_id', pa.string()), ('source', pa.string()),
           ('source_task_id', pa.string()), ('method', pa.string()), ('functional', pa.string()),
           ('x', pa.list_(pa.float64())), ('y', pa.list_(pa.float64())), ('n_points', pa.int32()),
           ('status', pa.string())]
EDOS_SCHEMA = pa.schema(_COMMON + [('is_canonical', pa.bool_()), ('efermi', pa.float64()), ('spin_channels', pa.int8()),
                                   ('task_nval', pa.float64()), ('full_mass', pa.float64()),
                                   ('trunc_ratio', pa.float64())] + _CALC)
PHDOS_SCHEMA = pa.schema(_COMMON + [('asr', pa.float64()), ('cnsr', pa.float64())] + _CALC)


def refuse(path: Path):
    if path.exists():
        raise FileExistsError(f'refusing existing output: {path}')


# ---------------------------------------------------------------- index
def build_index(out: Path):
    """DOS task per material, task -> material map and phonon metadata (no summary needed)."""
    refuse(out)
    dos_task = {}
    for doc in load_pages(RAW / 'electronic_structure'):
        dos = doc.get('dos') or {}
        if dos.get('task_id'):
            dos_task[doc['material_id']] = strip_mp(dos['task_id'])
    task_owner, conflicts = {}, Counter()
    for doc in load_pages(RAW_CORE / 'core'):
        for task in doc.get('task_ids') or []:
            task = strip_mp(task)
            if task in task_owner and task_owner[task] != doc['material_id']:
                conflicts['task_in_two_materials'] += 1
            task_owner[task] = doc['material_id']
    phonon = {}
    for doc in load_pages(RAW / 'phonon'):
        rules = doc.get('sum_rules_breaking') or {}
        phonon.setdefault(strip_mp(doc['identifier']), []).append(
            {'method': doc.get('phonon_method'), 'asr': rules.get('asr'), 'cnsr': rules.get('cnsr')})
    owners = Counter(task_owner.get(t) is not None for t in phonon)
    index = {'dos_task': dos_task, 'task_owner': task_owner, 'phonon': phonon,
             'counts': {'dos_tasks': len(dos_task), 'tasks': len(task_owner),
                        'phonon_docs': sum(map(len, phonon.values())),
                        'phonon_tasks_with_material': owners[True], 'phonon_tasks_without_material': owners[False],
                        'dos_tasks_owned_by_their_material': sum(task_owner.get(t) == m for m, t in dos_task.items()),
                        **conflicts}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index))
    print(json.dumps(index['counts']))
    return index


def summary_meta():
    meta = {}
    for doc in load_pages(RAW / 'summary'):
        builder = doc.get('builder_meta') or {}
        meta[doc['material_id']] = {
            'deprecated': bool(doc.get('deprecated')), 'license': builder.get('license'),
            'database_version': builder.get('database_version'), 'formula': doc.get('formula_pretty'),
            'nsites_mp': doc.get('nsites'), 'structure': doc.get('structure')}
    return meta


def load_index():
    return json.loads((STAGING / 'index.json').read_text())


# ---------------------------------------------------------------- shared helpers
def calc_structure(structure: dict):
    lattice = np.asarray(structure['lattice']['matrix'], dtype=np.float64)
    numbers, frac, valence = [], [], 0.0
    from ase.data import atomic_numbers
    for site in structure['sites']:
        species = site['species']
        numbers.append(atomic_numbers[species[0]['element']])
        frac.append(site['abc'])
        for item in species:
            valence += float(item.get('occu', 1.0)) * float(ZVAL[item['element']]['zval'])
    volume = abs(float(np.linalg.det(lattice)))
    return {'calc_lattice': lattice.reshape(-1).tolist(), 'calc_numbers': numbers,
            'calc_frac': np.asarray(frac, dtype=np.float64).reshape(-1).tolist(),
            'calc_volume_per_atom': volume / max(len(numbers), 1)}, valence


class PartWriter:
    def __init__(self, path: Path, schema: pa.Schema):
        self.path, self.schema, self.rows, self.writer = path, schema, [], None

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= CHUNK:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression='zstd')
        self.writer.write_table(table)
        self.rows = []

    def close(self):
        self.flush()
        if self.writer is None:  # keep an empty, typed part so readers see the schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression='zstd')
        self.writer.close()


# ---------------------------------------------------------------- eDOS
def edos_record(row: dict, run_type: str, material_id: str, is_canonical: bool) -> dict:
    """Store row of one Delta total-dos task (native spectrum, Q1 truncation status, calc structure)."""
    task = row['identifier']
    record = {'spectrum_id': f'edos:mp:{task}', 'material_id': material_id, 'source': 'mp',
              'source_task_id': task, 'method': run_type, 'functional': run_type,
              'is_canonical': is_canonical, 'efermi': None, 'x': [], 'y': [], 'n_points': 0, 'spin_channels': 0,
              'task_nval': None, 'full_mass': None, 'trunc_ratio': None, 'status': '',
              'calc_lattice': [], 'calc_numbers': [], 'calc_frac': [], 'calc_volume_per_atom': None}
    try:
        calc, task_nval = calc_structure(row['structure'])
        record.update(calc, task_nval=task_nval)
        energies = np.asarray(row['energies'], dtype=np.float64)
        up = np.asarray(row['spin_up_densities'], dtype=np.float64)
        down = row['spin_down_densities']
        has_down = down is not None and np.ndim(down) != 0 and len(down) == len(up)
        down = np.asarray(down, dtype=np.float64) if has_down else np.zeros_like(up)
        efermi = row['efermi']
        if efermi is None or not np.isfinite(efermi) or len(energies) != len(up) or len(up) < 2:
            raise ValueError('invalid arrays or E_F')
        x, y = energies - float(efermi), up + down
        finite = np.isfinite(x) & np.isfinite(y)
        xs, ys = x[finite], y[finite]
        order = np.argsort(xs, kind='stable')
        xs, ys = xs[order], ys[order]
        ux, inverse = np.unique(xs, return_inverse=True)
        if len(ux) != len(xs):
            sums, cnt = np.zeros(len(ux)), np.zeros(len(ux))
            np.add.at(sums, inverse, ys)
            np.add.at(cnt, inverse, 1)
            xs, ys = ux, sums / cnt
        full_mass = float(np.trapz(ys, xs))
        record.update(efermi=float(efermi), x=x.tolist(), y=y.tolist(), n_points=int(len(x)),
                      spin_channels=2 if has_down else 1, full_mass=full_mass,
                      trunc_ratio=full_mass / task_nval if task_nval > 0 else None)
        record['status'] = 'ok' if record['trunc_ratio'] is not None and record['trunc_ratio'] >= TRUNC_MIN \
            else 'truncated'
    except Exception as exc:  # recorded, never silently dropped
        record['status'] = f'unreadable:{type(exc).__name__}'
    return record


def _edos_file(job):
    path, run_type, wanted, canonical, part = job
    writer = PartWriter(part, EDOS_SCHEMA)
    counts = Counter()
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=256, columns=[
            'identifier', 'structure', 'energies', 'spin_up_densities', 'spin_down_densities', 'efermi']):
        ids = batch.column('identifier').to_pylist()
        take = [i for i, task in enumerate(ids) if task in wanted]
        if not take:
            continue
        for row in batch.take(pa.array(take)).to_pylist():
            record = edos_record(row, run_type, wanted[row['identifier']],
                                 canonical.get(wanted[row['identifier']]) == row['identifier'])
            counts[record['status'].split(':')[0]] += 1
            writer.add(record)
    writer.close()
    return str(path), dict(counts)


def delta_files(table: str, key: str):
    files = []
    for path in sorted((DELTA / table).rglob('*.parquet')):
        if '_delta_log' in path.parts:
            continue
        value = next(part.split('=', 1)[1] for part in path.parts if part.startswith(f'{key}='))
        files.append((path, urllib.parse.unquote(value)))
    return files


def build_edos(out_dir: Path, workers: int):
    refuse(out_dir)
    index = load_index()
    # Every DOS task owned by a material is stored; MP's canonical task is flagged. The
    # first-round views use canonical tasks only (user decision 2026-11-10).
    canonical = index['dos_task']
    files = delta_files('core/electronic-structure/total-dos', 'run_type')
    present = set()
    for path, _ in files:
        present |= set(pq.read_table(path, columns=['identifier']).column(0).to_pylist())
    wanted = {task: index['task_owner'][task] for task in present if task in index['task_owner']}
    out_dir.mkdir(parents=True)
    jobs = [(path, run_type, wanted, canonical, out_dir / f'part-{i:02d}.parquet')
            for i, (path, run_type) in enumerate(delta_files('core/electronic-structure/total-dos', 'run_type'))]
    jobs.sort(key=lambda job: -job[0].stat().st_size)
    totals = Counter()
    with Pool(min(workers, len(jobs))) as pool:
        for path, counts in pool.imap_unordered(_edos_file, jobs):
            totals.update(counts)
            print(Path(path).parent.name, counts, flush=True)
    report = {'delta_tasks': len(present), 'owned_tasks_stored': len(wanted), 'status': dict(totals),
              'canonical_tasks': len(canonical), 'canonical_in_local_delta': sum(t in present for t in canonical.values())}
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)


# ---------------------------------------------------------------- phDOS
def build_phdos(out_dir: Path):
    refuse(out_dir)
    index = load_index()
    owner, api = index['task_owner'], index['phonon']
    out_dir.mkdir(parents=True)
    writer = PartWriter(out_dir / 'part-00.parquet', PHDOS_SCHEMA)
    counts = Counter()
    for path, method in delta_files('phonon/electronic-structure/dos', 'phonon_method'):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=['identifier', 'dos']):
            for row in batch.to_pylist():
                task = strip_mp(row['identifier'])
                dos = row['dos'] or {}
                meta = next((m for m in api.get(task, []) if m['method'] == method), {})
                record = {'spectrum_id': f'phdos:mp:{method}:{task}', 'material_id': owner.get(task),
                          'source': 'mp', 'source_task_id': task, 'method': method,
                          'functional': 'unverified', 'x': [], 'y': [], 'n_points': 0,
                          'asr': meta.get('asr'), 'cnsr': meta.get('cnsr'), 'status': '',
                          'calc_lattice': [], 'calc_numbers': [], 'calc_frac': [], 'calc_volume_per_atom': None}
                try:
                    if dos.get('structure'):
                        calc, _ = calc_structure(dos['structure'])
                        record.update(calc)
                    x = np.asarray(dos.get('frequencies') or [], dtype=np.float64)
                    y = np.asarray(dos.get('densities') or [], dtype=np.float64)
                    if len(x) < 10 or len(x) != len(y):
                        raise ValueError('fewer than 10 points or length mismatch')
                    record.update(x=(x * THZ2CM).tolist(), y=(y / THZ2CM).tolist(), n_points=int(len(x)))
                    record['status'] = 'ok' if record['material_id'] else 'no_material'
                except Exception as exc:
                    record['status'] = f'unreadable:{type(exc).__name__}'
                counts[(method, record['status'].split(':')[0])] += 1
                writer.add(record)
    writer.close()
    report = {'status': {f'{m}/{s}': n for (m, s), n in counts.items()}}
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)


# ---------------------------------------------------------------- materials
def _standardize(job):
    material_id, structure = job
    try:
        import torch
        from pymatgen.core import Lattice, Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from utils.relative_features import build_cell_from_lattice
        raw = Structure(Lattice(structure['lattice']['matrix']), [s['label'] for s in structure['sites']],
                        [s['abc'] for s in structure['sites']])
        prim = SpacegroupAnalyzer(raw, symprec=SYMPREC).get_primitive_standard_structure()
        a, b, c = prim.lattice.abc
        params = np.asarray([[a, b, 1 / c], prim.lattice.angles], dtype=np.float32)
        cell, _ = build_cell_from_lattice(torch.from_numpy(params[None]).float())
        numbers = np.asarray(prim.atomic_numbers, dtype=np.uint8)
        valence = float(sum(ZVAL[str(site.specie.symbol)]['zval'] for site in prim.sites))
        return {'material_id': material_id, 'numbers': numbers.tolist(),
                'frac': np.asarray(prim.frac_coords, dtype=np.float32).reshape(-1).tolist(),
                'cell': cell.numpy()[0].reshape(-1).tolist(), 'n_atoms': int(len(numbers)),
                'nvalence': valence, 'nvalence_method': 'element_table', 'structure_status': 'ok'}
    except Exception as exc:
        return {'material_id': material_id, 'structure_status': f'failed:{type(exc).__name__}'}


def legacy_arrays():
    rows = {}
    for split in SPLITS_LEGACY:
        d = LEGACY / split
        ids = np.load(d / f'{split}_index.npy', allow_pickle=False).astype(str)
        numbers = np.load(d / f'atomic_numbers_{split}.npy')
        frac = np.load(d / f'frac_coords_{split}.npy')
        cell = np.load(d / f'cell_{split}.npy')
        offsets = np.load(d / f'atom_offsets_{split}.npy')
        nval = np.load(d / f'nvalence_{split}.npy')
        for i, material_id in enumerate(ids):
            sl = slice(offsets[i], offsets[i + 1])
            rows[material_id] = {'material_id': material_id, 'legacy_split': split, 'legacy_row': i,
                                 'numbers': numbers[sl].astype(int).tolist(),
                                 'frac': frac[sl].reshape(-1).astype(float).tolist(),
                                 'cell': cell[i].reshape(-1).astype(float).tolist(),
                                 'n_atoms': int(offsets[i + 1] - offsets[i]), 'nvalence': float(nval[i]),
                                 'nvalence_method': 'legacy_frozen', 'structure_status': 'legacy_frozen'}
    return rows


def read_parts(directory: Path, columns=None):
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in sorted(directory.glob('part-*.parquet'))])


def spectrum_materials(path: Path):
    # Every material owning a readable spectrum (canonical or not) gets an input structure,
    # so the eDOS task rule can be chosen per view without rebuilding this stage.
    rows = read_parts(path, ['material_id', 'status']).to_pylist()
    return {r['material_id'] for r in rows if r['material_id'] and r['status'] in ('ok', 'truncated')}


def build_materials(out: Path, workers: int):
    refuse(out)
    meta = summary_meta()
    legacy = legacy_arrays()
    with_spectra = spectrum_materials(STAGING / 'edos') | spectrum_materials(STAGING / 'phdos')
    wanted = (with_spectra - set(legacy)) & set(meta)
    structures = {m: meta[m]['structure'] for m in wanted}
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_standardize, sorted(structures.items()), chunksize=64)):
            rows.append(row)
            if (i + 1) % 10000 == 0:
                print('standardized', i + 1, flush=True)
    rows.extend(legacy.values())
    status = Counter()
    for row in rows:
        info = meta.get(row['material_id'], {})
        row.update({'deprecated': info.get('deprecated'), 'license': info.get('license'),
                    'database_version': info.get('database_version'), 'formula': info.get('formula'),
                    'in_current_mp': bool(info), 'legacy_split': row.get('legacy_split'),
                    'legacy_row': row.get('legacy_row')})
        if row.get('numbers'):
            row['composition'] = composition_key(row['numbers'])
            row['reduced_composition'] = reduced_key(row['composition'])
        status[row['structure_status'].split(':')[0]] += 1
    pq.write_table(pa.Table.from_pylist(rows), out, compression='zstd')
    print({'materials': len(rows), 'legacy': len(legacy), 'new': len(rows) - len(legacy), **status})


# ---------------------------------------------------------------- legacy labels
def build_legacy(out: Path):
    refuse(out)
    matched = {}
    for task, name in (('edos', 'geometry'), ('phdos', 'geometry_phdos')):
        import csv
        with open(AUDIT / name / 'geometry_by_material.csv') as stream:
            for row in csv.DictReader(stream):
                if row['task'] == task and row['gate_class'] in ('exact', 'clipped'):
                    matched[(task, row['material_id'])] = (row['dos_task'], row['run_type'], row['gate_class'])
    rows = []
    for split in SPLITS_LEGACY:
        d = LEGACY / split
        ids = np.load(d / f'{split}_index.npy', allow_pickle=False).astype(str)
        for task, bins in (('edos', 'E0'), ('phdos', 'P0')):
            values = np.load(d / f'{task}_tgtdos_{split}.npy')
            masks = np.load(d / f'{task}_mask_{split}.npy')
            for i, material_id in enumerate(ids):
                source = matched.get((task, material_id))
                rows.append({'spectrum_id': f'{task}:legacy:{material_id}', 'type': task, 'material_id': material_id,
                             'representation': 'legacy_binned', 'grid': bins, 'legacy_split': split,
                             'values': values[i].astype(float).tolist(), 'mask': masks[i].astype(bool).tolist(),
                             'matched_task': source[0] if source else None,
                             'matched_method': source[1] if source else None,
                             'gate_class': source[2] if source else 'unmatched'})
    pq.write_table(pa.Table.from_pylist(rows), out, compression='zstd')
    print({'legacy_spectra': len(rows), 'matched': sum(r['matched_task'] is not None for r in rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['index', 'edos', 'phdos', 'materials', 'legacy', 'all'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    stages = ['index', 'edos', 'phdos', 'materials', 'legacy'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        print('== stage', stage, flush=True)
        if stage == 'index':
            build_index(STAGING / 'index.json')
        elif stage == 'edos':
            build_edos(STAGING / 'edos', args.workers)
        elif stage == 'phdos':
            build_phdos(STAGING / 'phdos')
        elif stage == 'materials':
            build_materials(STAGING / 'materials.parquet', args.workers)
        elif stage == 'legacy':
            build_legacy(STAGING / 'legacy_spectra.parquet')


if __name__ == '__main__':
    main()
