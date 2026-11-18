"""Build the dos_v1 dataset from spectral_db_v1.

Rules confirmed 2026-11-15. The database is not modified. Training arrays go to
data/datasets/dos_v1/. Test arrays go under sealed/ so a normal training directory
does not contain them.

Stages, each reused when its cache already exists:

- materials:  standard primitive cells, symprec 0.1
- records:    canonical eDOS, MP phonons, PhononDB (no JARVIS)
- match:      attach records to MP materials, or mark them unmatched
- assemble:   one sample per MP material or per matched group of unmatched records
- views:      exact-bin arrays, provenance, train normalizer

Matching keeps the previous StructureMatcher parameters and does not reduce cells
to primitive first. Hint pairs with a volume ratio within 10% that still fail the
matcher are counted. More than 500 stops the build before any training array is
written (--accept-near-miss overrides that stop).

PhononDB DOS is normalized by phonopy to the primitive cell. The stored label is
scaled by the input cell atom count divided by that primitive count. Structure
matching still uses the PhononDB unit cell. A PhononDB archive without a primitive
cell is skipped. MP phonon and eDOS keep the calculation-cell atom count.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from common import ROOT, exact_bin, file_sha256
from make_view_v2 import W_MAX, edos_window_ratio
from splits_v3 import RARE_ELEMENTS, SALT, assign

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DB = ROOT / 'data/spectral_db_v1'
OUT = ROOT / 'data/datasets/dos_v1'
CACHE = OUT / 'cache'
ZVAL = json.loads((ROOT / 'data/reference/valence_by_element.json').read_text())
SYMPREC = 0.1
VOLUME_LIMIT = 0.10
NEAR_MISS_STOP = 500
DISPUTED_E = 0.5
MAX_ATOMS = 160
RANGES = {'edos': (-6.0, 6.0), 'phdos': (-280.0, 980.0)}
VIEWS = {'dos_v1': {'edos': 128, 'phdos': 64}, 'dos_v1_x2': {'edos': 256, 'phdos': 128}}
IMAG_LIMIT = -20.0
SOURCE_RANK = {'mp_pheasy': 0, 'mp_dfpt': 1, 'phonondb': 2}
SOURCE_CODES = {'none': 0, 'mp': 2, 'mp_dfpt': 4, 'phonondb': 5}
LINK_CODES = {'none': 0, 'hint': 1, 'rehung': 3, 'own': 4}
DEFAULT = dict(ltol=.2, stol=.3, angle_tol=5, scale=True)
TIGHT = dict(ltol=.15, stol=.225, angle_tol=3.75, scale=True)


def grid(task, bins):
    lo, hi = RANGES[task]
    return np.linspace(lo, hi, bins + 1)


def element_symbol(label: str) -> str:
    return str(label).split(':', 1)[0]


def valence_of(symbols) -> float:
    return float(sum(ZVAL[element_symbol(s)]['zval'] for s in symbols))


def volume_per_atom(lattice, n: int) -> float:
    cell = np.asarray(lattice, np.float64).reshape(3, 3)
    return abs(float(np.linalg.det(cell))) / max(int(n), 1)


def mode_distance(area_atom) -> float:
    if area_atom is None or not np.isfinite(area_atom):
        return 1e9
    return abs(float(area_atom) - 3.0)


def imag_fraction(x, y) -> float:
    """Share of absolute DOS area below -20 cm^-1."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return 0.0
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    absolute = float(np.trapz(np.abs(y), x))
    if absolute <= 0:
        return 0.0
    below = x < IMAG_LIMIT
    if below.sum() < 2 and not (below.any() and (~below).any()):
        neg = float(np.trapz(np.abs(y[below]), x[below])) if below.sum() >= 2 else 0.0
    else:
        # Integrate |y| on (-inf, -20], including the crossing segment.
        y_abs = np.abs(y)
        neg = 0.0
        for i in range(len(x) - 1):
            x0, x1, y0, y1 = x[i], x[i + 1], y_abs[i], y_abs[i + 1]
            if x1 <= IMAG_LIMIT:
                neg += (y0 + y1) * (x1 - x0) / 2
            elif x0 < IMAG_LIMIT < x1 and x1 != x0:
                t = (IMAG_LIMIT - x0) / (x1 - x0)
                y_cut = y0 + t * (y1 - y0)
                neg += (y0 + y_cut) * (IMAG_LIMIT - x0) / 2
    return neg / absolute


def pick_representative(rows: list) -> dict:
    """The training record of a merged group: source order, then mode count nearer 3, then id."""
    return min(rows, key=lambda r: (SOURCE_RANK.get(r['kind'], 9), mode_distance(r.get('area_atom')), r['record_id']))


def cluster_unmatched(rows: list, fits) -> list:
    """Greedy cliques. A record joins a group only when it matches every member."""
    groups = []
    for row in sorted(rows, key=lambda r: (r['formula'], SOURCE_RANK.get(r['kind'], 9), r['record_id'])):
        placed = False
        for group in groups:
            if group[0]['formula'] != row['formula']:
                continue
            if all(row['cs'] == other['cs'] and fits(other, row) for other in group):
                group.append(row)
                placed = True
                break
        if not placed:
            groups.append([row])
    return groups


def attach_record(record, materials, by_formula, fit_default, fit_tight, limit=VOLUME_LIMIT):
    """Return attachment dict. `fit(material, record)` compares opaque structure fields."""
    hint_id = record.get('hint')
    hint = materials.get(hint_id)
    near_miss = False
    if hint is not None and hint['formula'] == record['formula']:
        ratio = record['volume'] / hint['volume']
        matched = bool(fit_default(hint, record))
        if matched and abs(ratio - 1) <= limit:
            return {'material_id': hint_id, 'reason': 'hint', 'volume_ratio': float(ratio),
                    'near_miss': False, 'label_ok': True}
        if (not matched) and abs(ratio - 1) <= limit:
            near_miss = True
    hits = []
    for material_id in by_formula.get(record['formula'], ()):
        if material_id == hint_id:
            continue
        material = materials[material_id]
        if material['cs'] != record['cs']:
            continue
        ratio = record['volume'] / material['volume']
        if abs(ratio - 1) > limit:
            continue
        if fit_default(material, record):
            hits.append((material_id, float(ratio)))
    if len(hits) == 1:
        material_id, ratio = hits[0]
        return {'material_id': material_id, 'reason': 'rehung', 'volume_ratio': ratio,
                'near_miss': near_miss, 'label_ok': True}
    if len(hits) > 1:
        tight_hits = [(material_id, ratio) for material_id, ratio in hits if fit_tight(materials[material_id], record)]
        if len(tight_hits) == 1:
            material_id, ratio = tight_hits[0]
            return {'material_id': material_id, 'reason': 'tight_unique', 'volume_ratio': ratio,
                    'near_miss': near_miss, 'label_ok': False}
    return {'material_id': None, 'reason': 'none', 'volume_ratio': None,
            'near_miss': near_miss, 'label_ok': False}


# ---------------------------------------------------------------- structures
def _standardize(lattice, species, frac):
    import torch
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from utils.relative_features import build_cell_from_lattice
    symbols = [element_symbol(s) for s in species]
    raw = Structure(Lattice(np.asarray(lattice, np.float64).reshape(3, 3)), symbols,
                    np.asarray(frac, np.float64).reshape(-1, 3))
    analyzer = SpacegroupAnalyzer(raw, symprec=SYMPREC)
    crystal = analyzer.get_crystal_system()
    prim = analyzer.get_primitive_standard_structure()
    a, b, c = prim.lattice.abc
    params = np.asarray([[a, b, 1 / c], prim.lattice.angles], dtype=np.float32)
    cell, _ = build_cell_from_lattice(torch.from_numpy(params[None]).float())
    numbers = [int(z) for z in prim.atomic_numbers]
    from pymatgen.core import Element
    elements = [Element.from_Z(z).symbol for z in numbers]
    formula = __import__('pymatgen.core', fromlist=['Composition']).Composition(
        {e: elements.count(e) for e in set(elements)}).reduced_formula
    lattice_out = cell.numpy()[0].reshape(-1).astype(np.float64)
    frac_out = np.asarray(prim.frac_coords, np.float32).reshape(-1)
    return {'lattice': lattice_out.tolist(), 'species': elements, 'frac': frac_out.tolist(),
            'n': len(numbers), 'nvalence': valence_of(elements), 'formula': formula,
            'elements': sorted(set(elements)), 'cs': crystal,
            'volume': volume_per_atom(lattice_out, len(numbers))}


def _standardize_material(job):
    material_id, lattice, species, frac = job
    try:
        row = _standardize(lattice, species, frac)
        row.update(material_id=material_id, status='ok')
    except Exception as exc:
        row = {'material_id': material_id, 'status': f'failed:{type(exc).__name__}'}
    return row


def build_materials(workers: int):
    path = CACHE / 'materials.parquet'
    if path.exists():
        print('materials cache', path, flush=True)
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(DB / 'materials.parquet', columns=['material_id', 'lattice', 'site_elements', 'site_frac'])
    jobs = [(m, lat, el, fr) for m, lat, el, fr in zip(
        table.column('material_id').to_pylist(), table.column('lattice').to_pylist(),
        table.column('site_elements').to_pylist(), table.column('site_frac').to_pylist())]
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_standardize_material, jobs, chunksize=16)):
            rows.append(row)
            if (i + 1) % 20000 == 0:
                print('standardized', i + 1, flush=True)
    ok = []
    for row in rows:
        if row['status'] != 'ok':
            continue
        row['too_big'] = row['n'] > MAX_ATOMS
        ok.append(row)
    dropped = Counter(r['status'] for r in rows)
    dropped['over_160'] = sum(r['too_big'] for r in ok)
    pq.write_table(pa.Table.from_pylist(ok), path, compression='zstd')
    (CACHE / 'materials_report.json').write_text(json.dumps({'standardized': len(ok), **dropped}, indent=2) + '\n')
    print('materials standardized', len(ok), dict(dropped), flush=True)


def _area_atom(x, y, n):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2 or n <= 0:
        return None
    order = np.argsort(x[keep], kind='stable')
    return float(np.trapz(y[keep][order], x[keep][order]) / n)


def _record_from_calc(record_id, kind, hint, lattice, numbers, frac, x=None, y=None, dos_n=None):
    from pymatgen.core import Element, Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    symbols = [Element.from_Z(int(z)).symbol for z in numbers]
    structure = Structure(Lattice(np.asarray(lattice, np.float64).reshape(3, 3)), symbols,
                          np.asarray(frac, np.float64).reshape(-1, 3))
    formula = __import__('pymatgen.core', fromlist=['Composition']).Composition(
        {e: symbols.count(e) for e in set(symbols)}).reduced_formula
    crystal = SpacegroupAnalyzer(structure, symprec=SYMPREC).get_crystal_system()
    n = len(symbols)
    # dos_n is the cell the spectrum integrates over. PhononDB passes the primitive
    # count; MP and eDOS leave it empty and use this calculation cell.
    norm = n if dos_n is None else int(dos_n)
    if norm <= 0:
        raise ValueError('dos normalization count must be positive')
    return {'record_id': record_id, 'kind': kind, 'hint': hint, 'formula': formula,
            'cs': crystal, 'n': n, 'dos_n': norm, 'volume': volume_per_atom(lattice, n),
            'lattice': [float(v) for v in np.asarray(lattice, np.float64).reshape(-1)],
            'species': symbols, 'frac': [float(v) for v in np.asarray(frac, np.float64).reshape(-1)],
            'nvalence': valence_of(symbols), 'area_atom': None if x is None else _area_atom(x, y, norm)}


def _record_job(job):
    record_id, kind, hint, lattice, numbers, frac, x, y, dos_n = job
    try:
        return _record_from_calc(record_id, kind, hint, lattice, numbers, frac, x, y, dos_n)
    except Exception as exc:
        return {'record_id': record_id, 'status': f'failed:{type(exc).__name__}', 'kind': kind}


def build_records(workers: int):
    path = CACHE / 'records.parquet'
    if path.exists():
        print('records cache', path, flush=True)
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    materials = pq.read_table(DB / 'materials.parquet', columns=['material_id', 'canonical_dos_task'])
    named = {task: material for material, task in zip(materials.column('material_id').to_pylist(),
                                                      materials.column('canonical_dos_task').to_pylist()) if task}
    present = set(materials.column('material_id').to_pylist())
    jobs = []
    columns = ['source_task_id', 'calc_lattice', 'calc_numbers', 'calc_frac', 'parse_error']
    for batch in pq.ParquetFile(DB / 'spectra_edos.parquet').iter_batches(batch_size=4096, columns=columns):
        tasks, lattices, numbers, fracs, errors = (batch.column(i).to_pylist() for i in range(5))
        for task, lattice, number, frac, error in zip(tasks, lattices, numbers, fracs, errors):
            if task in named and not error and number:
                jobs.append((f'edos:mp:{task}', 'mp_edos', named[task], lattice, number, frac, None, None, None))
    phonon = pq.read_table(DB / 'spectra_phdos_mp.parquet',
                           columns=['source_task_id', 'method', 'material_id', 'x', 'y', 'calc_lattice',
                                    'calc_numbers', 'calc_frac', 'parse_error'])
    for record in phonon.to_pylist():
        if record['parse_error'] or not record['calc_numbers']:
            continue
        kind = 'mp_pheasy' if record['method'] == 'pheasy' else 'mp_dfpt'
        jobs.append((f"phdos:mp:{record['method']}:{record['source_task_id']}", kind, record['material_id'],
                     record['calc_lattice'], record['calc_numbers'], record['calc_frac'], record['x'], record['y'], None))
    phonondb = pq.read_table(DB / 'spectra_phdos_phonondb.parquet',
                             columns=['source_id', 'mp_reference', 'x', 'y', 'unit_cell', 'primitive_cell', 'parse_error'])
    from pymatgen.core import Element
    skipped = Counter()
    for record in phonondb.to_pylist():
        if record['parse_error'] or not record['unit_cell']:
            continue
        cell = json.loads(record['unit_cell'])
        primitive = json.loads(record['primitive_cell']) if record.get('primitive_cell') else None
        points = primitive.get('points') if isinstance(primitive, dict) else None
        if not points:
            skipped['phonondb_no_primitive'] += 1
            continue
        numbers = [Element(p['symbol']).Z for p in cell['points']]
        hint = _legacy_mp_id(record['mp_reference'])
        jobs.append((f"phdos:phonondb:{record['source_id']}", 'phonondb', hint if hint in present else None,
                     cell['lattice'], numbers, [p['coordinates'] for p in cell['points']], record['x'], record['y'],
                     len(points)))
    rows, failed = [], Counter()
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_record_job, jobs, chunksize=8)):
            if row.get('status'):
                failed[row['kind']] += 1
            else:
                rows.append(row)
            if (i + 1) % 10000 == 0:
                print('records prepared', i + 1, flush=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')
    print('records', len(rows), dict(failed), 'skipped', dict(skipped), flush=True)


def _legacy_mp_id(reference: str):
    """Numeric PhononDB MP id to the 8-letter id used in spectral_db_v1.

    Same encoding as emmet ``AlphaID(n, prefix='mp')`` with the body padded to 8
    using ``a`` (a = 0 in a base-26 alphabet). Current emmet-core no longer ships
    ``AlphaID``, so the conversion lives here.
    """
    text = str(reference or '')
    if not text.startswith('mp-') or not text[3:].isdigit():
        return None
    return 'mp-' + _integer_to_mp_alpha(int(text[3:]))


def _integer_to_mp_alpha(integer: int) -> str:
    from math import floor, log
    from string import ascii_lowercase
    alphabet = ascii_lowercase
    if integer < 0:
        raise ValueError('MP id cannot be negative')
    if integer == 0:
        body = alphabet[0]
    else:
        base = len(alphabet)
        max_pow = floor(log(integer) / log(base))
        body = ''
        remaining = integer
        for power in range(max_pow, -1, -1):
            if remaining == 0:
                body += alphabet[0]
                continue
            place = base ** power
            for coeff in range(base - 1, -1, -1):
                if coeff * place <= remaining:
                    body += alphabet[coeff]
                    remaining -= coeff * place
                    break
    return body.rjust(8, 'a')


# ---------------------------------------------------------------- match
_MATCH_MATERIALS = {}
_MATCH_BY_FORMULA = {}


def _init_match(materials, by_formula):
    global _MATCH_MATERIALS, _MATCH_BY_FORMULA
    _MATCH_MATERIALS = materials
    _MATCH_BY_FORMULA = by_formula


def _match_batch(records):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure
    default, tight = StructureMatcher(**DEFAULT), StructureMatcher(**TIGHT)

    def as_structure(row):
        return Structure(Lattice(np.asarray(row['lattice'], np.float64).reshape(3, 3)), row['species'],
                         np.asarray(row['frac'], np.float64).reshape(-1, 3))

    cache = {}

    def fit(matcher, material, record):
        key = (id(matcher), material['material_id'], record['record_id'])
        if key not in cache:
            try:
                cache[key] = bool(matcher.fit(as_structure(material), as_structure(record)))
            except Exception:
                cache[key] = False
        return cache[key]

    out = []
    for record in records:
        # Drop bulky geometry before returning; attachment only needs the decision.
        decision = attach_record(
            record, _MATCH_MATERIALS, _MATCH_BY_FORMULA,
            lambda material, rec: fit(default, material, rec),
            lambda material, rec: fit(tight, material, rec))
        decision['record_id'] = record['record_id']
        out.append(decision)
    return out


def build_matches(workers: int):
    path = CACHE / 'matches.parquet'
    partial = CACHE / 'matches_partial.parquet'
    if path.exists():
        print('matches cache', path, flush=True)
        return
    done = {}
    if partial.exists():
        done = {r['record_id']: r for r in pq.read_table(partial).to_pylist()}
        print('resuming matches', len(done), flush=True)
    materials = {r['material_id']: r for r in pq.read_table(CACHE / 'materials.parquet').to_pylist()}
    by_formula = {}
    for material in materials.values():
        by_formula.setdefault(material['formula'], []).append(material['material_id'])
    records = [r for r in pq.read_table(CACHE / 'records.parquet', columns=[
        'record_id', 'kind', 'hint', 'formula', 'cs', 'volume', 'lattice', 'species', 'frac']).to_pylist()
               if r['record_id'] not in done]
    chunks = [records[i:i + 24] for i in range(0, len(records), 24)]
    rows = list(done.values())
    with Pool(workers, initializer=_init_match, initargs=(materials, by_formula)) as pool:
        for i, part in enumerate(pool.imap_unordered(_match_batch, chunks)):
            rows.extend(part)
            if (i + 1) % 200 == 0:
                pq.write_table(pa.Table.from_pylist(rows), partial, compression='zstd')
                print('matched batches', i + 1, 'of', len(chunks), 'saved', len(rows), flush=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')
    if partial.exists():
        partial.unlink()
    print('matches', len(rows), 'near_miss', sum(bool(r['near_miss']) for r in rows), flush=True)


# ---------------------------------------------------------------- samples and views
def _load_spectra(record_ids: set) -> dict:
    wanted_tasks = {rid.split(':')[-1] for rid in record_ids if rid.startswith('edos:mp:')}
    spectra = {}
    if wanted_tasks:
        columns = ['source_task_id', 'x', 'y', 'parse_error']
        for batch in pq.ParquetFile(DB / 'spectra_edos.parquet').iter_batches(batch_size=2048, columns=columns):
            for task, x, y, error in zip(*(batch.column(i).to_pylist() for i in range(4))):
                if task in wanted_tasks and not error:
                    spectra[f'edos:mp:{task}'] = (x, y)
    phonon_ids = {rid for rid in record_ids if rid.startswith('phdos:mp:')}
    if phonon_ids:
        table = pq.read_table(DB / 'spectra_phdos_mp.parquet', columns=['method', 'source_task_id', 'x', 'y', 'parse_error'])
        for record in table.to_pylist():
            rid = f"phdos:mp:{record['method']}:{record['source_task_id']}"
            if rid in phonon_ids and not record['parse_error']:
                spectra[rid] = (record['x'], record['y'])
    pdb_ids = {rid for rid in record_ids if rid.startswith('phdos:phonondb:')}
    if pdb_ids:
        table = pq.read_table(DB / 'spectra_phdos_phonondb.parquet', columns=['source_id', 'x', 'y', 'parse_error'])
        for record in table.to_pylist():
            rid = f"phdos:phonondb:{record['source_id']}"
            if rid in pdb_ids and not record['parse_error']:
                spectra[rid] = (record['x'], record['y'])
    return spectra


def _cluster_formula(rows):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure
    matcher = StructureMatcher(**DEFAULT)

    def fits(left, right):
        try:
            return bool(matcher.fit(
                Structure(Lattice(np.asarray(left['lattice']).reshape(3, 3)), left['species'],
                          np.asarray(left['frac']).reshape(-1, 3)),
                Structure(Lattice(np.asarray(right['lattice']).reshape(3, 3)), right['species'],
                          np.asarray(right['frac']).reshape(-1, 3))))
        except Exception:
            return False

    return [[row['record_id'] for row in group] for group in cluster_unmatched(rows, fits)]


def assemble(accept_near_miss: bool):
    matches = pq.read_table(CACHE / 'matches.parquet').to_pandas().set_index('record_id')
    near = int(matches.near_miss.sum())
    print('near_miss', near, flush=True)
    if near > NEAR_MISS_STOP and not accept_near_miss:
        raise SystemExit(f'{near} hint pairs are within 10% volume but fail the matcher; '
                         f'stopping before writing views. Re-run with --accept-near-miss to keep going.')
    records = {r['record_id']: r for r in pq.read_table(CACHE / 'records.parquet').to_pylist()}
    materials = {r['material_id']: r for r in pq.read_table(CACHE / 'materials.parquet').to_pylist()}
    units = {}
    for material_id, material in materials.items():
        if material.get('too_big'):
            continue
        units[material_id] = {'kind': 'material', 'formula': material['formula'], 'elements': material['elements'],
                              'structure': material, 'candidates': {'edos': [], 'phdos': []}}
    members, exclusions = [], []
    unmatched = []
    for record_id, record in records.items():
        match = matches.loc[record_id]
        if match.reason == 'none' or (isinstance(match.material_id, float) and np.isnan(match.material_id)) or match.material_id is None:
            if record['kind'] == 'mp_edos':
                exclusions.append({'record_id': record_id, 'unit_id': None, 'task': 'edos', 'reason': 'edos_unmatched'})
            else:
                unmatched.append(record)
            continue
        unit = units.get(match.material_id)
        if unit is None:
            reason = 'over_160' if materials.get(match.material_id, {}).get('too_big') else 'material_not_standardized'
            exclusions.append({'record_id': record_id, 'unit_id': match.material_id, 'task': _task(record['kind']),
                               'reason': reason})
            continue
        task = _task(record['kind'])
        # eDOS is only the canonical task of the material that named it. A spectrum that
        # matches some other material is not that material's label.
        own_edos = task == 'edos' and bool(match.label_ok) and match.reason == 'hint' and record.get('hint') == match.material_id
        usable = own_edos or (task == 'phdos' and bool(match.label_ok))
        if not usable:
            if match.reason == 'tight_unique':
                reason = 'tight_unique'
            elif task == 'edos' and match.reason != 'hint':
                reason = 'edos_not_own_canonical'
            else:
                reason = 'not_a_label_source'
            exclusions.append({'record_id': record_id, 'unit_id': match.material_id, 'task': task, 'reason': reason})
            members.append({'unit_id': match.material_id, 'record_id': record_id, 'role': reason})
            continue
        unit['candidates'][task].append(record_id)
        members.append({'unit_id': match.material_id, 'record_id': record_id, 'role': 'candidate',
                        'reason': match.reason, 'volume_ratio': None if match.volume_ratio is None else float(match.volume_ratio)})
    buckets = {}
    for record in unmatched:
        buckets.setdefault(record['formula'], []).append(record)
    print('unmatched', len(unmatched), 'formulas', len(buckets), flush=True)
    grouped = []
    with Pool(max(1, min(6, len(buckets) or 1))) as pool:
        for part in pool.imap_unordered(_cluster_formula, list(buckets.values()), chunksize=4):
            grouped.extend(part)
    for group_ids in grouped:
        group = [records[rid] for rid in group_ids]
        chosen = pick_representative(group)
        try:
            structure = _standardize(chosen['lattice'], chosen['species'], chosen['frac'])
        except Exception as exc:
            for record in group:
                exclusions.append({'record_id': record['record_id'], 'unit_id': chosen['record_id'],
                                   'task': 'phdos', 'reason': f'standardize_failed:{type(exc).__name__}'})
            continue
        if structure['n'] > MAX_ATOMS:
            for record in group:
                exclusions.append({'record_id': record['record_id'], 'unit_id': chosen['record_id'],
                                   'task': 'phdos', 'reason': 'over_160'})
            continue
        own_ratio = chosen['volume'] / structure['volume']
        if abs(own_ratio - 1) > VOLUME_LIMIT:
            for record in group:
                exclusions.append({'record_id': record['record_id'], 'unit_id': chosen['record_id'],
                                   'task': 'phdos', 'reason': 'volume'})
            continue
        unit_id = chosen['record_id']
        units[unit_id] = {'kind': 'entry', 'formula': structure['formula'], 'elements': structure['elements'],
                          'structure': structure, 'own_volume_ratio': float(own_ratio),
                          'candidates': {'edos': [], 'phdos': [chosen['record_id']]}}
        for record in group:
            role = 'chosen' if record['record_id'] == chosen['record_id'] else 'merged_out'
            members.append({'unit_id': unit_id, 'record_id': record['record_id'], 'role': role, 'reason': 'group'})
            if role == 'merged_out':
                exclusions.append({'record_id': record['record_id'], 'unit_id': unit_id, 'task': 'phdos',
                                   'reason': 'merged_not_selected'})
    # Choose one label per task. eDOS must be the canonical record and pass W.
    spectra_needed = set()
    for unit in units.values():
        for task, ids in unit['candidates'].items():
            spectra_needed.update(ids)
    spectra = _load_spectra(spectra_needed)
    labels = {}
    for unit_id, unit in list(units.items()):
        labels[unit_id] = {}
        for task, ids in unit['candidates'].items():
            ranked = sorted((records[rid] for rid in ids if rid in records),
                            key=lambda r: (SOURCE_RANK.get(r['kind'], 0 if r['kind'] == 'mp_edos' else 9),
                                           mode_distance(r.get('area_atom')), r['record_id']))
            chosen = None
            for record in ranked:
                if task == 'edos':
                    xy = spectra.get(record['record_id'])
                    if xy is None or edos_window_ratio(xy[0], xy[1], record['nvalence']) > W_MAX:
                        exclusions.append({'record_id': record['record_id'], 'unit_id': unit_id, 'task': task,
                                           'reason': 'anomalous_W' if xy is not None else 'spectrum_missing'})
                        continue
                if record['record_id'] not in spectra:
                    exclusions.append({'record_id': record['record_id'], 'unit_id': unit_id, 'task': task,
                                       'reason': 'spectrum_missing'})
                    continue
                chosen = record
                break
            if chosen is None:
                continue
            for record in ranked:
                if record['record_id'] != chosen['record_id']:
                    exclusions.append({'record_id': record['record_id'], 'unit_id': unit_id, 'task': task,
                                       'reason': 'lower_priority'})
            link = 'own' if unit['kind'] == 'entry' else str(matches.loc[chosen['record_id']].reason)
            ratio = unit.get('own_volume_ratio', matches.loc[chosen['record_id']].volume_ratio)
            labels[unit_id][task] = {'record': chosen, 'link': link, 'volume_ratio': _finite(ratio),
                                     'others': [r['record_id'] for r in ranked if r['record_id'] != chosen['record_id']]}
        if not labels[unit_id]:
            del units[unit_id]
            del labels[unit_id]
    kept = {uid: units[uid] for uid, unit in units.items() if labels.get(uid)}
    return kept, labels, members, exclusions, spectra, near


def _task(kind: str) -> str:
    return 'edos' if kind == 'mp_edos' else 'phdos'


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _bin_label(record, unit, spectra, task, bins):
    x, y = spectra[record['record_id']]
    y = np.asarray(y, np.float64) * (unit['structure']['n'] / record['dos_n'])
    values, coverage = exact_bin(x, y, grid(task, bins))
    values = np.maximum(values, 0)
    return values, coverage >= 1 - 1e-6


def write_views(units, labels, members, exclusions, spectra, near_miss: int):
    from utils.spectral_balance import fit_task_normalizer
    order = {'train': [], 'valid': [], 'test': []}
    for unit_id, unit in units.items():
        order[assign(unit['formula'], set(unit['elements']))].append(unit_id)
    for ids in order.values():
        ids.sort()
    # Disputed flags use the main-grid per-atom energy and the train normalizer.
    main_bins = VIEWS['dos_v1']
    per_atom = {}
    record_n = {r['record_id']: r['dos_n'] for r in pq.read_table(
        CACHE / 'records.parquet', columns=['record_id', 'dos_n']).to_pylist()}
    for unit_id, lab in labels.items():
        for task, item in lab.items():
            values, _ = _bin_label(item['record'], units[unit_id], spectra, task, main_bins[task])
            per_atom[(unit_id, task, item['record']['record_id'])] = values / units[unit_id]['structure']['n']
    other_ids = {other for lab in labels.values() for item in lab.values() for other in item['others']}
    other_spectra = _load_spectra(other_ids - set(spectra))
    spectra.update(other_spectra)
    for unit_id, lab in labels.items():
        for task, item in lab.items():
            for other in item['others']:
                if other not in spectra or other not in record_n:
                    continue
                x, y = spectra[other]
                values, _ = exact_bin(x, y, grid(task, main_bins[task]))
                per_atom[(unit_id, task, other)] = np.maximum(values, 0) / record_n[other]
    train_energy = {}
    for task, bins in main_bins.items():
        rows, atoms = [], []
        for unit_id in order['train']:
            if task not in labels[unit_id]:
                continue
            values, _ = _bin_label(labels[unit_id][task]['record'], units[unit_id], spectra, task, bins)
            rows.append(values)
            atoms.append(units[unit_id]['structure']['n'])
        train_energy[task] = fit_task_normalizer(np.asarray(rows, np.float64), np.asarray(atoms, np.float64),
                                                 (RANGES[task][1] - RANGES[task][0]) / bins)['energy']
    disputed = {}
    for unit_id, lab in labels.items():
        for task, item in lab.items():
            base = per_atom[(unit_id, task, item['record']['record_id'])]
            worst = 0.0
            dx = (RANGES[task][1] - RANGES[task][0]) / main_bins[task]
            for other in item['others']:
                key = (unit_id, task, other)
                if key not in per_atom:
                    continue
                worst = max(worst, float(dx * np.square(base - per_atom[key]).sum() / train_energy[task]))
            disputed[(unit_id, task)] = worst > DISPUTED_E
    for name, bins in VIEWS.items():
        _write_one_view(name, bins, units, labels, spectra, order, disputed, train_energy if name == 'dos_v1' else None)
    _write_tables(units, labels, members, exclusions, order, near_miss, disputed)


def _write_one_view(name, bins, units, labels, spectra, order, disputed, train_energy_unused):
    from utils.spectral_balance import fit_task_normalizer
    manifest = {'schema': 'crystal-q1-v1', 'cell_convention': 'row lattice vectors, angstrom; legacy canonical frame',
                'frac_convention': 'dimensionless, same primitive basis and atom order',
                'primitive_policy': {'source': 'MP summary structure', 'symprec': SYMPREC},
                'geometry_precision': 'float32', 'label_contract': 'v2', 'binning': 'exact_piecewise_linear',
                'partial_labels': True, 'dataset': 'dos_v1', 'database': 'spectral_db_v1',
                'dos_norm_atoms': {'mp': 'calculation cell', 'phonondb': 'primitive cell',
                                   'structure_match': 'phonondb unit cell; mp calculation cell'},
                'grid': {t: {'range': list(RANGES[t]), 'bins': b} for t, b in bins.items()},
                'source_codes': SOURCE_CODES, 'link_codes': LINK_CODES,
                'split_rule': {'salt': SALT, 'rare_elements': sorted(RARE_ELEMENTS)},
                'imaginary_modes': {'definition': 'share of absolute DOS area below -20 cm^-1', 'flag_above': 0.05},
                'splits': {}, 'files': {}}
    sealed_manifest = {'dataset': name, 'splits': {}, 'files': {}}
    built = {}
    for split, ids in order.items():
        n = len(ids)
        arrays = {}
        counts = np.asarray([units[u]['structure']['n'] for u in ids], np.int32)
        arrays['cell'] = np.asarray([np.asarray(units[u]['structure']['lattice'], np.float32).reshape(3, 3) for u in ids])
        arrays['atomic_numbers'] = np.concatenate([np.asarray([_z(s) for s in units[u]['structure']['species']], np.uint8) for u in ids]) if n else np.zeros(0, np.uint8)
        arrays['frac_coords'] = np.concatenate([np.asarray(units[u]['structure']['frac'], np.float32).reshape(-1, 3) for u in ids]) if n else np.zeros((0, 3), np.float32)
        arrays['n_atoms'] = counts
        arrays['atom_offsets'] = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        arrays['nvalence'] = np.asarray([units[u]['structure']['nvalence'] for u in ids], np.float32)
        arrays['is_entry'] = np.asarray([units[u]['kind'] == 'entry' for u in ids], bool)
        arrays['index'] = np.asarray(ids)
        imag = np.full(n, np.nan, np.float32)
        imag_flag = np.zeros(n, bool)
        for task, width in bins.items():
            values = np.zeros((n, width), np.float32)
            masks = np.zeros((n, width), bool)
            has = np.zeros(n, bool)
            source = np.zeros(n, np.int8)
            dis = np.zeros(n, bool)
            link = np.zeros(n, np.int8)
            for i, unit_id in enumerate(ids):
                item = labels[unit_id].get(task)
                if item is None:
                    continue
                record = item['record']
                binned, mask = _bin_label(record, units[unit_id], spectra, task, width)
                values[i], masks[i], has[i] = binned, mask, True
                source[i] = SOURCE_CODES['mp'] if record['kind'] in ('mp_edos', 'mp_pheasy') else SOURCE_CODES.get(record['kind'], 0)
                dis[i] = disputed[(unit_id, task)]
                link[i] = LINK_CODES.get(item['link'], 0)
                if task == 'phdos':
                    x, y = spectra[record['record_id']]
                    frac = imag_fraction(x, y)
                    imag[i] = frac
                    imag_flag[i] = frac > 0.05
            arrays[f'{task}_tgtdos'] = values
            arrays[f'{task}_mask'] = masks
            arrays[f'has_{task}'] = has
            arrays[f'source_{task}'] = source
            arrays[f'disputed_{task}'] = dis
            arrays[f'link_{task}'] = link
        arrays['imag_frac_phdos'] = imag
        arrays['imag_flag_phdos'] = imag_flag
        built[split] = arrays
        manifest['splits'][split] = {'n': n, 'has_edos': int(arrays['has_edos'].sum()),
                                     'has_phdos': int(arrays['has_phdos'].sum()),
                                     'imag_flag_phdos': int(imag_flag.sum())}
    for task, width in bins.items():
        values, has, atoms = built['train'][f'{task}_tgtdos'], built['train'][f'has_{task}'], built['train']['n_atoms']
        manifest.setdefault('normalizer_train', {})[task] = fit_task_normalizer(
            values[has].astype(np.float64), atoms[has].astype(np.float64), (RANGES[task][1] - RANGES[task][0]) / width)
    for split, arrays in built.items():
        root = OUT / 'sealed' / 'views' / name if split == 'test' else OUT / 'views' / name
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        target = sealed_manifest if split == 'test' else manifest
        for key, array in arrays.items():
            filename = f'{split}_index.npy' if key == 'index' else f'{key}_{split}.npy'
            np.save(directory / filename, array, allow_pickle=False)
            target['files'][f'{split}/{filename}'] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                                      'sha256': file_sha256(directory / filename)}
        if split == 'test':
            sealed_manifest['splits'][split] = manifest['splits'][split]
    (OUT / 'views' / name / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'sealed' / 'views' / name / 'manifest.json').write_text(json.dumps(sealed_manifest, indent=2) + '\n')
    print(name, {s: manifest['splits'][s] for s in manifest['splits']}, flush=True)


def _z(symbol: str) -> int:
    from pymatgen.core import Element
    return Element(element_symbol(symbol)).Z


def _write_tables(units, labels, members, exclusions, order, near_miss, disputed):
    import pandas as pd
    label_rows = []
    for split, ids in order.items():
        for unit_id in ids:
            for task, item in labels[unit_id].items():
                label_rows.append({'unit_id': unit_id, 'task': task, 'set': split, 'origin': item['record']['record_id'],
                                   'kind': item['record']['kind'], 'link': item['link'],
                                   'volume_ratio': item['volume_ratio'], 'disputed': disputed[(unit_id, task)],
                                   'is_entry': units[unit_id]['kind'] == 'entry'})
    public = OUT / 'tables'
    public.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(label_rows).to_parquet(public / 'labels.parquet', index=False)
    pd.DataFrame(exclusions).to_parquet(public / 'exclusions.parquet', index=False)
    pd.DataFrame(members).to_parquet(public / 'members.parquet', index=False)
    report = {'near_miss_volume_match_fail': near_miss,
              'splits': {s: len(ids) for s, ids in order.items()},
              'exclusions': dict(Counter(r['reason'] for r in exclusions)),
              'labels': {f'{task}/{kind}': count for (task, kind), count in Counter(
                  (r['task'], r['kind']) for r in label_rows).items()}}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['materials', 'records', 'match', 'assemble', 'all'])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--accept-near-miss', action='store_true')
    args = parser.parse_args()
    stages = ['materials', 'records', 'match', 'assemble'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        print('==', stage, flush=True)
        if stage == 'materials':
            build_materials(args.workers)
        elif stage == 'records':
            build_records(args.workers)
        elif stage == 'match':
            build_matches(args.workers)
        elif stage == 'assemble':
            units, labels, members, exclusions, spectra, near = assemble(args.accept_near_miss)
            write_views(units, labels, members, exclusions, spectra, near)


if __name__ == '__main__':
    main()
