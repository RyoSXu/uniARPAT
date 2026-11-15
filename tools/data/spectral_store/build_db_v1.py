"""Build spectral_db_v1: source tables and provenance, no dataset decisions.

The database keeps what the sources say and the minimum needed to read a spectrum.
It does not standardize structures, choose labels, score quality, match records or
split materials. Those belong to a dataset built from this database.

Outputs (new directory only; data/spectral_store_v1 is never modified):

    data/spectral_db_v1/materials.parquet
    data/spectral_db_v1/spectra_edos.parquet
    data/spectral_db_v1/spectra_phdos_mp.parquet
    data/spectral_db_v1/spectra_phdos_phonondb.parquet
    data/spectral_db_v1/spectra_phdos_jarvis.parquet
    data/spectral_db_v1/manifest.json
    data/spectral_db_v1/build_stats.json
    data/spectral_db_v1/README.md

Stages refuse to overwrite an existing output unless --force is given.
`verify` compares the new tables with the staging tables they replace.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from build_store import PartWriter, calc_structure, delta_files, edos_record
from common import STORE, THZ2CM, file_sha256, load_pages, strip_mp

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/spectral_db_v1'
DELTA = ROOT / 'tools/data/raw/delta_mp'
RAW_MP = STORE / 'raw_mp'
RAW_CORE = STORE / 'raw_mp_core'
SCHEMA = 'spectral-db-v1'

SNAPSHOT_SUMMARY = 'mp-api-0.88.1/summary/2026-11-10'
SNAPSHOT_ELECTRONIC = 'mp-api-0.88.1/electronic_structure/2026-11-10'
SNAPSHOT_CORE = 'mp-api-0.88.1/core/2026-11-10'
SNAPSHOT_PHONON_API = 'mp-api-0.88.1/phonon/2026-11-10'
SNAPSHOT_DELTA_EDOS = 'mp-delta/total-dos/2026-11-05'
SNAPSHOT_DELTA_PHONON = 'mp-delta/phonon-dos/2026-11-05'
SNAPSHOT_PHONONDB = 'phonondb-mdr/recomputed'
SNAPSHOT_JARVIS = 'jarvis-dft/pages+jdft_3d-12-12-2022'

_STRUCTURE = [
    ('calc_lattice', pa.list_(pa.float64())),
    ('calc_numbers', pa.list_(pa.int16())),
    ('calc_elements', pa.list_(pa.string())),
    ('calc_frac', pa.list_(pa.float64())),
    ('calc_magmom', pa.list_(pa.float64())),
    ('calc_volume_per_atom', pa.float64()),
]
EDOS_SCHEMA = pa.schema([
    ('spectrum_id', pa.string()), ('material_id', pa.string()), ('source', pa.string()),
    ('source_task_id', pa.string()), ('run_type', pa.string()), ('calc_type', pa.string()),
    ('is_canonical', pa.bool_()), ('efermi', pa.float64()), ('spin_channels', pa.int8()),
    ('x', pa.list_(pa.float64())), ('y', pa.list_(pa.float64())), ('n_points', pa.int32()),
    *_STRUCTURE, ('parse_error', pa.string()), ('snapshot', pa.string()),
])
PHDOS_MP_SCHEMA = pa.schema([
    ('spectrum_id', pa.string()), ('material_id', pa.string()), ('source', pa.string()),
    ('source_task_id', pa.string()), ('method', pa.string()),
    ('asr', pa.float64()), ('cnsr', pa.float64()),
    ('x', pa.list_(pa.float64())), ('y', pa.list_(pa.float64())), ('n_points', pa.int32()),
    *_STRUCTURE, ('parse_error', pa.string()), ('snapshot', pa.string()),
])
MATERIALS_SCHEMA = pa.schema([
    ('material_id', pa.string()), ('formula', pa.string()), ('nsites_mp', pa.int32()),
    ('deprecated', pa.bool_()), ('license', pa.string()), ('database_version', pa.string()),
    ('emmet_version', pa.string()), ('pymatgen_version', pa.string()), ('build_date', pa.string()),
    ('canonical_dos_task', pa.string()),
    ('lattice', pa.list_(pa.float64())), ('site_elements', pa.list_(pa.string())),
    ('site_frac', pa.list_(pa.float64())), ('site_magmom', pa.list_(pa.float64())),
    ('n_sites', pa.int32()), ('source', pa.string()), ('snapshot', pa.string()),
])


def refuse(path: Path, force: bool):
    if path.exists():
        if not force:
            raise FileExistsError(f'refusing existing output (pass --force): {path}')
        if path.is_dir():
            for child in path.iterdir():
                child.unlink()
        else:
            path.unlink()


def site_facts(structure: dict | None):
    """Species string, magmom and disorder count. One string per site: 'El:occu' or several joined.

    Lattice, atomic numbers and fractional coordinates stay in calc_structure so they match
    the staging tables. Magmom and every species occupancy are kept here because the staging
    tables dropped them. Forces, velocities, charges and oxidation states are not kept.
    """
    elements, magmom, disordered = [], [], 0
    for site in (structure or {}).get('sites') or []:
        species = site.get('species') or []
        if len(species) != 1:
            disordered += 1
        elements.append(','.join(f"{item.get('element')}:{item.get('occu', 1)}" for item in species))
        mag = (site.get('properties') or {}).get('magmom')
        magmom.append(None if mag is None else float(mag))
    return elements, magmom, disordered


def assemble_edos(row: dict, run_type: str, material_id, is_canonical: bool, calc_type, snapshot: str) -> dict:
    """One Delta total-dos row, same arrays as build_store.edos_record, without quality scores."""
    base = edos_record(row, run_type, material_id, is_canonical)
    elements, magmom, disordered = site_facts(row.get('structure'))
    status = base['status']
    return {
        'spectrum_id': base['spectrum_id'], 'material_id': material_id, 'source': 'mp',
        'source_task_id': base['source_task_id'], 'run_type': run_type, 'calc_type': calc_type,
        'is_canonical': bool(is_canonical), 'efermi': base['efermi'], 'spin_channels': base['spin_channels'],
        'x': base['x'], 'y': base['y'], 'n_points': base['n_points'],
        'calc_lattice': base['calc_lattice'], 'calc_numbers': base['calc_numbers'],
        'calc_elements': elements, 'calc_frac': base['calc_frac'], 'calc_magmom': magmom,
        'calc_volume_per_atom': base['calc_volume_per_atom'],
        'parse_error': None if status in ('ok', 'truncated') else status,
        'snapshot': snapshot, '_disordered_sites': disordered,
    }


def jarvis_structure(atoms: dict):
    """Lattice and fractional coordinates from a JARVIS atoms dict.

    The downloaded atoms use Cartesian coordinates when ``cartesian`` is true (the files we
    have). Fractional coordinates are ``coords @ inv(lattice)``, so the lattice and the
    elements are the downloaded values and the fractional coordinates are that conversion.
    """
    lattice = np.asarray(atoms['lattice_mat'], dtype=np.float64).reshape(3, 3)
    coords = np.asarray(atoms['coords'], dtype=np.float64).reshape(-1, 3)
    elements = [str(item) for item in atoms['elements']]
    if len(elements) != len(coords):
        raise ValueError(f'{len(elements)} elements, {len(coords)} coordinates')
    frac = coords @ np.linalg.inv(lattice) if atoms.get('cartesian', True) else coords
    return lattice.reshape(-1).tolist(), elements, np.asarray(frac, dtype=np.float64).reshape(-1).tolist()


def _load_mp_index():
    return json.loads((STORE / 'staging/index.json').read_text())


# ---------------------------------------------------------------- materials
def build_materials(force: bool) -> dict:
    out = OUT / 'materials.parquet'
    refuse(out, force)
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = {}
    for doc in load_pages(RAW_MP / 'electronic_structure'):
        dos = doc.get('dos') or {}
        if dos.get('task_id'):
            canonical[doc['material_id']] = strip_mp(dos['task_id'])
    writer = PartWriter(out, MATERIALS_SCHEMA)
    stats = Counter()
    for doc in load_pages(RAW_MP / 'summary'):
        builder = doc.get('builder_meta') or {}
        structure = doc.get('structure') or {}
        lattice, elements, frac, magmom = [], [], [], []
        if structure.get('sites'):
            encoded, _ = calc_structure(structure)
            elements, magmom, disordered = site_facts(structure)
            lattice, frac = encoded['calc_lattice'], encoded['calc_frac']
            stats['disordered_sites'] += disordered
            stats['with_structure'] += 1
        else:
            stats['no_structure'] += 1
        if doc.get('deprecated'):
            stats['deprecated'] += 1
        stats['license:' + str(builder.get('license'))] += 1
        stats['db:' + str(builder.get('database_version'))] += 1
        writer.add({
            'material_id': doc['material_id'], 'formula': doc.get('formula_pretty'),
            'nsites_mp': doc.get('nsites'), 'deprecated': bool(doc.get('deprecated')),
            'license': builder.get('license'), 'database_version': builder.get('database_version'),
            'emmet_version': builder.get('emmet_version'), 'pymatgen_version': builder.get('pymatgen_version'),
            'build_date': builder.get('build_date'), 'canonical_dos_task': canonical.get(doc['material_id']),
            'lattice': lattice, 'site_elements': elements, 'site_frac': frac, 'site_magmom': magmom,
            'n_sites': len(elements), 'source': 'mp', 'snapshot': SNAPSHOT_SUMMARY,
        })
        stats['rows'] += 1
        if doc['material_id'] in canonical:
            stats['with_canonical_dos_task'] += 1
    writer.close()
    named = {key: stats[key] for key in stats if key.startswith(('license:', 'db:'))}
    summary = {key: stats[key] for key in
               ('rows', 'with_structure', 'no_structure', 'deprecated', 'disordered_sites', 'with_canonical_dos_task')}
    summary['by_license'] = {key.split(':', 1)[1]: value for key, value in named.items() if key.startswith('license:')}
    summary['by_database_version'] = {key.split(':', 1)[1]: value for key, value in named.items() if key.startswith('db:')}
    ids = set(pq.read_table(out, columns=['material_id']).column(0).to_pylist())
    summary['duplicate_material_ids'] = stats['rows'] - len(ids)
    summary['canonical_without_summary'] = sum(material not in ids for material in canonical)
    summary['snapshot_canonical_task'] = SNAPSHOT_ELECTRONIC
    print('materials', summary)
    return summary


# ---------------------------------------------------------------- eDOS
OWNER: dict = {}
CANONICAL: dict = {}
CALC_TYPES: dict = {}


def _init_edos(owner, canonical, calc_types):
    global OWNER, CANONICAL, CALC_TYPES
    OWNER, CANONICAL, CALC_TYPES = owner, canonical, calc_types


def _edos_file(job):
    path, run_type, part = job
    writer = PartWriter(Path(part), EDOS_SCHEMA)
    stats = Counter()
    reader = pq.ParquetFile(path)
    columns = ['identifier', 'structure', 'energies', 'spin_up_densities', 'spin_down_densities', 'efermi']
    for batch in reader.iter_batches(batch_size=128, columns=columns):
        for row in batch.to_pylist():
            task = row['identifier']
            material = OWNER.get(task)
            is_canonical = material is not None and CANONICAL.get(material) == task
            record = assemble_edos(row, run_type, material, is_canonical, CALC_TYPES.get(task), SNAPSHOT_DELTA_EDOS)
            stats['disordered_sites'] += record.pop('_disordered_sites')
            stats['rows'] += 1
            stats['unlisted' if material is None else 'owned'] += 1
            if is_canonical:
                stats['canonical'] += 1
            if record['parse_error']:
                stats['unreadable'] += 1
            if record['spin_channels'] == 2:
                stats['spin_resolved'] += 1
            writer.add(record)
    writer.close()
    return dict(stats)


def build_edos(force: bool, workers: int) -> dict:
    parts = OUT / '_parts_edos'
    dest = OUT / 'spectra_edos.parquet'
    refuse(dest, force)
    if parts.exists() and force:
        for child in parts.glob('*.parquet'):
            child.unlink()
    parts.mkdir(parents=True, exist_ok=True)
    index = _load_mp_index()
    calc_types = {}
    for doc in load_pages(RAW_CORE / 'core_calc_types'):
        for task, calc_type in (doc.get('calc_types') or {}).items():
            calc_types[strip_mp(task)] = calc_type
    files = delta_files('core/electronic-structure/total-dos', 'run_type')
    jobs = [(str(path), run_type, str(parts / f'part-{i:02d}.parquet')) for i, (path, run_type) in enumerate(files)]
    totals = Counter()
    with Pool(min(workers, len(jobs)), initializer=_init_edos,
              initargs=(index['task_owner'], index['dos_task'], calc_types)) as pool:
        for counts in pool.imap_unordered(_edos_file, jobs):
            totals.update(counts)
            print('edos part', dict(counts), flush=True)
    _merge_parts(parts, dest, EDOS_SCHEMA)
    for child in parts.glob('*.parquet'):
        child.unlink()
    parts.rmdir()
    summary = dict(totals)
    print('edos', summary)
    return summary


def _merge_parts(parts: Path, dest: Path, schema: pa.Schema):
    writer = pq.ParquetWriter(dest, schema, compression='zstd')
    try:
        for path in sorted(parts.glob('*.parquet')):
            writer.write_table(pq.read_table(path, schema=schema))
    finally:
        writer.close()


# ---------------------------------------------------------------- MP phDOS
def build_phdos_mp(force: bool) -> dict:
    dest = OUT / 'spectra_phdos_mp.parquet'
    refuse(dest, force)
    OUT.mkdir(parents=True, exist_ok=True)
    index = _load_mp_index()
    owner, api = index['task_owner'], index['phonon']
    writer = PartWriter(dest, PHDOS_MP_SCHEMA)
    stats = Counter()
    for path, method in delta_files('phonon/electronic-structure/dos', 'phonon_method'):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=['identifier', 'dos']):
            for row in batch.to_pylist():
                task = strip_mp(row['identifier'])
                dos = row['dos'] or {}
                meta = next((item for item in api.get(task, []) if item['method'] == method), {})
                record = {
                    'spectrum_id': f'phdos:mp:{method}:{task}', 'material_id': owner.get(task), 'source': 'mp',
                    'source_task_id': task, 'method': method, 'asr': meta.get('asr'), 'cnsr': meta.get('cnsr'),
                    'x': [], 'y': [], 'n_points': 0, 'calc_lattice': [], 'calc_numbers': [], 'calc_elements': [],
                    'calc_frac': [], 'calc_magmom': [], 'calc_volume_per_atom': None, 'parse_error': None,
                    'snapshot': SNAPSHOT_DELTA_PHONON,
                }
                try:
                    if dos.get('structure'):
                        calc, _ = calc_structure(dos['structure'])
                        elements, magmom, disordered = site_facts(dos['structure'])
                        record.update(calc, calc_elements=elements, calc_magmom=magmom)
                        stats['disordered_sites'] += disordered
                    else:
                        stats['no_structure'] += 1
                    frequencies = np.asarray(dos.get('frequencies') or [], dtype=np.float64)
                    densities = np.asarray(dos.get('densities') or [], dtype=np.float64)
                    if len(frequencies) < 10 or len(frequencies) != len(densities):
                        raise ValueError('fewer than 10 points or length mismatch')
                    record.update(x=(frequencies * THZ2CM).tolist(), y=(densities / THZ2CM).tolist(),
                                  n_points=int(len(frequencies)))
                except Exception as exc:
                    record['parse_error'] = f'unreadable:{type(exc).__name__}'
                stats['rows'] += 1
                stats['owned' if record['material_id'] else 'no_material'] += 1
                stats[f'{method}/{"error" if record["parse_error"] else "ok"}'] += 1
                if meta:
                    stats['with_sum_rules'] += 1
                writer.add(record)
    writer.close()
    summary = {key: stats[key] for key in stats}
    print('phdos_mp', summary)
    return summary


# ---------------------------------------------------------------- PhononDB
PHONONDB_COLUMNS = [
    'spectrum_id', 'source', 'source_id', 'method', 'formula', 'reduced_formula', 'mp_reference',
    'x', 'y', 'n_points', 'freq_thz', 'dos_thz', 'unit_cell', 'primitive_cell', 'supercell_matrix',
    'born', 'dielectric', 'mesh', 'phonopy_version', 'status',
]


def build_phonondb(force: bool) -> dict:
    """Copy the already recomputed PhononDB table and record each archive hash.

    Recomputing 10,034 archives with phonopy is the step that produced staging/phdos_phonondb.
    This stage does not recompute it. The phonopy version and the mesh are on each row.
    """
    dest = OUT / 'spectra_phdos_phonondb.parquet'
    refuse(dest, force)
    OUT.mkdir(parents=True, exist_ok=True)
    source = STORE / 'staging/phdos_phonondb'
    zips = STORE / 'raw_phonondb/zips'
    stats = Counter()
    rows = []
    for path in sorted(source.glob('part-*.parquet')):
        for record in pq.read_table(path, columns=PHONONDB_COLUMNS).to_pylist():
            archive = zips / f"{record['source_id']}.zip"
            status = record.pop('status')
            record['archive_sha256'] = file_sha256(archive) if archive.exists() else None
            record['parse_error'] = None if status == 'ok' else status
            record['functional_in_archive'] = False
            record['functional_on_card'] = 'PBEsol'
            record['snapshot'] = SNAPSHOT_PHONONDB
            stats['rows'] += 1
            stats['archive_missing' if record['archive_sha256'] is None else 'archive_hashed'] += 1
            if record['parse_error']:
                stats['unreadable'] += 1
            rows.append(record)
    pq.write_table(pa.Table.from_pylist(rows), dest, compression='zstd')
    print('phonondb', dict(stats))
    return dict(stats)


# ---------------------------------------------------------------- JARVIS
def _parse_floats(text):
    if text is None:
        return []
    out = []
    for item in str(text).split(','):
        item = item.strip().strip("'\"")
        if item:
            try:
                out.append(float(item))
            except ValueError:
                pass
    return out


def build_jarvis(force: bool) -> dict:
    dest = OUT / 'spectra_phdos_jarvis.parquet'
    refuse(dest, force)
    OUT.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(STORE / 'raw_jarvis/jdft_3d-12-12-2022.json.zip')
    meta = {record['jid']: record for record in json.loads(archive.read(archive.namelist()[0]))}
    stats = Counter()
    rows = []
    seen = set()
    for path in sorted((STORE / 'raw_jarvis/pages').glob('part-*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                page = json.loads(line)
                jid = page['jid']
                if jid in seen or (page.get('error') or '').startswith('failed'):
                    stats['skipped_failed_page'] += 1
                    continue
                seen.add(jid)
                card = meta.get(jid, {})
                fields = page.get('fields') or {}
                frequencies = _parse_floats(fields.get('phonon_dos_frequencies'))
                intensity = _parse_floats(fields.get('phonon_dos_intensity'))
                if len(frequencies) >= 10 and len(frequencies) == len(intensity):
                    parse_error, x, y = None, frequencies, intensity
                elif not frequencies:
                    parse_error, x, y = 'no_dos', [], []
                else:
                    parse_error, x, y = 'unreadable:length_mismatch', [], []
                lattice, elements, frac = [], [], []
                atoms = card.get('atoms')
                if atoms:
                    try:
                        lattice, elements, frac = jarvis_structure(atoms)
                        stats['cartesian' if atoms.get('cartesian', True) else 'fractional'] += 1
                    except Exception as exc:
                        parse_error = parse_error or f'structure:{type(exc).__name__}'
                        stats['structure_failed'] += 1
                else:
                    stats['no_atoms'] += 1
                reference = card.get('reference')
                mode_min = _parse_floats(fields.get('min_fd_phonon_mode'))
                rows.append({
                    'spectrum_id': f'phdos:jarvis:{jid}', 'source': 'jarvis', 'source_id': jid,
                    'method': 'finite_difference', 'functional': card.get('func') or None,
                    'formula': card.get('formula'), 'reference': None if reference is None else str(reference),
                    'spacegroup': card.get('spg_symbol'), 'x': x, 'y': y, 'n_points': len(x),
                    'phonon_modes': _parse_floats(fields.get('phonon_modes')),
                    'min_fd_phonon_mode': mode_min[0] if mode_min else None,
                    'fetched': page.get('fetched'), 'calc_lattice': lattice, 'calc_elements': elements,
                    'calc_frac': frac, 'parse_error': parse_error, 'snapshot': SNAPSHOT_JARVIS,
                })
                stats['rows'] += 1
                stats[parse_error or 'ok'] += 1
    pq.write_table(pa.Table.from_pylist(rows), dest, compression='zstd')
    print('jarvis', dict(stats))
    return dict(stats)


# ---------------------------------------------------------------- manifest and readme
def _git_commit() -> str:
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def _table_fact(path: Path) -> dict:
    schema = pq.read_schema(path)
    return {'rows': pq.ParquetFile(path).metadata.num_rows, 'bytes': path.stat().st_size,
            'sha256': file_sha256(path), 'columns': schema.names}


def _write_readme(facts: dict, stats: dict):
    materials = stats['materials']
    edos = stats['edos']
    phonon = stats['phdos_mp']
    phonondb = stats['phonondb']
    jarvis = stats['jarvis']
    text = f"""# 谱数据库 spectral_db_v1

构建时间：{stats['built_at']}（北京时间）。代码提交：`{stats['git_commit']}`。
本目录是数据库本身。清洗、筛选、去重、切分和训练视图都不在这里，那些是数据集的事。

## 这个库记什么

每条记录只保留来源给出的事实，以及把谱读出来所必需的换算。下面这些**不在库里**：

- 旧数据集 `crystal_q1_v1` 的谱和标签。旧材料如果现在还在 MP 里，就只是普通的 MP 材料。
- 结构匹配、体积比、原型是否一致、标签该不该用。这些是数据集构建时算的。
- 质量判断：积分是否够、谱是否异常、有没有虚频、原子数上限。库里不打这些分。
- 任何 train / valid / test 划分。

`data/spectral_store_v1/` 是之前的工作库，里面混有视图、切分和旧标签，本库不改它。

## 表

| 文件 | 行数 | 一行是什么 |
|---|---:|---|
| `materials.parquet` | {facts['materials.parquet']['rows']:,} | MP summary 里的一个材料，结构按下载的原样 |
| `spectra_edos.parquet` | {facts['spectra_edos.parquet']['rows']:,} | Delta 里的一条电子总态密度 |
| `spectra_phdos_mp.parquet` | {facts['spectra_phdos_mp.parquet']['rows']:,} | Delta 里的一条 MP 声子态密度 |
| `spectra_phdos_phonondb.parquet` | {facts['spectra_phdos_phonondb.parquet']['rows']:,} | 一份 PhononDB 压缩包重算出的声子态密度 |
| `spectra_phdos_jarvis.parquet` | {facts['spectra_phdos_jarvis.parquet']['rows']:,} | 一条已下载的 JARVIS 声子页 |

每张表的 SHA-256 在 `manifest.json`。

### materials.parquet

来源：MP API `materials/summary`，接口 0.88.1，下载于 2026-11-10。`canonical_dos_task` 来自同一次下载的 `materials/electronic_structure`（MP 指定的 DOS 任务号，去掉 `mp-` 前缀），不是我们挑的。

- 材料 {materials['rows']:,} 个，其中 MP 标记 deprecated 的 {materials['deprecated']:,} 个。
- 有结构的 {materials['with_structure']:,} 个，没有结构的 {materials['no_structure']:,} 个。
- 许可计数：{json.dumps(materials['by_license'], ensure_ascii=False)}。
- 数据库版本计数：{json.dumps(materials['by_database_version'], ensure_ascii=False)}。
- MP 给出规范 DOS 任务、且该材料在 summary 里的：{materials['with_canonical_dos_task']:,}。规范任务指向 summary 之外材料的：{materials['canonical_without_summary']:,}。
- 结构字段是下载的晶格矩阵、每个位点的元素和占位（`元素:占位`，多个物种用逗号接）、分数坐标、磁矩。没有做原胞标准化。位点上的力、速度、电荷、氧化态没有保留。无序位点（不止一个物种）共 {materials['disordered_sites']:,} 个。

### spectra_edos.parquet

来源：MP Delta 表 `s3://materialsproject-parsed/core/electronic-structure/total-dos/`，本地下载于 2026-11-05（`tools/data/raw/delta_mp`，清单 `manifest.json`）。材料归属来自 MP API `materials/core` 的 `task_ids`（2026-11-10）。`calc_type` 来自同一次 `calc_types`。`run_type` 是 Delta 的分区值（GGA、GGA+U、HSE06、PBE、PBEsol、r2SCAN、unknown），不是另行核实过的泛函。

- 共 {edos.get('rows', 0):,} 条。能挂到当前材料的 {edos.get('owned', 0):,} 条，Delta 里有、但 2026-11-10 的 `task_ids` 已经不列的 {edos.get('unlisted', 0):,} 条（`material_id` 为空，谱仍保留）。
- 归属材料自己把这条任务指定为规范 DOS 的：{edos.get('canonical', 0):,}。材料表里点名的规范任务有 {edos.get('canonical_task_in_delta', 0):,} 条能在 Delta 里找到；其中 {edos.get('canonical_task_owned_by_other', 0):,} 条的 core 归属不是点名它的那个材料，所以没有标成规范。Delta 里找不到的规范任务号：{edos.get('canonical_task_missing_from_delta', 0):,}。
- 有自旋向下通道、密度已按通道相加的：{edos.get('spin_resolved', 0):,}。只有一个通道的，密度就是该通道。分开的自旋通道没有保留。
- 读不出来的：{edos.get('unreadable', 0):,}（`parse_error` 记录原因，行不删）。
- 无序位点：{edos.get('disordered_sites', 0):,}。
- 横轴是能量减去该任务自己的 `efermi`（eV），`efermi` 另存，绝对能量可以加回去。纵轴是自旋相加后的态密度。点的顺序和 Delta 里一样，没有为了入库而删点、排序或合并重复能量；积分用的清洗不在本库。

### spectra_phdos_mp.parquet

来源：Delta 表 `phonon/electronic-structure/dos/`，下载于 2026-11-05。`method` 是 Delta 的 `phonon_method`（pheasy 或 dfpt）。`asr`、`cnsr` 是 MP API `materials/phonon` 给出的求和规则偏离；接口没有给出的留空。Delta 和接口都没有给出可核实的泛函，所以这张表没有泛函列。

- 共 {phonon.get('rows', 0):,} 条。挂到材料的 {phonon.get('owned', 0):,}，没有材料归属的 {phonon.get('no_material', 0):,}。
- 有求和规则数值的 {phonon.get('with_sum_rules', 0):,}。能读出的谱：dfpt {phonon.get('dfpt/ok', 0):,}，pheasy {phonon.get('pheasy/ok', 0):,}（这里只表示谱本身可读，材料归属见上一条）。
- 横轴是频率乘以 33.35641，从 THz 换成 cm⁻¹；纵轴是密度除以 33.35641。原始 THz 数组没有另存，用这个常数可以换回去。
- 无序位点：{phonon.get('disordered_sites', 0):,}。没有计算结构的：{phonon.get('no_structure', 0):,}。

### spectra_phdos_phonondb.parquet

来源：NIMS MDR PhononDB，索引来自 `atztogo/phonondb` 的 MDR 说明，压缩包在 `data/spectral_store_v1/raw_phonondb/zips/`。每行有 `archive_sha256`。谱不是压缩包里的成谱，是用 phonopy 从力常数重算的（`MP = 20 20 20`，`TETRAHEDRON = .TRUE.`，默认频率采样）。`phonopy_version` 和 `mesh` 在每行上。本表直接沿用这次重算，没有再跑一遍 phonopy。

- 共 {phonondb.get('rows', 0):,} 条，压缩包哈希已写入 {phonondb.get('archive_hashed', 0):,} 条，缺包 {phonondb.get('archive_missing', 0):,}。
- 压缩包里没有泛函字段。数据卡写的是 PBEsol，记在 `functional_on_card`，`functional_in_archive` 为 false。
- `mp_reference` 是数据卡上的旧整数 MP 编号（形如 `mp-25`），不是现在的 `mp-` 加字母编号，没有和当前 MP 逐条核对。
- 同时保留了 cm⁻¹ 和原始 THz 两套数组，以及原胞、惯胞、超胞矩阵、Born 电荷和介电张量（phonopy 输入里的 JSON）。

### spectra_phdos_jarvis.parquet

来源：JARVIS-DFT 网页上的声子字段（频率、强度、声子模、最小有限差分模），加上 `jdft_3d-12-12-2022.json.zip` 里的结构、化学式、泛函、参考编号和空间群。只收录实际下载过声子页的条目，不是 zip 里全部 75,993 条 dft_3d。

- 共 {jarvis.get('rows', 0):,} 条。有谱 {jarvis.get('ok', 0):,}，页面没有 DOS {jarvis.get('no_dos', 0):,}，长度对不上 {jarvis.get('unreadable:length_mismatch', 0):,}。
- 坐标：下载文件里是笛卡尔坐标的 {jarvis.get('cartesian', 0):,} 条，分数坐标的 {jarvis.get('fractional', 0):,} 条。库里的 `calc_frac` 在笛卡尔时由 `coords @ inv(lattice)` 换算，晶格和元素保持原值。
- `functional` 只用数据卡上的 `func`。卡上没有的就空着，不再填默认值。`reference` 保留数据卡原文，不判断它是不是 MP 编号。
- 强度是页面上的原始数值，没有按 3N 重标定。

JARVIS 和 PhononDB 的许可原文不在下载文件里，本库不补写。MP 材料的许可是接口给出的，写在材料表的 `license` 列。

## 重建

```bash
python tools/data/spectral_store/build_db_v1.py all --workers 8
python tools/data/spectral_store/build_db_v1.py verify
```

已有输出时会拒绝覆盖，需要同时加 `--force`。`verify` 把本库和 `spectral_store_v1` 的 staging 表逐条对谱。eDOS 和 MP 声子是建库时从 Delta 重读的，再和 staging 对；PhononDB 和 staging 里已经重算好的表对；JARVIS 是按网页和 zip 重读后再对。
"""
    (OUT / 'README.md').write_text(text)


def reconcile_canonical(stats: dict) -> dict:
    """How MP's named DOS task relates to the task's core owner. Computed from the tables."""
    materials = pq.read_table(OUT / 'materials.parquet', columns=['material_id', 'canonical_dos_task']).to_pylist()
    named = {row['material_id']: row['canonical_dos_task'] for row in materials if row['canonical_dos_task']}
    flagged, present = set(), set()
    for batch in pq.ParquetFile(OUT / 'spectra_edos.parquet').iter_batches(
            batch_size=65536, columns=['source_task_id', 'material_id', 'is_canonical']):
        tasks, owners, flags = (batch.column(i).to_pylist() for i in range(3))
        for task, owner, flag in zip(tasks, owners, flags):
            present.add(task)
            if flag:
                flagged.add(task)
    named_tasks = set(named.values())
    stats = dict(stats)
    edos = dict(stats.get('edos', {}))
    edos['canonical_task_in_delta'] = len(named_tasks & present)
    edos['canonical_flagged'] = len(flagged)
    edos['canonical_task_owned_by_other'] = len(named_tasks & present) - len(flagged)
    edos['canonical_task_missing_from_delta'] = len(named_tasks - present)
    stats['edos'] = edos
    return stats


def build_manifest(stats: dict):
    stats = reconcile_canonical(stats)
    facts = {name: _table_fact(OUT / name) for name in (
        'materials.parquet', 'spectra_edos.parquet', 'spectra_phdos_mp.parquet',
        'spectra_phdos_phonondb.parquet', 'spectra_phdos_jarvis.parquet')}
    stats = dict(stats)
    stats['schema'] = SCHEMA
    stats['built_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    stats['git_commit'] = _git_commit()
    (OUT / 'build_stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False) + '\n')
    manifest = {'schema': SCHEMA, 'built_at': stats['built_at'], 'git_commit': stats['git_commit'],
                'tables': facts,
                'processing': {
                    'edos': 'energies shifted by the task efermi; spin channels summed; order unchanged',
                    'phdos_mp': 'THz multiplied by 33.35641; density divided by 33.35641',
                    'phonondb': 'phonopy total DOS, mesh 20 20 20, tetrahedron; not recomputed in this step',
                    'jarvis_coordinates': 'Cartesian coords converted by coords @ inv(lattice) when cartesian is true',
                }}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    _write_readme(facts, stats)
    print(json.dumps({name: {'rows': item['rows'], 'sha256': item['sha256']} for name, item in facts.items()},
                     indent=2))


# ---------------------------------------------------------------- verify
def _digest(parts: list):
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, (bytes, bytearray)):
            digest.update(part)
        elif isinstance(part, str):
            digest.update(part.encode())
        elif part is None:
            digest.update(b'-')
        elif isinstance(part, (bool, np.bool_)):
            digest.update(b'1' if part else b'0')
        elif isinstance(part, (int, np.integer)):
            digest.update(str(int(part)).encode())
        elif isinstance(part, (float, np.floating)):
            digest.update(np.asarray([part], np.float64).tobytes())
        else:
            digest.update(np.asarray(list(part), np.float64).tobytes())
        digest.update(b'|')
    return digest.hexdigest()


def _column_hashes(path: Path, key: str, fields: list, predicate=None) -> dict:
    out = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=[key, *fields]):
        for record in batch.to_pylist():
            if predicate and not predicate(record):
                continue
            out[record[key]] = _digest([record.get(field) for field in fields])
    return out


def _compare(name: str, left: dict, right: dict) -> dict:
    missing = [key for key in left if key not in right]
    mismatched = [key for key in left if key in right and left[key] != right[key]]
    report = {'left': len(left), 'right': len(right), 'missing': len(missing), 'mismatched': len(mismatched),
              'missing_example': missing[:5], 'mismatched_example': mismatched[:5]}
    print(name, {k: report[k] for k in ('left', 'right', 'missing', 'mismatched')})
    return report


def verify() -> dict:
    """New tables against spectral_store_v1 staging. Raises if a shared spectrum differs."""
    report = {}
    edos_fields = ['material_id', 'source_task_id', 'is_canonical', 'efermi', 'spin_channels', 'n_points', 'x', 'y',
                   'calc_lattice', 'calc_numbers', 'calc_frac']
    staging_edos = STORE / 'staging/edos'
    left = {}
    for path in sorted(staging_edos.glob('part-*.parquet')):
        left.update(_column_hashes(path, 'spectrum_id', edos_fields))
    right = _column_hashes(OUT / 'spectra_edos.parquet', 'spectrum_id',
                           ['run_type' if f == 'run_type' else f for f in edos_fields])
    report['edos'] = _compare('edos', left, right)
    report['edos']['rows'] = pq.ParquetFile(OUT / 'spectra_edos.parquet').metadata.num_rows
    report['edos']['duplicate_ids'] = report['edos']['rows'] - report['edos']['right']

    phonon_fields = ['material_id', 'source_task_id', 'method', 'asr', 'cnsr', 'n_points', 'x', 'y',
                     'calc_lattice', 'calc_numbers', 'calc_frac']
    left = _column_hashes(STORE / 'staging/phdos/part-00.parquet', 'spectrum_id', phonon_fields)
    right = _column_hashes(OUT / 'spectra_phdos_mp.parquet', 'spectrum_id', phonon_fields)
    report['phdos_mp'] = _compare('phdos_mp', left, right)

    phonondb_fields = ['x', 'y', 'n_points', 'freq_thz', 'dos_thz', 'unit_cell', 'primitive_cell', 'phonopy_version']
    left = {}
    for path in sorted((STORE / 'staging/phdos_phonondb').glob('part-*.parquet')):
        left.update(_column_hashes(path, 'source_id', phonondb_fields))
    right = _column_hashes(OUT / 'spectra_phdos_phonondb.parquet', 'source_id', phonondb_fields)
    report['phonondb'] = _compare('phonondb', left, right)

    jarvis_fields = ['x', 'y', 'n_points', 'phonon_modes', 'min_fd_phonon_mode', 'formula']
    left = _column_hashes(STORE / 'staging/phdos_jarvis/part-0000.parquet', 'source_id', jarvis_fields)
    right = _column_hashes(OUT / 'spectra_phdos_jarvis.parquet', 'source_id', jarvis_fields)
    report['jarvis'] = _compare('jarvis', left, right)

    materials = pq.read_table(OUT / 'materials.parquet', columns=['material_id'])
    ids = materials.column(0).to_pylist()
    report['materials_unique'] = len(ids) == len(set(ids))
    report['materials_rows'] = len(ids)
    print('materials unique', report['materials_unique'], 'rows', len(ids))
    problems = [name for name, item in report.items()
                if isinstance(item, dict) and (item.get('missing') or item.get('mismatched') or item.get('duplicate_ids'))]
    if problems or not report['materials_unique']:
        raise SystemExit(f'verify failed: {problems or "materials"}')
    (OUT / 'verify.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['materials', 'edos', 'phdos_mp', 'phonondb', 'jarvis', 'manifest', 'all', 'verify'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.stage == 'verify':
        verify()
        return
    stages = ['materials', 'edos', 'phdos_mp', 'phonondb', 'jarvis', 'manifest'] if args.stage == 'all' else [args.stage]
    stats = {}
    stats_path = OUT / 'build_stats.json'
    if stats_path.exists() and args.stage != 'all':
        stats = json.loads(stats_path.read_text())
    for stage in stages:
        print('==', stage, flush=True)
        if stage == 'materials':
            stats['materials'] = build_materials(args.force)
        elif stage == 'edos':
            stats['edos'] = build_edos(args.force, args.workers)
        elif stage == 'phdos_mp':
            stats['phdos_mp'] = build_phdos_mp(args.force)
        elif stage == 'phonondb':
            stats['phonondb'] = build_phonondb(args.force)
        elif stage == 'jarvis':
            stats['jarvis'] = build_jarvis(args.force)
        elif stage == 'manifest':
            build_manifest(stats)
    if args.stage != 'manifest':
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'build_stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
