"""PhononDB and JARVIS phonon DOS into spectral-store-v1 (design 2026-11-11 steps 1c-1d).

Subcommands (incremental: each run adds only archives/pages not yet ingested):

- phonondb: recompute every downloaded MDR archive exactly as the legacy pipeline did
  (phonopy CLI, `MP = 20 20 20`, `TETRAHEDRON = .TRUE.`, default frequency sampling) and
  store the native DOS (x = THz x 33.35641 cm^-1, y = density / 33.35641, also the raw THz
  arrays) with the archive's unit cell, primitive cell, supercell matrix, Born charges and
  dielectric tensor. Archives that fail are stored with status unreadable:<reason>.
  -> staging/phdos_phonondb/part-XXXX.parquet
- jarvis: parse fetched JARVIS pages; x = frequencies (cm^-1), y = raw intensity
  (arbitrary units; the legacy 3N rescaling depends on the input cell and is applied at
  export), plus phonon modes and dft_3d metadata (formula, MP reference, spacegroup).
  -> staging/phdos_jarvis/part-XXXX.parquet
- gate: link legacy phDOS labels without an MP source to these spectra. Candidates have
  the label's reduced formula; a candidate is the original when the legacy rule
  (box_average on P0 after the legacy conversion; JARVIS scaled to 3N / native area with N
  the input primitive cell) reproduces the frozen label (common.classify_label, <= 4
  spike-clipped bins). Preference: PhononDB before JARVIS (legacy priority), exact before
  clipped, then the smallest id. Writes staging/phdos_legacy_links.parquet (train/valid)
  and sealed/phdos_legacy_links_test.parquet; prints counts only.

Every PhononDB archive and JARVIS page is kept, including those duplicating MP phonons.
"""
from __future__ import annotations

import argparse
import gzip
import json
import lzma
import os
import subprocess
import tempfile
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from common import STORE, THZ2CM, box_average, classify_label

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STAGING = STORE / 'staging'
P0 = np.linspace(-280.0, 980.0, 65)
MAX_CLIPPED = 4
_SOURCE = [('spectrum_id', pa.string()), ('source', pa.string()), ('source_id', pa.string()),
           ('method', pa.string()), ('functional', pa.string()), ('formula', pa.string()),
           ('reduced_formula', pa.string()), ('mp_reference', pa.string()),
           ('x', pa.list_(pa.float64())), ('y', pa.list_(pa.float64())), ('n_points', pa.int32()),
           ('status', pa.string())]
PHONONDB_SCHEMA = pa.schema(_SOURCE + [('freq_thz', pa.list_(pa.float64())), ('dos_thz', pa.list_(pa.float64())),
                                       ('unit_cell', pa.string()), ('primitive_cell', pa.string()),
                                       ('supercell_matrix', pa.string()), ('born', pa.string()),
                                       ('dielectric', pa.string()), ('mesh', pa.list_(pa.int32())),
                                       ('phonopy_version', pa.string())])
JARVIS_SCHEMA = pa.schema(_SOURCE + [('phonon_modes', pa.list_(pa.float64())), ('min_fd_phonon_mode', pa.float64()),
                                     ('spacegroup', pa.string()), ('fetched', pa.string())])


def parse_floats(text):
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


def reduced(formula):
    from pymatgen.core import Composition
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def next_part(directory: Path) -> Path:
    return directory / f'part-{len(list(directory.glob("part-*.parquet"))):04d}.parquet'


def ingested(directory: Path) -> set:
    parts = sorted(directory.glob('part-*.parquet'))
    if not parts:
        return set()
    return set(pa.concat_tables([pq.read_table(p, columns=['source_id']) for p in parts]).column(0).to_pylist())


# ---------------------------------------------------------------- PhononDB
def _recompute(job):
    path, meta = job
    import phonopy
    import yaml
    serial = path.stem
    row = {'spectrum_id': f'phdos:phonondb:{serial}', 'source': 'phonondb', 'source_id': serial,
           'method': 'phonopy_fd_recomputed', 'functional': 'PBEsol (dataset card; not in archive)',
           'formula': meta.get('name'), 'reduced_formula': meta.get('reduced'),
           'mp_reference': f"mp-{meta['mp_int']}" if meta.get('mp_int') is not None else None,
           'x': [], 'y': [], 'n_points': 0, 'status': '', 'freq_thz': [], 'dos_thz': [], 'unit_cell': None,
           'primitive_cell': None, 'supercell_matrix': None, 'born': None, 'dielectric': None,
           'mesh': [20, 20, 20], 'phonopy_version': phonopy.__version__}
    try:
        raw = lzma.decompress(zipfile.ZipFile(path).read('phonopy_params.yaml.xz')).decode()
        doc = yaml.safe_load(raw)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / 'phonopy.conf').write_text(raw)
            (Path(td) / 'mesh.conf').write_text('MP = 20 20 20\nTETRAHEDRON = .TRUE.\n')
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            result = subprocess.run(['phonopy', '-c', 'phonopy.conf', '--dos', 'mesh.conf'], cwd=td,
                                    capture_output=True, text=True, timeout=3600, env=env)
            dos_file = Path(td) / 'total_dos.dat'
            if not dos_file.exists():
                raise RuntimeError(f'no total_dos.dat (rc {result.returncode}): {result.stderr[-120:]}')
            data = np.loadtxt(dos_file, comments='#', ndmin=2)
        freq, dos = data[:, 0], data[:, 1]
        row.update(freq_thz=freq.tolist(), dos_thz=dos.tolist(), x=(freq * THZ2CM).tolist(),
                   y=(dos / THZ2CM).tolist(), n_points=len(freq), status='ok',
                   unit_cell=json.dumps(doc.get('unit_cell')), primitive_cell=json.dumps(doc.get('primitive_cell')),
                   supercell_matrix=json.dumps(doc.get('supercell_matrix')),
                   born=json.dumps(doc.get('born_effective_charge')),
                   dielectric=json.dumps(doc.get('dielectric_constant')))
    except Exception as exc:
        row['status'] = f'unreadable:{type(exc).__name__}:{str(exc)[:80]}'
    return row


def ingest_phonondb(workers: int):
    out = STAGING / 'phdos_phonondb'
    out.mkdir(parents=True, exist_ok=True)
    index = {r['serial']: r for r in json.loads((STORE / 'raw_phonondb/index.json').read_text())}
    done = ingested(out)
    priority = json.loads((STORE / 'raw_phonondb/priority_serials.json').read_text())
    zips = {p.stem: p for p in (STORE / 'raw_phonondb/zips').glob('*.zip')}
    todo = [zips[s] for s in priority if s in zips and s not in done]
    todo += [p for s, p in sorted(zips.items()) if s not in done and s not in set(priority)]
    rows = []
    with Pool(workers) as pool:
        for row in pool.imap_unordered(_recompute, [(p, index.get(p.stem, {})) for p in todo], chunksize=4):
            rows.append(row)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows, schema=PHONONDB_SCHEMA), next_part(out), compression='zstd')
    print(json.dumps({'new': len(rows), 'status': dict(Counter(r['status'].split(':')[0] for r in rows)),
                      'total_ingested': len(done) + len(rows), 'archives_present': len(zips)}))


# ---------------------------------------------------------------- JARVIS
def ingest_jarvis():
    out = STAGING / 'phdos_jarvis'
    out.mkdir(parents=True, exist_ok=True)
    done = ingested(out)
    archive = zipfile.ZipFile(STORE / 'raw_jarvis/jdft_3d-12-12-2022.json.zip')
    meta = {r['jid']: r for r in json.loads(archive.read(archive.namelist()[0]))}
    rows = []
    for path in sorted((STORE / 'raw_jarvis/pages').glob('part-*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                page = json.loads(line)
                jid = page['jid']
                if jid in done or (page['error'] or '').startswith('failed'):
                    continue
                done.add(jid)
                m = meta.get(jid, {})
                fields = page['fields'] or {}
                x = parse_floats(fields.get('phonon_dos_frequencies'))
                y = parse_floats(fields.get('phonon_dos_intensity'))
                status = 'ok' if len(x) >= 10 and len(x) == len(y) else (
                    'no_dos' if not x else 'unreadable:length_mismatch')
                mode_min = parse_floats(fields.get('min_fd_phonon_mode'))
                rows.append({'spectrum_id': f'phdos:jarvis:{jid}', 'source': 'jarvis', 'source_id': jid,
                             'method': 'finite_difference', 'functional': m.get('func') or 'OptB88vdW',
                             'formula': m.get('formula'), 'reduced_formula': reduced(m.get('formula')),
                             'mp_reference': m.get('reference') if str(m.get('reference', '')).startswith('mp-') else None,
                             'x': x if status == 'ok' else [], 'y': y if status == 'ok' else [],
                             'n_points': len(x), 'status': status,
                             'phonon_modes': parse_floats(fields.get('phonon_modes')),
                             'min_fd_phonon_mode': mode_min[0] if mode_min else None,
                             'spacegroup': m.get('spg_symbol'), 'fetched': page['fetched']})
    if rows:
        pq.write_table(pa.Table.from_pylist(rows, schema=JARVIS_SCHEMA), next_part(out), compression='zstd')
    print(json.dumps({'new': len(rows), 'status': dict(Counter(r['status'].split(':')[0] for r in rows)),
                      'total_ingested': len(done)}))


# ---------------------------------------------------------------- gate
def legacy_binned(row, n_input):
    x, y = np.asarray(row['x'], np.float64), np.asarray(row['y'], np.float64)
    if row['source'] == 'jarvis':  # legacy: 3N / native full-spectrum area (input primitive cell)
        area = float(np.trapz(y, x)) if len(x) > 1 else 0.0
        if area > 0:
            y = y * (3 * n_input / area)
    return box_average(x, y, P0)[0]


def gate():
    from pymatgen.core import Composition, Element
    tables = []
    for name in ('phdos_phonondb', 'phdos_jarvis'):
        parts = sorted((STAGING / name).glob('part-*.parquet'))
        if parts:
            tables.append(pa.concat_tables([pq.read_table(p, columns=['source', 'source_id', 'reduced_formula', 'x',
                                                                      'y', 'status']) for p in parts]))
    candidates = {}
    for table in tables:
        for r in table.to_pylist():
            if r['status'] == 'ok':
                candidates.setdefault(r['reduced_formula'], []).append(r)
    materials = {m['material_id']: m for m in pq.read_table(
        STAGING / 'materials.parquet', columns=['material_id', 'legacy_split', 'numbers', 'n_atoms']).to_pylist()}
    mp_phonon = pq.read_table(STAGING / 'phdos/part-00.parquet', columns=['source_task_id', 'material_id', 'status'])
    mp_tasks = set(mp_phonon.column('source_task_id').to_pylist())
    mp_materials = {r['material_id'] for r in mp_phonon.to_pylist() if r['status'] == 'ok'}
    labels = [r for r in pq.read_table(STAGING / 'legacy_spectra.parquet').to_pylist()
              if r['type'] == 'phdos' and r['matched_task'] not in mp_tasks and r['material_id'] not in mp_materials]
    links = {'public': [], 'sealed': []}
    report = Counter()
    rank = {'phonondb': 0, 'jarvis': 1}
    for label in labels:
        m = materials[label['material_id']]
        formula = Composition({Element.from_Z(z).symbol: m['numbers'].count(z) for z in set(m['numbers'])}).reduced_formula
        old, mask = np.asarray(label['values'], np.float32), np.asarray(label['mask'], bool)
        hits = []
        for c in candidates.get(formula, []):
            kind, off = classify_label(legacy_binned(c, m['n_atoms']), old, mask, MAX_CLIPPED)
            if kind != 'differs':
                hits.append(((rank[c['source']], kind != 'exact', off, c['source_id']), c['source'], c['source_id'], kind))
        side = 'sealed' if m['legacy_split'] == 'test' else 'public'
        if hits:
            hits.sort()
            _, source, source_id, kind = hits[0]
            links[side].append({'material_id': label['material_id'], 'source': source, 'source_id': source_id,
                                'gate_class': kind, 'n_matching': len(hits)})
            report[f"{m['legacy_split']}/{source}/{kind}"] += 1
        else:
            report[f"{m['legacy_split']}/unmatched/{'no_candidate' if formula not in candidates else 'no_gate_pass'}"] += 1
    pq.write_table(pa.Table.from_pylist(links['public']), STAGING / 'phdos_legacy_links.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pylist(links['sealed']), STORE / 'sealed' / 'phdos_legacy_links_test.parquet',
                   compression='zstd')
    print(json.dumps({'labels': len(labels), **dict(sorted(report.items()))}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['phonondb', 'jarvis', 'gate'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    {'phonondb': lambda: ingest_phonondb(args.workers), 'jarvis': ingest_jarvis, 'gate': gate}[args.command]()


if __name__ == '__main__':
    main()
