"""Read-only audit of structure use and the data ceiling (design 2026-11-05).

Subcommands:
  infer     blind per-material errors of frozen checkpoints on a non-test split
  geometry  identify each material's MP DOS task by label reconstruction, then
            compare input and DOS-calculation geometry
  noise     target-to-target error of near-identical same-composition structures
  baseline  composition-only vs composition+structure descriptor regressors

No training, no test arrays, no change to frozen data, protocols or checkpoints.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / 'data/crystal_q1_v1'
SPLITS = ('train', 'valid')
E0_EDGES = np.linspace(-6, 6, 129)
P0_EDGES = np.linspace(-280, 980, 65)
THZ2CM = 33.35641
STRICT = dict(ltol=.01, stol=.05, angle_tol=.5, scale=False)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + '\n')


def refuse_existing(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f'refusing to overwrite {path}')


# ---------------------------------------------------------------- data access

def load_split(split: str) -> dict:
    """Raw arrays of one frozen split restricted to the default training/evaluation view."""
    if split not in SPLITS:
        raise ValueError('the audit reads only train/valid')
    d = DATA / split
    arrays = {name: np.load(d / f'{name}_{split}.npy') for name in (
        'atom_offsets', 'atomic_numbers', 'cell', 'frac_coords', 'n_atoms',
        'edos_tgtdos', 'phdos_tgtdos', 'edos_mask', 'phdos_mask')}
    arrays['ids'] = np.load(d / f'{split}_index.npy').astype(str)
    excluded = set(json.loads((DATA / 'manifest.json').read_text())['training_exclusions'])
    arrays['keep'] = np.asarray([i not in excluded for i in arrays['ids']])
    return arrays


def structures(arrays: dict, rows=None):
    from pymatgen.core import Lattice, Structure
    offsets = arrays['atom_offsets']
    for i in (range(len(arrays['ids'])) if rows is None else rows):
        lo, hi = offsets[i], offsets[i + 1]
        yield i, Structure(Lattice(arrays['cell'][i].astype(np.float64)),
                           [int(z) for z in arrays['atomic_numbers'][lo:hi]],
                           arrays['frac_coords'][lo:hi].astype(np.float64))


def formula_key(numbers) -> str:
    """Reduced composition key used to propose candidate DOS tasks."""
    from pymatgen.core import Element
    return reduced_from_symbols([Element.from_Z(int(z)).symbol for z in numbers])


def shape(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0, None)
    total = values.sum(-1, keepdims=True)
    return np.divide(values, total, out=np.zeros_like(values), where=total > 0)


def total_variation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return .5 * np.abs(shape(left) - shape(right)).sum(-1)


def box_average(x, y, edges):
    """Vectorised equivalent of tools/data/c2b_grids.box_average (trapezoid mean per bin)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    out = np.zeros(len(edges) - 1)
    cover = np.zeros(len(edges) - 1, dtype=bool)
    lo = np.searchsorted(x, edges[:-1], side='left')
    hi = np.searchsorted(x, edges[1:], side='left')
    for b in np.nonzero(hi > lo)[0]:
        xs, ys = x[lo[b]:hi[b]], y[lo[b]:hi[b]]
        out[b] = np.trapz(ys, xs) / (edges[b + 1] - edges[b]) if len(xs) > 1 else ys[0]
        cover[b] = True
    return out, cover


# ---------------------------------------------------------------- infer

def infer(run_dirs, split: str, out: Path) -> None:
    import torch

    from tools.eval.lg_valid_export import load_run, select_best
    from tools.train import evaluate_split
    from utils.builder import ConfigBuilder
    if split not in SPLITS:
        raise ValueError('inference is limited to train/valid')
    torch.set_num_threads(1)
    summary = {}
    for run_dir in map(Path, run_dirs):
        saved, params, contract = load_run(run_dir)
        picked = select_best(run_dir, contract['selection_metric'], saved)
        samples = out / f'{run_dir.name}_{split}_samples.csv'
        spectra = out / f'{run_dir.name}_{split}_spectra.npz'
        refuse_existing(samples, spectra)
        builder = ConfigBuilder(**saved['config'])
        model = builder.get_model()
        model.model['transformer'].load_state_dict(picked['checkpoint']['model'], strict=True)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        loader = builder.get_dataloader(split=split, dos_minmax=True,
                                        batch_size=saved['cli']['batch_size'], dos_sumnorm=True)
        order = list(loader.sampler)
        ids = [str(value) for value in loader.dataset.material_ids]
        metrics = evaluate_split(model, loader, sumnorm=True, return_sample_level=True,
                                 return_spectra=True, sample_ids=[ids[i] for i in order])
        frame = metrics.pop('sample_df')
        arrays = metrics.pop('sample_spectra')
        if len(frame) != len(ids) or frame.material_id.duplicated().any():
            raise ValueError(f'{split} inference is not a one-to-one partition: {run_dir}')
        if not np.isfinite(frame.select_dtypes('number').to_numpy()).all():
            raise ValueError(f'non-finite metrics: {run_dir}')
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(samples, index=False)
        np.savez(spectra, material_ids=frame.material_id.to_numpy(dtype=str), **arrays)
        summary[run_dir.name] = {
            'split': split, 'samples': len(frame), 'epoch': int(picked['checkpoint']['epoch']),
            'blind_energy_edos_mean': float(frame.blind_energy_edos.mean()),
            'blind_energy_phdos_mean': float(frame.blind_energy_phdos.mean())}
        print(json.dumps({run_dir.name: summary[run_dir.name]}), flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(out / f'infer_{split}_summary.json', summary)


# ---------------------------------------------------------------- geometry

GATE_EXACT_TV = 1e-6
GATE_MAX_CLIPPED_BINS = 3


def label_gate(reconstructed: np.ndarray, target: np.ndarray) -> dict:
    """Is ``target`` this DOS after the frozen per-bin spike winsorisation?

    The stored label equals the box average up to one overall scale (cell size)
    except in at most a few winsorised bins, where the label can only be lower.
    """
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    tv = float(total_variation(reconstructed, target))
    both = (reconstructed > 0) & (target > 0)
    if not both.any():
        empty = not (reconstructed > 0).any() and not (target > 0).any()
        return {'tv': tv, 'scale': np.nan, 'class': 'exact' if empty else 'fail', 'off_bins': 0}
    scale = float(np.median(target[both] / reconstructed[both]))
    tol = 1e-4 * max(float(target.max()), 1e-12) + 1e-6 * np.abs(target)
    off = np.abs(reconstructed * scale - target) > tol
    lowered = target[off] < reconstructed[off] * scale
    if tv <= GATE_EXACT_TV:
        label = 'exact'
    elif off.sum() <= GATE_MAX_CLIPPED_BINS and lowered.all():
        label = 'clipped'
    else:
        label = 'fail'
    return {'tv': tv, 'scale': scale, 'class': label, 'off_bins': int(off.sum())}


def site_symbols(structure: dict) -> list:
    return [max(site['species'], key=lambda s: s['occu'])['element'] for site in structure['sites']]


def reduced_from_symbols(symbols) -> str:
    """Reduced formula key (gcd-reduced element counts); cheaper than pymatgen Composition."""
    from collections import Counter
    from math import gcd
    from functools import reduce
    counts = Counter(symbols)
    divisor = reduce(gcd, counts.values())
    return ' '.join(f'{element}{count // divisor}' for element, count in sorted(counts.items()))


def table_files(table: Path):
    """Active parquet files of a downloaded Delta table, partition values attached."""
    from deltalake import DeltaTable
    from urllib.parse import unquote
    active = {unquote(Path(uri).as_posix().split(table.name + '/', 1)[-1])
              for uri in DeltaTable(str(table)).file_uris()}
    files = []
    for path in sorted(table.rglob('*.parquet')):
        if '_delta_log' in path.parts:
            continue
        rel = unquote(path.relative_to(table).as_posix())
        if rel not in active and not any(rel.endswith(a.split('/')[-1]) for a in active):
            continue
        partition = dict(part.split('=', 1) for part in Path(rel).parts[:-1] if '=' in part)
        files.append((path, partition))
    if not files:
        raise FileNotFoundError(f'no active parquet files under {table}')
    return files


def _scan_edos_file(job):
    """Both passes over one total-DOS file: candidate selection, then E0 reconstruction."""
    import pyarrow.parquet as pq
    path, partition, wanted_formulas, wanted_ids = job
    reader = pq.ParquetFile(path)
    tasks = {}
    for batch in reader.iter_batches(batch_size=256, columns=['identifier', 'structure']):
        for identifier, structure in zip(batch.column('identifier').to_pylist(),
                                         batch.column('structure').to_pylist()):
            if structure is None:
                continue
            formula = reduced_from_symbols(site_symbols(structure))
            if identifier in wanted_ids or formula in wanted_formulas:
                tasks[identifier] = {'formula': formula, 'structure': structure,
                                     'run_type': partition.get('run_type'), 'e0': None}
    for batch in reader.iter_batches(batch_size=128, columns=[
            'identifier', 'energies', 'spin_up_densities', 'spin_down_densities', 'efermi']):
        for row in batch.to_pylist():
            task = tasks.get(row['identifier'])
            if task is None:
                continue
            up = np.asarray(row['spin_up_densities'] or [], dtype=np.float64)
            down = row['spin_down_densities']
            down = (np.asarray(down, dtype=np.float64) if down is not None and np.ndim(down) != 0
                    and len(down) == len(up) else np.zeros_like(up))
            efermi = row['efermi']
            if efermi is None or not np.isfinite(efermi) or not len(up):
                continue
            task['e0'] = box_average(np.asarray(row['energies'], dtype=np.float64) - efermi,
                                     up + down, E0_EDGES)[0]
    return tasks


def scan_edos_tasks(table: Path, wanted_formulas: set, wanted_ids: set, workers: int = 4):
    """E0 reconstruction for every total-DOS task whose reduced formula occurs in the audit."""
    from multiprocessing import Pool
    jobs = [(path, partition, wanted_formulas, wanted_ids) for path, partition in table_files(table)]
    jobs.sort(key=lambda job: -job[0].stat().st_size)
    tasks = {}
    with Pool(min(workers, len(jobs))) as pool:
        for part in pool.imap_unordered(_scan_edos_file, jobs):
            tasks.update(part)
    return tasks


def scan_phonon_tasks(table: Path, wanted_ids: set, wanted_formulas: set):
    """P0 reconstruction for phonon records whose identifier or reduced formula occurs in the audit."""
    import pyarrow.parquet as pq
    records = []
    for path, partition in table_files(table):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                dos = row.get('dos')
                if not dos or not dos.get('structure'):
                    continue
                formula = reduced_from_symbols(site_symbols(dos['structure']))
                if row.get('identifier') not in wanted_ids and formula not in wanted_formulas:
                    continue
                x = np.asarray(dos.get('frequencies') or [], dtype=np.float64)
                y = np.asarray(dos.get('densities') or [], dtype=np.float64)
                if len(x) < 10 or len(x) != len(y):
                    continue
                records.append({'identifier': row.get('identifier'), 'formula': formula,
                                'method': row.get('phonon_method') or partition.get('phonon_method'),
                                'structure': dos['structure'],
                                'p0': box_average(x * THZ2CM, y / THZ2CM, P0_EDGES)[0]})
    return records


def composition_key(numbers) -> tuple:
    z, counts = np.unique(np.asarray(numbers, dtype=int), return_counts=True)
    return tuple((int(a), int(c)) for a, c in zip(z, counts))


def material_table() -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        a = load_split(split)
        offsets = a['atom_offsets']
        for i, material in enumerate(a['ids']):
            if not a['keep'][i]:
                continue
            numbers = a['atomic_numbers'][offsets[i]:offsets[i + 1]]
            rows.append({'material_id': material, 'split': split, 'row': i,
                         'n_atoms': int(a['n_atoms'][i]),
                         'volume_per_atom': float(abs(np.linalg.det(a['cell'][i].astype(np.float64)))
                                                  / a['n_atoms'][i]),
                         'composition': repr(composition_key(numbers)),
                         'formula': formula_key(numbers)})
    return pd.DataFrame(rows)


def _match_one(job):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    material, split, row, structure_dict = job
    a = load_split.cache[split]
    lo, hi = a['atom_offsets'][row], a['atom_offsets'][row + 1]
    from pymatgen.core import Lattice
    left = Structure(Lattice(a['cell'][row].astype(np.float64)),
                     [int(z) for z in a['atomic_numbers'][lo:hi]], a['frac_coords'][lo:hi].astype(np.float64))
    right = Structure.from_dict(structure_dict)
    volume_ratio = (right.volume / len(right)) / (left.volume / len(left))
    strict = StructureMatcher(**STRICT).fit(left, right)
    loose = StructureMatcher().fit(left, right)
    return {'material_id': material, 'calc_n_atoms': len(right), 'calc_volume_ratio': float(volume_ratio),
            'strict_match': bool(strict), 'default_match': bool(loose)}


def _init_worker():
    load_split.cache = {split: load_split(split) for split in SPLITS}


def group_contrast(frame: pd.DataFrame, flag: str, value: str, seed: int = 20261105, draws: int = 2000) -> dict:
    """Within composition groups that contain both classes: mean(value | flag) - mean(value | not flag)."""
    deltas = []
    for _, group in frame.groupby('composition'):
        yes, no = group[group[flag]], group[~group[flag]]
        if len(yes) and len(no):
            deltas.append(yes[value].mean() - no[value].mean())
    deltas = np.asarray(deltas)
    if not len(deltas):
        return {'groups': 0}
    rng = np.random.default_rng(seed)
    boot = deltas[rng.integers(0, len(deltas), (draws, len(deltas)))].mean(1)
    return {'groups': int(len(deltas)), 'mean_delta': float(deltas.mean()),
            'ci95': np.percentile(boot, [2.5, 97.5]).tolist()}


def geometry(delta_dir: Path, energies: Path, out: Path, workers: int, limit: int | None,
             tasks=('edos', 'phdos')) -> None:
    from multiprocessing import Pool
    report_path = out / 'geometry_report.json'
    refuse_existing(report_path)
    materials = material_table()
    if limit:
        materials = materials.groupby('split', group_keys=False).head(limit).reset_index(drop=True)
    arrays = {split: load_split(split) for split in SPLITS}
    wanted_ids = {m.removeprefix('mp-') for m in materials.material_id}
    edos = scan_edos_tasks(delta_dir / 'core/electronic-structure/total-dos',
                           set(materials.formula), wanted_ids, workers) if 'edos' in tasks else {}
    by_formula = {}
    for identifier, task in edos.items():
        if task.get('e0') is not None:
            by_formula.setdefault(task['formula'], []).append(identifier)

    gate_rows, jobs = [], {'edos': [], 'phdos': []}
    for m in (materials.itertuples() if 'edos' in tasks else ()):
        target = arrays[m.split]['edos_tgtdos'][m.row]
        own = m.material_id.removeprefix('mp-')
        candidates = ([own] if own in edos and edos[own].get('e0') is not None else []) + \
            [c for c in by_formula.get(m.formula, []) if c != own]
        best = None
        for c in candidates:
            gate = label_gate(edos[c]['e0'], target)
            if best is None or gate['tv'] < best[1]['tv']:
                best = (c, gate)
            if c == own and gate['class'] != 'fail':
                break
        row = {'material_id': m.material_id, 'task': 'edos', 'candidates': len(candidates)}
        if best is not None:
            row.update(dos_task=best[0], same_id=best[0] == own, run_type=edos[best[0]]['run_type'],
                       gate_tv=best[1]['tv'], gate_class=best[1]['class'], gate_scale=best[1]['scale'],
                       gate_off_bins=best[1]['off_bins'])
            if best[1]['class'] != 'fail':
                jobs['edos'].append((m.material_id, m.split, m.row, edos[best[0]]['structure']))
        else:
            row.update(gate_class='no_candidate')
        gate_rows.append(row)
    del edos

    phonon = scan_phonon_tasks(delta_dir / 'phonon/electronic-structure/dos', wanted_ids,
                               set(materials.formula)) if 'phdos' in tasks else []
    by_id, by_phonon_formula = {}, {}
    for k, record in enumerate(phonon):
        by_id.setdefault(record['identifier'], []).append(k)
        by_phonon_formula.setdefault(record['formula'], []).append(k)
    for m in (materials.itertuples() if 'phdos' in tasks else ()):
        target = arrays[m.split]['phdos_tgtdos'][m.row]
        own = m.material_id.removeprefix('mp-')
        own_records = by_id.get(own, [])
        candidates = own_records + [k for k in by_phonon_formula.get(m.formula, []) if k not in own_records]
        row = {'material_id': m.material_id, 'task': 'phdos', 'candidates': len(candidates)}
        if not candidates:
            row.update(gate_class='no_candidate')
        else:
            k, gate = min(((k, label_gate(phonon[k]['p0'], target)) for k in candidates),
                          key=lambda item: (item[1]['tv'], item[0] not in own_records))
            r = phonon[k]
            row.update(dos_task=r['identifier'], same_id=r['identifier'] == own, run_type=r['method'],
                       gate_tv=gate['tv'], gate_class=gate['class'], gate_scale=gate['scale'],
                       gate_off_bins=gate['off_bins'])
            if gate['class'] != 'fail':
                jobs['phdos'].append((m.material_id, m.split, m.row, r['structure']))
        gate_rows.append(row)
    del phonon
    gates = pd.DataFrame(gate_rows)

    matched = []
    with Pool(workers, initializer=_init_worker) as pool:
        for task, task_jobs in jobs.items():
            for result in pool.imap(_match_one, task_jobs, chunksize=16):
                matched.append({**result, 'task': task})
    matches = pd.DataFrame(matched)
    table = gates.merge(materials, on='material_id').merge(matches, on=['material_id', 'task'], how='left')
    table['volume_dev'] = (table.calc_volume_ratio - 1).abs()
    table['mismatch'] = table.strict_match.eq(False) | table.volume_dev.gt(.01)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'geometry_by_material.csv', index=False)
    import gzip
    for task, task_jobs in jobs.items():
        with gzip.open(out / f'calc_structures_{task}.json.gz', 'wt') as handle:
            json.dump({material: structure for material, _, _, structure in task_jobs}, handle, default=float)

    report = {'scope': 'train/valid default views; MP Delta total-dos and phonon dos tables; no test',
              'gate': {'exact_tv': GATE_EXACT_TV, 'max_clipped_bins': GATE_MAX_CLIPPED_BINS},
              'matcher_strict': STRICT, 'volume_threshold': .01, 'limit_per_split': limit, 'tasks': {}}
    error = pd.read_csv(energies) if energies and Path(energies).exists() else None
    for task in tasks:
        t = table[table.task == task]
        entry = {'gate_counts': {split: t[t.split == split].gate_class.value_counts().to_dict()
                                 for split in SPLITS}}
        passed = t[t.gate_class.isin(['exact', 'clipped'])]
        entry['same_id_share_of_passed'] = float(passed.same_id.mean()) if len(passed) else None
        for split in SPLITS:
            s = passed[passed.split == split]
            entry[split] = {'passed': int(len(s)),
                            'strict_mismatch_rate': float((~s.strict_match.astype(bool)).mean()) if len(s) else None,
                            'volume_dev_gt_1pct_rate': float(s.volume_dev.gt(.01).mean()) if len(s) else None,
                            'mismatch_rate': float(s.mismatch.mean()) if len(s) else None,
                            'volume_ratio_quantiles': s.calc_volume_ratio.quantile([0, .05, .5, .95, 1]).tolist()
                            if len(s) else None}
        if error is not None:
            joined = passed.merge(error[['material_id', f'ensemble_energy_{task}']], on='material_id')
            for split in SPLITS:
                s = joined[joined.split == split].assign(mismatch=lambda f: f.mismatch.astype(bool))
                entry[split]['error_within_composition'] = group_contrast(s, 'mismatch', f'ensemble_energy_{task}')
                entry[split]['error_mean'] = {
                    'mismatch': float(s[s.mismatch][f'ensemble_energy_{task}'].mean()) if s.mismatch.any() else None,
                    'match': float(s[~s.mismatch][f'ensemble_energy_{task}'].mean()) if (~s.mismatch).any() else None}
        report['tasks'][task] = entry
    write_json(report_path, report)
    print(json.dumps(report, indent=1, default=float)[:4000])


# ---------------------------------------------------------------- noise floor

PROTOCOL = ROOT / 'results/runs/spectral_residual_20261030/protocol.json'


def energy_constants() -> dict:
    tasks = json.loads(PROTOCOL.read_text())['tasks']
    return {task: (tasks[task]['bin_width'], tasks[task]['energy']) for task in ('edos', 'phdos')}


def pair_energy(left, right, n_left, n_right, bin_width, energy) -> float:
    """E of using one material's per-atom label as the prediction for the other (symmetric)."""
    return float(bin_width * (((np.asarray(left, np.float64) / n_left
                                - np.asarray(right, np.float64) / n_right) ** 2).sum()) / energy)


def _pair_match(job):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    key, left, right = job
    a, b = Structure.from_dict(left), Structure.from_dict(right)
    return key, bool(StructureMatcher(**STRICT).fit(a, b)), \
        float((b.volume / len(b)) / (a.volume / len(a)))


def noise(geometry_dirs, energies: Path, out: Path, workers: int) -> None:
    import gzip
    from itertools import combinations
    from multiprocessing import Pool
    report_path = out / 'noise_report.json'
    refuse_existing(report_path)
    sources = {}
    for directory in map(Path, geometry_dirs):
        for task in json.loads((directory / 'geometry_report.json').read_text())['tasks']:
            sources[task] = directory  # a later directory overrides an earlier one for its tasks
    model = pd.read_csv(energies).set_index('material_id')
    constants = energy_constants()
    arrays = {split: load_split(split) for split in SPLITS}
    report = {'scope': 'same-absolute-composition pairs within train+valid whose DOS-calculation '
                       'structures match strictly; both labels pass the identity gate; no test',
              'geometry_sources': {task: str(path) for task, path in sources.items()},
              'matcher_strict': STRICT, 'tasks': {}}
    pair_rows = []
    for task in ('edos', 'phdos'):
        with gzip.open(sources[task] / f'calc_structures_{task}.json.gz', 'rt') as handle:
            calc = json.load(handle)
        table = pd.read_csv(sources[task] / 'geometry_by_material.csv')
        passed = table[(table.task == task) & table.gate_class.isin(['exact', 'clipped'])
                       & table.material_id.isin(calc.keys())]
        jobs, meta = [], {}
        for composition, group in passed.groupby('composition'):
            for (_, a), (_, b) in combinations(group.iterrows(), 2):
                key = (a.material_id, b.material_id)
                jobs.append((key, calc[a.material_id], calc[b.material_id]))
                meta[key] = (a, b, composition)
        with Pool(workers) as pool:
            results = pool.map(_pair_match, jobs, chunksize=8)
        bin_width, energy = constants[task]
        for key, strict, volume_ratio in results:
            a, b, composition = meta[key]
            ya = arrays[a.split][f'{task}_tgtdos'][a.row]
            yb = arrays[b.split][f'{task}_tgtdos'][b.row]
            pair_rows.append({
                'task': task, 'left_id': key[0], 'right_id': key[1], 'composition': composition,
                'splits': f'{a.split}-{b.split}', 'calc_strict_match': strict, 'calc_volume_ratio': volume_ratio,
                'input_strict_match_left': a.strict_match, 'input_strict_match_right': b.strict_match,
                'run_type_left': a.run_type, 'run_type_right': b.run_type,
                'target_pair_energy': pair_energy(ya, yb, a.n_atoms, b.n_atoms, bin_width, energy),
                'target_shape_tv': float(total_variation(ya, yb)),
                'model_energy_left': float(model.loc[key[0], f'ensemble_energy_{task}']),
                'model_energy_right': float(model.loc[key[1], f'ensemble_energy_{task}'])})
    pairs = pd.DataFrame(pair_rows)
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / 'noise_pairs.csv', index=False)
    for task in ('edos', 'phdos'):
        t = pairs[pairs.task == task]
        near = t[t.calc_strict_match]
        same_setting = near[near.run_type_left == near.run_type_right]
        entry = {'pairs': int(len(t)), 'strict_calc_pairs': int(len(near)),
                 'strict_calc_pairs_same_run_type': int(len(same_setting))}
        for name, s in (('strict', near), ('strict_same_run_type', same_setting), ('all_pairs', t)):
            if not len(s):
                continue
            model_e = np.concatenate([s.model_energy_left, s.model_energy_right])
            entry[name] = {
                'groups': int(s.composition.nunique()),
                'target_pair_energy_quantiles': s.target_pair_energy.quantile([.1, .5, .9]).tolist(),
                'floor_estimate_median': float(s.target_pair_energy.median() / 2),
                'target_shape_tv_quantiles': s.target_shape_tv.quantile([.1, .5, .9]).tolist(),
                'model_energy_quantiles': np.quantile(model_e, [.1, .5, .9]).tolist(),
                'floor_over_model_median': float(s.target_pair_energy.median() / 2 / np.median(model_e)),
                'pair_energy_median_ge_model_median': bool(s.target_pair_energy.median() >= np.median(model_e))}
        report['tasks'][task] = entry
    report['floor_note'] = ('if two labels are independent noisy draws around one structure-determined '
                            'spectrum, their expected pair E is twice the per-label noise; floor = median/2')
    write_json(report_path, report)
    print(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- descriptor baseline

ELEMENT_PROPERTIES = ('atomic_weight', 'covalent_radius_cordero', 'en_pauling', 'ionization_1',
                      'electron_affinity', 'group_id', 'period', 'mendeleev_number', 'dipole_polarizability')
RDF_EDGES = np.arange(.5, 6.0001, .25)
COS_EDGES = np.linspace(-1, 1, 11)


def element_table() -> pd.DataFrame:
    table = pd.read_csv(ROOT / 'data/reference/atom_features_mendeleev.csv').set_index('Z')
    table = table[list(ELEMENT_PROPERTIES)].astype(float)
    return table.fillna(table.median())


def composition_features(numbers, elements: pd.DataFrame) -> np.ndarray:
    z, counts = np.unique(np.asarray(numbers, dtype=int), return_counts=True)
    weights = counts / counts.sum()
    props = elements.loc[z].to_numpy()
    mean = weights @ props
    std = np.sqrt(weights @ (props - mean) ** 2)
    fractions = np.zeros(100)
    fractions[np.minimum(z, 100) - 1] = weights
    return np.concatenate([fractions, mean, std, props.min(0), props.max(0), [len(z)]])


def structure_features(numbers, cell, frac, elements: pd.DataFrame) -> np.ndarray:
    """Radial, property-weighted radial, coordination and bond-angle histograms of one cell."""
    from pymatgen.core import Lattice, Structure
    structure = Structure(Lattice(cell.astype(np.float64)), [int(z) for z in numbers], frac.astype(np.float64))
    n = len(structure)
    en = elements.loc[np.asarray(numbers, dtype=int), 'en_pauling'].to_numpy()
    rdf = np.zeros(len(RDF_EDGES) - 1)
    en_rdf = np.zeros(len(RDF_EDGES) - 1)
    angles = np.zeros(len(COS_EDGES) - 1)
    nearest, coordination = [], []
    for i, neighbors in enumerate(structure.get_all_neighbors(6.0)):
        if not neighbors:
            nearest.append(6.0)
            coordination.append(0)
            continue
        d = np.array([nb.nn_distance for nb in neighbors])
        idx = np.array([nb.index for nb in neighbors])
        rdf += np.histogram(d, RDF_EDGES)[0]
        en_rdf += np.histogram(d, RDF_EDGES, weights=np.abs(en[idx] - en[i]))[0]
        dmin = d.min()
        nearest.append(dmin)
        shell = np.nonzero(d <= 1.2 * dmin)[0]
        coordination.append(len(shell))
        if len(shell) > 1:
            vectors = np.array([neighbors[k].coords - structure[i].coords for k in shell])
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            cos = (vectors @ vectors.T)[np.triu_indices(len(shell), 1)]
            angles += np.histogram(np.clip(cos, -1, 1), COS_EDGES)[0] / max(len(cos), 1)
    shells = 4 * np.pi * RDF_EDGES[1:] ** 2 * np.diff(RDF_EDGES)
    density = n / structure.volume
    rdf = rdf / n / shells / density
    en_rdf = en_rdf / n / shells / density
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        number = SpacegroupAnalyzer(structure, symprec=.1).get_space_group_number()
    except Exception:  # noqa: BLE001 - symmetry search failure leaves the label unknown
        number = 0
    system = np.zeros(8)
    system[np.searchsorted([0, 2, 15, 74, 142, 167, 194, 230], number)] = 1
    return np.concatenate([[structure.volume / n, structure.density, n, np.mean(nearest), np.std(nearest),
                            np.mean(coordination), np.std(coordination)], rdf, en_rdf, angles / n, system])


def _features_one(job):
    split, row = job
    a = load_split.cache[split]
    lo, hi = a['atom_offsets'][row], a['atom_offsets'][row + 1]
    numbers = a['atomic_numbers'][lo:hi]
    elements = load_split.elements
    return (composition_features(numbers, elements),
            structure_features(numbers, a['cell'][row], a['frac_coords'][lo:hi], elements))


def _init_feature_worker():
    _init_worker()
    load_split.elements = element_table()


def baseline(lr2_valid, out: Path, workers: int, limit: int | None) -> None:
    from multiprocessing import Pool

    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    from tools.eval.lg_structure_diagnosis import paired_group_interval, pair_errors, summarize_pairs
    report_path = out / 'baseline_report.json'
    refuse_existing(report_path)
    out.mkdir(parents=True, exist_ok=True)
    data, features = {}, {}
    for split in SPLITS:
        a = load_split(split)
        rows = np.nonzero(a['keep'])[0]
        if limit:
            rows = rows[:limit]
        data[split] = {'rows': rows, 'ids': a['ids'][rows],
                       'keys': [repr(composition_key(a['atomic_numbers'][a['atom_offsets'][r]:a['atom_offsets'][r + 1]]))
                                for r in rows],
                       **{task: shape(a[f'{task}_tgtdos'][rows]) for task in ('edos', 'phdos')}}
        cache = out / f'features_{split}.npz'
        if cache.exists():
            loaded = np.load(cache)
            if not np.array_equal(loaded['ids'], data[split]['ids']):
                raise ValueError(f'feature cache rows differ: {cache}')
            features[split] = (loaded['composition'], loaded['structure'])
            continue
        with Pool(workers, initializer=_init_feature_worker) as pool:
            computed = pool.map(_features_one, [(split, int(r)) for r in rows], chunksize=32)
        features[split] = (np.stack([c for c, _ in computed]), np.stack([s for _, s in computed]))
        np.savez(cache, ids=data[split]['ids'], composition=features[split][0], structure=features[split][1])

    lr2 = {}
    if lr2_valid:
        shapes = {'edos': [], 'phdos': []}
        for path in lr2_valid:
            arrays = np.load(path)
            order = pd.Index(arrays['material_ids'].astype(str)).get_indexer(data['valid']['ids'])
            if (order < 0).any():
                raise ValueError(f'LR2 export misses valid materials: {path}')
            for task in shapes:
                shapes[task].append(arrays[f'pred_shape_{task}'][order].astype(np.float64))
        lr2 = {task: shape(np.mean(values, 0)) for task, values in shapes.items()}

    report = {'scope': 'descriptor regressors fitted on train only, evaluated on valid; SumNorm shapes',
              'features': {'composition': int(features['train'][0].shape[1]),
                           'structure': int(features['train'][1].shape[1])},
              'pca_components': 32, 'limit_per_split': limit, 'tasks': {}}
    predictions = {}
    for task in ('edos', 'phdos'):
        y_train, y_valid = data['train'][task], data['valid'][task]
        nonzero = y_train.sum(1) > 0
        pca = PCA(32, random_state=0).fit(y_train[nonzero])
        z_train = pca.transform(y_train)
        predictions[task] = {}
        for variant in ('composition', 'composition_structure'):
            def stack(split):
                c, s = features[split]
                return c if variant == 'composition' else np.concatenate([c, s], 1)
            scaler = StandardScaler().fit(stack('train'))
            x_train, x_valid = scaler.transform(stack('train')), scaler.transform(stack('valid'))
            knn = KNeighborsRegressor(10, weights='distance').fit(x_train, y_train)
            predictions[task][f'knn_{variant}'] = shape(knn.predict(x_valid))
            z_valid = np.stack([HistGradientBoostingRegressor(max_iter=300, learning_rate=.05,
                                                              random_state=0).fit(x_train, z_train[:, k])
                                .predict(x_valid) for k in range(z_train.shape[1])], 1)
            predictions[task][f'gbm_{variant}'] = shape(pca.inverse_transform(z_valid))
        if lr2:
            predictions[task]['lr2_ensemble'] = lr2[task]
        keys, ids = data['valid']['keys'], data['valid']['ids']
        zero = pair_errors(np.zeros_like(y_valid), y_valid, keys, ids)
        frames = {name: pair_errors(values, y_valid, keys, ids) for name, values in predictions[task].items()}
        entry = {'zero_response': summarize_pairs(zero)}
        for name, values in predictions[task].items():
            frame = frames[name]
            item = {'shape_tv_mean': float(total_variation(values, y_valid).mean()),
                    'pairs': summarize_pairs(frame),
                    'minus_zero': paired_group_interval(zero, frame),
                    'predicted_over_target_tv_median': float(frame.predicted_tv.median() / frame.target_tv.median())}
            if 'lr2_ensemble' in frames and name != 'lr2_ensemble':
                item['minus_lr2_ensemble'] = paired_group_interval(frames['lr2_ensemble'], frame)
            entry[name] = item
        for model in ('knn', 'gbm'):
            entry[f'{model}_structure_gain_shape_tv'] = (entry[f'{model}_composition']['shape_tv_mean']
                                                         - entry[f'{model}_composition_structure']['shape_tv_mean'])
        report['tasks'][task] = entry
        pd.concat([f.assign(predictor=name) for name, f in frames.items()]).to_csv(
            out / f'baseline_pairs_{task}.csv', index=False)
    write_json(report_path, report)
    print(json.dumps(report, indent=1, default=float)[:6000])


# ---------------------------------------------------------------- ensemble energies

def ensemble_energies(pairs, out: Path) -> None:
    """Per-material blind E of the pointwise mean of several exported models.

    Each member's per-material E is ``scale * sum((pred - target)**2)``; the scale
    (bin width, atom count, train constant) is recovered from the members and must agree.
    """
    frames = []
    for split, members in pairs.items():
        loaded = []
        for samples, spectra in members:
            frame = pd.read_csv(samples)
            arrays = np.load(spectra)
            ids = arrays['material_ids'].astype(str)
            if not np.array_equal(ids, frame.material_id.to_numpy(dtype=str)):
                raise ValueError(f'sample and spectra order differ: {spectra}')
            loaded.append((frame.set_index('material_id'), arrays, ids))
        ids = loaded[0][2]
        result = pd.DataFrame({'material_id': ids, 'split': split})
        for task in ('edos', 'phdos'):
            target = loaded[0][1][f'tgt_{task}']
            scales, preds = [], []
            for frame, arrays, member_ids in loaded:
                order = pd.Index(member_ids).get_indexer(ids)
                if (order < 0).any():
                    raise ValueError('ensemble members cover different materials')
                pred = arrays[f'pred_blind_{task}'][order]
                if not np.allclose(arrays[f'tgt_{task}'][order], target, atol=1e-7):
                    raise ValueError('ensemble members carry different targets')
                squared = ((pred.astype(np.float64) - target) ** 2).sum(1)
                energy = frame.loc[ids, f'blind_energy_{task}'].to_numpy()
                scales.append(np.divide(energy, squared, out=np.full_like(energy, np.nan), where=squared > 0))
                preds.append(pred.astype(np.float64))
                result[f'member{len(preds)}_energy_{task}'] = energy
            scale = np.nanmedian(np.stack(scales), 0)
            spread = np.nanmax(np.abs(np.stack(scales) / scale - 1))
            if spread > 1e-4:
                raise ValueError(f'member energy scales disagree ({spread:.2e}) for {task}')
            result[f'ensemble_energy_{task}'] = scale * ((np.mean(preds, 0) - target) ** 2).sum(1)
        frames.append(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(out, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('infer')
    p.add_argument('--run-dir', type=Path, action='append', required=True)
    p.add_argument('--split', choices=SPLITS, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('ensemble')
    p.add_argument('--member', nargs=3, action='append', metavar=('SPLIT', 'SAMPLES', 'SPECTRA'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('geometry')
    p.add_argument('--delta', type=Path, default=ROOT / 'tools/data/raw/delta_mp')
    p.add_argument('--energies', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='materials per split, for the pre-check only')
    p.add_argument('--tasks', default='edos,phdos')
    p = sub.add_parser('noise')
    p.add_argument('--geometry', type=Path, action='append', required=True)
    p.add_argument('--energies', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('baseline')
    p.add_argument('--lr2-valid', type=Path, action='append', help='LR2 valid spectra exports to average')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='materials per split, for the pre-check only')
    args = parser.parse_args()
    if args.command == 'noise':
        noise(args.geometry, args.energies, args.out, args.workers)
    elif args.command == 'baseline':
        baseline(args.lr2_valid, args.out, args.workers, args.limit)
    elif args.command == 'infer':
        infer(args.run_dir, args.split, args.out)
    elif args.command == 'ensemble':
        members = {}
        for split, samples, spectra in args.member:
            members.setdefault(split, []).append((Path(samples), Path(spectra)))
        refuse_existing(args.out)
        ensemble_energies(members, args.out)
    elif args.command == 'geometry':
        geometry(args.delta, args.energies, args.out, args.workers, args.limit, tuple(args.tasks.split(',')))


if __name__ == '__main__':
    main()
