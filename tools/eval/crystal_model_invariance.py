"""Frozen LR2 E+S output checks on native frames and equivalent cells; no labels."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from model.transformer import Transformer
from tools.eval.lg_valid_export import load_run, select_best
from tools.eval.periodic_geometry_acceptance import TOLERANCES, compare_records, e3_operations
from utils.b7_cif_inference import load_b7_grids
from utils.crystal import CrystalGeometry
from utils.crystal_cache import PackedCrystalStore
from utils.paths import file_sha256, input_path
from utils.spectral_balance import reconstruct_balanced_spectra
from utils.zproj_alignment import state_hash

DEFAULT_RUN = ROOT / 'results/runs/ablation_m1_sbal_lr2es20261030_lr2_s42_e35'
OUTPUT_TOLERANCES = {'float32': {'atol': 2e-6, 'rtol': 2e-5},
                     'float64': {'atol': 1e-10, 'rtol': 1e-8}}
EXTRA_CASES = ('repeat', 'permutation', 'integer_images', 'origin_wrapped',
               'basis_swap', 'basis_shear', 'supercell_211', 'supercell_121', 'supercell_112')
CASE_NAMES = tuple(name for name, _, _ in e3_operations()) + EXTRA_CASES


@dataclass(frozen=True)
class CrystalCase:
    numbers: torch.Tensor
    geometry: CrystalGeometry
    origin: np.ndarray
    images: np.ndarray
    basis: np.ndarray
    orthogonal: np.ndarray
    receiver_copy: np.ndarray


def transform_case(numbers, geometry, name):
    """Retain an exact integer map back to the original receiver's neighborhood."""
    if numbers.ndim != 1 or geometry.cell.shape != (3, 3) or geometry.frac_coords.shape != (len(numbers), 3):
        raise ValueError('one complete unpadded crystal required')
    if name not in CASE_NAMES:
        raise ValueError(f'unknown physical check: {name}')
    cell, frac = (value.detach().cpu().double().numpy() for value in geometry)
    size = len(numbers)
    origin, images = np.arange(size), np.zeros((size, 3), dtype=np.int64)
    copies, basis, orthogonal = np.zeros(size, dtype=int), np.eye(3, dtype=int), np.eye(3)
    changed = frac.copy()
    operations = {key: (q, t) for key, q, t in e3_operations()}
    if name in operations:
        orthogonal, translation = operations[name]
        changed += translation @ np.linalg.inv(cell @ orthogonal)
    elif name == 'permutation':
        origin = np.random.default_rng(42).permutation(size)
        changed = frac[origin]
    elif name == 'integer_images':
        images = np.random.default_rng(42).integers(-2, 3, size=(size, 3))
        changed += images
    elif name == 'origin_wrapped':
        changed += np.array([.371, -.229, .613])
        images = -np.floor(changed).astype(np.int64)
        changed += images
    elif name in ('basis_swap', 'basis_shear'):
        basis = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) if name == 'basis_swap' else np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        changed = frac @ np.linalg.inv(basis)
        offsets = -np.floor(changed).astype(np.int64)
        changed += offsets
        images = offsets @ basis
    elif name.startswith('supercell_'):
        axis = {'supercell_211': 0, 'supercell_121': 1, 'supercell_112': 2}[name]
        basis[axis, axis] = 2
        origin = np.tile(origin, 2)
        copies = np.repeat([0, 1], size)
        images = np.zeros((2*size, 3), dtype=np.int64)
        images[size:, axis] = 1
        changed = (frac[origin] + images) @ np.linalg.inv(basis)
    index = torch.as_tensor(origin, device=numbers.device)
    return CrystalCase(numbers[index], CrystalGeometry(
        torch.as_tensor(basis @ cell @ orthogonal, device=geometry.device, dtype=geometry.dtype),
        torch.as_tensor(changed, device=geometry.device, dtype=geometry.dtype)),
        origin, images, basis, orthogonal, copies)


def record_arrays(records):
    return {'keys': np.column_stack([value.detach().cpu().numpy() for value in
            (records.batch, records.dst, records.src, records.shifts)]).astype(np.int64),
            'r': records.vectors.detach().cpu().double().numpy(),
            'd': records.distances.detach().cpu().double().numpy()}


def mapped_geometry_report(reference, actual, case):
    keys = actual['keys'].copy()
    dst, src = keys[:, 1].copy(), keys[:, 2].copy()
    keys[:, 3:] = keys[:, 3:] @ case.basis + case.images[src] - case.images[dst]
    keys[:, 1], keys[:, 2] = case.origin[dst], case.origin[src]
    expected = {**reference, 'r': reference['r'] @ case.orthogonal}
    result = []
    for copy in np.unique(case.receiver_copy):
        chosen = case.receiver_copy[dst] == copy
        mapped = {'keys': keys[chosen], 'r': actual['r'][chosen], 'd': actual['d'][chosen]}
        try:
            row = compare_records(expected, mapped, tolerance=TOLERANCES[case.geometry.dtype])
        except AssertionError as error:
            row = {'status': 'fail', 'reason': str(error)}
        result.append({'receiver_copy': int(copy), **row})
    return {'status': 'fail' if any(r['status'] == 'fail' for r in result) else
            'boundary_limited' if any(r['status'] == 'boundary_limited' for r in result) else 'pass',
            'reference_records': len(reference['d']), 'actual_records': len(actual['d']), 'copies': result}


def physical_outputs(outputs, n_atoms, protocol):
    spectra = reconstruct_balanced_spectra(outputs, [n_atoms], protocol)
    result = {}
    for i, task in enumerate(('edos', 'phdos')):
        result[f'shape_{task}'] = spectra[f'shape_{task}'].detach().cpu().double().numpy()
        result[f'mass_{task}'] = (F.softplus(outputs['mass_logits'][:, i:i+1]) *
                                 protocol['tasks'][task]['mass_per_atom_reference']).detach().cpu().double().numpy()
        result[f'per_atom_{task}'] = (spectra[f'blind_{task}'] / n_atoms).detach().cpu().double().numpy()
        result[f'total_{task}'] = spectra[f'blind_{task}'].detach().cpu().double().numpy()
    return result


def output_comparison(reference, actual, n_atoms, multiplier, protocol, dtype):
    tolerance = OUTPUT_TOLERANCES[str(dtype).split('.')[-1]]
    rows = {}
    for task in ('edos', 'phdos'):
        entry = protocol['tasks'][task]
        for kind in ('shape', 'mass', 'per_atom', 'total'):
            key = f'{kind}_{task}'
            expected = reference[key] * (multiplier if kind == 'total' else 1)
            scale = {'shape': 1., 'mass': entry['mass_per_atom_reference'],
                     'per_atom': entry['mass_per_atom_reference']/entry['bin_width'],
                     'total': n_atoms*multiplier*entry['mass_per_atom_reference']/entry['bin_width']}[kind]
            left, right = expected/scale, actual[key]/scale
            difference = abs(left-right)
            allowed = tolerance['atol'] + tolerance['rtol'] * np.maximum(abs(left), abs(right))
            finite = bool(np.isfinite(actual[key]).all() and np.isfinite(expected).all())
            rows[key] = {'passed': bool(finite and (difference <= allowed).all()),
                         'max_abs_error': float(abs(expected-actual[key]).max()),
                         'max_normalized_error': float(difference.max()),
                         'max_tolerance_ratio': float((difference/allowed).max())}
    return {'passed': all(row['passed'] for row in rows.values()), 'fields': rows}


def select_train_rows(store, count=16):
    """Selection uses only complete atom counts, lattice shape, and seed 42."""
    excluded = set(store.manifest.get('training_exclusions', []))
    allowed = np.array([i for i, value in enumerate(store.material_ids) if str(value) not in excluded])
    if count < 4 or count > len(allowed):
        raise ValueError('at least four available train structures required')
    counts = store.n_atoms.numpy()
    ordered = allowed[np.argsort(counts[allowed], kind='stable')]
    chosen = list(dict.fromkeys(ordered[np.linspace(0, len(ordered)-1, min(10, count)).astype(int)].tolist()))
    condition = np.linalg.cond(store.cell.numpy().astype(float))
    if len(chosen) < count:
        chosen.append(int(allowed[np.argmax(condition[allowed])]))
    chosen = list(dict.fromkeys(chosen))
    for row in np.random.default_rng(42).permutation(allowed):
        if len(chosen) == count:
            break
        if int(row) not in chosen:
            chosen.append(int(row))
    return chosen


def policy():
    return {'schema': 'crystal-model-invariance-policy-v1', 'checkpoint': str(DEFAULT_RUN),
            'selected_epoch': 29, 'real_train_structures': 16, 'seed': 42,
            'synthetic_cells_A': [5.99998, 6., 6.00002], 'cases': list(CASE_NAMES),
            'dtypes': ['float32', 'float64'], 'output_tolerances': OUTPUT_TOLERANCES,
            'normalization': 'shape:1; mass:m_ref; per_atom:m_ref/dx; total:N_changed*m_ref/dx',
            'geometry_tolerances_A': {str(k): v for k, v in TOLERANCES.items()},
            'cutoff': 'strict d<6 retained; changed boundary topology and all output violations reported',
            'primary': 'both tasks shape, per-atom mass/density, and total scaled by atom multiplier',
            'cache': 'main sweep uncached; first real structure repeated with cold/warm production cache',
            'scope': 'finite frozen-model consistency checks; no training, labels, valid/test inference, or accuracy claim'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def run(out, *, max_wall_s=900):
    out = input_path(out)
    if out.exists():
        raise FileExistsError(f'refusing existing audit directory: {out}')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('this frozen full-model run requires the measured CUDA device')
    started = time.monotonic()
    saved, params, contract = load_run(DEFAULT_RUN)
    picked = select_best(DEFAULT_RUN, contract['selection_metric'], saved)
    if picked['checkpoint']['epoch'] != 29 or params['local_route'] != 'LR2' or params['local_layout'] != 'staged':
        raise ValueError('audit must bind the adopted staged LR2 E+S ep29')
    data = input_path(saved['config']['dataset']['train']['data_dir'])
    manifest = json.loads((data/'manifest.json').read_text())
    if file_sha256(data/'manifest.json') != picked['checkpoint']['data_contract']['manifest_sha256']:
        raise ValueError('data manifest differs from the selected checkpoint')
    input_paths = [DEFAULT_RUN/'config_used.yaml', DEFAULT_RUN/'checkpoint_best.pth', data/'manifest.json',
                   input_path(saved['cli']['spectral_balance_config']), ROOT/'data/reference/grids.json',
                   ROOT/'results/reports/history_m1_sbal_lr2es20261030_lr2_s42_e35.csv']
    for name in ('cell', 'frac_coords', 'atomic_numbers', 'atom_offsets', 'n_atoms', 'train_index'):
        relative = f'train/{name}_train.npy' if name != 'train_index' else 'train/train_index.npy'
        path = data/relative
        if file_sha256(path) != manifest['files'][relative]['sha256']:
            raise ValueError(f'train structure payload differs: {relative}')
        input_paths.append(path)
    input_hashes = {str(p): file_sha256(p) for p in input_paths}
    store = PackedCrystalStore(data, 'train')
    rows = select_train_rows(store)
    samples = [(str(store.material_ids[row]), *store[row]) for row in rows]
    for length in policy()['synthetic_cells_A']:
        samples.append((f'synthetic_cubic_{length:.5f}', torch.tensor([13]),
                        CrystalGeometry(torch.eye(3, dtype=torch.float64)*length,
                                        torch.zeros(1, 3, dtype=torch.float64))))
    sources = [p for directory in ('model', 'utils') for p in (ROOT/directory).rglob('*.py')]
    sources += [Path(__file__), ROOT/'tools/eval/lg_valid_export.py', ROOT/'tools/eval/periodic_geometry_acceptance.py']
    out.mkdir(parents=True)
    write_json(out/'protocol.json', {**policy(), 'input_sha256': input_hashes,
               'source_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in sources},
               'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'selected_train_rows': rows, 'selected_ids': [mid for mid, _, _ in samples[:len(rows)]]})
    for source in sources:
        dest = out/'source_snapshot'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    weights = picked['checkpoint']['model']
    protocol = contract['spectral_balance']
    del picked
    grid_values = load_b7_grids()
    # Match production grid quantization, including in the FP64 reference run.
    grids = [torch.as_tensor(value, dtype=torch.float32) for value in grid_values]
    comparisons, arrays, state_checks, cache_checks = [], {}, {}, []
    with (out/'comparisons.jsonl').open('w') as stream:
        for dtype in (torch.float32, torch.float64):
            name = str(dtype).split('.')[-1]
            model = Transformer(**params).to(device='cuda', dtype=dtype).eval()
            model.load_state_dict(weights, strict=True)
            model.requires_grad_(False)
            before = state_hash(model.state_dict())
            coordinates = [grid.to(device='cuda', dtype=dtype) for grid in grids]
            captured = []
            handle = model.encoder.register_forward_pre_hook(lambda module, args: captured.append(record_arrays(args[1])))

            def predict(numbers, geometry, transformer=model):
                if time.monotonic()-started > max_wall_s:
                    raise TimeoutError('frozen audit wall budget exceeded; no automatic rerun')
                numbers = numbers.to('cuda')
                geometry = geometry.to(device='cuda', dtype=dtype)
                captured.clear()
                with torch.inference_mode():
                    outputs = transformer(numbers[None], numbers[None].eq(0),
                                    CrystalGeometry(geometry.cell[None], geometry.frac_coords[None]), *coordinates)
                    result = physical_outputs(outputs, len(numbers), protocol)
                if len(captured) != 1:
                    raise RuntimeError('actual Encoder records were not captured exactly once')
                return result, captured[0]

            try:
                for sample_index, (mid, numbers, source_geometry) in enumerate(samples):
                    geometry = source_geometry.to(dtype=dtype)
                    base, baseline_records = predict(numbers, geometry)
                    for field, values in base.items():
                        arrays[f'{name}_{sample_index}_baseline_{field}'] = values
                    for operation in CASE_NAMES:
                        case = transform_case(numbers, geometry, operation)
                        actual, records = predict(case.numbers, case.geometry)
                        comparison = output_comparison(base, actual, len(numbers), len(case.numbers)/len(numbers), protocol, dtype)
                        geometry_report = mapped_geometry_report(baseline_records, records, case)
                        row = {'material_id': mid, 'sample_index': sample_index, 'synthetic': mid.startswith('synthetic_'),
                               'dtype': name, 'operation': operation, 'n_atoms': len(numbers),
                               'changed_atoms': len(case.numbers), 'geometry': geometry_report, 'output': comparison}
                        comparisons.append(row)
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        stream.flush()
                        for field, values in actual.items():
                            arrays[f'{name}_{sample_index}_{operation}_{field}'] = values
                    print(json.dumps({'dtype': name, 'sample': sample_index+1, 'of': len(samples),
                                      'material': mid, 'seconds': round(time.monotonic()-started, 1)}), flush=True)
                    if sample_index == 0:
                        model.enable_local_geometry_cache(enabled=True)
                        for operation in ('repeat', 'rotation_1', 'supercell_211'):
                            case = transform_case(numbers, geometry, operation)
                            for temperature in ('cold', 'warm'):
                                actual, _ = predict(case.numbers, case.geometry)
                                cache_checks.append({'dtype': name, 'operation': operation, 'cache': temperature,
                                    **output_comparison(base, actual, len(numbers), len(case.numbers)/len(numbers), protocol, dtype)})
                        assert model._local_geometry_cache.hits >= 3
                        model.enable_local_geometry_cache(enabled=False)
            finally:
                handle.remove()
            state_checks[name] = {'before': before, 'after': state_hash(model.state_dict())}
            if state_checks[name]['before'] != state_checks[name]['after']:
                raise RuntimeError('frozen inference mutated model tensors')
            del predict, model
            torch.cuda.empty_cache()
    for path, digest in input_hashes.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f'audit input changed: {path}')
    np.savez_compressed(out/'predictions.npz', **arrays)
    failures = [row for row in comparisons if not row['output']['passed'] or row['geometry']['status'] == 'fail']
    stable_failures = [row for row in failures if row['geometry']['status'] != 'boundary_limited']
    result = {'schema': 'crystal-model-invariance-result-v1', 'completed': True,
              'overall_pass': not failures and all(row['passed'] for row in cache_checks),
              'stable_topology_pass': not stable_failures, 'comparison_cases': len(comparisons),
              'real_structures': len(rows), 'synthetic_structures': len(samples)-len(rows),
              'cases_by_dtype': dict(Counter(row['dtype'] for row in comparisons)),
              'boundary_topology_cases': sum(row['geometry']['status'] == 'boundary_limited' for row in comparisons),
              'output_failures': len([row for row in comparisons if not row['output']['passed']]),
              'stable_failures': len(stable_failures), 'failures': failures, 'cache_checks': cache_checks,
              'state_checks': state_checks, 'seconds': time.monotonic()-started,
              'peak_allocated_MiB': torch.cuda.max_memory_allocated()/1024**2,
              'gpu': torch.cuda.get_device_name(0), 'dtype_note': 'FP64 casts the stored FP32 geometry and learned values; no new physical precision',
              'labels_read': False, 'test_evaluated': False, 'trained': False,
              'limitations': 'finite cases; pass does not prove all crystals, continuity at a hard cutoff, accuracy, or structural-difference generalization',
              'artifact_sha256': {p.name: file_sha256(p) for p in (out/'protocol.json', out/'comparisons.jsonl', out/'predictions.npz')}}
    write_json(out/'report.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('failures', 'cache_checks')}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.out)


if __name__ == '__main__':
    main()
