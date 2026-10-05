"""Compare no-update full-model cost on fixed, eligible structural train batches.

The reference is authenticated Git source, never a checkpoint. Labels are not
loaded. Timings include geometry and profiler gradient diagnostics, not an
optimizer or a complete training epoch.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from model.periodic_messages import PeriodicLocalMessage
from model.transformer import Transformer
from tools.checks.check_crystal_input import profile
from tools.checks.check_local_global import _WithSpectralGrids
from utils.b7_cif_inference import load_b7_grids
from utils.bucket_batch import automatic_pair_budget, crystal_collate, split_crystal_batch
from utils.crystal_cache import PackedCrystalStore
from utils.local_global_experiment import initialize_comparison_model
from utils.paths import CRYSTAL_Q1_DATA_ROOT, file_sha256
from utils.periodic_geometry import build_periodic_records
from utils.zproj_alignment import state_hash

REFERENCE_COMMIT = '553c64d'


def reference_module(commit=REFERENCE_COMMIT):
    revision = subprocess.check_output(['git', 'rev-parse', '--verify', f'{commit}^{{commit}}'], cwd=ROOT).decode().strip()
    source = subprocess.check_output(['git', 'show', f'{revision}:model/periodic_messages.py'], cwd=ROOT)
    digest = hashlib.sha256(source).hexdigest()
    name = f'_periodic_message_reference_{digest[:12]}'
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__file__ = f'git:{revision}:model/periodic_messages.py'
        sys.modules[name] = module  # dataclasses resolve the defining module.
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return sys.modules[name], {'commit': revision, 'source_sha256': digest}


def structural_batches(store, sizes):
    excluded = set(store.manifest['training_exclusions'])
    eligible = [i for i, name in enumerate(store.material_ids) if str(name) not in excluded]
    order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(0)).tolist()
    selected = [eligible[i] for i in order]
    largest = max(eligible, key=lambda i: int(store.n_atoms[i]))
    batches = {}
    for size in sizes:
        if size < 1 or size > len(eligible):
            raise ValueError('batch sizes must fit the eligible train view')
        batches[f'random_b{size}'] = selected[:size]
        if size == 32:
            batches['tail_b32'] = [i for i in selected if i != largest][:31] + [largest]
    return batches


def forward_profile(model, chunks, device, repeats):
    def run():
        with torch.no_grad():
            for atoms, geometry in chunks:
                model(atoms, atoms.eq(0), geometry)
    for _ in range(2):
        run()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        run()
        torch.cuda.synchronize(device)
        times.append(1000 * (time.perf_counter() - start))
    return {'median_ms': float(np.median(times)), 'timings_ms': times,
            'peak_allocated_MiB': torch.cuda.max_memory_allocated(device) / 1024**2}


def verify(out, *, precision='fp32'):
    """Full-model outputs and all registered gradients against Git source."""
    from tools.eval.periodic_message_gpu_check import compare_gradients

    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing existing report: {out}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    dtype = {'fp32': torch.float32, 'fp64': torch.float64}[precision]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-10
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    old, identity = reference_module()
    store = PackedCrystalStore(CRYSTAL_Q1_DATA_ROOT, 'train')
    definitions = structural_batches(store, [32])
    cases = {}
    for label, indices in definitions.items():
        raw = crystal_collate([store[i] for i in indices])
        cases[label] = [[c[0].to(device), c[1].to(device=device, dtype=dtype)]
                        for c in split_crystal_batch(raw, automatic_pair_budget(raw))]
    grids = tuple(torch.as_tensor(grid, device=device, dtype=dtype) for grid in load_b7_grids())
    report = {'precision': precision, 'hardware': torch.cuda.get_device_name(device),
              'reference': identity, 'updates': 0, 'labels_read': False,
              'source_sha256': {name: file_sha256(ROOT / name) for name in (
                  'model/periodic_messages.py', 'model/transformer.py', 'model/local_global_encoder.py',
                  'tools/checks/check_local_message_speed.py')}, 'cases': definitions, 'arms': []}

    def outputs_and_grads(model, chunks):
        wrapped = _WithSpectralGrids(model, grids).eval()
        wrapped.zero_grad(set_to_none=True)
        outputs = {}
        total = sum(len(c[0]) for c in chunks)
        for atoms, geometry in chunks:
            values = wrapped(atoms, atoms.eq(0), geometry)
            loss = sum(values[name].square().mean() for name in ('edos', 'phdos', 'eta'))
            (loss * len(atoms) / total).backward()
            for name, value in values.items():
                outputs.setdefault(name, []).append(value.detach().cpu())
        gradients = {name: param.grad.detach().cpu() for name, param in model.named_parameters()
                     if param.grad is not None}
        missing = [name for name, param in model.named_parameters() if param.grad is None]
        wrapped.zero_grad(set_to_none=True)
        return {name: torch.cat(values) for name, values in outputs.items()}, gradients, missing

    before_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        for route in ('A', 'B+'):
            for layout in ('staged', 'alternating'):
                params = dict(d_model=512, nhead=8, dim_feedforward=2048, num_decoder_layers=6,
                              dropout=.05, activation='gelu', atom_feat_mode='z_only_proj',
                              scale_mode='eta', q1_coord=True, local_route=route,
                              local_layout=layout, local_depth=2)
                current = Transformer(**params)
                init = initialize_comparison_model(current, params, 42)
                current.to(device).eval()
                with patch('model.local_global_encoder.PeriodicLocalMessage', old.PeriodicLocalMessage):
                    reference = Transformer(**params).to(device).eval()
                reference.load_state_dict(current.state_dict(), strict=True)
                if state_hash(reference.state_dict()) != init['state_sha256']:
                    raise AssertionError('reference did not receive identical complete weights')
                arm = {'route': route, 'layout': layout, 'init': init, 'cases': {}}
                for label, chunks in cases.items():
                    expected, expected_gradients, expected_missing = outputs_and_grads(reference, chunks)
                    actual, actual_gradients, actual_missing = outputs_and_grads(current, chunks)
                    if expected_missing != actual_missing:
                        raise AssertionError('optimization changed gradient connectivity')
                    errors = {}
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name], atol=tolerance, rtol=tolerance)
                        errors[name] = float((actual[name] - expected[name]).abs().max())
                    gradients = compare_gradients(expected_gradients, actual_gradients, dtype)
                    if not gradients['pass']:
                        raise AssertionError(f'full-model gradient mismatch: {route}/{layout}/{label}')
                    arm['cases'][label] = {'output_max_abs': errors, 'gradients': gradients,
                                          'unconnected_parameters_equal': expected_missing}
                    print(f'{precision} {route}/{layout} {label}: outputs/gradients matched', flush=True)
                for model in (reference, current):
                    if state_hash(model.state_dict()) != init['state_sha256']:
                        raise AssertionError('equivalence check updated model weights or buffers')
                arm['weights_unchanged'] = True
                report['arms'].append(arm)
                del reference, current, expected_gradients, actual_gradients
                gc.collect(); torch.cuda.empty_cache()
    finally:
        torch.set_default_dtype(before_dtype)
    report['status'] = 'passed'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {out}', flush=True)
    return report


def check(out, *, mode, batch_sizes=(32,), repeats=5, recompute_angles=True):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing existing report: {out}')
    if mode not in ('reference', 'current') or repeats < 1:
        raise ValueError('explicit reference/current mode and positive repeats required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU cost substituted')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    reference, identity = reference_module()
    local_class = reference.PeriodicLocalMessage if mode == 'reference' else PeriodicLocalMessage
    store = PackedCrystalStore(CRYSTAL_Q1_DATA_ROOT, 'train')
    definitions = structural_batches(store, batch_sizes)
    cases = {}
    for label, indices in definitions.items():
        raw = crystal_collate([store[i] for i in indices])
        chunks = [[c[0].to(device), c[1].to(device)]
                  for c in split_crystal_batch(raw, automatic_pair_budget(raw))]
        record_count, pair_count = 0, 0
        for atoms, geometry in chunks:
            records = build_periodic_records(geometry, atoms.eq(0), radius=6.)
            degree = torch.bincount(records.batch * atoms.shape[1] + records.dst,
                                    minlength=atoms.numel())
            record_count += len(records.distances)
            pair_count += int((degree * (degree - 1) // 2).sum())
        cases[label] = {'indices': indices, 'n_atoms': store.n_atoms[indices].tolist(),
                        'records': record_count, 'unordered_neighbor_pairs': pair_count,
                        'computation_batch_sizes': [len(c[0]) for c in chunks], 'chunks': chunks}
    grids = tuple(torch.as_tensor(grid, device=device, dtype=torch.float32) for grid in load_b7_grids())
    report = {'mode': mode, 'hardware': torch.cuda.get_device_name(device), 'torch': torch.__version__,
              'dtype': 'fp32', 'amp': False, 'tf32': False, 'depth': 2, 'split': 'train',
              'labels_read': False, 'updates': 0, 'optimizer_created': False,
              'recompute_angles': recompute_angles, 'reference': identity,
              'manifest_sha256': file_sha256(CRYSTAL_Q1_DATA_ROOT / 'manifest.json'),
              'source_sha256': {name: file_sha256(ROOT / name) for name in (
                  'model/periodic_messages.py', 'model/local_global_encoder.py', 'model/transformer.py',
                  'utils/periodic_geometry.py', 'utils/local_global_experiment.py',
                  'tools/checks/check_crystal_input.py', 'tools/checks/check_local_message_speed.py')},
              'cases': {name: {k: v for k, v in case.items() if k != 'chunks'} for name, case in cases.items()},
              'arms': []}
    with patch('model.local_global_encoder.PeriodicLocalMessage', local_class):
        for route in ('A', 'B+'):
            for layout in ('staged', 'alternating'):
                params = dict(d_model=512, nhead=8, dim_feedforward=2048, num_decoder_layers=6,
                              dropout=.05, activation='gelu', atom_feat_mode='z_only_proj',
                              scale_mode='eta', q1_coord=True, local_route=route,
                              local_layout=layout, local_depth=2)
                model = Transformer(**params)
                init = initialize_comparison_model(model, params, 42)
                if route == 'A':
                    for layer in model.encoder.local_layers:
                        for block in layer.message.blocks:
                            block.recompute_angles = recompute_angles
                model.to(device)
                wrapped = _WithSpectralGrids(model, grids)
                initial = state_hash(model.state_dict())
                arm = {'route': route, 'layout': layout, 'init': init,
                       'parameters': sum(p.numel() for p in model.parameters()), 'cases': {}}
                print(f'{mode} {route}/{layout}: initialized', flush=True)
                for label, case in cases.items():
                    wrapped.eval()
                    forward = forward_profile(wrapped, case['chunks'], device, repeats)
                    wrapped.train()
                    backward = profile(wrapped, case['chunks'], device, repeats)
                    arm['cases'][label] = {'eval_forward': forward, 'train_forward_backward': backward}
                    print(f'{mode} {route}/{layout} {label}: '
                          f"eval={forward['median_ms']:.2f} ms, train f/b={backward['median_ms']:.2f} ms, "
                          f"peak={backward['peak_allocated_MiB']:.2f} MiB", flush=True)
                    wrapped.zero_grad(set_to_none=True)
                    gc.collect(); torch.cuda.empty_cache()
                if initial != state_hash(model.state_dict()):
                    raise AssertionError('no-update profiling changed weights or buffers')
                arm['weights_unchanged'] = True
                report['arms'].append(arm)
                del model, wrapped
                gc.collect(); torch.cuda.empty_cache()
    report['status'] = 'passed'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {out}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('reference', 'current', 'verify'), required=True)
    parser.add_argument('--precision', choices=('fp32', 'fp64'), default='fp32')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[32])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--no-recompute-angles', action='store_true')
    args = parser.parse_args()
    if args.mode == 'verify':
        verify(args.out, precision=args.precision)
    else:
        if args.precision != 'fp32':
            parser.error('cost comparisons are FP32; FP64 is available for verification')
        check(args.out, mode=args.mode, batch_sizes=args.batch_sizes, repeats=args.repeats,
              recompute_angles=not args.no_recompute_angles)


if __name__ == '__main__':
    main()
