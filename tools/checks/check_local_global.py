"""Full-size four-arm CUDA preflight on structural train inputs, with no updates.

Checks finite gradients, complete spectral E(3) outputs and short-batch cost.
No spectral labels, optimizer, checkpoint or train/test selection are used.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from pymatgen.core import Element

from model.transformer import Transformer
from tools.checks.check_crystal_input import profile
from tools.eval.periodic_geometry_acceptance import e3_operations
from utils.b7_cif_inference import load_b7_grids
from utils.bucket_batch import crystal_collate
from utils.crystal import CrystalGeometry
from utils.crystal_cache import PackedCrystalStore
from utils.local_global_experiment import initialize_comparison_model
from utils.paths import CRYSTAL_Q1_DATA_ROOT, file_sha256
from utils.periodic_geometry import build_periodic_records
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import state_hash


class _WithSpectralGrids(torch.nn.Module):
    """Supply the fixed E0/P0 queries to the existing no-update profiler."""
    def __init__(self, model, grids):
        super().__init__()
        self.core, self.grids = model, grids

    def forward(self, atoms, mask, geometry):
        return self.core(atoms, mask, geometry, *self.grids)


def check(out, *, depth, repeats=3, precision='fp32', include_largest=False):
    if depth < 1 or repeats < 1:
        raise ValueError('positive depth and repeat count required')
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'refusing existing preflight report: {out}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA preflight requires CUDA; no CPU timing substituted')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    dtype = {'fp32': torch.float32, 'fp64': torch.float64}[precision]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-8
    store = PackedCrystalStore(CRYSTAL_Q1_DATA_ROOT, 'train')
    indices = [i for i, n in enumerate(store.n_atoms.tolist()) if 2 <= n <= 8][:3]
    if len(indices) != 3:
        raise ValueError('three small train structures required')
    atoms, geometry = crystal_collate([store[i] for i in indices])
    atoms, geometry = atoms.to(device), geometry.to(device=device, dtype=dtype)
    mask = atoms.eq(0)
    with (ROOT / 'data/reference/valence_by_element.json').open() as stream:
        table = json.load(stream)
    valence = [sum(table[Element.from_Z(int(z)).symbol]['zval'] for z in row if z)
               for row in atoms.cpu().tolist()]
    counts = (~mask).sum(-1)
    grids = tuple(torch.as_tensor(grid, device=device, dtype=dtype) for grid in load_b7_grids())
    base_records = build_periodic_records(geometry, mask, radius=6.)
    report = {'hardware': torch.cuda.get_device_name(device), 'torch': torch.__version__,
              'dtype': precision, 'amp': False, 'tf32': False, 'depth': depth,
              'updates': 0, 'training': False, 'split': 'train', 'labels_read': False,
              'manifest_sha256': file_sha256(CRYSTAL_Q1_DATA_ROOT / 'manifest.json'),
              'samples': [{'index': i, 'id': str(store.material_ids[i]), 'n_atoms': int(store.n_atoms[i])}
                          for i in indices],
              'record_count': base_records.distances.numel(),
              'nearest_inside_cutoff_A': float((6. - base_records.distances).amin()),
              'source_sha256': {name: file_sha256(ROOT / name) for name in (
                  'model/transformer.py', 'model/local_global_encoder.py', 'model/periodic_messages.py',
                  'utils/local_global_experiment.py', 'utils/spectral_inference.py',
                  'utils/periodic_geometry.py', 'tools/checks/check_local_global.py')}, 'arms': []}
    batches = [[atoms, geometry]]
    if include_largest:
        index = int(store.n_atoms.argmax())
        large = crystal_collate([store[index]])
        batches.append([large[0].to(device), large[1].to(device=device, dtype=dtype)])
        report['large_structural_case'] = {'index': index, 'id': str(store.material_ids[index]),
                                          'n_atoms': int(store.n_atoms[index]),
                                          'labels_read': False}
    before_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        for route in ('A', 'B+'):
            for layout in ('staged', 'alternating'):
                params = dict(d_model=512, nhead=8, dim_feedforward=2048, num_decoder_layers=6,
                    dropout=.05, activation='gelu', atom_feat_mode='z_only_proj', scale_mode='eta',
                    q1_coord=True, local_route=route, local_layout=layout, local_depth=depth)
                model = Transformer(**params)
                init = initialize_comparison_model(model, params, 42)
                model.to(device).eval()
                wrapped = _WithSpectralGrids(model, grids)
                arm = {'route': route, 'layout': layout, 'init': init,
                       'parameters': sum(p.numel() for p in model.parameters()),
                       'execution_order': model.encoder.execution_order, 'e3': []}
                initial = state_hash(model.state_dict())
                with torch.no_grad():
                    expected = wrapped(atoms, mask, geometry)
                    expected.update(reconstruct_h1_spectra(expected, counts, valence))
                    if not all(torch.isfinite(value).all() for value in expected.values()):
                        raise AssertionError('non-finite baseline spectrum')
                    for name, matrix, translation in e3_operations():
                        q = torch.as_tensor(matrix, device=device, dtype=dtype)
                        t = torch.as_tensor(translation, device=device, dtype=dtype)
                        cell = geometry.cell @ q
                        frac = geometry.frac_coords + torch.bmm(t.expand(len(atoms), 1, 3), torch.linalg.inv(cell))
                        changed = CrystalGeometry(cell, frac)
                        records = build_periodic_records(changed, mask, radius=6.)
                        same_topology = (torch.equal(records.batch, base_records.batch)
                            and torch.equal(records.dst, base_records.dst)
                            and torch.equal(records.src, base_records.src)
                            and torch.equal(records.shifts, base_records.shifts))
                        if not same_topology:
                            raise AssertionError(f'cutoff/record topology changed under {name}')
                        actual = wrapped(atoms, mask, changed)
                        actual.update(reconstruct_h1_spectra(actual, counts, valence))
                        errors = {}
                        for key in expected:
                            torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
                            diff = (actual[key] - expected[key]).abs()
                            errors[key] = {'max_abs': float(diff.amax()),
                                'max_relative_to_clamped_bin': float((diff / expected[key].abs().clamp_min(1e-12)).amax())}
                        arm['e3'].append({'operation': name, 'record_topology_equal': True, 'outputs': errors})
                arm['profiles'] = [profile(wrapped, [batch], device, repeats) for batch in batches]
                arm['gradient_checks'] = {}
                for prefix in ('tok_emb', 'atom_norm', 'atom_proj', 'encoder.local_layers', 'encoder.global_layers',
                               'decoder', 'edos_out_head', 'phdos_out_head', 'eta_head', 'edos_trunk', 'phdos_trunk'):
                    selected = [p for key, p in model.named_parameters() if key.startswith(prefix)]
                    if not selected or not all(p.requires_grad and p.grad is not None for p in selected):
                        raise AssertionError(f'frozen or disconnected parameters: {prefix}')
                    grads = [p.grad for p in selected]
                    if not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
                        raise AssertionError(f'absent/non-finite/zero gradients: {prefix}')
                    # Q1 fc1 may have zero first-step gradients while the
                    # zero-initialized output projection receives a gradient.
                    arm['gradient_checks'][prefix] = {'parameter_tensors': len(selected),
                        'all_trainable_present_finite': True,
                        'total_abs_gradient': sum(float(g.abs().sum()) for g in grads),
                        'zero_gradient_tensors': sum(int(not g.any()) for g in grads)}
                if state_hash(model.state_dict()) != initial:
                    raise AssertionError('preflight changed model parameters or buffers')
                arm['weights_unchanged'] = True
                report['arms'].append(arm)
                print(f'{precision} {route}/{layout}: full outputs, E(3), gradients and weights passed', flush=True)
                wrapped.zero_grad(set_to_none=True)
                del model, wrapped
                gc.collect(); torch.cuda.empty_cache()
        if len({arm['init']['common_state_sha256'] for arm in report['arms']}) != 1:
            raise AssertionError('four-arm common initialization mismatch')
    finally:
        torch.set_default_dtype(before_dtype)
    report['status'] = 'passed'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {out}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--precision', choices=('fp32', 'fp64'), default='fp32')
    parser.add_argument('--include-largest', action='store_true')
    args = parser.parse_args()
    check(args.out, depth=args.depth, repeats=args.repeats,
          precision=args.precision, include_largest=args.include_largest)


if __name__ == '__main__':
    main()
