"""Acceptance of the extracted interface, gradients and scientific contracts.

CPU only, without optimizer steps. Historical probes are numerical oracles;
the new core never imports them. No full DOS model or labels are evaluated.
"""

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch
from torch import nn

from model.periodic_messages import PeriodicLocalMessage, _prepare_message_geometry, high_order_pair_features
from tools.eval.periodic_geometry_acceptance import e3_operations, pack_cell
from tools.eval.periodic_message_high_order_compare import create_models, low_moment_pair
from tools.eval.periodic_message_module_check import aligned_local_models
from tools.eval.periodic_message_precheck import compare_features, synthetic_cases
from tools.eval.periodic_message_prototypes import geometry_from_pos
from utils.periodic_geometry import PeriodicNeighborRecords, build_periodic_records

ROOT = Path(__file__).resolve().parents[1]


def records_from_probe(geometry, mask):
    return PeriodicNeighborRecords.from_edges(geometry.edges, geometry.vectors, mask, geometry.radius)


class LocalModuleContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            manifest = json.loads((ROOT / "results/periodic_message_precheck_q1_r6_20261003/manifest.json").read_text())
            with torch.random.fork_rng(devices=[]):
                references, _ = create_models(manifest, torch.float64)
                cls.references = {dtype: {r: copy.deepcopy(m).to(dtype) for r, m in references.items()}
                                  for dtype in (torch.float32, torch.float64)}
                cls.models = {dtype: aligned_local_models(cls.references[dtype], dtype)
                              for dtype in (torch.float32, torch.float64)}
        finally:
            torch.set_default_dtype(previous)

    def atoms(self, mask, dtype):
        return torch.randn((*mask.shape, 512), dtype=dtype,
                           generator=torch.Generator().manual_seed(123))

    def test_core_import_does_not_load_upstream_downstream_or_diagnostics(self):
        code = """
import sys
import numpy  # Initialize MKL before Torch/libgomp in this environment.
from utils.periodic_geometry import build_periodic_records
assert not any(k.startswith('model') or k.startswith('tools.eval') for k in sys.modules)
from model.periodic_messages import PeriodicLocalMessage
for route in ('A', 'B+'):
    PeriodicLocalMessage(route)
assert not any(k.startswith('tools.eval') for k in sys.modules)
assert not any(k in sys.modules for k in ('model.model', 'model.transformer', 'model.periodic_manybody', 'utils.builder', 'run_ablation_experiments'))
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lazy_model_package_preserves_public_class_identity(self):
        code = """
import numpy  # Initialize MKL before Torch/libgomp in this environment.
import model
assert {'basemodel', 'Transformer'} <= set(dir(model))
from model import basemodel, Transformer
from model.model import basemodel as original_base
from model.transformer import Transformer as original_transformer
assert basemodel is original_base and Transformer is original_transformer
assert model.basemodel is basemodel and model.Transformer is Transformer
try:
    model.unknown_component
except AttributeError:
    pass
else:
    raise AssertionError('unknown name should raise AttributeError')
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_standalone_construction_counts_and_no_baseline_parameter_alias(self):
        for dtype, models in self.models.items():
            for route, model in models.items():
                self.assertEqual(sum(p.numel() for p in model.parameters()), {"A": 261568, "B+": 184772}[route])
                original = dict(self.references[dtype][route].named_parameters())
                self.assertEqual(set(original), set(dict(model.named_parameters())))
                for name, param in model.named_parameters():
                    self.assertTrue(torch.equal(param, original[name]))
                    self.assertNotEqual(param.data_ptr(), original[name].data_ptr())
        with self.assertRaises(ValueError):
            PeriodicLocalMessage("unknown")

    def test_physical_geometry_matches_frozen_builder_for_all_synthetic_cases(self):
        for dtype in self.models:
            for name, pos, mask, _ in synthetic_cases(dtype):
                old = geometry_from_pos(pos, mask)
                records = build_periodic_records(pos, mask, 6.)
                with self.subTest(dtype=dtype, case=name):
                    for field in ("batch", "dst", "src", "shifts", "distances"):
                        self.assertTrue(torch.equal(getattr(records, field), old.edges[field]), field)
                    self.assertTrue(torch.equal(records.vectors, old.vectors))
                    self.assertFalse(hasattr(records, "radial"))
                    self.assertFalse(hasattr(records, "unit"))
                    self.assertFalse(hasattr(records, "cutoff"))

    def test_forward_states_counts_and_padding_match_frozen_prototypes(self):
        for dtype, models in self.models.items():
            for name, pos, mask, _ in synthetic_cases(dtype):
                g = geometry_from_pos(pos, mask)
                records = build_periodic_records(pos, mask, 6.)
                h = self.atoms(mask, dtype)
                h[mask] = float("nan")
                before = h.clone()
                for route, model in models.items():
                    with torch.no_grad():
                        reference = self.references[dtype][route](h, mask, g, True)
                        actual = model(h, records, mask, return_debug=True)
                    metrics = compare_features(reference, actual, dtype, "direct")
                    self.assertTrue(metrics["pass"], (dtype, name, route, metrics))
                    for field in ("degree", "effective_count", "pair_count"):
                        self.assertTrue(torch.equal(reference[1][field], actual[1][field]))
                    if route == "B+":
                        for a, b in zip(reference[1]["angular_invariants"], actual[1]["angular_invariants"]):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                    self.assertTrue((actual[0][mask] == 0).all())
                    torch.testing.assert_close(h, before, atol=0, rtol=0, equal_nan=True)

    def test_input_parameter_and_geometry_gradients_match_frozen_prototypes(self):
        for dtype, models in self.models.items():
            _, pos, mask, _ = synthetic_cases(dtype)[1]
            original_g = geometry_from_pos(pos, mask)
            for route, template in models.items():
                old, new = copy.deepcopy(self.references[dtype][route]), copy.deepcopy(template)
                results = []
                for model, extracted in ((old, False), (new, True)):
                    h = self.atoms(mask, dtype).requires_grad_()
                    vectors = original_g.vectors.clone().requires_grad_()
                    distances = original_g.edges["distances"].clone().requires_grad_()
                    edges = dict(original_g.edges, distances=distances)
                    before = {name: value.clone() for name, value in model.state_dict().items()}
                    if extracted:
                        records = PeriodicNeighborRecords.from_edges(edges, vectors, mask, 6.)
                        output = model(h, records, mask)
                    else:
                        from tools.eval.periodic_message_prototypes import prepare_geometry
                        output = model(h, mask, prepare_geometry(edges, vectors, mask, 6.))
                    params = tuple(model.parameters())
                    grads = torch.autograd.grad(output.square().mean(), (h, vectors, distances) + params)
                    self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
                    self.assertTrue(all(grad.abs().max().item() > 0 for grad in grads))
                    for name, value in model.state_dict().items():
                        self.assertTrue(torch.equal(before[name], value))
                    results.append((output, grads))
                atol, rtol = (2e-6, 2e-5) if dtype == torch.float32 else (2e-12, 2e-10)
                torch.testing.assert_close(results[0][0], results[1][0], atol=atol, rtol=rtol)
                for a, b in zip(results[0][1], results[1][1]):
                    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)

    def test_shared_interface_in_registered_pipeline_backpropagates_to_zp(self):
        class Pipeline(nn.Module):
            def __init__(self, route):
                super().__init__()
                self.zp = nn.Sequential(nn.Embedding(118, 512), nn.LayerNorm(512), nn.Linear(512, 512))
                self.local = PeriodicLocalMessage(route)
                # Minimal scalar downstream stub; no production Encoder/head.
                self.downstream = nn.Linear(512, 3)

            def forward(self, elements, records, mask):
                return self.downstream(self.local(self.zp(elements), records, mask))

        _, pos, mask, elements = synthetic_cases(torch.float32)[1]
        records = build_periodic_records(pos, mask, 6.)
        for route in ("A", "B+"):
            pipeline = Pipeline(route).to(dtype=pos.dtype)
            before = {name: value.clone() for name, value in pipeline.state_dict().items()}
            pipeline(elements, records, mask).square().mean().backward()
            expected = [p for module in (pipeline.zp, pipeline.local, pipeline.downstream) for p in module.parameters()]
            registered = list(pipeline.parameters())
            self.assertEqual({id(p) for p in expected}, {id(p) for p in registered})
            self.assertEqual(len(registered), len({id(p) for p in registered}))
            for name, param in pipeline.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertGreater(param.grad.abs().max().item(), 0, name)
            for name, value in pipeline.state_dict().items():
                self.assertTrue(torch.equal(before[name], value))

    def test_local_rotation_reflection_and_equivariant_debug_states(self):
        # e3nn's representation-matrix factory also uses the default dtype.
        # Match the established float64 physical-frame comparison protocol.
        previous_dtype = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous_dtype)
        torch.set_default_dtype(torch.float64)
        for dtype, models in self.models.items():
            _, pos, mask, _ = synthetic_cases(dtype)[1]
            records = build_periodic_records(pos, mask, 6.)
            h = self.atoms(mask, dtype)
            with torch.no_grad():
                for route, model in models.items():
                    baseline = model(h, records, mask, return_debug=True)
                    for name, q, _ in e3_operations():
                        if name == "translation":
                            continue
                        changed = replace(records, vectors=records.vectors @ torch.tensor(q, dtype=dtype))
                        actual = model(h, changed, mask, return_debug=True)
                        metrics = compare_features(baseline, actual, dtype, "direct", orthogonal=q)
                        self.assertTrue(metrics["pass"], (route, dtype, name, metrics))

    def test_permutation_integer_images_and_record_order_consistency(self):
        dtype = torch.float64
        _, pos, mask, _ = synthetic_cases(dtype)[1]
        h = self.atoms(mask, dtype)
        records = build_periodic_records(pos, mask, 6.)
        order = torch.tensor([2, 0, 1])
        permuted = build_periodic_records(torch.cat((pos[:, :2], pos[:, 2:][:, order]), 1), mask[:, order], 6.)
        unwrapped = pos.clone()
        unwrapped[:, 2:] += torch.tensor([[[2., -1., 0.], [-2., 0., 1.], [1., 2., -1.]]], dtype=dtype)
        images = build_periodic_records(unwrapped, mask, 6.)
        reverse = torch.arange(len(records.distances) - 1, -1, -1)
        reversed_records = replace(records, **{name: getattr(records, name)[reverse]
                                   for name in ("batch", "dst", "src", "shifts", "vectors", "distances")})
        for model in self.models[dtype].values():
            with torch.no_grad():
                baseline = model(h, records, mask)
                torch.testing.assert_close(model(h[:, order], permuted, mask[:, order]), baseline[:, order], atol=2e-10, rtol=2e-10)
                torch.testing.assert_close(model(h, images, mask), baseline, atol=2e-10, rtol=2e-10)
                torch.testing.assert_close(model(h, reversed_records, mask), baseline, atol=2e-10, rtol=2e-10)

    def test_all_images_reverse_vectors_and_pair_counts(self):
        pos = pack_cell(np.eye(3)*3, np.array([[0., 0., 0.]]), torch.float64)
        mask = torch.zeros(1, 1, dtype=torch.bool)
        records = build_periodic_records(pos, mask, 6.)
        self.assertEqual(len(records.distances), 26)
        keys = {tuple(shift.tolist()): index for index, shift in enumerate(records.shifts)}
        for key, index in keys.items():
            reverse = keys[tuple(-x for x in key)]
            torch.testing.assert_close(records.vectors[index], -records.vectors[reverse], atol=0, rtol=0)
        for model in self.models[torch.float64].values():
            _, debug = model(self.atoms(mask, torch.float64), records, mask, return_debug=True)
            self.assertEqual(debug["degree"].item(), 26)
            self.assertEqual(debug["pair_count"].item(), 325)

    def test_dtype_transfer_preserves_indices_layout_and_gradient_connection(self):
        _, pos, mask, _ = synthetic_cases(torch.float64)[1]
        original = build_periodic_records(pos, mask, 6.)
        vectors = original.vectors.clone().requires_grad_()
        distances = original.distances.clone().requires_grad_()
        records = replace(original, vectors=vectors, distances=distances)
        changed = records.to(device="cpu", dtype=torch.float32)
        for field in ("batch", "dst", "src", "shifts"):
            self.assertEqual(getattr(changed, field).dtype, torch.long)
            self.assertTrue(torch.equal(getattr(records, field), getattr(changed, field)))
        self.assertEqual(changed.padding_mask.dtype, torch.bool)
        output = self.models[torch.float32]["B+"](self.atoms(mask, torch.float32), changed, mask)
        grads = torch.autograd.grad(output.square().mean(), (vectors, distances))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads))
        with self.assertRaises(ValueError):
            records.to(dtype=torch.float16)

    def test_mismatched_masks_dtypes_and_invalid_records_are_rejected(self):
        _, pos, mask, _ = synthetic_cases(torch.float64)[1]
        records = build_periodic_records(pos, mask, 6.)
        h = self.atoms(mask, torch.float64)
        model = self.models[torch.float64]["B+"]
        bad_mask = mask.clone()
        bad_mask[0, -1] = True
        for atoms, supplied_mask in ((h, bad_mask), (h.float(), mask), (h[..., :-1], mask), (torch.full_like(h, float("nan")), mask)):
            with self.assertRaises(ValueError):
                model(atoms, records, supplied_mask)
        mask[0, 0] = True
        self.assertFalse(records.padding_mask.any())  # Caller mutation does not alter metadata.
        with self.assertRaises(ValueError):
            replace(records, shifts=records.shifts.float())
        with self.assertRaises(ValueError):
            replace(records, distances=records.distances * 2)
        with self.assertRaises(ValueError):
            replace(records, dst=torch.full_like(records.dst, 999))
        with self.assertRaises(ValueError):
            replace(records, radius=float("inf"))

    def test_invalid_structure_or_zero_length_is_not_silently_empty(self):
        _, pos, mask, _ = synthetic_cases(torch.float64)[1]
        for radius in (float("nan"), float("inf"), 0., -1.):
            with self.assertRaises(ValueError):
                build_periodic_records(pos, mask, radius)
        invalid = pos.clone()
        invalid[0, 2] = float("nan")
        with self.assertRaises(ValueError):
            build_periodic_records(invalid, mask, 6.)
        invalid = pos.clone()
        invalid[0, 1, 0] = 180
        with self.assertRaises(ValueError):
            build_periodic_records(invalid, mask, 6.)
        invalid = pos.clone()
        invalid[0, 0, 2] = -1
        with self.assertRaises(ValueError):
            build_periodic_records(invalid, mask, 6.)
        invalid = pos.clone()
        invalid[0, 3] = invalid[0, 2]
        with self.assertRaisesRegex(ValueError, "zero-length"):
            build_periodic_records(invalid, mask, 6.)

    def test_high_order_counterexample_and_ablation_are_preserved(self):
        from e3nn import o3
        geometries, mask, ids = low_moment_pair(torch.float64, True)
        h = self.atoms(mask[:, :1], torch.float64).repeat(1, ids.shape[1], 1)
        enhanced = copy.deepcopy(self.models[torch.float64]["B+"])
        features = []
        # Unit weights isolate angular capacity; this is explicitly an ablation,
        # not a claim about the weak native cutoff response at 5.85 Angstrom.
        for old_geometry in geometries:
            records = records_from_probe(old_geometry, mask)
            g = _prepare_message_geometry(records)
            weights = torch.ones(len(g.cutoff), 8, dtype=torch.float64)
            harmonics = o3.spherical_harmonics([3, 4], g.unit, normalize=False, normalization="component")
            features.append(high_order_pair_features(g, weights, harmonics))
            enhanced.include_high_order = False
            from tools.eval.periodic_message_prototypes import prepare_geometry
            # Prepare native fields, not the c=1 probe fields.
            native = prepare_geometry(old_geometry.edges, old_geometry.vectors, mask, 6.)
            with torch.no_grad():
                actual = enhanced(h, records, mask)
                expected = self.references[torch.float64]["B"](h, mask, native)
            torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
        self.assertGreater(float((features[0][0, 0]-features[1][0, 0]).abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
