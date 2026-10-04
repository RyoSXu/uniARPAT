"""Scientific checks for the bounded, untrained high-order B enhancement."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from e3nn import o3

from tools.eval.periodic_geometry_acceptance import e3_operations, pack_cell
from tools.eval.periodic_message_high_order import high_order_pair_features
from tools.eval.periodic_message_high_order_compare import create_models, low_moment_pair, sheared_geometry
from tools.eval.periodic_message_precheck import SEED, state_digest, synthetic_cases
from tools.eval.periodic_message_prototypes import geometry_from_pos, prepare_geometry


class HighOrderMessageContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        manifest = Path(__file__).resolve().parents[1]/"results/periodic_message_precheck_q1_r6_20261003/manifest.json"
        cls.manifest = json.loads(manifest.read_text())
        cls.models, cls.entry = create_models(cls.manifest, torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls.previous_dtype)

    def test_pair_statistic_matches_explicit_weighted_pairs(self):
        pos = pack_cell(np.eye(3)*3, np.array([[0., 0., 0.]]), torch.float64)
        g = geometry_from_pos(pos, torch.zeros(1, 1, dtype=torch.bool))
        self.assertEqual(len(g.cutoff), 26)  # All images, including self across cells.
        rng = torch.Generator().manual_seed(70)
        weights = torch.randn(26, 8, generator=rng)
        harmonics = o3.spherical_harmonics([3, 4], g.unit, normalize=False, normalization="component")
        actual = high_order_pair_features(g, weights, harmonics)
        q = g.unit@g.unit.T
        polynomials = ((5*q**3-3*q)/2, (35*q**4-30*q**2+3)/8)
        i, j = torch.triu_indices(26, 26, offset=1)
        expected = torch.stack([(weights[i]*weights[j]*p[i, j, None]).sum(0)/np.sqrt(325)
                                for p in polynomials], -1).reshape(1, 1, 16)
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)

    def test_existing_weights_preserved_and_added_inputs_ablate_to_B(self):
        self.assertEqual(sum(p.numel() for p in self.models["B+"].parameters()), 184772)
        original, enhanced = self.models["B"].state_dict(), self.models["B+"].state_dict()
        for name, value in original.items():
            new = enhanced[name]
            if name.endswith("update.0.weight"):
                new = new[:, :78]
            self.assertTrue(torch.equal(value, new), name)
        model = copy.deepcopy(self.models["B+"])
        model.include_high_order = False
        geometries, mask, ids = low_moment_pair(torch.float64, True)
        for g in geometries:
            with torch.no_grad():
                atoms = self.entry(ids, mask)
                torch.testing.assert_close(model(atoms, mask, g), self.models["B"](atoms, mask, g), atol=2e-14, rtol=2e-14)

    def test_high_order_statistics_rotation_and_reflection_invariant(self):
        geometries, mask, _ = low_moment_pair(torch.float64, True)
        g = geometries[0]
        weights = torch.arange(1, len(g.cutoff)*8+1).reshape(-1, 8)/100
        previous = high_order_pair_features(g, weights, o3.spherical_harmonics([3, 4], g.unit, normalize=False, normalization="component"))
        for name, q, _ in e3_operations():
            if name == "translation":
                continue
            transformed = prepare_geometry(g.edges, g.vectors@torch.tensor(q), mask)
            high = o3.spherical_harmonics([3, 4], transformed.unit, normalize=False, normalization="component")
            actual = high_order_pair_features(transformed, weights, high)
            torch.testing.assert_close(previous, actual, atol=3e-12, rtol=3e-12)

    def test_known_counterexample_resolved_in_float64_with_fixed_radial_fields(self):
        geometries, mask, ids = low_moment_pair(torch.float64, True)
        self.assertTrue(torch.equal(geometries[0].radial, geometries[1].radial))
        self.assertTrue(torch.equal(geometries[0].degree, geometries[1].degree))
        atoms = self.entry(ids, mask)
        deltas = {}
        for route, model in self.models.items():
            with torch.no_grad():
                outputs = [model(atoms, mask, g) for g in geometries]
                deltas[route] = float((outputs[0][0, 0]-outputs[1][0, 0]).abs().max())
        self.assertGreater(deltas["A"], 1e-10)
        self.assertLess(deltas["B"], 2e-10)
        self.assertGreater(deltas["B+"], 1e-10)

    def test_new_weight_columns_and_geometry_receive_finite_nonzero_gradients(self):
        geometries, mask, ids = low_moment_pair(torch.float64, True)
        model = copy.deepcopy(self.models["B+"])
        before = state_digest(model)
        g = geometries[0]
        vectors = g.vectors.clone().requires_grad_()
        distances = g.edges["distances"].clone().requires_grad_()
        differentiable = prepare_geometry(dict(g.edges, distances=distances), vectors, mask)
        # Normal cutoff is intentionally restored, testing the actual adopted formula.
        output = model(self.entry(ids, mask), mask, differentiable)
        params = tuple(model.parameters())
        grads = torch.autograd.grad(output.square().mean(), params+(vectors, distances))
        self.assertTrue(all(torch.isfinite(x).all() for x in grads))
        for (name, _), grad in zip(model.named_parameters(), grads):
            if name.endswith("update.0.weight"):
                self.assertGreater(float(grad[:, 78:].abs().max()), 0, name)
        self.assertGreater(float(grads[-1].abs().max()), 0)
        self.assertGreater(float(grads[-2].abs().max()), 0)
        self.assertEqual(before, state_digest(model))

    def test_padding_empty_and_single_neighbor_features_are_exact_zero(self):
        for dtype in (torch.float32, torch.float64):
            model = copy.deepcopy(self.models["B+"]).to(dtype)
            entry = copy.deepcopy(self.entry).to(dtype)
            for name, pos, mask, ids in synthetic_cases(dtype):
                g = geometry_from_pos(pos, mask)
                atoms = entry(ids, mask)
                atoms[mask] = float("nan")
                with torch.no_grad():
                    output, debug = model(atoms, mask, g, True)
                self.assertTrue(torch.isfinite(output).all(), name)
                self.assertTrue((output[mask] == 0).all(), name)
                for features in debug["angular_invariants"]:
                    self.assertTrue(torch.isfinite(features).all(), name)
                    self.assertTrue((features[g.pair_count == 0] == 0).all(), name)

    def test_physical_shear_rebuild_uses_full_position_contract(self):
        pos = pack_cell(np.array([[4.1, 0., 0.], [-1.2, 5.3, 0.], [.7, .9, 6.2]]),
                        np.array([[.13, .27, .39], [.57, .63, .79]]), torch.float64)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        original = geometry_from_pos(pos, mask)
        identity = sheared_geometry(pos, mask, amount=0.)
        torch.testing.assert_close(original.edges["distances"], identity.edges["distances"], atol=2e-7, rtol=2e-7)
        changed = sheared_geometry(pos, mask)
        self.assertGreater(float((original.edges["distances"]-changed.edges["distances"]).abs().max()), 1e-3)
        self.assertTrue(torch.isfinite(changed.vectors).all())


if __name__ == "__main__":
    unittest.main()
