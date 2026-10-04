"""Scientific contracts for standalone untrained periodic message candidates."""

import copy
import unittest

import numpy as np
import torch

from tools.eval.periodic_geometry_acceptance import e3_operations, pack_cell
from tools.eval.periodic_message_prototypes import (
    EDGE_IRREPS, STATE_IRREPS, PeriodicMessageProbe, geometry_from_pos, prepare_geometry,
)


def star(skew=False, distance=5.0, dtype=torch.float64):
    axes = np.eye(3)
    if skew:
        axes[2] = [.25, .25, np.sqrt(.875)]
    directions = np.concatenate((axes, -axes))
    cart = np.concatenate((np.zeros((1, 3)), distance*directions))
    return pack_cell(np.eye(3)*30, cart/30+.5, dtype), torch.zeros(1, 7, dtype=torch.bool)


def skew_position(dtype=torch.float64):
    basis = np.array([[4.1, 0, 0], [-1.2, 5.3, 0], [.7, .9, 6.2]])
    frac = np.array([[.13, .27, .39], [.57, .63, .79], [.89, .11, .43]])
    return pack_cell(basis, frac, dtype), torch.zeros(1, 3, dtype=torch.bool)


class PeriodicMessageContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(20261003)
        cls.models = {route: PeriodicMessageProbe(route) for route in ("invariant", "equivariant")}
        for name in ("local_input", "local_output"):
            getattr(cls.models["equivariant"], name).load_state_dict(
                getattr(cls.models["invariant"], name).state_dict())
        torch.set_default_dtype(old)

    def atoms(self, mask, dtype=torch.float64):
        generator = torch.Generator().manual_seed(123)
        return torch.randn((*mask.shape, 512), generator=generator, dtype=dtype)

    def test_parameter_counts_and_common_scalar_projections(self):
        self.assertEqual(sum(p.numel() for p in self.models["invariant"].parameters()), 261568)
        self.assertEqual(sum(p.numel() for p in self.models["equivariant"].parameters()), 180676)
        self.assertIsNone(self.models["equivariant"].blocks[-1].direction_gate)
        for name in ("local_input", "local_output"):
            first, second = (getattr(self.models[route], name) for route in self.models)
            for a, b in zip(first.parameters(), second.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_all_images_counts_and_unordered_pairs_are_preserved(self):
        pos = pack_cell(np.eye(3)*3, np.array([[0., 0., 0.]]), torch.float64)
        g = geometry_from_pos(pos, torch.zeros(1, 1, dtype=torch.bool))
        self.assertEqual(g.degree.item(), 26)
        self.assertEqual(g.pair_count.item(), 325)
        self.assertTrue((g.edges["src"] == g.edges["dst"]).all())
        self.assertEqual(sum(len(group)*(len(group)-1)//2 for group in g.groups), 325)

    def test_invalid_inputs_are_errors_instead_of_empty_geometry(self):
        pos, mask = skew_position()
        for radius in (0., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                geometry_from_pos(pos, mask, radius)
        changed = pos.clone()
        changed[0, 2] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            geometry_from_pos(changed, mask)
        changed = pos.clone()
        changed[0, 3] = changed[0, 2]
        with self.assertRaisesRegex(ValueError, "zero-length"):
            geometry_from_pos(changed, mask)

    def test_vector_distance_mismatch_and_padding_records_are_rejected(self):
        pos, mask = skew_position()
        g = geometry_from_pos(pos, mask)
        with self.assertRaisesRegex(ValueError, "disagree"):
            prepare_geometry(g.edges, g.vectors*1.01, mask)
        with self.assertRaisesRegex(ValueError, "padding"):
            prepare_geometry(g.edges, g.vectors, torch.ones_like(mask))

    def test_mixed_padding_nan_slots_empty_neighborhood_and_zero_length_arrays(self):
        pos, mask = skew_position()
        pos = pos.repeat(2, 1, 1)
        mask = torch.tensor([[False, True, False], [True, True, True]])
        pos[0, 3] = float("nan")
        pos[1, 2:] = float("nan")
        sparse = pack_cell(np.eye(3)*30, np.array([[0., 0., 0.]]), torch.float64)
        for p, m in ((pos, mask), (sparse, torch.zeros(1, 1, dtype=torch.bool)),
                     (sparse[:, :2], torch.zeros(1, 0, dtype=torch.bool))):
            g = geometry_from_pos(p, m)
            atoms = self.atoms(m)
            atoms[m] = float("nan")
            for route, model in self.models.items():
                with self.subTest(route=route, slots=m.shape):
                    with torch.no_grad():
                        output, debug = model(atoms, m, g, True)
                    self.assertEqual(output.shape, (*m.shape, 512))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue((output[m] == 0).all())
                    for state in debug["states"]:
                        self.assertTrue((state[m] == 0).all())

    def test_spherical_representation_convention_and_all_e3_local_outputs(self):
        from e3nn import o3
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            for dtype, tolerance in ((torch.float64, 2e-10), (torch.float32, 2e-5)):
                pos, mask = skew_position(dtype)
                g, atoms = geometry_from_pos(pos, mask), self.atoms(mask, dtype)
                for route, template in self.models.items():
                    model = copy.deepcopy(template).to(dtype).eval()
                    with torch.no_grad():
                        output, debug = model(atoms, mask, g, True)
                        for name, q, _ in e3_operations():
                            Q = torch.tensor(q, dtype=dtype)
                            changed = prepare_geometry(g.edges, g.vectors@Q, mask)
                            actual, state = model(atoms, mask, changed, True)
                            with self.subTest(route=route, dtype=dtype, transform=name):
                                torch.testing.assert_close(actual, output, atol=tolerance, rtol=tolerance)
                                if route == "equivariant":
                                    D = STATE_IRREPS.D_from_matrix(torch.tensor(q.T, dtype=torch.float64)).to(dtype)
                                    Y = EDGE_IRREPS.D_from_matrix(torch.tensor(q.T, dtype=torch.float64)).to(dtype)
                                    original_y = o3.spherical_harmonics(EDGE_IRREPS, g.unit, normalize=False)
                                    changed_y = o3.spherical_harmonics(EDGE_IRREPS, changed.unit, normalize=False)
                                    torch.testing.assert_close(changed_y, original_y@Y.T,
                                                               atol=tolerance, rtol=tolerance)
                                    for left, right in zip(debug["states"], state["states"]):
                                        torch.testing.assert_close(right, left@D.T,
                                                                   atol=tolerance, rtol=tolerance)
        finally:
            torch.set_default_dtype(old)

    def test_permutation_and_edge_order_do_not_become_features(self):
        pos, mask = skew_position()
        order = torch.tensor([2, 0, 1])
        changed = torch.cat((pos[:, :2], pos[:, 2:][:, order]), 1)
        g = geometry_from_pos(pos, mask)
        permuted = geometry_from_pos(changed, mask[:, order])
        reverse = torch.arange(len(g.cutoff)-1, -1, -1)
        edges = {key: value[reverse] if key in ("batch", "dst", "src", "shifts", "distances") else value
                 for key, value in g.edges.items()}
        reversed_edges = prepare_geometry(edges, g.vectors[reverse], mask)
        atoms = self.atoms(mask)
        for route, model in self.models.items():
            with torch.no_grad():
                baseline = model(atoms, mask, g)
                shuffled = model(atoms[:, order], mask[:, order], permuted)
                reordered = model(atoms, mask, reversed_edges)
            torch.testing.assert_close(shuffled, baseline[:, order], atol=2e-10, rtol=2e-10)
            torch.testing.assert_close(reordered, baseline, atol=2e-10, rtol=2e-10)

    def test_integer_coordinate_images_preserve_local_state(self):
        pos, mask = skew_position()
        changed = pos.clone()
        changed[:, 2:] += torch.tensor([[[2., -1., 0.], [-2., 0., 1.], [1., 2., -1.]]])
        g, other = geometry_from_pos(pos, mask), geometry_from_pos(changed, mask)
        atoms = self.atoms(mask)
        for model in self.models.values():
            with torch.no_grad():
                torch.testing.assert_close(model(atoms, mask, g), model(atoms, mask, other),
                                           atol=2e-10, rtol=2e-10)

    def test_angle_response_when_radial_data_and_vector_sum_are_identical(self):
        pos, mask = star(False)
        changed, _ = star(True)
        g, other = geometry_from_pos(pos, mask), geometry_from_pos(changed, mask)
        self.assertEqual(g.degree.tolist(), [[6, 1, 1, 1, 1, 1, 1]])
        self.assertTrue(torch.equal(g.degree, other.degree))
        for geometry in (g, other):
            central = geometry.unit[geometry.edges["dst"] == 0]
            self.assertLess(central.sum(0).norm().item(), 1e-12)
        atoms = self.atoms(torch.zeros(1, 1, dtype=torch.bool)).repeat(1, 7, 1)
        for model in self.models.values():
            with torch.no_grad():
                delta = (model(atoms, mask, g)[0, 0]-model(atoms, mask, other)[0, 0]).abs().max().item()
            self.assertGreater(delta, 1e-10)

    def test_radial_and_count_changes_reach_scalar_content(self):
        pos, mask = star(True)
        distance, _ = star(True, 5.2)
        few = mask.clone()
        few[0, -1] = True
        atoms = self.atoms(torch.zeros(1, 1, dtype=torch.bool)).repeat(1, 7, 1)
        g = geometry_from_pos(pos, mask)
        for model in self.models.values():
            with torch.no_grad():
                baseline = model(atoms, mask, g)[0, 0]
                for p, m in ((distance, mask), (pos, few)):
                    actual = model(atoms, m, geometry_from_pos(p, m))[0, 0]
                    self.assertGreater((actual-baseline).abs().max().item(), 1e-10)

    def test_chunk_recomputation_preserves_forward_and_parameter_gradients(self):
        pos, mask = star(True)
        g = geometry_from_pos(pos, mask)
        models = [copy.deepcopy(self.models["invariant"]) for _ in range(2)]
        for block in models[0].blocks:
            block.recompute_angles = False
            block.pair_chunk = 4096
        for block in models[1].blocks:
            block.recompute_angles = True
            block.pair_chunk = 2
        inputs = [self.atoms(mask).requires_grad_() for _ in models]
        outputs = [model(atoms, mask, g) for model, atoms in zip(models, inputs)]
        for output in outputs:
            output.square().mean().backward()
        torch.testing.assert_close(outputs[0], outputs[1], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, atol=1e-12, rtol=1e-10)
        for (_, first), (_, second) in zip(models[0].named_parameters(), models[1].named_parameters()):
            self.assertIsNotNone(first.grad)
            self.assertIsNotNone(second.grad)
            torch.testing.assert_close(first.grad, second.grad, atol=1e-12, rtol=1e-10)

    def test_backward_reaches_both_blocks_and_leaves_parameters_unchanged(self):
        pos, mask = star(True)
        g = geometry_from_pos(pos, mask)
        for route, template in self.models.items():
            model = copy.deepcopy(template)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            output = model(self.atoms(mask), mask, g)
            output.square().mean().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (route, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (route, name))
                self.assertGreater(parameter.grad.abs().max().item(), 0., (route, name))
            for index in range(2):
                active = sum(p.grad.abs().sum().item() for p in model.blocks[index].parameters())
                self.assertGreater(active, 0.)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))


if __name__ == "__main__":
    unittest.main()
