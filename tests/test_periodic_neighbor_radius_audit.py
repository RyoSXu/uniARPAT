"""Analytical geometry and invalid-input contracts for the radius audit."""

import unittest

import numpy as np
import torch

from tools.eval.periodic_neighbor_radius_audit import (
    SPLITS, inspect_crystal, positive_values, shell_bounds, validate_inputs,
)
from utils.g2_periodic_edges import build_g2_edges, g2_indegrees


def cubic(side, fractions):
    pos = torch.zeros(1, len(fractions) + 2, 3)
    pos[0, 0] = torch.tensor([side, side, 1 / side])
    pos[0, 1] = 90
    pos[0, 2:] = torch.tensor(fractions)
    return pos


class RadiusAuditContracts(unittest.TestCase):
    def test_finite_explicit_radius_and_train_valid_only(self):
        self.assertEqual(SPLITS, ("train", "valid"))
        for values in ([float("inf")], [float("nan")], [0], [-1], [4, 3], [3, 3]):
            with self.assertRaises(ValueError):
                positive_values(values, "radii")

    def test_nan_valid_coordinate_is_rejected_but_padding_is_ignored(self):
        pos = cubic(3, [[0, 0, 0], [float("nan"), 0, 0]])
        with self.assertRaises(ValueError):
            validate_inputs(pos, torch.zeros(1, 2, dtype=torch.bool))
        validate_inputs(pos, torch.tensor([[False, True]]))

    def test_cubic_shells_strict_boundary_and_counts(self):
        result = inspect_crystal(cubic(3, [[0, 0, 0]]), torch.zeros(1, 1, dtype=torch.bool),
                                 [3, 3.01, 4.3, 5.3], [0.001, 0.01], 16)
        np.testing.assert_array_equal(result["degrees"], [[0, 6, 18, 26]])
        np.testing.assert_allclose(result["shell_bounds"][0, 0], [3, np.sqrt(18), np.sqrt(27)], atol=1e-5)
        self.assertFalse(result["censored"].any())
        self.assertEqual(result["self_count"].tolist(), [0, 6, 18, 26])

    def test_sparse_crystal_adaptive_reference_and_right_censoring(self):
        pos, mask = cubic(10, [[0, 0, 0]]), torch.zeros(1, 1, dtype=torch.bool)
        result = inspect_crystal(pos, mask, [3], [0.001], 32)
        np.testing.assert_array_equal(result["degrees"], [[0]])
        np.testing.assert_allclose(result["shell_bounds"][0, 0], [10, np.sqrt(200), np.sqrt(300)], atol=1e-5)
        self.assertGreater(result["reference_radius"], np.sqrt(300))
        limited = inspect_crystal(pos, mask, [3], [0.001], 8)
        self.assertTrue(limited["censored"].all())
        self.assertTrue(np.isnan(limited["shell_bounds"]).all())

    def test_shell_groups_use_anchor_without_chaining(self):
        starts, upper = shell_bounds(np.array([1., 1.0008, 1.0016, 2., 2.0008, 3.]), 0.001)
        np.testing.assert_allclose(starts, [1, 1.0016, 2])
        np.testing.assert_allclose(upper, [1.0008, 1.0016, 2.0008])

    def test_filtering_matches_direct_enumeration_with_noncontiguous_padding(self):
        pos = cubic(3, [[2.04, -.1, .2], [float("nan"), 0, 0], [-.06, .8, .7]])
        mask = torch.tensor([[False, True, False]])
        radii = [2.5, 4.5, 5.5]
        result = inspect_crystal(pos, mask, radii, [0.001], 16)
        for column, radius in enumerate(radii):
            direct = build_g2_edges(pos, mask, r_cut=radius)
            degree = g2_indegrees(direct["batch"], direct["dst"], 1, 3)[0, [0, 2]].numpy()
            np.testing.assert_array_equal(result["degrees"][:, column], degree)

    def test_coincident_distinct_atoms_stop_audit(self):
        with self.assertRaisesRegex(ValueError, "zero-distance"):
            inspect_crystal(cubic(3, [[0, 0, 0], [0, 0, 0]]), torch.zeros(1, 2, dtype=torch.bool),
                            [3.01], [0.001], 16)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
