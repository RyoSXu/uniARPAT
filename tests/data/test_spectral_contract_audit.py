"""Analytic spectral counterexamples, not learned-model performance tests."""
import unittest

import numpy as np

from tools.eval.spectral_contract_audit import (
    SPLITS, coverage_layout, energy_window_ratios, fixed_shape_optimum,
    padding_decomposition, spectral_r2,
)


class SpectralContractAuditTests(unittest.TestCase):
    def test_state_window_can_exceed_occupied_count(self):
        # One occupied band plus an empty band: total states need not <= electrons.
        below, above = energy_window_ratios(np.array([[1., 1.]]), [-1., 0., 1.], [1.])
        np.testing.assert_allclose(below, [1.])
        np.testing.assert_allclose(above, [1.])
        self.assertGreater(float((below + above)[0]), 1)

    def test_integration_uses_widths_and_rejects_zero_crossing_bin(self):
        below, above = energy_window_ratios(np.array([[2., 1.]]), [-2., 0., 3.], [2.])
        np.testing.assert_allclose(below, [2.])
        np.testing.assert_allclose(above, [1.5])
        with self.assertRaisesRegex(ValueError, "exact edge"):
            energy_window_ratios(np.array([[1., 1.]]), [-1., .1, 1.], [1.])

    def test_padding_alone_changes_r2_for_identical_active_errors(self):
        target = np.array([[1., 3., 0., 0.]])
        prediction = np.array([[2., 2., 0., 0.]])
        mask = np.array([[True, True, False, False]])
        result = padding_decomposition(prediction, target, mask)
        # Covered SST=2, full SST=6; SSE=2 in both. Covered R2~0, padded R2~2/3.
        np.testing.assert_allclose(result["covered"], [0.], atol=1e-8)
        np.testing.assert_allclose(result["full"], [2 / 3], atol=1e-8)
        np.testing.assert_allclose(result["denominator_effect"], [2 / 3], atol=1e-8)
        np.testing.assert_allclose(result["outside_penalty"], [0.])

    def test_outside_prediction_error_is_separate_from_padding_reference(self):
        target = np.array([[1., 3., 0., 0.]])
        prediction = np.array([[2., 2., 1., 1.]])
        result = padding_decomposition(prediction, target, np.array([[True, True, False, False]]))
        np.testing.assert_allclose(result["full"], [1 / 3], atol=1e-8)
        np.testing.assert_allclose(result["outside_penalty"], [1 / 3], atol=1e-8)

    def test_all_covered_bins_do_not_change_scores(self):
        target = np.array([[1., 3., 2.]])
        prediction = np.array([[1.5, 2., 2.5]])
        result = padding_decomposition(prediction, target, np.ones_like(target, dtype=bool))
        np.testing.assert_allclose(result["full"], result["covered"])
        np.testing.assert_allclose(result["denominator_effect"], 0)

    def test_zero_and_constant_targets_are_not_hidden_in_covered_scores(self):
        target = np.array([[0., 0.], [2., 2.]])
        result = padding_decomposition(np.ones_like(target), target, np.ones_like(target, dtype=bool))
        self.assertEqual(result["usable"].tolist(), [False, False])
        self.assertEqual(result["zero_target"].tolist(), [True, False])

    def test_internal_holes_are_not_exterior_padding(self):
        target = np.array([[0., 1., 0., 2., 0.]])
        mask = np.array([[False, True, False, True, False]])
        result = coverage_layout(target, mask)
        self.assertEqual(result["internal_holes"].tolist(), [1])
        self.assertEqual(result["exterior_uncovered"].tolist(), [2])

    def test_true_total_is_not_optimal_scale_when_shape_is_wrong(self):
        target = np.array([[1., 3.]])
        shape = np.array([[.5, .5]])
        result = fixed_shape_optimum(shape, target, np.array([10.]))
        np.testing.assert_allclose(result["optimal_total"], [4.])
        np.testing.assert_allclose(result["unbounded_r2"], [0.], atol=1e-8)
        # Different erroneous shape: optimal total is about 3.824, not true total 4.
        result = fixed_shape_optimum(np.array([[.2, .8]]), target, np.array([10.]))
        self.assertAlmostEqual(result["optimal_total"][0], 2.6 / .68)
        best = result["unbounded_r2"][0]
        oracle = spectral_r2(np.array([[.8, 3.2]]), target)[0]
        self.assertGreater(best, oracle)

    def test_bound_changes_reachability_without_changing_shape(self):
        target = np.array([[1., 3.]])
        result = fixed_shape_optimum(np.array([[.25, .75]]), target, np.array([2.]))
        np.testing.assert_allclose(result["unbounded_r2"], [1.])
        self.assertLess(result["bounded_r2"][0], 0)

    def test_invalid_values_fail_instead_of_silently_dropping_data(self):
        self.assertEqual(SPLITS, ("train", "valid"))
        for value in (np.array([[np.nan, 1.]]), np.array([[-1., 1.]])):
            with self.assertRaises(ValueError):
                spectral_r2(value, value)
        with self.assertRaisesRegex(ValueError, "zero-filled"):
            padding_decomposition(np.ones((1, 2)), np.ones((1, 2)), np.array([[True, False]]))

    def test_existing_zero_target_w1_has_a_nonphysical_shape_preference(self):
        # This documents the current boundary condition; it does not change loss.
        import torch
        from model.losses import sumnorm_klw_loss

        target = torch.zeros(1, 4)
        coverage = torch.ones(1, 4, dtype=torch.bool)
        first = torch.tensor([[.97, .01, .01, .01]]).log()
        last = torch.tensor([[.01, .01, .01, .97]]).log()
        kl_only = sumnorm_klw_loss(first, target, coverage, False, 0, 0, 1.)
        first_loss = sumnorm_klw_loss(first, target, coverage, False, 1, 0, 1.)
        last_loss = sumnorm_klw_loss(last, target, coverage, False, 1, 0, 1.)
        self.assertAlmostEqual(float(kl_only), 0.)
        self.assertAlmostEqual(float(first_loss), .985, places=6)
        self.assertAlmostEqual(float(last_loss), .265, places=6)


if __name__ == "__main__":
    unittest.main()
