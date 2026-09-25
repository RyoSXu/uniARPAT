import unittest

import numpy as np
import pandas as pd

from tools.eval.edos_error_attribution import (
    bootstrap_difference_ci,
    bootstrap_median_ci,
    masked_spectral_r2,
    ordered_valid_reference_r2,
    shape_error_descriptors,
    spectral_descriptors,
    structural_descriptors,
)


class EdosErrorAttributionTests(unittest.TestCase):
    def test_spectral_descriptors_are_scale_invariant(self):
        spectra = np.zeros((2, 16), dtype=np.float64)
        spectra[0, 7] = 1.0
        spectra[1] = 1.0

        original = spectral_descriptors(spectra, edge_bins=2)
        scaled = spectral_descriptors(spectra * 17.0, edge_bins=2)

        for name in original:
            np.testing.assert_allclose(original[name], scaled[name])
        self.assertAlmostEqual(original["roughness"][0], 1.0)
        self.assertAlmostEqual(original["entropy_norm"][0], 0.0)
        self.assertAlmostEqual(original["entropy_norm"][1], 1.0)
        self.assertAlmostEqual(original["peak_share"][0], 1.0)

    def test_shape_error_descriptors_are_zero_for_matching_shapes(self):
        target = np.array([[0.0, 0.25, 0.5, 0.25, 0.0]])

        original = shape_error_descriptors(target, target)
        rescaled = shape_error_descriptors(target * 3.0, target * 7.0)

        for name, values in original.items():
            np.testing.assert_allclose(values, rescaled[name])
        np.testing.assert_allclose(original["roughness_bias"], [0.0])
        np.testing.assert_allclose(original["peak_share_bias"], [0.0])
        np.testing.assert_allclose(original["peak_bin_shift_abs"], [0.0])
        np.testing.assert_allclose(original["slope_error_mae"], [0.0])

    def test_shape_error_descriptors_detect_smoother_prediction(self):
        target = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
        prediction = np.array([[0.0, 0.25, 0.5, 0.25, 0.0]])

        result = shape_error_descriptors(prediction, target)

        self.assertLess(result["roughness_bias"][0], 0.0)
        self.assertLess(result["peak_share_bias"][0], 0.0)
        self.assertEqual(result["peak_bin_shift_abs"][0], 0.0)

    def test_shape_error_descriptors_report_peak_shift_and_gradient_regions(self):
        target = np.array([[0.0, 0.8, 0.2, 0.0, 0.0]])
        prediction = np.array([[0.0, 0.4, 0.2, 0.2, 0.2]])
        shifted_target = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        shifted_prediction = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])

        result = shape_error_descriptors(prediction, target, high_gradient_fraction=0.25)
        shifted = shape_error_descriptors(shifted_prediction, shifted_target)

        self.assertGreater(result["slope_error_high_gradient_mae"][0], result["slope_error_other_gradient_mae"][0])
        self.assertGreaterEqual(result["slope_error_high_gradient_share"][0], 0.0)
        self.assertLessEqual(result["slope_error_high_gradient_share"][0], 1.0)
        self.assertEqual(shifted["peak_bin_shift"][0], 1.0)

    def test_masked_r2_ignores_unsupported_bins(self):
        target = np.array([[1.0, 3.0, 0.0]])
        prediction = np.array([[1.0, 3.0, 100.0]])
        mask = np.array([[True, True, False]])

        score = masked_spectral_r2(prediction, target, mask)

        np.testing.assert_allclose(score, [1.0])

    def test_structural_descriptor_uses_inverse_c_contract_for_c_axis(self):
        elements = np.zeros((1, 82), dtype=np.int64)
        elements[0, 2] = 14
        positions = np.zeros((1, 82, 3), dtype=np.float64)
        positions[0, 0] = [2.0, 2.0, 0.5]
        positions[0, 1] = [90.0, 90.0, 90.0]

        descriptors = structural_descriptors(elements, positions)

        self.assertEqual(descriptors["natoms"][0], 1.0)
        self.assertEqual(descriptors["n_unique_elements"][0], 1.0)
        self.assertAlmostEqual(descriptors["volume_per_atom"][0], 8.0)

    def test_bootstrap_interval_is_reproducible(self):
        high = np.array([0.0, 0.1, 0.2, 0.3])
        other = np.array([0.5, 0.6, 0.7, 0.8])

        first = bootstrap_difference_ci(high, other, "median", replicates=100, seed=7)
        second = bootstrap_difference_ci(high, other, "median", replicates=100, seed=7)

        self.assertEqual(first, second)
        self.assertLess(first[1], 0.0)

    def test_bootstrap_median_interval_is_reproducible(self):
        values = np.array([-0.5, -0.4, -0.3, -0.2, -0.1])

        first = bootstrap_median_ci(values, replicates=100, seed=11)
        second = bootstrap_median_ci(values, replicates=100, seed=11)

        self.assertEqual(first, second)
        self.assertLess(first[0], np.median(values))
        self.assertGreater(first[1], np.median(values))

    def test_valid_reference_r2_is_sorted_by_sample_index(self):
        reference = pd.DataFrame(
            {
                "sample_index": [1, 0, 1, 0],
                "task": ["edos", "edos", "phdos", "phdos"],
                "normalized_r2": [0.7, 0.4, 0.8, 0.5],
            }
        )

        result = ordered_valid_reference_r2(reference, expected_count=2)

        np.testing.assert_allclose(result, [0.4, 0.7])

    def test_valid_reference_r2_rejects_incomplete_sample_order(self):
        reference = pd.DataFrame(
            {"sample_index": [0, 2], "task": ["edos", "edos"], "normalized_r2": [0.4, 0.7]}
        )

        with self.assertRaisesRegex(ValueError, "sample order"):
            ordered_valid_reference_r2(reference, expected_count=2)


if __name__ == "__main__":
    unittest.main()
