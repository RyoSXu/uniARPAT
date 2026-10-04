"""Negative controls for the precheck's feature comparison and frame lifting."""

import unittest

import numpy as np
import torch

from tools.eval.periodic_geometry_acceptance import e3_operations, pack_cell
from tools.eval.periodic_message_precheck import changed_geometry, compare_features, feature_error
from tools.eval.periodic_message_prototypes import STATE_IRREPS, geometry_from_pos


class PrecheckEvidenceContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_unchanged_vector_arrays_fail_under_external_rotation(self):
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            state = torch.zeros(1, 1, 108)
            state[0, 0, 64] = 1.
            output = torch.zeros(1, 1, 512)
            reference = (output, {"states": [state]})
            q = e3_operations()[1][1]
            wrong = compare_features(reference, reference, torch.float64, "direct", orthogonal=q)
            self.assertFalse(wrong["pass"])
            representation = STATE_IRREPS.D_from_matrix(torch.tensor(q.T.copy()))
            right = (output, {"states": [state@representation.T]})
            self.assertTrue(compare_features(reference, right, torch.float64, "direct", orthogonal=q)["pass"])
        finally:
            torch.set_default_dtype(old)

    def test_error_metric_detects_scalar_leak_nan_and_empty_shape(self):
        expected = torch.zeros(2, 3, dtype=torch.float64)
        actual = expected.clone()
        actual[0, 0] = .01
        self.assertFalse(feature_error(expected, actual, (1e-6, 1e-6))["pass"])
        actual[0, 0] = float("nan")
        with self.assertRaises(AssertionError):
            feature_error(expected, actual, (1e-6, 1e-6))
        self.assertTrue(feature_error(expected[:0], expected[:0], (1e-6, 1e-6))["pass"])

    def test_rebuilt_frames_preserve_physical_vectors_and_image_keys(self):
        basis = np.array([[4.1, 0., 0.], [-1.2, 5.3, 0.], [.7, .9, 6.2]])
        frac = np.array([[.13, .27, .39], [.57, .63, .79], [.89, .11, .43]])
        pos = pack_cell(basis, frac, torch.float64)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        g = geometry_from_pos(pos, mask)
        for kind in ("permutation", "integer_images", "basis_swap", "basis_shear"):
            _, _, metrics = changed_geometry(pos, mask, g, kind)
            self.assertEqual(metrics["status"], "pass")
        for name, q, t in e3_operations():
            _, _, metrics = changed_geometry(pos, mask, g, "physical_e3", q, t)
            self.assertEqual(metrics["status"], "pass", name)


if __name__ == "__main__":
    unittest.main()
