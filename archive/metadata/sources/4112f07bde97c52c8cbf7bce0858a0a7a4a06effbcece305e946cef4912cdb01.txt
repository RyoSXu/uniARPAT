"""Analytical and adversarial CPU contracts for periodic geometry at 6 Angstrom."""

import unittest

import numpy as np
import torch

from tools.eval.periodic_geometry_acceptance import (
    TOLERANCES, checked_records, compare_records, e3_operations,
    independent_reference, pack_cell, physical_e3_case,
    representation_case, select_indices, supercell_cases,
)


def position(lengths=(4.1, 5.3, 6.7), angles=(79., 97., 108.), dtype=torch.float64):
    result = torch.zeros(1, 5, 3, dtype=dtype)
    result[0, 0] = torch.tensor([lengths[0], lengths[1], 1/lengths[2]], dtype=dtype)
    result[0, 1] = torch.tensor(angles, dtype=dtype)
    result[0, 2:] = torch.tensor([[.13, .27, .39], [.57, .63, .79], [.89, .11, .43]], dtype=dtype)
    return result


class GeometryAcceptanceContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_analytical_cubic_records_at_six_angstrom(self):
        for dtype in (torch.float32, torch.float64):
            pos = position((3., 3., 3.), (90., 90., 90.), dtype)[:, :3]
            actual = checked_records(pos, torch.zeros(1, 1, dtype=torch.bool))
            shifts = actual["keys"][:, 3:]
            expected = {(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)} - {(0, 0, 0)}
            self.assertEqual({tuple(shift) for shift in shifts}, expected)
            self.assertEqual(actual["metrics"]["n_edges"], 26)
            self.assertEqual(actual["metrics"]["cross_cell_self_records"], 26)
            np.testing.assert_allclose(actual["r"], shifts*3, atol=1e-5)

    def test_independent_enumeration_skew_aspect_and_unwrapped_coordinates(self):
        for cell in (np.diag([1.1, 4.2, 8.3]),
                     np.array([[3.4, 0, 0], [2.8, 1.3, 0], [-.8, .9, 5.1]]),
                     np.array([[4.1, 0, 0], [-1.2, 5.3, 0], [.7, .9, 6.2]])):
            pos = pack_cell(cell, np.array([[2.13, -.27, .39], [-.57, 1.63, .79]]), torch.float64)
            mask = torch.zeros(1, 2, dtype=torch.bool)
            actual = checked_records(pos, mask)
            reference = independent_reference(actual["cell"], actual["frac"], mask.numpy())
            result = compare_records(reference, actual, tolerance=TOLERANCES[torch.float64])
            self.assertEqual(result["status"], "pass")

    def test_e3_distances_and_lifted_vectors_including_reflections(self):
        for dtype in (torch.float32, torch.float64):
            pos, mask = position(dtype=dtype), torch.zeros(1, 3, dtype=torch.bool)
            baseline = checked_records(pos, mask)
            for name, orthogonal, translation in e3_operations():
                with self.subTest(dtype=dtype, operation=name):
                    result = physical_e3_case(pos, mask, baseline, orthogonal, translation, 6.)
                    self.assertEqual(result["status"], "pass")
                    self.assertLess(result["max_vector_error_A"], TOLERANCES[dtype])
        self.assertEqual(sum(np.linalg.det(q) < 0 for _, q, _ in e3_operations()), 3)

    def test_fixed_frame_arrays_cannot_replace_vector_equivariance(self):
        pos, mask = position(), torch.zeros(1, 3, dtype=torch.bool)
        baseline = checked_records(pos, mask)
        _, orthogonal, _ = e3_operations()[1]
        with self.assertRaisesRegex(AssertionError, "matched-record error"):
            compare_records(dict(baseline, r=baseline["r"] @ orthogonal), baseline)

    def test_representation_checks_align_exact_ids_not_distance_multisets(self):
        for dtype in (torch.float32, torch.float64):
            pos, mask = position(dtype=dtype), torch.tensor([[False, True, False]])
            pos[0, 3] = float("nan")
            baseline = checked_records(pos, mask)
            for kind in ("permutation", "integer_images", "basis_swap", "basis_shear"):
                with self.subTest(dtype=dtype, representation=kind):
                    result = representation_case(pos, mask, baseline, kind)
                    self.assertEqual(result["status"], "pass")

    def test_supercell_is_compared_per_receiver_copy(self):
        for dtype in (torch.float32, torch.float64):
            pos, mask = position(dtype=dtype), torch.zeros(1, 3, dtype=torch.bool)
            baseline = checked_records(pos, mask)
            for result in supercell_cases(pos, mask, baseline):
                self.assertEqual(result["status"], "pass")
                self.assertEqual(result["matched_records"], len(baseline["keys"]))
                self.assertEqual(result["supercell_total_records"], 2*len(baseline["keys"]))

    def test_mixed_batch_and_noncontiguous_padding_exclude_nan_slots(self):
        pos = position(dtype=torch.float32).repeat(2, 1, 1)
        mask = torch.tensor([[False, True, False], [True, True, True]])
        pos[0, 3] = float("nan")
        pos[1, 2:] = float("nan")
        actual = checked_records(pos, mask)
        self.assertTrue((actual["keys"][:, 0] == 0).all())
        self.assertFalse((actual["keys"][:, 1:3] == 1).any())
        reference = independent_reference(actual["cell"], actual["frac"], mask.numpy())
        self.assertEqual(compare_records(reference, actual)["status"], "pass")

    def test_empty_fields_for_sparse_all_padding_and_zero_slots(self):
        cases = [(position((10., 10., 10.), (90., 90., 90.))[:, :3], torch.zeros(1, 1, dtype=torch.bool)),
                 (position(), torch.ones(1, 3, dtype=torch.bool)),
                 (position()[:, :2], torch.zeros(1, 0, dtype=torch.bool))]
        for pos, mask in cases:
            actual = checked_records(pos, mask)
            self.assertEqual(actual["keys"].shape, (0, 6))
            self.assertEqual(actual["d"].shape, (0,))
            self.assertEqual(actual["r"].shape, (0, 3))

    def test_exact_cutoff_is_excluded(self):
        for dtype in (torch.float32, torch.float64):
            pos = position((6., 6., 6.), (90., 90., 90.), dtype)[:, :3]
            actual = checked_records(pos, torch.zeros(1, 1, dtype=torch.bool))
            self.assertEqual(len(actual["keys"]), 0)

    def test_invalid_inputs_rejected_before_calling_existing_builder(self):
        pos, mask = position(), torch.zeros(1, 3, dtype=torch.bool)
        for radius in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                checked_records(pos, mask, radius)
        pos[0, 2, 0] = float("nan")
        with self.assertRaises(ValueError):
            checked_records(pos, mask)

    def test_missing_edges_and_reversed_directions_are_detected(self):
        baseline = checked_records(position(), torch.zeros(1, 3, dtype=torch.bool))
        missing = {key: baseline[key][1:] for key in ("keys", "d", "r")}
        with self.assertRaisesRegex(AssertionError, "non-boundary record mismatch"):
            compare_records(baseline, missing)
        with self.assertRaisesRegex(AssertionError, "matched-record error"):
            compare_records(baseline, dict(baseline, r=-baseline["r"]))
        duplicated = {key: np.concatenate([baseline[key], baseline[key][:1]]) for key in ("keys", "d", "r")}
        with self.assertRaisesRegex(AssertionError, "duplicate mapped record"):
            compare_records(baseline, duplicated)

    def test_boundary_mismatch_is_reported_as_limited(self):
        reference = {"keys": np.array([[0, 0, 0, 1, 0, 0]]),
                     "d": np.array([6.-1e-7]), "r": np.array([[6.-1e-7, 0, 0]])}
        actual = {"keys": np.empty((0, 6), dtype=np.int64), "d": np.empty(0), "r": np.empty((0, 3))}
        result = compare_records(reference, actual)
        self.assertEqual(result["status"], "boundary_limited")
        self.assertEqual(result["boundary_missing"], 1)

    def test_real_sample_indices_remain_integers_with_empty_extra_list(self):
        for extra in ([], [3]):
            indices = select_indices(10, 3, [1, 9], extra)
            self.assertTrue(all(isinstance(index, int) for index in indices))
            self.assertEqual(indices, sorted(set(indices)))
            array = np.arange(10)
            for index in indices:
                self.assertEqual(array[index:index+1].item(), index)


if __name__ == "__main__":
    unittest.main()
