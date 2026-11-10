"""uMLIP phonon baseline (phase A): dataset binning rule and supercell sizing."""
from __future__ import annotations

import unittest

import numpy as np

from tools.data.umlip_phonon_baseline import BINS, WINDOW, bin_dos, supercell_matrix
from tools.eval.umlip_phonon_compare import oracle_scale


class Binning(unittest.TestCase):
    def test_constant_density_and_window(self):
        x = np.arange(WINDOW[0] - 5, WINDOW[1] + 5, 0.5)
        binned = bin_dos(x, np.full_like(x, 2.0))
        self.assertEqual(binned.shape, (BINS,))
        # in-bin trapezoid without edge interpolation loses less than two pitches per bin (dataset rule)
        self.assertTrue(np.all(binned > 2.0 * (1 - 1.0 / 19.6875)) and np.all(binned <= 2.0))

    def test_single_and_empty_bins(self):
        edges = np.linspace(WINDOW[0], WINDOW[1], BINS + 1)
        binned = bin_dos(np.array([edges[3] + 1.0]), np.array([5.0]))
        self.assertEqual(binned[3], 5.0)
        self.assertEqual(np.count_nonzero(binned), 1)


class Supercell(unittest.TestCase):
    def test_heights_reach_minimum(self):
        cell = np.diag([3.0, 5.0, 12.0])
        np.testing.assert_array_equal(np.diag(supercell_matrix(cell, 10.0)), [4, 2, 1])
        skew = np.array([[4.0, 0, 0], [2.0, 3.0, 0], [0, 0, 4.0]])
        matrix = np.diag(supercell_matrix(skew, 10.0))
        volume = abs(np.linalg.det(skew))
        heights = [volume / np.linalg.norm(np.cross(skew[(i + 1) % 3], skew[(i + 2) % 3])) for i in range(3)]
        self.assertTrue(all(m * h >= 10.0 for m, h in zip(matrix, heights)))


class OracleScale(unittest.TestCase):
    def test_matches_target_total(self):
        prediction, target = np.array([[1.0, 3.0], [0.0, 0.0]]), np.array([[2.0, 2.0], [1.0, 1.0]])
        scaled = oracle_scale(prediction, target)
        np.testing.assert_allclose(scaled[0].sum(), 4.0)
        np.testing.assert_allclose(scaled[1], [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
