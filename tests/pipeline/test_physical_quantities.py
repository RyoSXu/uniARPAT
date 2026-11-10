"""Physical quantities from DOS (phase A evaluation): thermodynamic limits and DOS gap calls."""
from __future__ import annotations

import unittest

import numpy as np

from tools.eval.physical_quantities import E_BINS, PH_BINS, R_GAS, centers, electronic, thermo, E_WINDOW, PH_WINDOW


class Thermo(unittest.TestCase):
    def test_low_frequency_reaches_dulong_petit(self):
        omega, _ = centers(PH_WINDOW, PH_BINS)
        dos = np.zeros((1, PH_BINS))
        dos[0, np.argmax(omega > 0)] = 5.0  # all modes in the lowest positive bin (about 10 cm^-1)
        cv = thermo(dos, np.array([2.0]), 300.0)['cv'][0]
        self.assertAlmostEqual(cv / (3 * R_GAS), 1.0, places=3)

    def test_scale_invariance(self):
        rng = np.random.default_rng(0)
        dos = rng.random((1, PH_BINS))
        a, b = thermo(dos, np.array([3.0]), 300.0), thermo(4 * dos, np.array([3.0]), 300.0)
        self.assertAlmostEqual(a['cv'][0], b['cv'][0])
        self.assertAlmostEqual(a['s'][0], b['s'][0])


class Gap(unittest.TestCase):
    def test_fermi_at_valence_top_and_metal(self):
        energy, width = centers(E_WINDOW, E_BINS)
        semiconductor = np.where((energy > 0) & (energy < 1.5), 0.0, 1.0)
        metal = np.ones(E_BINS)
        gaps = electronic(np.stack([semiconductor, metal]), np.array([1.0, 1.0]))['gap_0.01']
        self.assertAlmostEqual(gaps[0], 16 * width)  # bins centered in (0, 1.5) eV
        self.assertEqual(gaps[1], 0.0)


if __name__ == '__main__':
    unittest.main()
