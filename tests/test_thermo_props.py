import unittest
import numpy as np
from thermo_props import ThermodynamicCalculator

class TestThermoProps(unittest.TestCase):
    def setUp(self):
        self.calc = ThermodynamicCalculator()

    def test_slack_formula(self):
        # Create a mock phDOS with a reasonable Debye temperature (~500 K)
        phdos = np.zeros(64)
        # Put phonon modes around 300 cm^-1 (idx ~29)
        idx_300 = int((300.0 - (-280.0)) / 20.0)
        phdos[idx_300] = 1.0

        kappa = self.calc.compute_Slack_kappaL(
            phdos=phdos,
            M_avg=28.0855,
            volume_per_atom=20.0,
            n_atoms=2,
            T=300.0,
            gamma=1.5,
            A=3.1e-6
        )
        self.assertTrue(1.0 < kappa < 500.0, f"Calculated kappa {kappa} out of physical range!")

    def test_acoustic_suppression_factor(self):
        # Compare n_atoms=1 vs n_atoms=8 (should decrease by factor of 8^(2/3) = 4.0)
        phdos = np.zeros(64)
        idx_300 = int((300.0 - (-280.0)) / 20.0)
        phdos[idx_300] = 1.0

        kappa_1 = self.calc.compute_Slack_kappaL(phdos=phdos, M_avg=50.0, volume_per_atom=20.0, n_atoms=1)
        kappa_8 = self.calc.compute_Slack_kappaL(phdos=phdos, M_avg=50.0, volume_per_atom=20.0, n_atoms=8)
        ratio = kappa_1 / kappa_8
        self.assertAlmostEqual(ratio, 4.0, places=4, msg="n_atoms^(2/3) suppression scaling failed!")

    def test_heat_capacity_high_temp_limit(self):
        # At high temperature (T = 3000 K), Cv should approach Dulong-Petit limit 3 * R ~= 24.94 J/(mol-atom*K)
        frequencies = np.linspace(-280, 980, 64)
        phdos = np.zeros(64)
        # Positive modes between 200 and 600 cm^-1
        mask = (frequencies >= 200) & (frequencies <= 600)
        phdos[mask] = 1.0
        # Normalize so integral is 3
        d_omega = frequencies[1] - frequencies[0]
        phdos = phdos * (3.0 / (np.sum(phdos) * d_omega))

        cv_curve = self.calc.compute_Cv(phdos, T_range=np.array([3000.0]))
        cv_high_t = float(cv_curve[0])
        R_gas = 8.314462618
        dulong_petit = 3.0 * R_gas
        # Cv should be within 5% of Dulong-Petit limit at 3000 K
        self.assertAlmostEqual(cv_high_t, dulong_petit, delta=0.5, msg="High-temperature Cv did not converge to Dulong-Petit limit!")

if __name__ == '__main__':
    unittest.main()
