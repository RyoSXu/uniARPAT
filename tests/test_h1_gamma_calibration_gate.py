"""Contracts for the preregistered H1 gamma calibration gate."""

import unittest

import numpy as np

from tools.eval.h1_gamma_calibration_gate import (
    adjudicate_gate,
    apply_gamma_calibration,
    fit_gamma_calibration,
    soft_binary_cross_entropy,
)


class H1GammaCalibrationGateTests(unittest.TestCase):
    def test_identity_mapping_preserves_interior_probabilities(self):
        gamma = np.array([0.02, 0.25, 0.5, 0.81, 0.98])

        calibrated = apply_gamma_calibration(gamma, a=1.0, b=0.0)

        np.testing.assert_allclose(calibrated, gamma, rtol=0.0, atol=1e-14)

    def test_fit_repairs_known_logit_shrinkage(self):
        target = np.linspace(0.04, 0.96, 200)
        target_logit = np.log(target) - np.log1p(-target)
        prediction = 1.0 / (1.0 + np.exp(-(0.55 * target_logit + 0.2)))

        fit = fit_gamma_calibration(prediction, target)
        calibrated = apply_gamma_calibration(prediction, fit["a"], fit["b"])

        self.assertAlmostEqual(fit["a"], 1.0 / 0.55, places=4)
        self.assertAlmostEqual(fit["b"], -0.2 / 0.55, places=4)
        self.assertLess(
            soft_binary_cross_entropy(calibrated, target),
            soft_binary_cross_entropy(prediction, target),
        )

    def test_gate_requires_effect_mechanism_and_invariants(self):
        control = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        candidate = control + 0.03
        control_error = np.array([0.50, 0.40, 0.30, 0.20, 0.10])
        candidate_error = control_error - 0.08
        invariants = {"shape": True, "phdos": True, "parameters": True}

        decision = adjudicate_gate(
            control,
            candidate,
            control_error,
            candidate_error,
            invariants,
            replicates=200,
            seed=7,
        )

        self.assertEqual(decision["verdict"], "win")
        self.assertTrue(decision["main_effect_met"])
        self.assertTrue(decision["mechanism_met"])

        below_margin = adjudicate_gate(
            control,
            control + 0.0199,
            control_error,
            candidate_error,
            invariants,
            replicates=100,
            seed=7,
        )
        broken_invariant = adjudicate_gate(
            control,
            candidate,
            control_error,
            candidate_error,
            {**invariants, "phdos": False},
            replicates=100,
            seed=7,
        )
        self.assertEqual(below_margin["verdict"], "park")
        self.assertEqual(broken_invariant["verdict"], "park")

    def test_one_percentage_point_failure_increase_is_not_allowed(self):
        control = np.linspace(0.1, 1.0, 100)
        candidate = control + 0.03
        candidate[0] = -0.01
        control_error = np.linspace(0.1, 0.5, 100)
        candidate_error = control_error - 0.02

        decision = adjudicate_gate(
            control,
            candidate,
            control_error,
            candidate_error,
            {"all": True},
            replicates=100,
            seed=8,
        )

        self.assertEqual(decision["fail_delta_pp"], 1.0)
        self.assertFalse(decision["failure_guard_met"])
        self.assertEqual(decision["verdict"], "park")

    def test_invalid_gamma_is_rejected(self):
        for values in (
            np.array([-0.1, 0.5]),
            np.array([0.5, 1.1]),
            np.array([0.5, np.nan]),
        ):
            with self.assertRaises(ValueError):
                apply_gamma_calibration(values, a=1.0, b=0.0)
        with self.assertRaises(ValueError):
            apply_gamma_calibration(np.array([0.5]), a=0.0, b=0.0)


if __name__ == "__main__":
    unittest.main()
