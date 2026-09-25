"""Contracts for paired valid-only slope-pilot decisions."""

import unittest

import numpy as np
import pandas as pd

from tools.eval.edos_slope_pilot_verdict import (
    analyze_paired_samples,
    paired_bootstrap_interval,
)


def _sample_frame():
    return pd.DataFrame(
        {
            "mpid": ["a", "b", "c", "d"],
            "edos_spectral_roughness": [0.1, 0.2, 0.7, 0.8],
            "r2_edos_oracle_unmasked": [0.4, 0.3, 0.2, -0.1],
            "r2_edos_blind_unmasked": [0.35, 0.25, 0.1, -0.05],
            "r2_phdos_oracle_unmasked": [0.7, 0.6, 0.5, 0.4],
            "r2_phdos_blind_unmasked": [0.65, 0.55, 0.45, 0.35],
            "edos_roughness_bias": [-0.1, -0.2, -0.3, -0.4],
            "edos_slope_error_high_gradient_mae": [0.2, 0.3, 0.6, 0.7],
        }
    )


class TestEdosSlopePilotVerdict(unittest.TestCase):
    def test_paired_bootstrap_is_reproducible(self):
        control = np.array([0.2, -0.1, 0.4, 0.3])
        candidate = control + np.array([0.05, 0.1, -0.02, 0.03])

        first = paired_bootstrap_interval(control, candidate, replicates=200, seed=19)
        second = paired_bootstrap_interval(control, candidate, replicates=200, seed=19)

        self.assertEqual(first, second)

    def test_improvement_with_mechanism_support_is_win(self):
        control = _sample_frame()
        candidate = control.copy()
        candidate.loc[2:, "r2_edos_oracle_unmasked"] = [0.24, 0.12]
        candidate.loc[2:, "edos_slope_error_high_gradient_mae"] = [0.35, 0.45]

        metrics, verdict = analyze_paired_samples(
            control, candidate, replicates=200, seed=3, roughness_threshold=0.5
        )

        self.assertEqual(verdict["verdict"], "win")
        self.assertTrue(verdict["primary_met"])
        self.assertTrue(verdict["mechanism_supported"])
        self.assertIn("roughness_train_p90", set(metrics["population"]))

    def test_secondary_guard_breach_is_reject(self):
        control = _sample_frame()
        candidate = control.copy()
        candidate["r2_phdos_blind_unmasked"] -= 0.04

        _, verdict = analyze_paired_samples(
            control, candidate, replicates=100, seed=4, roughness_threshold=0.5
        )

        self.assertEqual(verdict["verdict"], "reject")
        self.assertFalse(verdict["guardrails_met"])

    def test_mismatched_sample_order_is_rejected(self):
        control = _sample_frame()
        candidate = _sample_frame().iloc[::-1].reset_index(drop=True)

        with self.assertRaisesRegex(ValueError, "IDs/order"):
            analyze_paired_samples(control, candidate, replicates=20, roughness_threshold=0.5)


if __name__ == "__main__":
    unittest.main()
