"""D4 read-only phDOS descriptor and B7 stratum contracts."""

import unittest

import numpy as np

from tools.eval.d4_phdos_spike_imaginary_audit import (
    b7_strata,
    p0_negative_center_count,
    phdos_descriptors,
    train_p90_thresholds,
)


class TestD4PhdosAudit(unittest.TestCase):
    def test_negative_centers_and_zero_total_are_safe(self):
        edges = np.linspace(-280.0, 980.0, 65)
        self.assertEqual(p0_negative_center_count(edges), 14)
        target = np.zeros((2, 64), dtype=np.float64)
        target[1, 0] = 2.0
        target[1, 20] = 1.0
        result = phdos_descriptors(target, np.ones_like(target, dtype=bool), 14)
        self.assertTrue(np.isfinite(result["negative_mass_fraction"]).all())
        self.assertEqual(result["negative_mass_fraction"][0], 0.0)
        self.assertAlmostEqual(result["negative_mass_fraction"][1], 2.0 / 3.0)

    def test_train_threshold_and_actionable_failure_delta(self):
        descriptors = {
            "total": np.ones(10),
            "negative_mass_fraction": np.arange(10, dtype=float),
            "peak_share": np.arange(10, dtype=float) / 10.0,
            "uncovered_mass_fraction": np.zeros(10),
        }
        thresholds = train_p90_thresholds(descriptors)
        self.assertAlmostEqual(thresholds["negative_mass_fraction"], 8.1)
        # Deliberately separate the two groups: only high values fail.
        r2 = np.array([1.0] * 8 + [-0.1, -0.2])
        summary = b7_strata(descriptors, thresholds, r2)
        row = summary[(summary.metric == "negative_mass_fraction") &
                      (summary.stratum == "high_minus_other")].iloc[0]
        self.assertGreater(row.fail_rate_phdos, 3.0)
        self.assertGreater(row.delta_fail_ci95_low, 0.0)


if __name__ == "__main__":
    unittest.main()
