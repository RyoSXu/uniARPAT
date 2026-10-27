"""Locked comparability rule: valid must be both statistically and materially smaller."""
import unittest

import numpy as np
import pandas as pd

from tools.eval.shape_aux_group_comparability import (
    classify,
    clearly_smaller,
    group_table,
    mean_difference_interval,
    relative_gap,
)


class TestShapeGapComparability(unittest.TestCase):
    def test_relative_gap_is_symmetric_and_zero_for_copies(self):
        scalar = relative_gap(np.array([2.0, 4.0]), np.array([2.0, 2.0]))
        self.assertEqual(scalar[0], 0.0)
        self.assertAlmostEqual(scalar[1], relative_gap(np.array([2.0]), np.array([4.0]))[0])
        left = np.array([[1.0, 0.0], [0.0, 2.0]])
        right = np.array([[1.0, 0.0], [2.0, 0.0]])
        vector = relative_gap(left, right)
        self.assertEqual(vector[0], 0.0)
        self.assertGreater(vector[1], 0.0)

    def test_group_skill_ignores_near_zero_targets(self):
        frame = pd.DataFrame({
            "composition": ["a", "a", "b"],
            "target_tv": [1e-8, 0.4, 0.2],
            "skill": [5.0, 0.5, -0.1],
            "volume_gap": [0.2, 0.4, 0.1],
            "radial_gap": [0.2, 0.2, 0.2],
            "flagged": [False, False, True],
        })
        groups = group_table(frame).set_index("composition")
        self.assertAlmostEqual(groups.loc["a", "skill"], 0.5)
        self.assertAlmostEqual(groups.loc["a", "target_tv"], 0.4)
        self.assertAlmostEqual(groups.loc["a", "volume_gap"], 0.3)

    def test_clearly_smaller_requires_interval_and_median_ratio(self):
        valid = np.array([0.4, 0.5])
        train = np.array([1.0, 1.1])
        interval = mean_difference_interval(valid, train, draws=200)
        self.assertTrue(clearly_smaller(valid, train, interval))
        mild = np.array([0.95, 0.96])
        mild_interval = {"ci95": [-0.2, -0.01]}
        self.assertFalse(clearly_smaller(mild, train, mild_interval))
        crossed = {"ci95": [-0.3, 0.01]}
        self.assertFalse(clearly_smaller(valid, train, crossed))

    def test_classification_stops_either_way(self):
        self.assertEqual(classify({"volume_gap": False, "radial_gap": False, "target_tv": False}, 10),
                         "same_class")
        self.assertEqual(classify({"volume_gap": True, "radial_gap": False, "target_tv": False}, 10),
                         "data_difference")
        self.assertEqual(classify({"volume_gap": False}, 1), "unresolved")


if __name__ == "__main__":
    unittest.main()
