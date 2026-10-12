"""Safeguards for C0/C1/C2 local-structure diagnosis helpers."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.eval.local_structure_diagnosis import (
    apply_ridge_probe,
    calibrated_errors,
    fit_global_scale,
    grouped_mean_frame,
    ridge_probe_cv,
    strict_match_report,
    support_count_strata,
)


def pair_frame(pairs=4, groups=("a", "a", "b", "b")):
    return pd.DataFrame({
        "i": list(range(pairs)),
        "j": list(range(pairs, 2 * pairs)),
        "composition": list(groups),
        "left_id": [str(i) for i in range(pairs)],
        "right_id": [str(i) for i in range(pairs, 2 * pairs)],
    })


class TestLocalStructureDiagnosis(unittest.TestCase):
    def test_global_scale_recovers_amplitude_on_train(self):
        target = np.array([[1.0, 0.0], [0.0, 1.0]])
        pred = 0.25 * target
        scale = fit_global_scale(pred[0] - pred[1], target[0] - target[1])
        self.assertAlmostEqual(scale, 4.0)
        err = calibrated_errors(pred[0] - pred[1], target[0] - target[1], scale)
        self.assertAlmostEqual(err["difference_error"][0], 0.0)

    def test_ridge_probe_locked_on_valid(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(40, 4))
        shape = rng.normal(size=(40, 4))
        pairs = pair_frame()
        left, right = pairs.i.to_numpy(), pairs.j.to_numpy()
        for li, ri in zip(left, right):
            shape[ri] = shape[li] + features[li] - features[ri]
        model, alpha, _, info = ridge_probe_cv(features, pairs, shape, pairs.composition.to_numpy())
        _, err = apply_ridge_probe(model, features, pairs, shape)
        # The relation is exactly linear; ridge shrinkage at the smallest alpha of the
        # grid leaves ~1e-4 relative error, so machine precision is not the right bar.
        self.assertLess(float(np.median(err)), 1e-3)
        self.assertGreater(alpha, 0)
        self.assertEqual(sorted(info["cv_median_pair_error"]), sorted(
            [str(value) for value in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0)]))
        self.assertEqual(info["groups"], 2)
        self.assertEqual(info["n_splits"], 2)

    def test_ridge_probe_flags_grid_edge_selection(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(40, 4))
        shape = rng.normal(size=(40, 4))
        pairs = pair_frame()
        left, right = pairs.i.to_numpy(), pairs.j.to_numpy()
        for li, ri in zip(left, right):
            shape[ri] = shape[li] + features[li] - features[ri]
        _, alpha, _, info = ridge_probe_cv(features, pairs, shape, pairs.composition.to_numpy())
        self.assertTrue(info["selected_at_grid_edge"])
        self.assertIn(alpha, (1e-4, 1000.0))

    def test_grouped_mean_resamples_compositions(self):
        frame = pd.DataFrame({
            "composition": ["a", "a", "b"],
            "difference_error": [0.1, 0.3, 0.2],
        })
        result = grouped_mean_frame(frame, "difference_error")
        self.assertEqual(result["groups"], 2)
        self.assertAlmostEqual(result["mean_group"], 0.2)

    def test_strict_match_report_keeps_counts_and_cases(self):
        payload = {"strict_matches": {
            "train": {"candidate_pairs": 6, "strict_count": 1, "settings": {"ltol": 0.01},
                      "pairs": [{"left_id": "t1", "right_id": "t2", "strict_match": True,
                                 "normalized_rms": 0.0002, "volume_per_atom_ratio": 1.0,
                                 "edos_target_tv": 0.2, "phdos_target_tv": 0.1,
                                 "formula": "AgBiS2"},
                                {"left_id": "t3", "right_id": "t4", "strict_match": False,
                                 "normalized_rms": None, "volume_per_atom_ratio": 1.0,
                                 "edos_target_tv": 0.9, "phdos_target_tv": 0.1, "formula": "X"}]},
            "valid": {"candidate_pairs": 45, "strict_count": 12, "settings": {"ltol": 0.01},
                      "pairs": [{"left_id": "v1", "right_id": "v2", "strict_match": True,
                                 "normalized_rms": 0.001, "volume_per_atom_ratio": 1.0,
                                 "edos_target_tv": 0.1, "phdos_target_tv": 0.1, "formula": "A"},
                                {"left_id": "v3", "right_id": "v4", "strict_match": True,
                                 "normalized_rms": 0.02, "volume_per_atom_ratio": 1.0,
                                 "edos_target_tv": 0.3, "phdos_target_tv": 0.1, "formula": "B"}]},
        }}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verification_reproducible.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = strict_match_report(path)
        self.assertEqual(report["valid"]["strict_count"], 12)
        self.assertEqual(report["valid"]["candidate_pairs"], 45)
        self.assertEqual(len(report["valid"]["close_pairs_with_large_edos_gap"]), 2)
        self.assertEqual(report["valid"]["close_pairs_with_large_edos_gap"][0]["formula"], "B")
        self.assertEqual(report["train"]["strict_count"], 1)
        self.assertEqual(len(report["train"]["close_pairs_with_large_edos_gap"]), 1)

    def test_support_count_strata_bins(self):
        frame = pd.DataFrame({"material_id": ["m0", "m1", "m2", "m3"]})
        metadata = pd.DataFrame({"material_id": ["m0", "m1", "m2", "m3"],
                                 "same_composition_train_count": [0, 1, 2, 5]})
        for mode in ("blind", "oracle"):
            for task in ("edos", "phdos"):
                frame[f"{mode}_r2_{task}"] = [0.1, 0.2, 0.3, 0.4]
                frame[f"{mode}_mae_{task}"] = [1.0, 2.0, 3.0, 4.0]
        result = support_count_strata(frame, metadata)
        self.assertEqual(result["composition_count_0"]["n"], 1)
        self.assertEqual(result["composition_count_1"]["n"], 1)
        self.assertEqual(result["composition_count_ge2"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
