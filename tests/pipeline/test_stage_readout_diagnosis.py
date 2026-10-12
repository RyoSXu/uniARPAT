"""Safeguards for the frozen stage readout diagnosis helpers.

Synthetic cases only: they pin the antisymmetric zero-intercept readout contract,
the composition-group-equal weighting, fold-local preprocessing, the train-only
regularization selection with its boundary rule, and the near-zero separation.
"""
import unittest

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import Ridge

from tools.eval.stage_readout_diagnosis import (
    ALPHAS,
    NEAR_ZERO_TV,
    StageFit,
    atom_ridge_path,
    decoder_ridge_path,
    feature_scale,
    fold_score,
    group_equal_weights,
    group_mean,
    identity_checks,
    method_metrics,
    pair_index_table,
    pair_targets,
    pooled_mean_rms,
    predict_materials,
    predict_pairs,
)


def synthetic_pairs(n_groups=6, per_group=3, dim=5, positions=4, seed=0):
    rng = np.random.default_rng(seed)
    materials = n_groups * per_group
    features = rng.normal(size=(materials, dim)).astype(np.float32)
    decoder = rng.normal(size=(materials, positions, dim)).astype(np.float32)
    shape = rng.dirichlet(np.ones(positions), size=materials)
    keys = [f"g{i // per_group}" for i in range(materials)]
    ids = [f"m{i}" for i in range(materials)]
    frame, sizes = pair_index_table(keys, ids)
    return features, decoder, shape.astype(np.float64), frame, sizes


class TestStageReadoutDiagnosis(unittest.TestCase):
    def test_pair_table_lists_all_within_group_pairs(self):
        _, _, _, frame, sizes = synthetic_pairs(n_groups=3, per_group=3)
        self.assertEqual(len(frame), 3 * 3)
        self.assertEqual(sorted(sizes.values()), [3, 3, 3])
        self.assertEqual(frame.composition.nunique(), 3)

    def test_pooled_mean_rms_ignores_padding(self):
        value = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, -50.0]]])
        mask = torch.tensor([[False, False, True]])
        pooled = pooled_mean_rms(value, mask)
        expected = torch.tensor([[2.0, 3.0, np.sqrt(5.0), np.sqrt(10.0)]], dtype=torch.float32)
        self.assertTrue(torch.allclose(pooled, expected))

    def test_readout_is_antisymmetric_and_zero_on_identical_features(self):
        features, decoder, shape, frame, _ = synthetic_pairs()
        target, _ = pair_targets(shape, frame)
        for stage, values in (("local_last", features), ("decoder_edos", decoder)):
            weights_shape = (values.shape[1], target.shape[1]) if values.ndim == 2 else (
                values.shape[1], values.shape[2])
            weights = np.ones(weights_shape) * 0.1
            scale = feature_scale(values, np.arange(len(values)))
            checks = identity_checks(stage, values, scale, weights, frame)
            self.assertLessEqual(checks["swap_antisymmetry_max_abs"], 1e-12)
            self.assertLessEqual(checks["identical_representation_max_abs"], 1e-12)

    def test_atom_ridge_matches_sklearn_weighted(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40, 6))
        y = rng.normal(size=(40, 3))
        weights = rng.uniform(0.5, 1.5, size=40)
        weights *= len(weights) / weights.sum()
        for alpha in (1e-4, 1.0, 1e3):
            ours = atom_ridge_path(x, y, (alpha,), weights)[alpha]
            reference = Ridge(alpha=alpha, fit_intercept=False).fit(x, y, sample_weight=weights)
            self.assertTrue(np.allclose(ours, reference.coef_.T, rtol=1e-8, atol=1e-10))

    def test_decoder_ridge_matches_sklearn_per_position(self):
        rng = np.random.default_rng(2)
        n, positions, dim = 30, 5, 4
        features = rng.normal(size=(n, positions, dim)).astype(np.float32)
        shape = rng.dirichlet(np.ones(positions), size=n)
        left = np.arange(0, n, 2)
        right = np.arange(1, n, 2)
        groups = np.array([f"g{i % 3}" for i in range(len(left))])
        weights = group_equal_weights(groups)
        scale = feature_scale(features, np.arange(n))
        target = shape[left] - shape[right]
        alpha = 0.7
        ours = decoder_ridge_path(features, left, right, scale, target, (alpha,), weights)[alpha]
        for position in range(positions):
            block = features[:, position, :].astype(np.float64)
            design = (block[left] - block[right]) / scale[position]
            reference = Ridge(alpha=alpha, fit_intercept=False).fit(
                design, target[:, position], sample_weight=weights)
            self.assertTrue(np.allclose(ours[position], reference.coef_, rtol=1e-8, atol=1e-10))

    def test_group_equal_weights_balance_composition_groups(self):
        groups = ["a"] * 4 + ["b"] * 2 + ["c"]
        weights = group_equal_weights(np.asarray(groups))
        totals = pd.Series(weights).groupby(pd.Series(groups)).sum()
        self.assertTrue(np.allclose(totals.to_numpy(), totals.to_numpy()[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_feature_scale_is_rms_without_centering(self):
        values = np.array([[1.0, 2.0], [3.0, 6.0]])
        scale = feature_scale(values, np.arange(2))
        expected = np.sqrt(np.array([5.0, 20.0]))
        self.assertTrue(np.allclose(scale, expected))
        shifted = values + 100.0
        self.assertFalse(np.allclose(feature_scale(shifted, np.arange(2)), scale))

    def test_group_mean_gives_groups_equal_weight(self):
        values = np.array([0.0, 0.0, 10.0])
        groups = np.array(["a", "a", "b"])
        self.assertAlmostEqual(group_mean(values, groups), 5.0)
        self.assertAlmostEqual(group_mean(values, groups, values > 1.0), 10.0)

    def test_fold_score_separates_near_zero_targets(self):
        target = np.array([[1.0, 0.0], [1e-9, 0.0]])
        pred = np.zeros_like(target)
        tv = 0.5 * np.abs(target).sum(-1)
        groups = np.array(["a", "b"])
        # Only the non-near-zero pair scores; its zero-response error is 0.5.
        self.assertAlmostEqual(fold_score(pred, target, tv, groups), 0.5)

    def test_stage_fit_keeps_composition_pairs_in_one_fold(self):
        features, _, shape, frame, _ = synthetic_pairs(n_groups=6, per_group=3)
        fit = StageFit("local_last", features, frame, shape)
        valid_counts = np.zeros(len(frame), dtype=int)
        for fold in fit.folds:
            valid_counts[fold["valid"]] += 1
            train_groups = set(frame.composition.to_numpy()[fold["train"]])
            valid_groups = set(frame.composition.to_numpy()[fold["valid"]])
            self.assertFalse(train_groups & valid_groups)
        self.assertTrue((valid_counts == 1).all())

    def test_stage_fit_selects_cv_minimum_and_extends_at_boundary(self):
        features, _, shape, frame, _ = synthetic_pairs(n_groups=6, per_group=3)
        fit = StageFit("local_last", features, frame, shape)
        alpha = fit.select()
        self.assertAlmostEqual(fit.trace[alpha], min(fit.trace.values()))
        self.assertIn(float(alpha), [float(value) for value in fit.alphas])
        # A pure shrinkage target makes the largest alpha the CV minimum; the
        # boundary rule must then extend the grid on train evidence alone.
        extended = StageFit("local_last", features, frame, shape)
        extended.target = np.zeros_like(extended.target)
        picked = extended.select()
        self.assertTrue(extended.extensions)
        self.assertAlmostEqual(extended.trace[picked], min(extended.trace.values()))
        self.assertGreater(picked, ALPHAS[-1])

    def test_locked_readout_recovers_linear_relation(self):
        features, _, shape, frame, _ = synthetic_pairs(n_groups=8, per_group=4, dim=6,
                                                      positions=5, seed=3)
        rng = np.random.default_rng(4)
        mapping = rng.normal(size=(6, 5)) * 0.1
        # Differences of this target are exactly linear in the feature
        # differences, so the locked readout must beat the zero response.
        shape = features @ mapping + 0.5
        fit = StageFit("local_last", features, frame, shape)
        fit.select()
        fit.final_fit()
        predicted = predict_materials("local_last", features, fit.scale, fit.weights)
        target, tv = pair_targets(shape, frame)
        delta = predicted[frame.i.to_numpy(int)] - predicted[frame.j.to_numpy(int)]
        error = 0.5 * np.abs(delta - target).sum(-1)
        self.assertLess(group_mean(error, frame.composition.to_numpy()),
                        group_mean(tv, frame.composition.to_numpy()))

    def test_near_zero_pairs_reported_separately(self):
        frame = pd.DataFrame({
            "target_tv": [0.2, NEAR_ZERO_TV / 2],
            "predicted_tv": [0.1, 0.0],
            "difference_error": [0.1, 0.0],
            "skill": [0.5, np.nan],
            "direction_cosine": [0.2, np.nan],
            "composition": ["a", "b"],
        })
        report = method_metrics(frame)
        self.assertEqual(report["main_pairs"], 1)
        self.assertEqual(report["near_zero_pairs"], 1)
        self.assertIn("near_zero", report)
        self.assertEqual(report["near_zero"]["group_mean_error"]["groups"], 1)

    def test_predict_pairs_matches_material_differences(self):
        features, _, shape, frame, _ = synthetic_pairs()
        fit = StageFit("local_last", features, frame, shape)
        fit.select()
        fit.final_fit()
        materials = predict_materials("local_last", features, fit.scale, fit.weights)
        left = frame.i.to_numpy(int)[:5]
        right = frame.j.to_numpy(int)[:5]
        pairs = predict_pairs("local_last", features, left, right, fit.scale, fit.weights)
        self.assertTrue(np.allclose(pairs, materials[left] - materials[right]))


if __name__ == "__main__":
    unittest.main()
