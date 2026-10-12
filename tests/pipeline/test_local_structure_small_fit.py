"""Selection integrity and train-only spectral fit criteria."""
import unittest

import numpy as np
import pandas as pd
import torch

from model.losses import edos_pair_contrast_loss
from tools.eval.local_structure_small_fit import check_initial_metrics, fit_metrics, select_pairs


class TestSmallFit(unittest.TestCase):
    def test_initial_metrics_allow_only_small_float_roundoff(self):
        initial = dict(pair_error_mean=.23, direction_cosine_mean=.005, pairs=16,
                       fit_demonstrated=False)
        rounded = {**initial, "pair_error_mean": .23 + 1e-8,
                   "direction_cosine_mean": .005 + 4e-6}
        self.assertTrue(check_initial_metrics(initial, rounded)["passed"])
        with self.assertRaisesRegex(ValueError, "FP32 metric differs"):
            check_initial_metrics(initial, {**initial, "pair_error_mean": .24})
        with self.assertRaisesRegex(ValueError, "discrete metric differs"):
            check_initial_metrics(initial, {**initial, "pairs": 15})

    def test_selection_is_label_independent_and_group_uniform(self):
        rows = []
        for group in range(10):
            for pair in range(2):
                i = group * 4 + pair * 2
                rows.append(dict(i=i, j=i + 1, left_id=f"m{i}", right_id=f"m{i+1}",
                                 composition=f"g{group}", target_tv=float(i)))
        frame = pd.DataFrame(rows)
        selected, eligible = select_pairs(frame, np.ones(40), 5)
        shuffled = frame.sample(frac=1, random_state=3)
        shuffled["target_tv"] = -1
        other, _ = select_pairs(shuffled, np.ones(40), 5)
        pd.testing.assert_frame_equal(selected, other)
        self.assertEqual(selected.composition.nunique(), 5)
        self.assertEqual(eligible["eligible_groups"], 10)

    def test_selection_rejects_bad_rows_and_never_crops_large_crystals(self):
        frame = pd.DataFrame([dict(i=0, j=1, left_id="a", right_id="b", composition="x"),
                              dict(i=2, j=3, left_id="c", right_id="d", composition="y")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_pairs(pd.concat([frame, frame.iloc[:1]]), [2, 2, 2, 2], 1)
        with self.assertRaisesRegex(ValueError, "too few"):
            select_pairs(frame, [2, 41, 2, 2], 2)
        chosen, _ = select_pairs(frame, [2, 41, 2, 2], 1)
        self.assertEqual(chosen.composition.tolist(), ["y"])

    def test_zero_and_perfect_prediction_have_expected_pair_skills(self):
        target = np.array([[.8, .2], [.2, .8]])
        keys, ids = ["g", "g"], ["a", "b"]
        zero, _ = fit_metrics(np.full_like(target, .5), target, keys, ids)
        perfect, _ = fit_metrics(target, target, keys, ids)
        self.assertEqual(zero["pair_error_over_zero"], 1)
        self.assertFalse(zero["fit_demonstrated"])
        self.assertEqual(perfect["pair_error_over_zero"], 0)
        self.assertTrue(perfect["fit_demonstrated"])

    def test_aggregate_gain_does_not_hide_failure_on_most_pairs(self):
        target = np.array([[.95, .05], [.05, .95], [.52, .48], [.48, .52],
                           [.52, .48], [.48, .52], [.52, .48], [.48, .52]])
        prediction = np.full_like(target, .5)
        prediction[:2] = target[:2]
        keys = ["a", "a", "b", "b", "c", "c", "d", "d"]
        metrics, _ = fit_metrics(prediction, target, keys, list(map(str, range(8))))
        self.assertLess(metrics["pair_error_over_zero"], .2)
        self.assertEqual(metrics["half_error_pairs"], 1)
        self.assertFalse(metrics["fit_demonstrated"])

    def test_existing_pair_loss_matches_metric_and_preserves_swap_symmetry(self):
        logits = torch.tensor([[.4, -.2], [-.6, .7]], requires_grad=True)
        target = torch.tensor([[.8, .2], [.2, .8]])
        loss = edos_pair_contrast_loss(logits[:1], logits[1:], target[:1], target[1:])
        swapped = edos_pair_contrast_loss(logits[1:], logits[:1], target[1:], target[:1])
        metrics, _ = fit_metrics(logits.softmax(-1).detach().numpy(), target.numpy(),
                                 ["g", "g"], ["a", "b"])
        self.assertEqual(loss.item(), swapped.item())
        self.assertAlmostEqual(loss.item(), metrics["pair_error_mean"], places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
