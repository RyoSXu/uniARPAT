"""固定小样本选样、全对拟合判据、冻结边界和预算合同。"""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from tools.eval.g2_small_fit_probe import (
    fit_arm, fit_verdict, memory_set_rms, pair_scores, select_train_pairs, state_change,
)


class TestSmallFitSelection(unittest.TestCase):
    def test_selection_ignores_labels_validation_and_row_order(self):
        rows = []
        for group in range(8):
            for split in ("train", "valid"):
                for pair in range(2):
                    a = 10 * group + pair * 2
                    rows.append(dict(split=split, group=str(group), reduced_group=str(group),
                                     sample_index_a=a, sample_index_b=a + 1,
                                     mpid_a=f"{split}{a}", mpid_b=f"{split}{a + 1}",
                                     primary_valid=split == "valid", target_error=float(a)))
        plan = pd.DataFrame(rows)
        expected = select_train_pairs(plan, n_pairs=4)
        shuffled = plan.sample(frac=1., random_state=3)
        shuffled["target_error"] = -1000.
        pd.testing.assert_frame_equal(expected, select_train_pairs(shuffled, n_pairs=4))
        self.assertEqual(expected.reduced_group.nunique(), 4)
        self.assertEqual(len(set(expected.sample_index_a) | set(expected.sample_index_b)), 8)
        self.assertTrue((expected.split == "train").all())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_train_pairs(pd.concat([plan, plan.iloc[[0]]]), n_pairs=4)


class TestSmallFitMetrics(unittest.TestCase):
    def test_all_pairs_gate_cannot_be_hidden_by_small_aggregate_error(self):
        target = torch.tensor([[.9, .1], [.1, .9], [.50001, .49999], [.49999, .50001]])
        prediction = target.clone()
        prediction[2], prediction[3] = target[3], target[2]
        scores = pair_scores(prediction, target, np.array([[0, 1], [2, 3]]))
        self.assertLess(scores.contrast_mse.mean() / scores.zero_mse.mean(), .01)
        self.assertEqual(scores.passed.tolist(), [True, False])

    def test_perfect_fit_and_zero_target_are_finite(self):
        target = torch.tensor([[.8, .2], [.2, .8], [.5, .5], [.5, .5]])
        scores = pair_scores(target, target, np.array([[0, 1], [2, 3]]))
        self.assertEqual(scores.informative.tolist(), [True, False])
        self.assertTrue(scores.passed.all())
        self.assertTrue(np.isfinite(scores.select_dtypes(include=[np.number])).all().all())
        wrong = target.clone()
        wrong[3] = torch.tensor([.6, .4])
        self.assertFalse(pair_scores(wrong, target, np.array([[2, 3]])).passed.iloc[0])

    def test_memory_alias_detection_allows_unrestricted_atom_permutation(self):
        memory = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        self.assertEqual(memory_set_rms(memory, memory[[2, 0, 1]]), 0.)
        changed = memory[[2, 0, 1]].clone()
        changed[0, 0] += .1
        self.assertGreater(memory_set_rms(memory, changed), 0.)

    def test_verdict_distinguishes_fit_witnesses_from_failure_to_fit(self):
        self.assertEqual(fit_verdict(dict(frozen=0, joint=100)), "both_networks_fit_small_train_set")
        self.assertEqual(fit_verdict(dict(frozen=None, joint=100)), "joint_adaptation_fit_only")
        self.assertEqual(fit_verdict(dict(frozen=100, joint=None)), "frozen_readout_fit_supported")
        self.assertEqual(fit_verdict(dict(frozen=None, joint=None)), "small_set_fit_not_demonstrated")


class TinyReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 4)
        self.dropout = torch.nn.Dropout(.5)

    def forward(self, memory, mask):
        return self.projection(self.dropout(memory[:, 0])).softmax(-1)


class TinyProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.readout = TinyReadout()
        self.eta_head = torch.nn.Linear(2, 1).requires_grad_(False)
        self.eval()

    def forward(self, batch):
        return self.readout(self.encoder(batch["src"]), batch["src_key_padding_mask"])


class TestSmallFitTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.prototype = TinyProbe()
        self.batch = dict(src=torch.tensor([[[1., 0.]], [[0., 1.]], [[1., 1.]], [[-1., 1.]]]),
                          src_key_padding_mask=torch.zeros(4, 1, dtype=torch.bool))
        with torch.no_grad():
            memory = self.prototype.encoder(self.batch["src"])
        self.cache = dict(memories=[value.clone() for value in memory], target=torch.randn(4, 4).softmax(-1))
        self.pairs = np.array([[0, 1], [2, 3]])

    def test_actual_optimizer_boundary_and_equal_full_batch_budget(self):
        cached = copy.deepcopy(self.cache)
        with tempfile.TemporaryDirectory() as directory:
            for arm in ("frozen", "joint"):
                probe = copy.deepcopy(self.prototype)
                history, trajectory, _ = fit_arm(probe, arm, self.cache, self.batch, self.pairs,
                                                  torch.device("cpu"), Path(directory), steps=6, eval_every=2)
                self.assertEqual(history.step.tolist(), [0, 2, 4, 6])
                self.assertEqual(len(trajectory), 8)
                self.assertTrue(all(not module.training for module in probe.modules()))
                encoder = state_change(self.prototype.encoder.state_dict(), probe.encoder)
                readout = state_change(self.prototype.readout.state_dict(), probe.readout)
                eta = state_change(self.prototype.eta_head.state_dict(), probe.eta_head)
                self.assertEqual(encoder["changed_values"] > 0, arm == "joint")
                self.assertGreater(readout["changed_values"], 0)
                self.assertEqual(eta["changed_values"], 0)
                saved = torch.load(Path(directory) / f"{arm}_final.pth", weights_only=True)
                self.assertEqual(saved["step"], 6)
        for before, after in zip(cached["memories"], self.cache["memories"]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(cached["target"], self.cache["target"], rtol=0, atol=0)

    def test_first_fit_witness_is_saved_and_budget_still_completed(self):
        with torch.no_grad():
            self.cache["target"] = self.prototype(self.batch)
        with tempfile.TemporaryDirectory() as directory:
            history, _, first_fit = fit_arm(copy.deepcopy(self.prototype), "frozen", self.cache, self.batch,
                                            self.pairs, torch.device("cpu"), Path(directory), steps=4, eval_every=2)
            self.assertEqual(first_fit, 0)
            self.assertEqual(history.iloc[-1].step, 4)
            saved = torch.load(Path(directory) / "frozen_first_fit.pth", weights_only=True)
            self.assertEqual(saved["step"], 0)


if __name__ == "__main__":
    unittest.main()
