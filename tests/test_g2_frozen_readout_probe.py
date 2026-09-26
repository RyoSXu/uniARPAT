"""冻结读出实验的对应、隔离、梯度与判据合同。"""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from model.transformer import Transformer
from tools.eval.g2_frozen_readout_probe import (
    FrozenEdosReadout, composition_keys, contrast_mse, group_weights,
    positive_logit_control, prepare_pair_plan, shuffled_donors, train_arm,
    verdict_from_comparisons,
)
from tools.eval.g2_structure_path_audit import FeatureProbe


class TestReadoutCorrespondence(unittest.TestCase):
    def test_reduced_composition_excludes_different_cell_sizes(self):
        elements = {
            "train": np.array([[126, 127, 14, 8, 0, 0], [126, 127, 14, 8, 0, 0]]),
            "valid": np.array([[126, 127, 14, 14, 8, 8], [126, 127, 14, 14, 8, 8],
                               [126, 127, 6, 7, 0, 0], [126, 127, 6, 7, 0, 0]]),
        }
        ids = {"train": np.array(["t0", "t1"]), "valid": np.array(["v0", "v1", "v2", "v3"])}
        rows = []
        for split, a, b in (("train", 0, 1), ("valid", 0, 1), ("valid", 2, 3)):
            rows.append(dict(split=split, arm="edge", group=composition_keys(elements[split])[0][a],
                             sample_index_a=a, sample_index_b=b, mpid_a=ids[split][a], mpid_b=ids[split][b]))
        result = prepare_pair_plan(pd.DataFrame(rows), elements, ids)
        self.assertEqual(result.primary_valid.tolist(), [False, False, True])
        rows[0]["mpid_a"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "IDs"):
            prepare_pair_plan(pd.DataFrame(rows), elements, ids)

    def test_shuffling_is_bijective_within_group_and_keeps_fixed_points(self):
        members = {"a": [0, 1], "b": [2, 3, 4]}
        group_of = {i: group for group, indices in members.items() for i in indices}
        indices = [0, 1, 0, 2, 3, 4]
        rng = np.random.default_rng(42)
        two_member_maps = set()
        for _ in range(100):
            donors = shuffled_donors(indices, group_of, members, rng)
            self.assertEqual(donors[0], donors[2])
            self.assertEqual(set(donors[:2]), {0, 1})
            self.assertEqual(set(donors[3:]), {2, 3, 4})
            two_member_maps.add(tuple(donors[:2]))
        self.assertEqual(two_member_maps, {(0, 1), (1, 0)})

    def test_epoch_weights_give_each_composition_equal_mass(self):
        groups = np.array(["a", "b", "b", "b"])
        weights = group_weights(groups)
        self.assertAlmostEqual(float(weights.mean()), 1.)
        self.assertAlmostEqual(float(weights[groups == "a"].sum()), float(weights[groups == "b"].sum()))

    def test_signed_contrast_objective_detects_reversed_structure(self):
        a, b = torch.tensor([[.8, .2]]), torch.tensor([[.2, .8]])
        self.assertEqual(contrast_mse(a, b, a, b).item(), 0.)
        self.assertGreater(contrast_mse(b, a, a, b).item(), 0.)

    def test_positive_control_can_fit_softmax_contrasts(self):
        target = torch.tensor([[.9, .1, 0.], [.1, .6, .3], [.2, .3, .5]])
        result = positive_logit_control(target, np.array([[0, 1], [1, 2]]), steps=300)
        self.assertTrue(result["passed"])


class TestFrozenReadoutBoundary(unittest.TestCase):
    def test_both_training_arms_keep_frozen_features_and_equal_update_budget(self):
        class TinyReadout(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(2, 4)

            def forward(self, memory, mask):
                return self.projection(memory[:, 0]).softmax(-1)

        torch.set_num_threads(1)
        cache = dict(memories=[torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]), torch.tensor([[1., 1.]])],
                     target=torch.tensor([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1]]),
                     local_of={0: 0, 1: 1, 2: 2})
        before = copy.deepcopy(cache)
        plan = pd.DataFrame(dict(sample_index_a=[0, 0, 1], sample_index_b=[1, 2, 2],
                                 group=["a"] * 3, reduced_group=["a"] * 3))
        with tempfile.TemporaryDirectory() as directory, \
                patch("tools.eval.g2_frozen_readout_probe.EPOCHS", 2), \
                patch("tools.eval.g2_frozen_readout_probe.PAIR_BATCH", 2), \
                patch("tools.eval.g2_frozen_readout_probe.file_hash", return_value="synthetic"):
            for arm in ("matched", "shuffled"):
                history = train_arm(TinyReadout(), cache, plan, arm, torch.device("cpu"), Path(directory))
                self.assertEqual(history[-1]["updates"], 4)
                self.assertTrue(np.isfinite([row["objective"] for row in history]).all())
                self.assertTrue((Path(directory) / f"{arm}_final.pth").is_file())
        for a, b in zip(cache["memories"], before["memories"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertIsNone(a.grad)
        torch.testing.assert_close(cache["target"], before["target"], rtol=0, atol=0)

    def test_cached_forward_equivalence_and_readout_only_optimization(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        source = Transformer(d_model=16, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
                             dim_feedforward=32, edos_num=128, phdos_num=64, dropout=.05,
                             use_g2=True, scale_mode="eta").eval().requires_grad_(False)
        state = copy.deepcopy(source.state_dict())
        inp = torch.tensor([[126, 127, 14, 8, 0], [126, 127, 14, 6, 0]])
        pos = torch.zeros(2, 5, 3)
        pos[:, 0] = torch.tensor([5., 5., .2])
        pos[:, 1] = 90.
        pos[:, 3] = torch.tensor([.25, .25, .25])
        mask = inp.eq(0)
        processed = [inp, pos, mask] + [None] * 15
        probe = FeatureProbe(source)
        try:
            with torch.no_grad():
                direct, features = probe.forward(processed)
        finally:
            probe.close()
        readout = FrozenEdosReadout(source)
        memory = features["encoder"].clone()
        prediction = readout(memory, mask[:, 2:])
        torch.testing.assert_close(prediction, direct["edos"].softmax(-1), rtol=0, atol=1e-7)
        self.assertTrue(all(not module.training for module in readout.modules()))
        before = readout.head.layers[-1].weight.detach().clone()
        optimizer = torch.optim.AdamW(readout.parameters(), lr=1e-3)
        loss = (prediction[0] - torch.linspace(.001, .02, 128)).square().mean()
        loss.backward()
        self.assertIsNone(memory.grad)
        self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))
        self.assertIsNotNone(readout.query.grad)
        self.assertTrue(torch.isfinite(readout.query.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, readout.head.layers[-1].weight))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)


class TestReadoutVerdict(unittest.TestCase):
    def comparisons(self, train=.15, valid=.08, ci=.01):
        return pd.DataFrame([
            dict(population=population, reference=reference, relative_improvement=gain, ci_low=ci)
            for population, gain in (("train", train), ("valid_unseen_composition", valid))
            for reference in ("original", "shuffled", "zero_contrast")
        ])

    def test_requires_advantage_over_all_three_references(self):
        frame = self.comparisons()
        self.assertEqual(verdict_from_comparisons(frame, {"passed": True}), "heldout_recoverability_supported")
        frame.loc[(frame.population == "valid_unseen_composition") & (frame.reference == "original"), "relative_improvement"] = 0.
        self.assertEqual(verdict_from_comparisons(frame, {"passed": True}), "train_learning_only")

    def test_uncertainty_and_failed_intervention_do_not_claim_information_absent(self):
        frame = self.comparisons(train=0., ci=-.001)
        self.assertEqual(verdict_from_comparisons(frame, {"passed": True}), "readout_intervention_not_supported")
        self.assertEqual(verdict_from_comparisons(frame, {"passed": False}), "invalid_positive_control")


if __name__ == "__main__":
    unittest.main()
