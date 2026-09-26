"""联合适配实验的缓存、优化边界和因果对照合同。"""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from model.transformer import Transformer
from tools.eval.g2_encoder_adaptation_probe import (
    AdaptiveEdosProbe, clip_and_step, encoder_batch, gradient_equivalence,
    split_encoder_inputs, train_joint, verdict_from_comparisons,
)
from tools.eval.g2_frozen_readout_probe import contrast_mse


class TestEncoderAdaptationBoundary(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.source = Transformer(d_model=16, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
                                  dim_feedforward=32, edos_num=128, phdos_num=64, dropout=.05,
                                  use_g2=True, scale_mode="eta").eval().requires_grad_(False)
        with torch.no_grad():
            self.source.encoder.g2_msgs[0].alpha.fill_(.1)
        src = torch.tensor([[126, 127, 14, 8, 0], [126, 127, 14, 0, 8], [126, 127, 14, 8, 6]])
        pos = torch.zeros(3, 5, 3)
        pos[:, 0] = 6.
        pos[:, 1] = 90.
        pos[0, 3], pos[1, 4] = .2, .3
        pos[2, 3], pos[2, 4] = .2, .4
        self.inputs, self.memories = [], []

        def capture(module, args, kwargs, output):
            self.inputs.extend(split_encoder_inputs(kwargs))
            for i, keep in enumerate(~kwargs["src_key_padding_mask"]):
                self.memories.append(output[i, keep].clone())

        handle = self.source.encoder.register_forward_hook(capture, with_kwargs=True)
        try:
            with torch.no_grad():
                self.original = self.source(src, src.eq(0), pos)
        finally:
            handle.remove()
        self.probe = AdaptiveEdosProbe(self.source)
        self.plan = pd.DataFrame(dict(sample_index_a=[0, 0, 1], sample_index_b=[1, 2, 2],
                                      reduced_group=["a"] * 3))
        self.cache = dict(memories=self.memories, local_of={0: 0, 1: 1, 2: 2},
                          target=torch.randn(3, 128).softmax(-1))

    def test_noncontiguous_padding_and_duplicate_materials_preserve_edges(self):
        batch = encoder_batch(self.inputs, [1, 0, 1, 2], torch.device("cpu"))
        edges = batch["g2_edges"]
        self.assertEqual(batch["src_key_padding_mask"].sum(-1).tolist(), [1, 1, 1, 0])
        for i, source in enumerate([1, 0, 1, 2]):
            selected = edges["batch"] == i
            self.assertTrue((edges["dst"][selected] < len(self.inputs[source]["atom_src"])).all())
            torch.testing.assert_close(edges["src"][selected], self.inputs[source]["edge_src"], rtol=0, atol=0)
            torch.testing.assert_close(edges["distances"][selected], self.inputs[source]["edge_dist"], rtol=0, atol=0)
        p = self.probe(batch)
        torch.testing.assert_close(p[0], p[2], rtol=0, atol=0)

    def test_forward_scale_equivalence_and_update_boundary(self):
        state = copy.deepcopy(self.source.state_dict())
        eta = copy.deepcopy(self.probe.eta_head.state_dict())
        encoder_before = self.probe.encoder.layers[0].rp_proj.weight.detach().clone()
        readout_before = self.probe.readout.query.detach().clone()
        p, gamma = self.probe(encoder_batch(self.inputs, [0, 1, 2], torch.device("cpu")), with_gamma=True)
        torch.testing.assert_close(p, self.original["edos"].softmax(-1), rtol=0, atol=1e-7)
        torch.testing.assert_close(gamma, self.original["eta"][:, 1], rtol=0, atol=1e-7)
        self.assertTrue(all(not module.training for module in self.probe.modules()))
        loss = contrast_mse(p[0], p[1], self.cache["target"][0], self.cache["target"][1])
        loss.backward()
        self.assertIsNotNone(self.probe.encoder.layers[0].rp_proj.weight.grad)
        self.assertIsNotNone(self.probe.readout.query.grad)
        self.assertTrue(all(p.grad is None for p in self.probe.eta_head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.source.parameters()))
        options = dict(lr=5e-5, betas=(.9, .99), weight_decay=.01)
        clip_and_step(self.probe, torch.optim.AdamW(self.probe.readout.parameters(), **options),
                      torch.optim.AdamW(self.probe.encoder.parameters(), **options))
        self.assertFalse(torch.equal(encoder_before, self.probe.encoder.layers[0].rp_proj.weight))
        self.assertFalse(torch.equal(readout_before, self.probe.readout.query))
        for name, value in self.probe.eta_head.state_dict().items():
            torch.testing.assert_close(value, eta[name], rtol=0, atol=0)
        for name, value in self.source.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertTrue(all(not value.requires_grad for row in self.inputs for value in row.values()))

    def test_cached_and_recomputed_readout_gradients_match(self):
        result = gradient_equivalence(self.probe, {"inputs": self.inputs}, self.cache, self.plan, torch.device("cpu"))
        self.assertLess(result["readout_gradient_relative_rms"], 1e-3)
        self.assertTrue(all(p.requires_grad for p in self.probe.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.probe.parameters()))

    def test_training_keeps_tail_batch_budget_and_frozen_inputs(self):
        before = copy.deepcopy(self.inputs)
        with tempfile.TemporaryDirectory() as directory, \
                patch("tools.eval.g2_encoder_adaptation_probe.EPOCHS", 2), \
                patch("tools.eval.g2_encoder_adaptation_probe.PAIR_BATCH", 2), \
                patch("tools.eval.g2_encoder_adaptation_probe.file_hash", return_value="synthetic"):
            history = train_joint(self.probe, {"inputs": self.inputs}, self.cache, self.plan,
                                  torch.device("cpu"), Path(directory))
            self.assertEqual(history.updates.tolist(), [2, 4])
            self.assertTrue(np.isfinite(history.objective).all())
            saved = torch.load(Path(directory) / "joint_final.pth", weights_only=True)
            self.assertEqual(saved["updates"], 4)
        for old, new in zip(before, self.inputs):
            for name in old:
                torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)

    def test_encoder_gradient_size_does_not_change_readout_step(self):
        class Blocks(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(1, 1, bias=False)
                self.readout = torch.nn.Linear(1, 1, bias=False)

        left, right = Blocks(), Blocks()
        right.load_state_dict(left.state_dict())
        for model, factor in ((left, 1.), (right, 10000.)):
            model.encoder.weight.grad = torch.ones_like(model.encoder.weight) * factor
            model.readout.weight.grad = torch.ones_like(model.readout.weight) * 3.
            clip_and_step(model, torch.optim.SGD(model.readout.parameters(), lr=.1),
                          torch.optim.SGD(model.encoder.parameters(), lr=.1))
        torch.testing.assert_close(left.readout.weight, right.readout.weight, rtol=0, atol=0)


class TestJointVerdict(unittest.TestCase):
    def comparisons(self):
        return pd.DataFrame([dict(population=population, reference=reference, relative_improvement=gain, ci_low=.01)
                             for population, gain in (("train", .15), ("valid_unseen_composition", .08))
                             for reference in ("matched", "original", "shuffled", "zero_contrast")])

    def test_requires_causal_control_and_all_references(self):
        frame = self.comparisons()
        self.assertEqual(verdict_from_comparisons(frame), "heldout_joint_adaptation_supported")
        frame.loc[(frame.population == "valid_unseen_composition") & (frame.reference == "matched"), "relative_improvement"] = .01
        self.assertEqual(verdict_from_comparisons(frame), "joint_train_learning_only")
        frame = self.comparisons()
        frame.loc[(frame.population == "valid_unseen_composition") & (frame.reference == "zero_contrast"), "ci_low"] = -.01
        self.assertEqual(verdict_from_comparisons(frame), "joint_train_learning_only")

    def test_missing_reference_or_failed_training_does_not_claim_absence(self):
        frame = self.comparisons()
        self.assertEqual(verdict_from_comparisons(frame[frame.reference != "matched"]), "joint_adaptation_not_supported")
        frame["relative_improvement"] = 0.
        self.assertEqual(verdict_from_comparisons(frame), "joint_adaptation_not_supported")


if __name__ == "__main__":
    unittest.main()
