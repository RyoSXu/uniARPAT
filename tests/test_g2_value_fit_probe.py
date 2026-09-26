"""实际多层encoder的受限更新、跨冻结层梯度和磁盘拟合见证。"""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from model.transformer import Transformer
from tools.eval.g2_encoder_adaptation_probe import AdaptiveEdosProbe, encoder_batch, split_encoder_inputs
from tools.eval.g2_small_fit_probe import configure_encoder_scope, fit_arm, state_change
from tools.eval.g2_value_fit_probe import gradient_boundary_check, verify_checkpoints


class TestG2ValueFit(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        source = Transformer(d_model=16, nhead=4, num_encoder_layers=2, num_decoder_layers=1,
                             dim_feedforward=32, edos_num=128, phdos_num=64, dropout=.05,
                             use_g2=True, scale_mode="eta").eval().requires_grad_(False)
        with torch.no_grad():
            for message in source.encoder.g2_msgs:
                message.alpha.fill_(.1)
        src = torch.tensor([[126, 127, 14, 8], [126, 127, 14, 8], [126, 127, 14, 8]])
        positions = torch.zeros(3, 4, 3)
        positions[:, 0], positions[:, 1] = 6., 90.
        positions[:, 3, 0] = torch.tensor([.15, .3, .45])
        self.inputs, memories = [], []

        def capture(module, args, kwargs, output):
            self.inputs.extend(split_encoder_inputs(kwargs))
            memories.extend(value.clone() for value in output)

        handle = source.encoder.register_forward_hook(capture, with_kwargs=True)
        try:
            with torch.no_grad():
                source(src, src.eq(0), positions)
        finally:
            handle.remove()
        self.prototype = AdaptiveEdosProbe(source)
        self.batch = encoder_batch(self.inputs, range(3), torch.device("cpu"))
        self.cache = dict(memories=memories, target=torch.randn(3, 128).softmax(-1))
        self.pairs = np.array([[0, 1], [1, 2]])

    def test_scope_can_be_narrowed_and_restored_without_stale_gradients(self):
        probe = copy.deepcopy(self.prototype)
        for p in probe.encoder.parameters():
            p.grad = torch.ones_like(p)
        for arm in ("g2_only", "non_g2", "last_layer", "joint", "frozen", "g2_only"):
            configure_encoder_scope(probe, arm)
            for name, parameter in probe.encoder.named_parameters():
                expected = arm == "joint" or (arm == "g2_only" and name.startswith("g2_msgs.")) or (arm == "non_g2" and not name.startswith("g2_msgs."))
                expected = expected or (arm == "last_layer" and name.startswith("layers.1."))
                self.assertEqual(parameter.requires_grad, expected)
                self.assertIsNone(parameter.grad)
            self.assertTrue(all(p.requires_grad for p in probe.readout.parameters()))
            self.assertTrue(all(not p.requires_grad for p in probe.eta_head.parameters()))
        probe.encoder = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "requires existing G2"):
            configure_encoder_scope(probe, "g2_only")
        self.assertTrue(all(p.requires_grad for p in probe.encoder.parameters()))

    def test_backward_reaches_early_g2_through_frozen_later_layer(self):
        probe = copy.deepcopy(self.prototype)
        before = copy.deepcopy(probe.state_dict())
        report = gradient_boundary_check(probe, self.batch, self.cache["target"], self.pairs)
        self.assertEqual(len(report["layers"]), 2)
        self.assertTrue(all(row["gradient_l2"] > 0 for row in report["layers"]))
        self.assertLess(report["encoder_trainable_parameters"], report["encoder_total_parameters"])
        self.assertEqual(state_change(before, probe)["changed_values"], 0)
        self.assertTrue(all(p.grad is None for p in probe.parameters()))

    def test_detached_early_message_is_rejected_by_preflight(self):
        probe = copy.deepcopy(self.prototype)
        handle = probe.encoder.g2_msgs[0].register_forward_hook(lambda module, args, output: output.detach())
        try:
            with self.assertRaisesRegex(RuntimeError, "missing or nonfinite gradient: encoder.g2_msgs.0"):
                gradient_boundary_check(probe, self.batch, self.cache["target"], self.pairs)
        finally:
            handle.remove()
        self.assertTrue(all(p.grad is None for p in probe.parameters()))

    def test_actual_training_updates_only_messages_and_readout_and_reloads(self):
        probe = copy.deepcopy(self.prototype)
        inputs = copy.deepcopy(self.inputs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history, trajectory, first_fit = fit_arm(probe, "g2_only", self.cache, self.batch, self.pairs,
                                                      torch.device("cpu"), root, steps=4, eval_every=2)
            self.assertEqual(history.step.tolist(), [0, 2, 4])
            self.assertEqual(len(trajectory), 6)
            for i in range(2):
                self.assertGreater(state_change(self.prototype.encoder.g2_msgs[i].state_dict(), probe.encoder.g2_msgs[i])["changed_values"], 0)
            fixed = {name: value for name, value in self.prototype.encoder.state_dict().items() if not name.startswith("g2_msgs.")}
            self.assertEqual(state_change(fixed, probe.encoder)["changed_values"], 0)
            self.assertEqual(state_change(self.prototype.eta_head.state_dict(), probe.eta_head)["changed_values"], 0)
            self.assertGreater(state_change(self.prototype.readout.state_dict(), probe.readout)["changed_values"], 0)
            with torch.no_grad():
                final = probe(self.batch)
            with patch("tools.eval.g2_value_fit_probe.STEPS", 4), patch("tools.eval.g2_value_fit_probe.REPO_ROOT", root):
                report = verify_checkpoints(self.prototype, root, self.cache, self.batch, self.pairs, first_fit, final)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["checkpoints"][-1]["max_prediction_tv_vs_final"], 0.)
        for before, after in zip(inputs, self.inputs):
            for name in before:
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)

    def test_reload_rejects_a_checkpoint_falsely_claimed_as_fit_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(dict(arm="g2_only", step=0, probe=self.prototype.state_dict()), root / "g2_only_first_fit.pth")
            with patch("tools.eval.g2_value_fit_probe.REPO_ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "reloaded witness failed"):
                    verify_checkpoints(self.prototype, root, self.cache, self.batch, self.pairs, 0,
                                       torch.zeros_like(self.cache["target"]))

    def test_reload_tolerance_accepts_roundoff_but_rejects_changed_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(dict(arm="g2_only", step=2000, probe=self.prototype.state_dict()), root / "g2_only_final.pth")
            with torch.no_grad():
                prediction = self.prototype(self.batch)
            rounded = prediction.clone()
            rounded[0, 0] += 1e-6
            rounded[0, 1] -= 1e-6
            with patch("tools.eval.g2_value_fit_probe.REPO_ROOT", root):
                report = verify_checkpoints(self.prototype, root, self.cache, self.batch, self.pairs, None, rounded)
                self.assertEqual(report["status"], "passed")
                self.assertLess(report["checkpoints"][0]["max_prediction_tv_vs_final"], 1e-5)
                rounded[0, 0] += 1e-3
                rounded[0, 1] -= 1e-3
                with self.assertRaisesRegex(RuntimeError, "differs from reported"):
                    verify_checkpoints(self.prototype, root, self.cache, self.batch, self.pairs, None, rounded)

    def test_per_pair_history_survives_final_checkpoint_export_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tools.eval.g2_small_fit_probe.atomic_torch_save", side_effect=OSError("simulated export failure")):
                with self.assertRaisesRegex(OSError, "simulated export"):
                    fit_arm(copy.deepcopy(self.prototype), "g2_only", self.cache, self.batch, self.pairs,
                            torch.device("cpu"), root, steps=4, eval_every=2)
            trajectory = pd.read_csv(root / "g2_only_pair_history.csv")
            self.assertEqual(trajectory.groupby("step").size().to_dict(), {0: 2, 2: 2, 4: 2})
            self.assertTrue(np.isfinite(trajectory.contrast_mse).all())

    def test_non_g2_backward_passes_through_frozen_messages(self):
        probe = copy.deepcopy(self.prototype)
        before = copy.deepcopy(probe.state_dict())
        report = gradient_boundary_check(probe, self.batch, self.cache["target"], self.pairs, arm="non_g2")
        self.assertEqual(report["arm"], "non_g2")
        self.assertEqual(len(report["layers"]), 2)
        self.assertTrue(all(row["gradient_l2"] > 0 for row in report["layers"]))
        self.assertTrue(all(not p.requires_grad for p in probe.encoder.g2_msgs.parameters()))
        self.assertEqual(state_change(before, probe)["changed_values"], 0)
        self.assertTrue(all(p.grad is None for p in probe.parameters()))

    def test_detaching_frozen_g2_breaks_earlier_trainable_encoder_gradient(self):
        probe = copy.deepcopy(self.prototype)
        handle = probe.encoder.g2_msgs[0].register_forward_hook(lambda module, args, output: output.detach())
        try:
            with self.assertRaisesRegex(RuntimeError, "missing or nonfinite gradient: encoder.layers.0"):
                gradient_boundary_check(probe, self.batch, self.cache["target"], self.pairs, arm="non_g2")
        finally:
            handle.remove()
        self.assertTrue(all(p.grad is None for p in probe.parameters()))

    def test_non_g2_optimization_preserves_messages_and_records_every_pair(self):
        probe = copy.deepcopy(self.prototype)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history, trajectory, first_fit = fit_arm(probe, "non_g2", self.cache, self.batch, self.pairs,
                                                      torch.device("cpu"), root, steps=4, eval_every=2)
            self.assertEqual(history.step.tolist(), [0, 2, 4])
            self.assertEqual(len(trajectory), 6)
            self.assertEqual(state_change(self.prototype.encoder.g2_msgs.state_dict(), probe.encoder.g2_msgs)["changed_values"], 0)
            self.assertEqual(state_change(self.prototype.eta_head.state_dict(), probe.eta_head)["changed_values"], 0)
            for old, new in zip(self.prototype.encoder.layers, probe.encoder.layers):
                self.assertGreater(state_change(old.state_dict(), new)["changed_values"], 0)
            self.assertGreater(state_change(self.prototype.readout.state_dict(), probe.readout)["changed_values"], 0)
            saved_pairs = pd.read_csv(root / "non_g2_pair_history.csv")
            self.assertEqual(saved_pairs.groupby("step").size().to_dict(), {0: 2, 2: 2, 4: 2})
            with torch.no_grad():
                final = probe(self.batch)
            with patch("tools.eval.g2_value_fit_probe.STEPS", 4), patch("tools.eval.g2_value_fit_probe.REPO_ROOT", root):
                report = verify_checkpoints(self.prototype, root, self.cache, self.batch, self.pairs, first_fit, final, arm="non_g2")
            self.assertEqual(report["arm"], "non_g2")
            self.assertEqual(report["status"], "passed")

    def test_last_layer_gradient_scope_keeps_actual_layer_index(self):
        probe = copy.deepcopy(self.prototype)
        before = copy.deepcopy(probe.state_dict())
        report = gradient_boundary_check(probe, self.batch, self.cache["target"], self.pairs, arm="last_layer")
        self.assertEqual([row["layer"] for row in report["layers"]], [1])
        self.assertGreater(report["layers"][0]["gradient_l2"], 0.)
        self.assertEqual(report["encoder_trainable_parameters"], sum(p.numel() for p in probe.encoder.layers[-1].parameters()))
        self.assertEqual(state_change(before, probe)["changed_values"], 0)
        self.assertTrue(all(p.grad is None for p in probe.parameters()))

    def test_detaching_final_frozen_g2_breaks_last_layer_gradient(self):
        probe = copy.deepcopy(self.prototype)
        handle = probe.encoder.g2_msgs[-1].register_forward_hook(lambda module, args, output: output.detach())
        try:
            with self.assertRaisesRegex(RuntimeError, "missing or nonfinite gradient: encoder.layers.1"):
                gradient_boundary_check(probe, self.batch, self.cache["target"], self.pairs, arm="last_layer")
        finally:
            handle.remove()
        self.assertTrue(all(p.grad is None for p in probe.parameters()))

    def test_last_layer_training_keeps_prefix_and_messages_unchanged(self):
        probe = copy.deepcopy(self.prototype)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history, trajectory, first_fit = fit_arm(probe, "last_layer", self.cache, self.batch, self.pairs,
                                                      torch.device("cpu"), root, steps=4, eval_every=2)
            self.assertEqual(history.step.tolist(), [0, 2, 4])
            self.assertEqual(len(trajectory), 6)
            frozen = {name: value for name, value in self.prototype.encoder.state_dict().items() if not name.startswith("layers.1.")}
            self.assertEqual(state_change(frozen, probe.encoder)["changed_values"], 0)
            self.assertEqual(state_change(self.prototype.eta_head.state_dict(), probe.eta_head)["changed_values"], 0)
            self.assertGreater(state_change(self.prototype.encoder.layers[-1].state_dict(), probe.encoder.layers[-1])["changed_values"], 0)
            self.assertGreater(state_change(self.prototype.readout.state_dict(), probe.readout)["changed_values"], 0)
            saved_pairs = pd.read_csv(root / "last_layer_pair_history.csv")
            self.assertEqual(saved_pairs.groupby("step").size().to_dict(), {0: 2, 2: 2, 4: 2})
            with torch.no_grad():
                final = probe(self.batch)
            with patch("tools.eval.g2_value_fit_probe.STEPS", 4), patch("tools.eval.g2_value_fit_probe.REPO_ROOT", root):
                report = verify_checkpoints(self.prototype, root, self.cache, self.batch, self.pairs, first_fit, final, arm="last_layer")
            self.assertEqual(report["arm"], "last_layer")
            self.assertTrue(all(row["frozen_state_changed_values"] == 0 for row in report["checkpoints"]))

    def test_reload_rejects_a_witness_with_changed_frozen_prefix(self):
        state = copy.deepcopy(self.prototype.state_dict())
        state["encoder.layers.0.rp_proj.bias"] += .001
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(dict(arm="last_layer", step=0, probe=state), root / "last_layer_first_fit.pth")
            with patch("tools.eval.g2_value_fit_probe.REPO_ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "reloaded checkpoint changed frozen state"):
                    verify_checkpoints(self.prototype, root, self.cache, self.batch, self.pairs, 0,
                                       torch.zeros_like(self.cache["target"]), arm="last_layer")


if __name__ == "__main__":
    unittest.main()
