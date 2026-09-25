"""Contracts for the opt-in eDOS slope-matching term."""

import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path

import torch
import yaml

from model.losses import (
    calibrate_additive_loss_weight,
    edos_slope_matching_loss,
)
from run_ablation_experiments import maybe_get_test_loader, write_edos_slope_calibration
from utils.experiment_config import ExperimentConfig


class TestEdosSlopeLoss(unittest.TestCase):
    def test_new_options_are_disabled_by_default(self):
        config = ExperimentConfig()

        self.assertEqual(config.edos_slope_ratio, 0.0)
        self.assertFalse(config.skip_test_eval)

    def test_matching_shape_has_zero_loss_and_finite_gradient(self):
        target = torch.tensor([[0.05, 0.15, 0.50, 0.30], [0.20, 0.30, 0.10, 0.40]])
        logits = target.log().requires_grad_()

        loss = edos_slope_matching_loss(logits, target)
        loss.backward()

        self.assertAlmostEqual(loss.item(), 0.0, places=7)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_loss_uses_normalized_probability_shape(self):
        target = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        logits = torch.tensor([[0.5, -0.3, 1.2, 0.0]])

        original = edos_slope_matching_loss(logits, target)
        shifted = edos_slope_matching_loss(logits + 17.0, target)

        torch.testing.assert_close(original, shifted)

    def test_calibration_sets_requested_gradient_norm_ratio(self):
        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.7, 1.1]))
        base = parameter.square().sum()
        added = (parameter * torch.tensor([1.0, 2.0, 3.0])).square().mean()

        weight, base_norm, added_norm = calibrate_additive_loss_weight(
            base, added, [parameter], target_ratio=0.1
        )

        self.assertAlmostEqual(weight * added_norm / base_norm, 0.1, places=6)

    def test_calibration_rejects_zero_gradient_norm(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        base = parameter.square().sum()
        added = (parameter * 0.0).sum()

        with self.assertRaisesRegex(ValueError, "nonzero"):
            calibrate_additive_loss_weight(base, added, [parameter], target_ratio=0.1)

    def test_slope_loss_does_not_touch_phdos_head(self):
        shared = torch.nn.Linear(3, 4)
        edos_head = torch.nn.Linear(4, 5)
        phdos_head = torch.nn.Linear(4, 3)
        shared_features = shared(torch.ones(2, 3))
        edos_logits = edos_head(shared_features)
        phdos_logits = phdos_head(shared_features)
        phdos_logits.retain_grad()
        target = torch.softmax(torch.randn(2, 5), dim=-1)

        edos_slope_matching_loss(edos_logits, target).backward()

        self.assertIsNotNone(shared.weight.grad)
        self.assertIsNotNone(edos_head.weight.grad)
        self.assertIsNone(phdos_head.weight.grad)
        self.assertIsNone(phdos_logits.grad)

    def test_skip_test_does_not_construct_test_loader(self):
        class Builder:
            def get_dataloader(self, **kwargs):
                raise AssertionError("test loader must not be constructed")

        self.assertIsNone(
            maybe_get_test_loader(Builder(), SimpleNamespace(skip_test_eval=True), True)
        )

    def test_default_test_loader_behavior_is_preserved(self):
        class Builder:
            def __init__(self):
                self.kwargs = None

            def get_dataloader(self, **kwargs):
                self.kwargs = kwargs
                return "test-loader"

        builder = Builder()

        loader = maybe_get_test_loader(
            builder, SimpleNamespace(skip_test_eval=False, batch_size=16), True
        )

        self.assertEqual(loader, "test-loader")
        self.assertEqual(
            builder.kwargs,
            {
                "split": "test",
                "dos_minmax": True,
                "batch_size": 16,
                "dos_sumnorm": True,
            },
        )

    def test_calibration_write_preserves_existing_runtime_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config_used.yaml"
            config_path.write_text(yaml.safe_dump({"runtime": {"existing": 3}}))

            write_edos_slope_calibration(
                config_path,
                {
                    "target_gradient_ratio": 0.1,
                    "lambda": 0.02,
                    "base_gradient_norm": 1.5,
                    "slope_gradient_norm": 0.75,
                },
            )

            runtime = yaml.safe_load(config_path.read_text())["runtime"]
            self.assertEqual(runtime["existing"], 3)
            self.assertEqual(runtime["edos_slope_lambda"], 0.02)
            self.assertEqual(runtime["edos_slope_gradient_norm"], 0.75)


if __name__ == "__main__":
    unittest.main()
