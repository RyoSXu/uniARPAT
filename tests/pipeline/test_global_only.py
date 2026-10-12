"""A true no-local control must retain the shared backbone and run identity."""
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

import torch
import pandas as pd

from model.local_global_encoder import LocalGlobalEncoder
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture, parameters
from tests.pipeline.test_local_structure_controls import predict
from tools.checks.check_long_budget import captured_config
from tools.eval.local_global_compare import training_command
from tools.eval import global_only_compare
from tools.train import build_arg_parser, configure_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.global_only_experiment import GLOBAL_ONLY_SCHEMA, LOCAL_PREFIX, global_only_config
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, deep_reference_state,
    initialize_comparison_model, validate_local_global_config,
)
from utils.local_structure_experiment import FROZEN
from utils.zproj_alignment import state_hash


def configuration():
    reference = ExperimentConfig(**{**FROZEN, "local_route": "B+", "tag": "_unit_reference"})
    return global_only_config(reference, "unit_global_only")


def model_parameters():
    return {**parameters("G0", depth=0), "global_depth": 6, "q1_coord": True}


class TestGlobalOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, dtype=torch.float32):
        before = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            params = model_parameters()
            model = Transformer(**params)
            info = initialize_comparison_model(model, params, 42)
        finally:
            torch.set_default_dtype(before)
        return model.eval(), info

    def test_zero_local_calls_and_no_local_parameters(self):
        model, info = self.create()
        self.assertEqual(len(model.encoder.local_layers), 0)
        self.assertEqual(info["update_counts"], {"local_calls": 0, "internal_local_blocks": 0,
                                               "global_attention": 6, "global_ffn": 6})
        self.assertEqual(model.encoder.execution_order, tuple(("global", i) for i in range(6)))
        self.assertFalse(any(key.startswith(LOCAL_PREFIX) for key in model.state_dict()))
        self.assertFalse(model.encoder.shared_geometry)

    def test_every_retained_tensor_matches_untrained_reference_and_preserves_rng(self):
        params = model_parameters()
        model = Transformer(**params)
        rng = torch.get_rng_state().clone()
        info = initialize_comparison_model(model, params, 42)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        reference, _ = deep_reference_state(42, {**params, "local_route": "B+", "local_depth": 2})
        self.assertEqual(info["reference_state_sha256"], state_hash(reference))
        shared = {key: value for key, value in reference.items() if not key.startswith(LOCAL_PREFIX)}
        self.assertEqual(set(model.state_dict()), set(shared))
        for key, expected in shared.items():
            actual = model.state_dict()[key]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertNotEqual(actual.data_ptr(), expected.data_ptr())
        self.assertEqual(info["state_sha256"], info["shared_state_sha256"])

    def test_no_geometry_record_building_or_local_execution_in_full_forward(self):
        for dtype in (torch.float32, torch.float64):
            model, _ = self.create(dtype)
            z, geometry = fixture(dtype)
            configure_local_global_execution(model)
            self.assertIsNone(model._local_geometry_cache)
            self.assertEqual(model._local_geometry_cache_disabled_reason, "no_geometry")
            with patch("utils.periodic_geometry.build_periodic_records", side_effect=AssertionError("geometry")), \
                    patch("model.local_global_encoder.LocalResidualLayer.forward",
                          side_effect=AssertionError("local update")), torch.no_grad():
                expected = predict(model, z, geometry)
                changed = CrystalGeometry(geometry.cell * 1.7, geometry.frac_coords + .23)
                actual = predict(model, z, changed)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "geometry-consuming"):
                model.enable_local_geometry_cache()
            with self.assertRaisesRegex(ValueError, "no geometry"):
                model.encoder(torch.rand(2, 4, 16), object(), z.eq(0))

    def test_permutation_padding_and_mixed_empty_row_remain_finite(self):
        model, _ = self.create()
        z, geometry = fixture()
        order = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            expected = predict(model, z, geometry)
            actual = predict(model, z[:, order], CrystalGeometry(geometry.cell, geometry.frac_coords[:, order]))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-5)
        mask = z.eq(0)
        mask[1] = True
        h = torch.randn(2, 4, 16)
        encoded = model.encoder(h, None, mask)
        self.assertTrue(torch.isfinite(encoded).all())
        self.assertEqual(int(encoded[mask].count_nonzero()), 0)

    def test_frozen_recipe_rejects_depth_seed_budget_and_variant_changes(self):
        cfg = configuration()
        validate_local_global_config(cfg)
        for change in ({"local_depth": 2}, {"local_depth": False}, {"local_depth": 0.},
                       {"global_depth": 2}, {"global_depth": 6.},
                       {"local_layout": "alternating"}, {"seed": 43},
                       {"epochs": 100}, {"lr": 1e-4}, {"use_amp": True}, {"lg_baseline": True},
                       {"tag": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_local_global_config(replace(cfg, **change))
        for depth in (False, 0., 1):
            with self.assertRaises(ValueError):
                LocalGlobalEncoder("G0", "staged", depth, global_depth=6)
        with self.assertRaises(ValueError):
            LocalGlobalEncoder("G0", "staged", 0)
        with self.assertRaises(ValueError):
            LocalGlobalEncoder("C0", "staged", 0, global_depth=6)

    def test_command_round_trip_and_actual_runner_configuration(self):
        cfg = configuration()
        parsed = ExperimentConfig.from_args(build_arg_parser().parse_args(training_command(cfg)[2:]))
        self.assertEqual(asdict(parsed), asdict(cfg))
        assembled = captured_config(cfg)
        params = assembled["model"]["params"]["sub_model"]["transformer"]
        self.assertEqual(params["local_route"], "G0")
        self.assertEqual(params["local_depth"], 0)
        self.assertEqual(params["global_depth"], 6)
        self.assertEqual(comparison_contract(cfg, params)["schema"], GLOBAL_ONLY_SCHEMA)

    def test_checkpoint_restores_optimizer_and_rejects_local_arm_contract(self):
        cfg = configuration()
        params = model_parameters()
        model, _ = self.create()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        z, geometry = fixture()
        sum(output.square().mean() for output in predict(model, z, geometry).values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        contract = comparison_contract(cfg, params)
        payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, .5,
            model_contract=contract, training_rng=capture_training_rng())
        fresh = Transformer(**params)
        new_optimizer = torch.optim.AdamW(fresh.parameters(), lr=cfg.lr)
        restore_ablation_checkpoint(payload, fresh, new_optimizer, False, model_contract=contract)
        self.assertEqual(state_hash(model.state_dict()), state_hash(fresh.state_dict()))
        for key, slots in optimizer.state_dict()["state"].items():
            for slot, expected in slots.items():
                torch.testing.assert_close(new_optimizer.state_dict()["state"][key][slot], expected, atol=0, rtol=0)
        wrong = comparison_contract(replace(cfg, local_route="C0", local_depth=2),
                                   {**params, "local_route": "C0", "local_depth": 2})
        with self.assertRaisesRegex(ValueError, "different model contract"):
            restore_ablation_checkpoint(payload, fresh, new_optimizer, False, model_contract=wrong)

    def test_completed_run_is_verified_and_never_retrained(self):
        cfg = configuration()
        with patch("sys.argv", ["global_only_compare", "--tag-prefix", "unit_global_only",
                                "--run", "--preflight", "/tmp/g0-unit-preflight.json"]), \
                patch.object(global_only_compare, "load_control_reference", return_value=(None, cfg)), \
                patch.object(global_only_compare, "verify_preflight", return_value={}), \
                patch.object(global_only_compare, "summarize", return_value=[{"status": "complete"}]), \
                patch.object(global_only_compare, "train_and_eval") as train, \
                patch.object(global_only_compare, "verify_completed") as verify:
            global_only_compare.main()
            train.assert_not_called()
            verify.assert_called_once()

    def test_report_keeps_original_baseline_separate_and_uses_arm_minus_g0(self):
        fields = {"best_epoch": 1, "best_score": .4, "epochs": 35,
                  "contract_schema": "unit", "init_protocol": "unit",
                  "epoch_time_s_median": 1., "epoch_time_s_sum": 35.,
                  "peak_vram_mb_median": 1., "peak_vram_mb_max": 1.}
        exports = {}
        for i, label in enumerate(("G0", "C0", "C1", "C2", "original_baseline")):
            frame = pd.DataFrame(index=["a", "b", "c"])
            for mode in ("blind", "oracle"):
                for task in ("edos", "phdos"):
                    frame[f"{mode}_r2_{task}"] = [-.2 + .05*i, .3 + .05*i, .5 + .05*i]
                    frame[f"{mode}_mae_{task}"] = [1. - .05*i, 2. - .05*i, 3. - .05*i]
            exports[label] = ({**fields, "tag": label}, frame)
        report = global_only_compare.comparison_report(exports, draws=50)
        self.assertEqual(report["comparison_reference"], "G0")
        self.assertEqual(report["runs"]["G0"]["tag"], "G0")
        self.assertEqual(report["runs"]["original_baseline"]["tag"], "original_baseline")
        self.assertAlmostEqual(report["comparisons"]["original_baseline"]["blind_edos_r2_median"]["difference"], .2)
        self.assertAlmostEqual(report["comparisons"]["C0"]["blind_edos_mae_median"]["difference"], -.05)


if __name__ == "__main__":
    unittest.main()
