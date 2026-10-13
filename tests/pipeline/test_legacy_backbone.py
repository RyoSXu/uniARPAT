"""Old/new global-backbone attribution must preserve the intended contrasts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import torch

from model.local_global_encoder import LegacyContentGlobalLayer, LocalGlobalEncoder
from model.transformer import Transformer, TransformerEncoderLayer
from tests.model.test_local_global_encoder import fixture, parameters
from tests.pipeline.test_local_structure_controls import predict
from tools.checks.check_long_budget import captured_config
from tools.eval import legacy_backbone_compare as runner
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, configure_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.legacy_backbone_experiment import (
    LEGACY_SCHEMA, LOCAL_PREFIX, baseline_parameters, legacy_backbone_configs, old_source_key,
)
from utils.lg_baseline_experiment import initialize_baseline_model
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, validate_local_global_config,
)
from utils.local_structure_experiment import FROZEN
from utils.zproj_alignment import state_hash


def configurations():
    return legacy_backbone_configs(ExperimentConfig(**{**FROZEN, "local_route": "B+"}), "unit_oldg")


def model_params(route):
    return {**parameters(route, depth=0 if route == "O0" else 2), "global_depth": 6, "q1_coord": True}


class TestLegacyBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, route, dtype=torch.float32):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            params = model_params(route)
            model = Transformer(**params)
            info = initialize_comparison_model(model, params, 42)
        finally:
            torch.set_default_dtype(old_dtype)
        return model.eval(), info

    def test_content_layer_matches_original_with_geometric_scores_zeroed(self):
        for dtype, tolerance in ((torch.float32, 2e-6), (torch.float64, 1e-12)):
            old = TransformerEncoderLayer(16, 4, 32, 0., "gelu").to(dtype)
            with torch.no_grad():
                old.rp_proj.weight.zero_()
                old.rp_proj.bias.zero_()
            content = LegacyContentGlobalLayer(16, 4, 32, 0., "gelu").to(dtype)
            content.layer.load_state_dict({key: value for key, value in old.state_dict().items()
                                          if not key.startswith("rp_proj.")}, strict=True)
            mask = torch.tensor([[False, False, True], [False, True, True]])
            h = torch.randn(2, 3, 16, dtype=dtype).masked_fill(mask[..., None], 0)
            expected = old(h, src_key_padding_mask=mask).masked_fill(mask[..., None], 0)
            actual = content(h, mask)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            actual.square().sum().backward()
            expected.square().sum().backward()
            for key, parameter in content.layer.named_parameters():
                torch.testing.assert_close(parameter.grad, dict(old.named_parameters())[key].grad,
                                           atol=tolerance, rtol=tolerance)
            self.assertFalse(hasattr(content.layer, "rp_proj"))
            with self.assertRaisesRegex(ValueError, "geometry-free"):
                content.layer(h, rp_base=torch.zeros(2, 3, 3, 576, dtype=dtype))
            with self.assertRaisesRegex(ValueError, "geometry-free"):
                content.layer(h, g1_adj=torch.ones(2, 3, 3, dtype=torch.bool))

    def test_remaining_baseline_and_c1_tensors_copy_exactly_with_rng_preserved(self):
        old_parts = []
        for route in ("O0", "O1"):
            params = model_params(route)
            model = Transformer(**params)
            rng = torch.get_rng_state().clone()
            info = initialize_comparison_model(model, params, 42)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                base_params = baseline_parameters(params)
                baseline = Transformer(**base_params)
                base_info = initialize_baseline_model(baseline, base_params, 42)
            self.assertEqual(info["baseline_initialization"], base_info)
            for key, actual in model.state_dict().items():
                if not key.startswith(LOCAL_PREFIX):
                    source = baseline.state_dict()[old_source_key(key)]
                    torch.testing.assert_close(actual, source, atol=0, rtol=0)
                    self.assertNotEqual(actual.data_ptr(), source.data_ptr())
            old_parts.append(info["shared_old_state_sha256"])
            if route == "O1":
                c1_params = {**params, "local_route": "C1"}
                c1 = Transformer(**c1_params)
                c1_info = initialize_comparison_model(c1, c1_params, 42)
                self.assertEqual(info["c1_initialization"], c1_info)
                for key, actual in model.state_dict().items():
                    if key.startswith(LOCAL_PREFIX):
                        torch.testing.assert_close(actual, c1.state_dict()[key], atol=0, rtol=0)
            else:
                self.assertEqual(info["n_local_tensors"], 0)
                self.assertEqual(len(model.encoder.local_layers), 0)
        self.assertEqual(*old_parts)

    def test_o0_ignores_geometry_and_o1_uses_only_radial_c1(self):
        z, geometry = fixture()
        for route in ("O0", "O1"):
            model, _ = self.create(route)
            self.assertFalse(any("rp_proj" in key or "rp_encoder" in key for key in model.state_dict()))
            with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old scoring")), \
                    patch("model.periodic_messages.o3.spherical_harmonics", side_effect=AssertionError("angles")), \
                    torch.no_grad():
                expected = predict(model, z, geometry)
                if route == "O0":
                    configure_local_global_execution(model)
                    self.assertEqual(model._local_geometry_cache_disabled_reason, "no_geometry")
                    with patch("utils.periodic_geometry.build_periodic_records", side_effect=AssertionError("records")):
                        actual = predict(model, z, CrystalGeometry(geometry.cell*1.9, geometry.frac_coords+.17))
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                    with self.assertRaises(ValueError):
                        model.enable_local_geometry_cache()
                else:
                    self.assertEqual(model.encoder.geometry_route, "C1")
                    actual = predict(model, z, CrystalGeometry(geometry.cell*1.3, geometry.frac_coords))
                    self.assertTrue(any(not torch.allclose(actual[key], expected[key]) for key in expected))

    def test_whole_o0_matches_original_baseline_with_scores_zeroed(self):
        model, _ = self.create("O0")
        base_params = baseline_parameters(model_params("O0"))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            baseline = Transformer(**base_params)
            initialize_baseline_model(baseline, base_params, 42)
        baseline.eval()
        z, geometry = fixture()
        with torch.no_grad():
            for layer in baseline.encoder.layers:
                layer.rp_proj.weight.zero_()
                layer.rp_proj.bias.zero_()
            expected = predict(baseline, z, geometry)
            actual = predict(model, z, geometry)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=2e-5)

    def test_full_outputs_e3_permutation_and_empty_rows(self):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            z, geometry = fixture(dtype)
            for route in ("O0", "O1"):
                model, _ = self.create(route, dtype)
                with torch.no_grad():
                    expected = predict(model, z, geometry)
                    for name, matrix, translation in e3_operations():
                        q, t = torch.as_tensor(matrix, dtype=dtype), torch.as_tensor(translation, dtype=dtype)
                        cell = geometry.cell @ q
                        frac = geometry.frac_coords + torch.bmm(t.expand(2, 1, 3), torch.linalg.inv(cell))
                        actual = predict(model, z, CrystalGeometry(cell, frac))
                        for key in expected:
                            torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
                    order = torch.tensor([2, 0, 3, 1])
                    actual = predict(model, z[:, order], CrystalGeometry(geometry.cell, geometry.frac_coords[:, order]))
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
                    empty_z = z.clone()
                    empty_z[1] = 0
                    empty = predict(model, empty_z, geometry)
                    self.assertTrue(all(torch.isfinite(value).all() for value in empty.values()))

    def test_frozen_recipe_and_runner_round_trip(self):
        for cfg in configurations():
            validate_local_global_config(cfg)
            parsed = ExperimentConfig.from_args(build_arg_parser().parse_args(training_command(cfg)[2:]))
            self.assertEqual(asdict(parsed), asdict(cfg))
            params = captured_config(cfg)["model"]["params"]["sub_model"]["transformer"]
            self.assertEqual(comparison_contract(cfg, params)["schema"], LEGACY_SCHEMA)
            for change in ({"seed": 43}, {"epochs": 100}, {"local_depth": 1}, {"local_depth": False},
                           {"local_depth": float(cfg.local_depth)}, {"global_depth": 6.},
                           {"local_layout": "alternating"}, {"lg_baseline": True}, {"use_amp": True}, {"tag": ""}):
                with self.subTest(route=cfg.local_route, change=change), self.assertRaises(ValueError):
                    validate_local_global_config(replace(cfg, **change))
        with self.assertRaises(ValueError):
            LocalGlobalEncoder("O0", "staged", 0)

    def test_finite_gradients_and_checkpoint_contract_isolation(self):
        z, geometry = fixture()
        for cfg in configurations():
            model, _ = self.create(cfg.local_route)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
            sum(value.square().mean() for value in predict(model, z, geometry).values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
            params = model_params(cfg.local_route)
            contract = comparison_contract(cfg, params)
            payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, .5,
                                                model_contract=contract, training_rng=capture_training_rng())
            fresh = Transformer(**params)
            fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=cfg.lr)
            restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=contract)
            self.assertEqual(state_hash(model.state_dict()), state_hash(fresh.state_dict()))
            for key, slots in optimizer.state_dict()["state"].items():
                for slot, expected in slots.items():
                    torch.testing.assert_close(fresh_optimizer.state_dict()["state"][key][slot], expected, atol=0, rtol=0)
            wrong_cfg = replace(cfg, local_route="C1", local_depth=2)
            wrong = comparison_contract(wrong_cfg, {**params, "local_route": "C1", "local_depth": 2})
            with self.assertRaisesRegex(ValueError, "different model contract"):
                restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=wrong)

    def test_factorial_difference_directions_and_interaction(self):
        common = {"best_epoch": 1, "best_score": .4, "epochs": 35, "contract_schema": "unit", "init_protocol": "unit",
                  "epoch_time_s_median": 1., "epoch_time_s_sum": 35., "peak_vram_mb_median": 1., "peak_vram_mb_max": 1.}
        exports = {}
        for label, offset in {"G0": 0., "C1": .05, "O0": .02, "O1": .22, "original_baseline": .12}.items():
            frame = pd.DataFrame(index=["a", "b", "c"])
            for mode in ("blind", "oracle"):
                for task in ("edos", "phdos"):
                    frame[f"{mode}_r2_{task}"] = [-.1+offset, .2+offset, .8+offset]
                    frame[f"{mode}_mae_{task}"] = [3.-offset, 2.-offset, 1.-offset]
            exports[label] = ({**common, "tag": label}, frame)
        report = runner.factorial_report(exports, draws=100)
        expected = {"baseline_minus_O0": .10, "O1_minus_O0": .20, "C1_minus_G0": .05,
                    "O0_minus_G0": .02, "O1_minus_C1": .17, "interaction_old_minus_new": .15}
        for name, difference in expected.items():
            self.assertAlmostEqual(report["contrasts"][name]["blind_edos_r2_median"]["difference"], difference)
            self.assertAlmostEqual(report["contrasts"][name]["blind_edos_mae_median"]["difference"], -difference)
        changed = dict(exports)
        changed["G0"] = (exports["G0"][0], exports["G0"][1].iloc[::-1])
        with self.assertRaisesRegex(ValueError, "material-ID"):
            runner.factorial_report(changed, draws=10)

    def test_completed_arms_are_verified_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "summarize", return_value=[{"status": "complete"}]), \
                patch.object(runner, "verify_completed") as verify, \
                patch.object(runner, "analyze") as analyze, \
                patch.object(runner.subprocess, "Popen") as process:
            controls = configurations()
            runner.execute(controls, {}, Path(directory))
            process.assert_not_called()
            self.assertEqual(verify.call_count, 2)
            analyze.assert_called_once()

    def test_composition_interaction_uses_equal_groups_and_checks_pair_ids(self):
        template = pd.DataFrame({"left_id": ["a", "b", "d"], "right_id": ["b", "c", "e"],
                                 "composition": ["X", "X", "Y"]})
        tables = {}
        values = {"G0": [1., 1., 10.], "C1": [.9, .9, 9.7], "O0": [.8, .8, 9.8],
                  "O1": [.5, .5, 9.5], "original_baseline": [.7, .7, 9.7]}
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                for label, errors in values.items():
                    tables[label, grouping, task] = template.assign(difference_error=errors)
        result = runner.composition_contrasts(tables)
        self.assertAlmostEqual(result["C1_minus_G0"]["valid_absolute_edos"]["mean_group_error_delta"], -.2)
        self.assertAlmostEqual(result["interaction_old_minus_new"]["valid_absolute_edos"]["mean_group_error_delta"], -.1)
        self.assertEqual(result["interaction_old_minus_new"]["valid_absolute_edos"]["groups"], 2)
        tables["C1", "absolute", "edos"] = tables["C1", "absolute", "edos"].iloc[::-1]
        with self.assertRaisesRegex(ValueError, "pair identities"):
            runner.composition_contrasts(tables)

    def test_failed_arm_stops_queue_without_running_next_arm(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.subprocess, "Popen", return_value=MagicMock(pid=123, wait=lambda: 9)) as process:
            with self.assertRaises(RuntimeError):
                runner.execute(configurations(), {}, Path(directory))
            self.assertEqual(process.call_count, 1)
            self.assertEqual(json.loads((Path(directory)/"queue.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
