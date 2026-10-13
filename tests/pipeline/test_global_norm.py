"""N1 must change only global normalization placement and isolate run identity."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import torch

from model.local_global_encoder import ScalarGlobalLayer
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture, parameters
from tests.pipeline.test_local_structure_controls import predict
from tools.checks.check_long_budget import captured_config
from tools.eval import global_norm_compare as runner
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, configure_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.global_norm_experiment import NORM_SCHEMA, c1_reference_parameters, global_norm_config
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, validate_local_global_config,
)
from utils.local_structure_experiment import FROZEN
from utils.zproj_alignment import state_hash


def configuration():
    return global_norm_config(ExperimentConfig(**{**FROZEN, "local_route": "B+"}), "unit_norm")


def model_params(route):
    return {**parameters(route, depth=2), "global_depth": 6, "q1_coord": True}


class TestGlobalNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, route="N1", dtype=torch.float32):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            params = model_params(route)
            model = Transformer(**params)
            info = initialize_comparison_model(model, params, 42)
        finally:
            torch.set_default_dtype(old_dtype)
        return model.eval(), info

    def test_post_norm_matches_standard_layer_outputs_and_gradients(self):
        for dtype, tolerance in ((torch.float32, 3e-6), (torch.float64, 1e-12)):
            actual_layer = ScalarGlobalLayer(16, 4, 32, 0., "gelu", norm_first=False).to(dtype)
            expected_layer = torch.nn.TransformerEncoderLayer(
                16, 4, 32, 0., "gelu", batch_first=True, norm_first=False).to(dtype)
            mapping = {"norm_attn": "norm1", "norm_ffn": "norm2", "attn": "self_attn",
                       "ffn.0": "linear1", "ffn.3": "linear2"}
            mapped = {}
            for key, value in actual_layer.state_dict().items():
                for source, target in mapping.items():
                    if key.startswith(source + "."):
                        mapped[target + key[len(source):]] = value
                        break
            expected_layer.load_state_dict(mapped, strict=True)
            mask = torch.tensor([[False, False, True], [False, True, True]])
            h = torch.randn(2, 3, 16, dtype=dtype)
            actual_h, expected_h = h.clone().requires_grad_(), h.clone().requires_grad_()
            actual = actual_layer(actual_h, mask)
            expected = expected_layer(expected_h, src_key_padding_mask=mask).masked_fill(mask[..., None], 0)
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(actual_h.grad, expected_h.grad, rtol=tolerance, atol=tolerance)
            expected_params = dict(expected_layer.named_parameters())
            for key, parameter in actual_layer.named_parameters():
                for source, target in mapping.items():
                    if key.startswith(source + "."):
                        torch.testing.assert_close(parameter.grad, expected_params[target + key[len(source):]].grad,
                                                   rtol=tolerance, atol=tolerance)
                        break

    def test_default_pre_norm_keeps_the_previous_formula(self):
        layer = ScalarGlobalLayer(16, 4, 32, 0., "gelu").eval()
        h = torch.randn(2, 3, 16)
        mask = torch.tensor([[False, False, True], [True, True, True]])
        with torch.no_grad():
            values = h[:1].masked_fill(mask[:1, :, None], 0)
            normalized = layer.norm_attn(values).masked_fill(mask[:1, :, None], 0)
            message = layer.attn(normalized, normalized, normalized,
                                 key_padding_mask=mask[:1], need_weights=False)[0]
            values = (values + layer.dropout_attn(message)).masked_fill(mask[:1, :, None], 0)
            values = (values + layer.dropout_ffn(layer.ffn(layer.norm_ffn(values)))).masked_fill(mask[:1, :, None], 0)
            expected = torch.cat([values, torch.zeros_like(h[1:])])
            torch.testing.assert_close(layer(h, mask), expected, atol=0, rtol=0)
        self.assertTrue(layer.norm_first)
        with self.assertRaises(ValueError):
            ScalarGlobalLayer(16, 4, 32, 0., "gelu", norm_first="post")

    def test_entire_initial_state_matches_untrained_c1_and_rng_is_preserved(self):
        c1, c1_info = self.create("C1")
        params = model_params("N1")
        n1 = Transformer(**params)
        rng = torch.get_rng_state().clone()
        info = initialize_comparison_model(n1, params, 42)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(info["c1_initialization"], c1_info)
        self.assertEqual(info["state_sha256"], c1_info["state_sha256"])
        self.assertEqual(n1.state_dict().keys(), c1.state_dict().keys())
        for key, value in n1.state_dict().items():
            torch.testing.assert_close(value, c1.state_dict()[key], atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), c1.state_dict()[key].data_ptr())
        self.assertEqual(sum(p.numel() for p in n1.parameters()), sum(p.numel() for p in c1.parameters()))
        for n1_layer, c1_layer in zip(n1.encoder.global_layers, c1.encoder.global_layers):
            self.assertFalse(n1_layer.norm_first)
            self.assertTrue(c1_layer.norm_first)
            self.assertEqual(n1_layer.attn.dropout, c1_layer.attn.dropout)
        self.assertEqual(n1.encoder.execution_order, c1.encoder.execution_order)
        self.assertEqual(n1.encoder.geometry_route, "C1")

    def test_whole_model_e3_permutation_and_empty_rows(self):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            model, _ = self.create(dtype=dtype)
            z, geometry = fixture(dtype)
            with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old scoring")), \
                    patch("model.periodic_messages.o3.spherical_harmonics", side_effect=AssertionError("angles")), \
                    torch.no_grad():
                expected = predict(model, z, geometry)
                for name, matrix, translation in e3_operations():
                    q = torch.as_tensor(matrix, dtype=dtype)
                    t = torch.as_tensor(translation, dtype=dtype)
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
                self.assertTrue(all(torch.isfinite(x).all() for x in predict(model, empty_z, geometry).values()))
            configure_local_global_execution(model)
            self.assertIsNone(model._local_geometry_cache)
            self.assertEqual(model._local_geometry_cache_disabled_reason, "cpu")
            model.enable_local_geometry_cache(max_bytes=2**20)
            self.assertEqual(model._local_geometry_cache.route, "C1")

    def test_frozen_recipe_cli_roundtrip_and_contract(self):
        cfg = configuration()
        validate_local_global_config(cfg)
        parsed = ExperimentConfig.from_args(build_arg_parser().parse_args(training_command(cfg)[2:]))
        self.assertEqual(asdict(parsed), asdict(cfg))
        params = captured_config(cfg)["model"]["params"]["sub_model"]["transformer"]
        contract = comparison_contract(cfg, params)
        self.assertEqual(contract["schema"], NORM_SCHEMA)
        self.assertTrue(contract["ablation"]["all_initial_tensors_identical"])
        self.assertIsNone(cfg.dropout)
        self.assertEqual(contract["ablation"]["attention_probability_dropout"], params["dropout"])
        for change in ({"seed": 43}, {"epochs": 100}, {"local_depth": False}, {"global_depth": 6.},
                       {"local_layout": "alternating"}, {"use_amp": True}, {"tag": ""}, {"norm": "minmax"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_local_global_config(replace(cfg, **change))
        with self.assertRaises(ValueError):
            c1_reference_parameters({**params, "normalize_before": True})

    def test_gradients_restore_and_rejection_of_same_shape_c1_checkpoint(self):
        model, _ = self.create()
        z, geometry = fixture()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        sum(v.square().mean() for v in predict(model, z, geometry).values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        params = model_params("N1")
        contract = comparison_contract(configuration(), params)
        payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, .5,
            model_contract=contract, training_rng=capture_training_rng())
        fresh, _ = self.create()
        fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=5e-5)
        restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=contract)
        self.assertEqual(state_hash(model.state_dict()), state_hash(fresh.state_dict()))
        for key, slots in optimizer.state_dict()["state"].items():
            for slot, expected in slots.items():
                torch.testing.assert_close(fresh_optimizer.state_dict()["state"][key][slot], expected, atol=0, rtol=0)
        c1_cfg = replace(configuration(), local_route="C1")
        wrong = comparison_contract(c1_cfg, {**params, "local_route": "C1"})
        with self.assertRaisesRegex(ValueError, "different model contract"):
            restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=wrong)

    def test_statistics_keep_n1_minus_reference_directions(self):
        common = {"best_epoch": 1, "best_score": .4, "epochs": 35, "contract_schema": "unit", "init_protocol": "unit",
                  "epoch_time_s_median": 1., "epoch_time_s_sum": 35., "peak_vram_mb_median": 1., "peak_vram_mb_max": 1.}
        exports = {}
        for label, offset in {"N1": .2, "C1": 0., "O1": .1, "original_baseline": .15}.items():
            frame = pd.DataFrame(index=["a", "b", "c"])
            for mode in ("blind", "oracle"):
                for task in ("edos", "phdos"):
                    frame[f"{mode}_r2_{task}"] = [-.1+offset, .2+offset, .8+offset]
                    frame[f"{mode}_mae_{task}"] = [3.-offset, 2.-offset, 1.-offset]
            exports[label] = ({**common, "tag": label}, frame)
        report = runner.comparison_report(exports, draws=100)
        for reference, expected in {"C1": .2, "O1": .1, "original_baseline": .05}.items():
            values = report["contrasts"][f"N1_minus_{reference}"]
            self.assertAlmostEqual(values["blind_edos_r2_median"]["difference"], expected)
            self.assertAlmostEqual(values["blind_edos_mae_median"]["difference"], -expected)
        changed = dict(exports)
        changed["C1"] = (exports["C1"][0], exports["C1"][1].iloc[::-1])
        with self.assertRaisesRegex(ValueError, "material-ID"):
            runner.comparison_report(changed, draws=10)

    def test_composition_bootstrap_checks_pairs_and_weights_groups_equally(self):
        template = pd.DataFrame({"left_id": ["a", "b", "d"], "right_id": ["b", "c", "e"],
                                 "composition": ["X", "X", "Y"]})
        pairs = {}
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                for label, values in {"N1": [.5, .5, 9.5], "C1": [1., 1., 10.],
                                      "O1": [.8, .8, 9.8], "original_baseline": [.7, .7, 9.7]}.items():
                    pairs[label, grouping, task] = template.assign(difference_error=values)
        report = runner.composition_contrasts(pairs)
        self.assertAlmostEqual(report["N1_minus_C1"]["valid_absolute_edos"]["mean_group_error_delta"], -.5)
        pairs["C1", "absolute", "edos"] = pairs["C1", "absolute", "edos"].iloc[::-1]
        with self.assertRaisesRegex(ValueError, "pair identities"):
            runner.composition_contrasts(pairs)

    def test_completed_run_is_verified_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "summarize", return_value=[{"status": "complete"}]), \
                patch.object(runner, "verify_completed") as verify, \
                patch.object(runner, "analyze") as analyze, \
                patch.object(runner.subprocess, "Popen") as process:
            runner.execute(configuration(), {}, Path(directory))
            process.assert_not_called()
            verify.assert_called_once()
            analyze.assert_called_once()

    def test_failed_run_stops_before_analysis(self):
        failed = MagicMock(pid=17)
        failed.wait.return_value = 1
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.subprocess, "Popen", return_value=failed), \
                patch.object(runner, "analyze") as analyze:
            with self.assertRaisesRegex(RuntimeError, "N1 failed"):
                runner.execute(configuration(), {}, Path(directory))
            analyze.assert_not_called()
            self.assertEqual(json.loads((Path(directory)/"queue.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
