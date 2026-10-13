"""D1 isolates probability dropout and must keep N1 tensors/other behavior."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_structure_controls import predict
from tools.checks.check_long_budget import captured_config
from tools.checks.check_global_norm import _train_batch_updates
from tools.eval import attention_dropout_compare as runner
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, configure_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.attention_dropout_experiment import (
    DROPOUT_SCHEMA, attention_dropout_config, n1_reference_parameters,
)
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, restore_training_rng,
    validate_local_global_config,
)
from utils.local_structure_experiment import FROZEN
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash


def configuration():
    return attention_dropout_config(ExperimentConfig(**{**FROZEN, "local_route": "B+"}), "unit_dropout")


class TestAttentionDropout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, route="D1", dtype=torch.float32):
        before = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            params = {**model_params(route), "dropout": .05}
            model = Transformer(**params)
            rng = rng_digest(rng_snapshot())
            info = initialize_comparison_model(model, params, 42)
            self.assertEqual(rng, rng_digest(rng_snapshot()))
        finally:
            torch.set_default_dtype(before)
        return model.eval(), info, params

    def test_complete_untrained_state_and_all_other_dropout_match_n1(self):
        d1, info, _ = self.create()
        n1, reference, _ = self.create("N1")
        self.assertEqual(info["n1_initialization"], reference)
        self.assertEqual(info["state_sha256"], reference["state_sha256"])
        self.assertEqual(d1.state_dict().keys(), n1.state_dict().keys())
        for name, value in d1.state_dict().items():
            torch.testing.assert_close(value, n1.state_dict()[name], atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), n1.state_dict()[name].data_ptr())
        self.assertEqual(info["parameter_count"], reference["parameter_count"])
        self.assertEqual(d1.encoder.execution_order, n1.encoder.execution_order)
        reference_modules = dict(n1.named_modules())
        for name, module in d1.named_modules():
            if isinstance(module, torch.nn.Dropout):
                self.assertEqual(module.p, reference_modules[name].p, name)
        for left, right in zip(d1.encoder.global_layers, n1.encoder.global_layers):
            self.assertFalse(left.norm_first)
            self.assertFalse(right.norm_first)
            self.assertEqual(left.attn.dropout, 0.)
            self.assertEqual(right.attn.dropout, .05)

    def test_eval_outputs_are_identical_to_n1(self):
        for dtype in (torch.float32, torch.float64):
            d1, _, _ = self.create(dtype=dtype)
            n1, _, _ = self.create("N1", dtype=dtype)
            z, geometry = fixture(dtype)
            with torch.no_grad():
                expected, actual = predict(n1, z, geometry), predict(d1, z, geometry)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_training_outputs_gradients_and_rng_match_manual_n1_intervention(self):
        for dtype in (torch.float32, torch.float64):
            d1, _, _ = self.create(dtype=dtype)
            n1, _, _ = self.create("N1", dtype=dtype)
            for layer in n1.encoder.global_layers:
                layer.attn.dropout = 0.
            z, geometry = fixture(dtype)
            rng = torch.get_rng_state().clone()
            results = []
            for model in (n1, d1):
                model.train()
                torch.set_rng_state(rng)
                output = predict(model, z, geometry)
                sum(v.square().mean() for v in output.values()).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                results.append((output, torch.get_rng_state().clone()))
            for key in results[0][0]:
                torch.testing.assert_close(results[0][0][key], results[1][0][key], atol=0, rtol=0)
            expected = dict(n1.named_parameters())
            for key, parameter in d1.named_parameters():
                torch.testing.assert_close(parameter.grad, expected[key].grad, atol=0, rtol=0)
            torch.testing.assert_close(results[0][1], results[1][1], atol=0, rtol=0)

    def test_real_batch_probe_switches_an_eval_model_to_train_before_updates(self):
        model, _, _ = self.create("N1")
        self.assertFalse(model.training)
        z, geometry = fixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        def update(batch, step):
            self.assertEqual(step, 0)
            self.assertTrue(model.training)
            self.assertTrue(all(layer.attn.training for layer in model.encoder.global_layers))
            optimizer.zero_grad(set_to_none=True)
            loss = sum(value.square().mean() for value in predict(model, *batch).values())
            loss.backward()
            optimizer.step()
            return {"loss": float(loss.detach())}

        report = _train_batch_updates(SimpleNamespace(model={"transformer": model}, train_one_step=update),
                                      [(z, geometry), (z, geometry)])
        self.assertEqual(report["batches"], 2)
        self.assertTrue(report["training_mode"])
        self.assertEqual(report["attention_training_modes"], [True] * 6)

    def test_symmetry_permutation_empty_rows_and_c1_cache(self):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            model, _, _ = self.create(dtype=dtype)
            z, geometry = fixture(dtype)
            with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old scoring")), \
                    patch("model.periodic_messages.o3.spherical_harmonics", side_effect=AssertionError("angles")), \
                    torch.no_grad():
                expected = predict(model, z, geometry)
                for _, matrix, translation in e3_operations():
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
                empty = z.clone()
                empty[1] = 0
                self.assertTrue(all(torch.isfinite(v).all() for v in predict(model, empty, geometry).values()))
            configure_local_global_execution(model)
            self.assertEqual(model._local_geometry_cache_disabled_reason, "cpu")
            model.enable_local_geometry_cache(max_bytes=2**20)
            self.assertEqual(model._local_geometry_cache.route, "C1")

    def test_initializer_rejects_accidental_changes_to_normalization_or_other_dropout(self):
        model, _, params = self.create()
        for attribute, value in (("norm_first", True), ("attn.dropout", .05),
                                 ("dropout_attn.p", 0.), ("dropout_ffn.p", 0.), ("ffn.2.p", 0.)):
            changed = deepcopy(model)
            parts = attribute.split(".")
            module = changed.encoder.global_layers[0]
            for name in parts[:-1]:
                module = module[int(name)] if name.isdigit() else getattr(module, name)
            setattr(module, parts[-1], value)
            with self.assertRaisesRegex(ValueError, "only attention-probability"):
                initialize_comparison_model(changed, params, 42)
        with self.assertRaises(ValueError):
            n1_reference_parameters({**params, "normalize_before": True})

    def test_frozen_recipe_cli_roundtrip_and_explicit_contract(self):
        cfg = configuration()
        validate_local_global_config(cfg)
        parsed = ExperimentConfig.from_args(build_arg_parser().parse_args(training_command(cfg)[2:]))
        self.assertEqual(asdict(cfg), asdict(parsed))
        params = captured_config(cfg)["model"]["params"]["sub_model"]["transformer"]
        contract = comparison_contract(cfg, params)
        self.assertEqual(contract["schema"], DROPOUT_SCHEMA)
        self.assertEqual(contract["ablation"]["attention_probability_dropout"], 0.)
        self.assertEqual(contract["ablation"]["residual_and_ffn_dropout"], .05)
        self.assertTrue(contract["ablation"]["all_initial_tensors_identical"])
        for change in ({"seed": 43}, {"epochs": 100}, {"local_depth": False}, {"global_depth": 6.},
                       {"local_layout": "alternating"}, {"use_amp": True}, {"tag": ""},
                       {"dropout": 0.}, {"norm": "minmax"}, {"init_ckpt": "N1.pth"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_local_global_config(replace(cfg, **change))

    def test_checkpoint_roundtrip_and_rejection_of_n1_c1_with_identical_shapes(self):
        model, _, params = self.create()
        model.train()
        z, geometry = fixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        sum(v.square().mean() for v in predict(model, z, geometry).values()).backward()
        optimizer.step()
        contract = comparison_contract(configuration(), params)
        rng = rng_digest(rng_snapshot())
        payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, .5,
            model_contract=contract, training_rng=capture_training_rng())
        fresh, _, _ = self.create()
        fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=5e-5)
        restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=contract)
        restore_training_rng(payload["training_rng"])
        self.assertEqual(rng, rng_digest(rng_snapshot()))
        self.assertEqual(state_hash(model.state_dict()), state_hash(fresh.state_dict()))
        for key, slots in optimizer.state_dict()["state"].items():
            for name, value in slots.items():
                torch.testing.assert_close(fresh_optimizer.state_dict()["state"][key][name], value, atol=0, rtol=0)
        for route in ("N1", "C1"):
            wrong = comparison_contract(replace(configuration(), local_route=route), {**params, "local_route": route})
            with self.assertRaisesRegex(ValueError, "different model contract"):
                restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=wrong)

    def test_statistics_pair_same_materials_and_keep_d1_minus_n1_direction(self):
        common = {"best_epoch": 1, "best_score": .4, "epochs": 35, "contract_schema": "unit", "init_protocol": "unit",
                  "epoch_time_s_median": 1., "epoch_time_s_sum": 35., "peak_vram_mb_median": 1., "peak_vram_mb_max": 1.}
        exports = {}
        for label, offset in {"D1": .2, "N1": 0., "O1": .1, "original_baseline": .15}.items():
            frame = pd.DataFrame(index=["a", "b", "c"])
            for mode in ("blind", "oracle"):
                for task in ("edos", "phdos"):
                    frame[f"{mode}_r2_{task}"] = [-.1+offset, .2+offset, .8+offset]
                    frame[f"{mode}_mae_{task}"] = [3.-offset, 2.-offset, 1.-offset]
            exports[label] = ({**common, "tag": label}, frame)
        report = runner.comparison_report(exports, draws=100)
        for reference, delta in {"N1": .2, "O1": .1, "original_baseline": .05}.items():
            entry = report["contrasts"][f"D1_minus_{reference}"]
            self.assertAlmostEqual(entry["blind_edos_r2_median"]["difference"], delta)
            self.assertAlmostEqual(entry["blind_edos_mae_median"]["difference"], -delta)
        changed = dict(exports)
        changed["N1"] = (exports["N1"][0], exports["N1"][1].iloc[::-1])
        with self.assertRaisesRegex(ValueError, "material-ID"):
            runner.comparison_report(changed, draws=10)

    def test_composition_bootstrap_pairs_correctly_and_weights_groups_equally(self):
        template = pd.DataFrame({"left_id": ["a", "b", "d"], "right_id": ["b", "c", "e"],
                                 "composition": ["X", "X", "Y"]})
        pairs = {}
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                for label, values in {"D1": [1., 1., 9.], "N1": [3., 3., 9.],
                                      "O1": [2., 2., 10.], "original_baseline": [1., 1., 9.]}.items():
                    pairs[label, grouping, task] = template.assign(difference_error=values)
        report = runner.composition_contrasts(pairs)
        self.assertAlmostEqual(report["D1_minus_N1"]["valid_absolute_edos"]["mean_group_error_delta"], -1.)
        pairs["N1", "absolute", "edos"] = pairs["N1", "absolute", "edos"].iloc[::-1]
        with self.assertRaisesRegex(ValueError, "pair identities"):
            runner.composition_contrasts(pairs)

    def test_preflight_rejects_changed_source_references_or_configuration(self):
        cfg = configuration()
        report = {"schema": "attention-dropout-preflight-v1", "passed": True,
                  "controls": {"D1": {"configuration": asdict(cfg)}}, "code_sha256": {"code": "hash"},
                  "references": {"N1": "identity"}, "reference_evidence_sha256": {"evidence": "hash"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"preflight.json"
            path.write_text(json.dumps(report))
            with patch.object(runner, "reference_identity", return_value=(None, None, report["references"])), \
                    patch.object(runner, "evidence_fingerprints", return_value=report["reference_evidence_sha256"]), \
                    patch("tools.checks.check_attention_dropout.code_fingerprints", return_value=report["code_sha256"]):
                self.assertEqual(runner.verify_preflight(path, Path(directory), cfg), report)
                for key, value in (("code_sha256", {"code": "changed"}), ("references", {}),
                                   ("reference_evidence_sha256", {}), ("passed", False)):
                    changed = {**report, key: value}
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        runner.verify_preflight(path, Path(directory), cfg)
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), replace(cfg, tag="different"))

    def test_complete_arm_is_verified_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner.norm, "summarize", return_value=[{"status": "complete"}]), \
                patch.object(runner.norm, "verify_completed") as verify, \
                patch.object(runner, "analyze") as analyze, \
                patch.object(runner.norm.subprocess, "Popen") as process:
            runner.execute(configuration(), {}, Path(directory))
            process.assert_not_called()
            verify.assert_called_once()
            analyze.assert_called_once()
            queue = json.loads((Path(directory)/"queue.json").read_text())
            self.assertEqual(queue["schema"], "attention-dropout-queue-v1")
            self.assertEqual(queue["arms"], {"D1": "complete"})

    def test_failure_stops_before_analysis_without_retry(self):
        failed = MagicMock(pid=17)
        failed.wait.return_value = 1
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner.norm, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.norm.subprocess, "Popen", return_value=failed) as start, \
                patch.object(runner, "analyze") as analyze:
            with self.assertRaisesRegex(RuntimeError, "D1 failed"):
                runner.execute(configuration(), {}, Path(directory))
            start.assert_called_once()
            analyze.assert_not_called()
            self.assertEqual(json.loads((Path(directory)/"queue.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
