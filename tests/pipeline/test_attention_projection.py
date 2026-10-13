"""Identity controls must separate QKVO initialization and trainability."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from model.local_global_encoder import LegacyContentGlobalLayer
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_structure_controls import predict
from tools.checks.check_long_budget import captured_config
from tools.eval import attention_projection_compare as runner
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, configure_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.attention_projection_experiment import (
    PROJECTION_SCHEMA, attention_projection_configs, is_projection_parameter,
    set_identity_projections, validate_projection_model,
)
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, restore_training_rng,
    validate_local_global_config,
)
from utils.local_structure_experiment import FROZEN
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash


def configurations():
    return attention_projection_configs(ExperimentConfig(**{**FROZEN, "local_route": "B+"}), "unit_projection")


class TestAttentionProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, route="I1", dtype=torch.float32):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            params = {**model_params(route), "dropout": .05}
            model = Transformer(**params)
            rng = rng_digest(rng_snapshot())
            info = initialize_comparison_model(model, params, 42)
            self.assertEqual(rng, rng_digest(rng_snapshot()))
        finally:
            torch.set_default_dtype(previous)
        return model.eval(), info, params

    def test_all_non_projection_tensors_copy_d1_and_i0_i1_share_complete_state(self):
        d1, reference, _ = self.create("D1")
        learned, left, _ = self.create("I1")
        fixed, right, _ = self.create("I0")
        self.assertEqual(left["d1_initialization"], reference)
        self.assertEqual(right["d1_initialization"], reference)
        self.assertEqual(left["state_sha256"], right["state_sha256"])
        self.assertNotEqual(left["state_sha256"], reference["state_sha256"])
        for name, value in learned.state_dict().items():
            torch.testing.assert_close(value, fixed.state_dict()[name], atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), fixed.state_dict()[name].data_ptr())
            if not is_projection_parameter(name):
                torch.testing.assert_close(value, d1.state_dict()[name], atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), d1.state_dict()[name].data_ptr())
        self.assertTrue(all(is_projection_parameter(k) for k in left["changed_initial_tensors"]))
        self.assertEqual(len(left["changed_initial_tensors"]), 12)
        self.assertEqual(left["n_retained_tensors"], left["n_tensors"] - 24)
        self.assertEqual(left["frozen_parameter_names"], [])
        self.assertEqual(len(right["frozen_parameter_names"]), 24)
        self.assertEqual(left["parameter_count"], right["parameter_count"])
        self.assertEqual(right["trainable_parameter_count"], left["parameter_count"] - left["projection_parameter_count"])

    def test_identity_projections_have_zero_bias_and_only_i0_is_frozen(self):
        for route in ("I1", "I0"):
            model, _, params = self.create(route)
            validate_projection_model(model, params, require_identity=True)
            for name, p in model.named_parameters():
                self.assertEqual(p.requires_grad, route == "I1" or not is_projection_parameter(name), name)
            reference_modules = dict(self.create("D1")[0].named_modules())
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    self.assertEqual(module.p, reference_modules[name].p)

    def test_initial_forward_common_gradients_and_rng_match_at_both_precisions(self):
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-11)):
            for training in (False, True):
                learned, _, _ = self.create("I1", dtype)
                fixed, _, _ = self.create("I0", dtype)
                z, geometry = fixture(dtype)
                rng = torch.get_rng_state().clone()
                results = []
                for model in (learned, fixed):
                    model.train(training)
                    torch.set_rng_state(rng)
                    output = predict(model, z, geometry)
                    sum(v.square().mean() for v in output.values()).backward()
                    results.append((output, torch.get_rng_state().clone()))
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            self.assertIsNotNone(p.grad, name)
                            self.assertTrue(torch.isfinite(p.grad).all(), name)
                        else:
                            self.assertIsNone(p.grad, name)
                for key in results[0][0]:
                    torch.testing.assert_close(results[0][0][key], results[1][0][key], atol=0, rtol=0)
                common = dict(fixed.named_parameters())
                for name, p in learned.named_parameters():
                    if not is_projection_parameter(name):
                        torch.testing.assert_close(p.grad, common[name].grad, atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(results[0][1], results[1][1], atol=0, rtol=0)

    def test_fixed_projections_survive_adamw_while_learned_projections_move(self):
        for route in ("I1", "I0"):
            model, info, params = self.create(route)
            model.train()
            z, geometry = fixture()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.01)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                sum(v.square().mean() for v in predict(model, z, geometry).values()).backward()
                optimizer.step()
            projections = {name: value for name, value in model.state_dict().items() if is_projection_parameter(name)}
            if route == "I0":
                validate_projection_model(model, params, require_identity=True)
                self.assertEqual(state_hash(projections), info["projection_state_sha256"])
                self.assertTrue(all(p.grad is None and p not in optimizer.state
                                    for name, p in model.named_parameters() if is_projection_parameter(name)))
            else:
                self.assertNotEqual(state_hash(projections), info["projection_state_sha256"])
            self.assertTrue(any(not torch.equal(p, before[name]) for name, p in model.named_parameters()
                                if not is_projection_parameter(name)))

    def test_identity_attention_matches_old_content_formula_with_matched_ffn(self):
        model, _, params = self.create("I0", torch.float64)
        new = model.encoder.global_layers[0]
        old = LegacyContentGlobalLayer(params["d_model"], params["nhead"], params["dim_feedforward"],
                                       params["dropout"], params.get("activation", "gelu")).double()
        names = {"linear1": "ffn.0", "linear2": "ffn.3", "norm1": "norm_attn", "norm2": "norm_ffn"}
        old.layer.load_state_dict({name + '.' + field: new.state_dict()[source + '.' + field]
                                  for name, source in names.items() for field in ("weight", "bias")})
        h = torch.randn(2, 4, params["d_model"], dtype=torch.float64)
        readout = torch.randn_like(h)
        mask = torch.tensor([[False, False, True, True], [True, True, True, True]])
        for training in (False, True):
            rng = torch.get_rng_state().clone()
            results = []
            for layer in (new, old):
                layer.train(training)
                torch.set_rng_state(rng)
                value = h.clone().requires_grad_()
                output = layer(value, mask)
                (output * readout).sum().backward()
                results.append((output, value.grad, torch.get_rng_state().clone()))
            for i in (0, 1):
                torch.testing.assert_close(results[0][i], results[1][i], atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(results[0][2], results[1][2], atol=0, rtol=0)

    def test_manual_d1_identity_intervention_matches_i1(self):
        learned, _, _ = self.create("I1")
        d1, _, _ = self.create("D1")
        set_identity_projections(d1)
        self.assertEqual(state_hash(learned.state_dict()), state_hash(d1.state_dict()))
        z, geometry = fixture()
        with torch.no_grad():
            left, right = predict(learned, z, geometry), predict(d1, z, geometry)
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)

    def test_e3_permutation_empty_rows_and_geometry_cache(self):
        for route in ("I1", "I0"):
            for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
                model, _, _ = self.create(route, dtype)
                z, geometry = fixture(dtype)
                with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old scoring")), \
                        patch("model.periodic_messages.o3.spherical_harmonics", side_effect=AssertionError("angles")), \
                        torch.no_grad():
                    expected = predict(model, z, geometry)
                    for _, matrix, translation in e3_operations():
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
                    empty = z.clone()
                    empty[1] = 0
                    self.assertTrue(all(torch.isfinite(v).all() for v in predict(model, empty, geometry).values()))
                configure_local_global_execution(model)
                model.enable_local_geometry_cache(max_bytes=2**20)
                self.assertEqual(model._local_geometry_cache.route, "C1")

    def test_invalid_trainability_normalization_and_dropout_are_rejected(self):
        for route in ("I1", "I0"):
            model, _, params = self.create(route)
            changed = deepcopy(model)
            changed.encoder.global_layers[0].attn.in_proj_weight.requires_grad_(route == "I0")
            with self.assertRaisesRegex(ValueError, "freezes only"):
                initialize_comparison_model(changed, params, 42)
            for attribute, value in (("norm_first", True), ("attn.dropout", .05), ("ffn.2.p", 0.)):
                changed = deepcopy(model)
                pieces = attribute.split('.')
                module = changed.encoder.global_layers[0]
                for name in pieces[:-1]:
                    module = module[int(name)] if name.isdigit() else getattr(module, name)
                setattr(module, pieces[-1], value)
                with self.assertRaisesRegex(ValueError, "preserve D1"):
                    initialize_comparison_model(changed, params, 42)
            with self.assertRaises(ValueError):
                initialize_comparison_model(model, {**params, "normalize_before": True}, 42)

    def test_frozen_recipe_cli_roundtrip_and_isolated_contracts(self):
        for cfg in configurations():
            validate_local_global_config(cfg)
            parsed = ExperimentConfig.from_args(build_arg_parser().parse_args(training_command(cfg)[2:]))
            self.assertEqual(asdict(cfg), asdict(parsed))
            params = captured_config(cfg)["model"]["params"]["sub_model"]["transformer"]
            contract = comparison_contract(cfg, params)
            self.assertEqual(contract["schema"], PROJECTION_SCHEMA)
            self.assertEqual(contract["ablation"]["projection_trainable"], cfg.local_route == "I1")
            for change in ({"seed": 43}, {"epochs": 100}, {"local_depth": False}, {"global_depth": 6.},
                           {"local_layout": "alternating"}, {"use_amp": True}, {"tag": ""},
                           {"dropout": 0.}, {"norm": "minmax"}, {"init_ckpt": "D1.pth"}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_local_global_config(replace(cfg, **change))

    def test_checkpoint_restore_keeps_fixed_flags_optimizer_rng_and_rejects_other_routes(self):
        cfg = configurations()[1]
        model, _, params = self.create("I0")
        model.train()
        z, geometry = fixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        sum(v.square().mean() for v in predict(model, z, geometry).values()).backward()
        optimizer.step()
        contract = comparison_contract(cfg, params)
        payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, .5,
                                            model_contract=contract, training_rng=capture_training_rng())
        rng = rng_digest(rng_snapshot())
        fresh, _, _ = self.create("I0")
        fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=5e-5)
        restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=contract)
        restore_training_rng(payload["training_rng"])
        self.assertEqual(rng, rng_digest(rng_snapshot()))
        self.assertEqual(state_hash(fresh.state_dict()), state_hash(model.state_dict()))
        validate_projection_model(fresh, params, require_identity=True)
        for key, slots in optimizer.state_dict()["state"].items():
            for name, value in slots.items():
                torch.testing.assert_close(fresh_optimizer.state_dict()["state"][key][name], value, atol=0, rtol=0)
        for route in ("I1", "D1", "N1"):
            wrong = comparison_contract(replace(cfg, local_route=route), {**params, "local_route": route})
            with self.assertRaisesRegex(ValueError, "different model contract"):
                restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=wrong)

    def test_paired_metrics_report_both_direct_contrasts_and_reject_different_partitions(self):
        common = {"best_epoch": 1, "best_score": .4, "epochs": 35, "contract_schema": "unit", "init_protocol": "unit",
                  "epoch_time_s_median": 1., "epoch_time_s_sum": 35., "peak_vram_mb_median": 1., "peak_vram_mb_max": 1.}
        exports = {}
        for label, offset in {"I0": .4, "I1": .2, "D1": 0., "O1": .1, "original_baseline": .15}.items():
            frame = pd.DataFrame(index=["a", "b", "c"])
            for mode in ("blind", "oracle"):
                for task in ("edos", "phdos"):
                    frame[f"{mode}_r2_{task}"] = [-.1 + offset, .2 + offset, .8 + offset]
                    frame[f"{mode}_mae_{task}"] = [3. - offset, 2. - offset, 1. - offset]
            exports[label] = ({**common, "tag": label}, frame)
        report = runner.comparison_report(exports, draws=100)
        self.assertEqual(set(report["direct_contrasts"]), {"I1_minus_D1", "I0_minus_I1"})
        self.assertEqual(len(report["contrasts"]), 7)
        for contrast in report["direct_contrasts"]:
            entry = report["contrasts"][contrast]
            self.assertAlmostEqual(entry["blind_edos_r2_median"]["difference"], .2)
            self.assertAlmostEqual(entry["blind_edos_mae_median"]["difference"], -.2)
        changed = dict(exports)
        changed["D1"] = (exports["D1"][0], exports["D1"][1].iloc[::-1])
        with self.assertRaisesRegex(ValueError, "material-ID"):
            runner.comparison_report(changed, draws=10)

    def test_group_bootstrap_reports_both_direct_contrasts(self):
        template = pd.DataFrame({"left_id": ["a", "b", "d"], "right_id": ["b", "c", "e"],
                                 "composition": ["X", "X", "Y"]})
        pairs = {}
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                for label, values in {"I0": [1., 1., 9.], "I1": [3., 3., 9.], "D1": [5., 5., 9.],
                                      "O1": [2., 2., 10.], "original_baseline": [1., 1., 9.]}.items():
                    pairs[label, grouping, task] = template.assign(difference_error=values)
        report = runner.composition_contrasts(pairs)
        for contrast in ("I1_minus_D1", "I0_minus_I1"):
            self.assertAlmostEqual(report[contrast]["valid_absolute_edos"]["mean_group_error_delta"], -1.)
        pairs["I1", "absolute", "edos"] = pairs["I1", "absolute", "edos"].iloc[::-1]
        with self.assertRaisesRegex(ValueError, "pair identities"):
            runner.composition_contrasts(pairs)

    def test_preflight_rejects_changed_source_references_config_or_initial_pairing(self):
        controls = configurations()
        entries = {cfg.local_route: {"configuration": asdict(cfg),
                   "initialization": self.create(cfg.local_route)[1], "training_start_rng_sha256": "rng"}
                   for cfg in controls}
        report = {"schema": "attention-projection-preflight-v1", "passed": True, "controls": entries,
                  "code_sha256": {"code": "hash"}, "references": {"D1": "identity"},
                  "reference_evidence_sha256": {"evidence": "hash"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(report))
            with patch.object(runner, "reference_identity", return_value=(None, None, report["references"])), \
                    patch.object(runner, "evidence_fingerprints", return_value=report["reference_evidence_sha256"]), \
                    patch("tools.checks.check_attention_projection.code_fingerprints", return_value=report["code_sha256"]):
                self.assertEqual(runner.verify_preflight(path, Path(directory), controls), report)
                for key, value in (("code_sha256", {}), ("references", {}), ("reference_evidence_sha256", {}), ("passed", False)):
                    path.write_text(json.dumps({**report, key: value}))
                    with self.assertRaises(ValueError):
                        runner.verify_preflight(path, Path(directory), controls)
                changed = deepcopy(report)
                changed["controls"]["I0"]["initialization"]["state_sha256"] = "different"
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "pair complete"):
                    runner.verify_preflight(path, Path(directory), controls)
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), [replace(controls[0], tag="changed"), controls[1]])

    def test_complete_arms_are_verified_and_not_retrained(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner.legacy, "summarize", return_value=[{"status": "complete"}]), \
                patch.object(runner.legacy, "verify_completed") as verify, \
                patch.object(runner, "analyze") as analyze, \
                patch.object(runner.legacy.subprocess, "Popen") as process:
            runner.execute(configurations(), {}, Path(directory))
            process.assert_not_called()
            self.assertEqual(verify.call_count, 2)
            analyze.assert_called_once()
            queue = json.loads((Path(directory) / "queue.json").read_text())
            self.assertEqual(queue["schema"], "attention-projection-queue-v1")
            self.assertEqual(queue["arms"], {"I1": "complete", "I0": "complete"})

    def test_failure_stops_before_second_arm_and_analysis_without_retry(self):
        failed = MagicMock(pid=17)
        failed.wait.return_value = 1
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner.legacy, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.legacy.subprocess, "Popen", return_value=failed) as start, \
                patch.object(runner, "analyze") as analyze:
            with self.assertRaisesRegex(RuntimeError, "I1 failed"):
                runner.execute(configurations(), {}, Path(directory))
            start.assert_called_once()
            analyze.assert_not_called()
            queue = json.loads((Path(directory) / "queue.json").read_text())
            self.assertEqual(queue["status"], "failed")
            self.assertEqual(queue["arms"]["I0"], "pending")

    def test_analysis_uses_verified_second_candidate_and_existing_export_pipeline(self):
        controls = configurations()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner.legacy, "verify_completed") as verify, \
                patch.object(runner.norm, "analyze", return_value={"passed": True}) as analyze:
            self.assertEqual(runner.analyze(controls, {}, Path(directory)), {"passed": True})
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(analyze.call_args.args[0].local_route, "I0")
            self.assertEqual(analyze.call_args.kwargs["additional_runs"], {"I1": runner.legacy.run_directory(controls[0])})
            self.assertIs(analyze.call_args.kwargs["comparison_fn"], runner.comparison_report)


if __name__ == "__main__":
    unittest.main()
