"""Outer normalization placement, matched A64 controls and isolated execution."""
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from torch.nn import functional as F

from model.local_global_encoder import LocalResidualLayer
from model.periodic_messages import prepare_shared_message_geometry
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_module import exports as module_exports
from tests.pipeline.test_local_structure_controls import predict
from tools.checks import check_local_norm as checker
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval import local_norm_compare as runner
from tools.eval.local_global_compare import training_command
from tools.eval.local_outer_norm_compare import OUTER_NORM
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, write_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import capture_training_rng, comparison_contract, initialize_comparison_model, validate_local_global_config
from utils.local_outer_norm_experiment import OUTER_NORM_SCHEMA, outer_norm_config, validate_outer_norm_model
from utils.local_structure_experiment import FROZEN
from utils.paths import file_sha256
from utils.periodic_geometry import build_periodic_records
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash


class TestLocalOuterNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = ExperimentConfig(**{**FROZEN, "local_route": "I3", "selection_metric": "balanced_blind_r2", "tag": "_unit_a64"})
        cls.cfg = outer_norm_config(cls.reference, "unit_outer")
        cls.params = {route: {**model_params(route), "dropout": .05} for route in ("I3", "LN1", "LN2")}
        cls.infos = {}
        for route, params in cls.params.items():
            model = Transformer(**params)
            cls.infos[route] = initialize_comparison_model(model, params, 42)

    def create(self, dtype=torch.float32):
        model = Transformer(**self.params["LN2"]).to(dtype)
        initialize_comparison_model(model, self.params["LN2"], 42)
        return model.eval()

    def test_both_outer_formulas_and_every_gradient_including_cached_geometry(self):
        for dtype, tolerance in ((torch.float32, 3e-5), (torch.float64, 1e-11)):
            z, geometry = fixture(dtype)
            records = build_periodic_records(geometry, z.eq(0), 6.)
            prepared = prepare_shared_message_geometry(records, "A")
            for first in (True, False):
                layer = LocalResidualLayer("A", 16, outer_norm_first=first).to(dtype)
                with torch.no_grad():
                    layer.norm.weight.uniform_(.5, 1.5)
                    layer.norm.bias.uniform_(-.3, .3)
                a = (torch.randn((*z.shape, 16), dtype=dtype)*3+2).requires_grad_()
                b = a.detach().clone().requires_grad_()
                norm = lambda values: F.layer_norm(values, (16,), layer.norm.weight, layer.norm.bias, layer.norm.eps)
                branch = layer.message(norm(b) if first else b, records, z.eq(0))
                expected = b+branch if first else norm(b+branch)
                expected = expected.masked_fill(z.eq(0)[..., None], 0)
                actual = layer(a, records, z.eq(0), prepared_geometry=prepared)
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                parameters = tuple(layer.parameters())
                weights = torch.randn_like(actual)
                ag = torch.autograd.grad((actual*weights).sum(), (a, *parameters))
                bg = torch.autograd.grad((expected*weights).sum(), (b, *parameters))
                for left, right in zip(ag, bg):
                    torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
                self.assertTrue(torch.equal(actual[z.eq(0)], torch.zeros_like(actual[z.eq(0)])))
        with self.assertRaises(ValueError):
            LocalResidualLayer("A", 16, outer_norm_first="post")

    def test_zero_local_branch_distinguishes_identity_from_normalized_residual(self):
        z, geometry = fixture()
        records = build_periodic_records(geometry, z.eq(0), 6.)
        h = torch.randn((*z.shape, 16))*3+2
        for first in (True, False):
            layer = LocalResidualLayer("A", 16, outer_norm_first=first)
            with torch.no_grad():
                layer.message.local_output.weight.zero_()
                layer.message.local_output.bias.zero_()
                layer.norm.bias.fill_(.7)
            expected = (h if first else layer.norm(h)).masked_fill(z.eq(0)[..., None], 0)
            torch.testing.assert_close(layer(h, records, z.eq(0)), expected, atol=0, rtol=0)

    def test_all_untrained_tensors_match_without_checkpoint_loading_or_rng_consumption(self):
        models = []
        with patch("torch.load", side_effect=AssertionError("trained weights used for initialization")):
            for route in ("I3", "LN2"):
                model = Transformer(**self.params[route])
                before = rng_digest(rng_snapshot())
                initialize_comparison_model(model, self.params[route], 42)
                self.assertEqual(before, rng_digest(rng_snapshot()))
                models.append(model)
        for key, value in models[0].state_dict().items():
            other = models[1].state_dict()[key]
            torch.testing.assert_close(value, other, atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), other.data_ptr())
        self.assertEqual(self.infos["LN2"]["pre_norm_initialization"], self.infos["I3"])
        self.assertEqual(self.infos["LN2"]["state_sha256"], self.infos["I3"]["state_sha256"])
        for model, first in zip(models, (True, False)):
            self.assertEqual(sum(p.numel() for p in model.parameters()), self.infos["I3"]["parameter_count"])
            self.assertEqual([layer.outer_norm_first for layer in model.encoder.local_layers], [first]*2)
            self.assertEqual([b.norm_first for layer in model.encoder.local_layers for b in layer.message.blocks], [False]*4)

    def test_full_candidate_symmetry_padding_empty_rows_and_finite_gradients(self):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            model = self.create(dtype)
            z, geometry = fixture(dtype)
            # A trained LN bias must not leak into padding states.
            with torch.no_grad():
                for layer in model.encoder.local_layers:
                    layer.norm.bias.fill_(.3)
            expected = predict(model, z, geometry)
            sum(value.square().mean() for value in expected.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            expected.update(reconstruct_h1_spectra(expected, [3, 2], [16., 8.]))
            with torch.no_grad():
                for _, matrix, translation in e3_operations():
                    q, t = torch.as_tensor(matrix, dtype=dtype), torch.as_tensor(translation, dtype=dtype)
                    cell = geometry.cell @ q
                    coords = geometry.frac_coords + torch.bmm(t.expand(2, 1, 3), torch.linalg.inv(cell))
                    actual = predict(model, z, CrystalGeometry(cell, coords))
                    actual.update(reconstruct_h1_spectra(actual, [3, 2], [16., 8.]))
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
                order = torch.tensor([2, 0, 3, 1])
                coords = geometry.frac_coords[:, order].clone()
                coords[z[:, order].eq(0)] = float("nan")
                actual = predict(model, z[:, order], CrystalGeometry(geometry.cell, coords))
                actual.update(reconstruct_h1_spectra(actual, [3, 2], [16., 8.]))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
                empty = torch.zeros_like(z)
                outputs = predict(model, empty, CrystalGeometry(geometry.cell, torch.full_like(geometry.frac_coords, float("nan"))))
                self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

    def test_recipe_assembly_model_validation_and_actual_normalization_record(self):
        args = build_arg_parser().parse_args(training_command(self.cfg)[2:])
        self.assertEqual(asdict(ExperimentConfig.from_args(args)), asdict(self.cfg))
        contract = comparison_contract(self.cfg, self.params["LN2"])
        self.assertEqual(contract["schema"], OUTER_NORM_SCHEMA)
        self.assertEqual(contract["ablation"]["outer_formula"], "LN512(h+L(h,R))")
        self.assertEqual(runner.restore_reference_assembly(captured_config(self.cfg), protocol=OUTER_NORM), captured_config(self.reference))
        for changes in ({"selection_metric": "balanced_oracle_mae"}, {"seed": 43}, {"epochs": 100},
                        {"batch_size": 64}, {"local_layout": "alternating"}, {"use_amp": True}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(self.cfg, **changes))
        model = self.create()
        validate_outer_norm_model(model, self.params["LN2"], require_identity=True)
        layer = model.encoder.local_layers[0]
        for target, flag in ((layer, "outer_norm_first"), (layer.message, "norm_first"), (layer.message.blocks[0], "norm_first")):
            setattr(target, flag, True)
            with self.assertRaises(ValueError):
                validate_outer_norm_model(model, self.params["LN2"])
            setattr(target, flag, False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.yaml"
            path.write_text("{}")
            record = write_local_global_execution(path, model, 0)
            self.assertEqual(record["internal_normalization"], ["post-norm"]*4)
            self.assertEqual(record["outer_normalization"], ["post-norm"]*2)

    def test_same_shape_a64_and_internal_pre_norm_checkpoints_are_rejected_both_directions(self):
        model = self.create()
        optimizer = torch.optim.AdamW(model.parameters())
        z, geometry = fixture()
        sum(value.square().mean() for value in predict(model, z, geometry).values()).backward()
        optimizer.step()
        contract = comparison_contract(self.cfg, self.params["LN2"])
        payload = build_ablation_checkpoint(2, "M1", 42, False, model, optimizer, .5,
            model_contract=contract, training_rng=capture_training_rng())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        before = state_hash(model.state_dict())
        restore_ablation_checkpoint(saved, model, optimizer, False, model_contract=contract)
        self.assertEqual(state_hash(model.state_dict()), before)
        for route in ("I3", "LN1"):
            cfg = replace(self.cfg, local_route=route)
            other = comparison_contract(cfg, self.params[route])
            for loaded, expected in ((saved, other), ({**saved, "model_contract": other}, contract)):
                with self.assertRaisesRegex(ValueError, "different model contract"):
                    restore_ablation_checkpoint(loaded, model, optimizer, False, model_contract=expected)

    def preflight(self):
        identity = {"evidence_dir": "/synthetic", "reference_commit": runner.REFERENCE_COMMIT, "data_order": {}}
        assembled = {"model": {"params": {"sub_model": {"transformer": self.params["LN2"]}}}}
        saved = {"config": runner.restore_reference_assembly(assembled, protocol=OUTER_NORM), "cli": {"data_contract": {}},
                 "runtime": {"local_global_init": self.infos["I3"], "structure_training_rng": [{"state_sha256": "rng"}]}}
        entry = {"configuration": asdict(self.cfg), "assembled_configuration": assembled,
            "initialization": self.infos["LN2"], "contract": comparison_contract(self.cfg, self.params["LN2"]),
            "data_order": {}, "training_start_rng_sha256": "rng", "forward_backward": {"all_parameter_gradients_finite": True},
            "evaluation_finite": True, "symmetry": {"torch.float32": {}, "torch.float64": {}},
            "pre_norm_checkpoint_rejected": True, "checkpoint_roundtrip": {"weights_optimizer_rng_equal": True, "foreign_contracts_rejected": True},
            "costs": [{}, {}, {}]}
        report = {"schema": "local-outer-norm-preflight-v1", "passed": True, "tag_prefix": "unit_outer", "code_sha256": {},
            "runtime": {"gpu": "unit"}, "data_contract": {}, "references": {"A64": identity},
            "reference_compatibility": {"passed": True}, "reference_prediction_replay": {"passed": True},
            "learning_rate_35": learning_rate_trace(35, self.cfg.lr), "controls": {"LN2": entry}}
        return report, saved, identity

    def test_preflight_binds_candidate_family_reference_code_recipe_and_checks(self):
        report, saved, identity = self.preflight()
        with patch.object(checker, "code_fingerprints", return_value={}), \
                patch.object(runner, "reference_identity", return_value=(saved, self.reference, identity)), \
                patch.object(runner.fresh, "runtime_identity", return_value=report["runtime"]):
            runner.validate_preflight(report, self.cfg, protocol=OUTER_NORM)
            with self.assertRaises(ValueError):
                runner.validate_preflight(report, self.cfg)
            for key in ("schema", "code_sha256", "runtime", "data_contract", "reference_compatibility", "reference_prediction_replay"):
                changed = deepcopy(report)
                changed[key] = {"drift": True}
                with self.assertRaises(ValueError):
                    runner.validate_preflight(changed, self.cfg, protocol=OUTER_NORM)
            for key in ("configuration", "contract", "initialization", "data_order", "training_start_rng_sha256",
                        "checkpoint_roundtrip", "pre_norm_checkpoint_rejected", "symmetry", "evaluation_finite", "costs"):
                changed = deepcopy(report)
                changed["controls"]["LN2"][key] = {"drift": True}
                with self.assertRaises(ValueError):
                    runner.validate_preflight(changed, self.cfg, protocol=OUTER_NORM)

    def test_completed_candidate_requires_both_actual_outer_and_internal_placement(self):
        entry = {"assembled_configuration": {"model": "unit"}, "contract": {"schema": "unit"}}
        record = {key: list(value) for key, value in OUTER_NORM.execution_fields}
        saved = {"config": entry["assembled_configuration"], "runtime": {"local_global_execution": [record]}}
        with patch.object(runner.queue, "verify_completed"), patch.object(runner, "load_run", return_value=(saved, {}, entry["contract"])):
            runner.verify_completed(self.cfg, {"controls": {"LN2": entry}}, protocol=OUTER_NORM)
            for key, _ in OUTER_NORM.execution_fields:
                original = record.pop(key)
                with self.assertRaises(ValueError):
                    runner.verify_completed(self.cfg, {"controls": {"LN2": entry}}, protocol=OUTER_NORM)
                record[key] = ["pre-norm"]*len(original)
                with self.assertRaises(ValueError):
                    runner.verify_completed(self.cfg, {"controls": {"LN2": entry}}, protocol=OUTER_NORM)
                record[key] = original

    def test_outer_candidate_report_uses_joint_blind_and_each_task_protection(self):
        data = module_exports()
        exports = {"LN2": data["I3"], "A64": data["I2"]}
        report = runner.comparison_report(exports, draws=20, protocol=OUTER_NORM)
        self.assertTrue(report["decisions"]["post_norm_conditions_met"])
        self.assertFalse(report["decisions"]["automatic_baseline_replacement"])
        bad = deepcopy(exports)
        bad["LN2"][1].loc[bad["LN2"][1].index[:2], "blind_r2_phdos"] = -.1
        self.assertFalse(runner.comparison_report(bad, draws=20, protocol=OUTER_NORM)["decisions"]["post_norm_conditions_met"])

    def test_analysis_only_infers_outer_candidate_and_reuses_reference_bytes(self):
        table = pd.DataFrame({"composition": ["x"], "difference_error": [.1]})
        tables = {(g, t): table for g in ("absolute", "reduced") for t in ("edos", "phdos")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = {"exports": {}}
            for suffix in runner.EXPORT_SUFFIXES:
                path = root/f"A64_{suffix}"
                path.write_bytes(b"immutable export")
                reference["exports"][path.name] = {"source": str(path), "sha256": file_sha256(path)}
            preflight = {"references": {"A64": reference}, "code_sha256": {}, "data_contract": {}, "controls": {"LN2": {"initialization": {}}}}
            def export(run, samples, summary, *, spectra_out):
                for path in (samples, summary, spectra_out):
                    path.write_bytes(b"candidate")
            def export_train(run, out, label):
                for suffix in ("samples.csv", "spectra.npz", "summary.json"):
                    (out/f"{label}_train_{suffix}").write_bytes(b"candidate train")
                return tables
            builder = MagicMock()
            builder.get_dataset.side_effect = [MagicMock(__len__=lambda _:18704), MagicMock(__len__=lambda _:2313)]
            with patch.object(runner, "validate_preflight"), patch.object(runner, "verify_completed"), \
                    patch.object(runner.fresh, "artifacts", return_value={}), patch.object(runner, "load_run", return_value=({"config": {}}, {}, {})), \
                    patch.object(runner, "ConfigBuilder", return_value=builder), patch.object(runner, "export", side_effect=export) as valid_export, \
                    patch.object(runner.bplus, "export_train", side_effect=export_train) as train_export, \
                    patch.object(runner.bplus, "assert_paired_targets"), patch.object(runner.bplus, "spectrum_pairs", return_value=tables), \
                    patch.object(runner, "load", return_value=({}, pd.DataFrame())), patch.object(runner, "comparison_report", return_value={}), \
                    patch.object(runner, "structural_report", return_value={}):
                result = runner.analyze([self.cfg], preflight, root/"analysis", protocol=OUTER_NORM)
            self.assertEqual(result["schema"], "local-outer-norm-analysis-v1")
            self.assertEqual(valid_export.call_count, 1)
            self.assertEqual(train_export.call_args.args[2], "LN2")
            self.assertTrue((root/"analysis/LN2_valid_spectra.npz").exists())
            for name, source in reference["exports"].items():
                self.assertEqual(file_sha256(root/"analysis"/name), source["sha256"])
                self.assertEqual(file_sha256(Path(source["source"])), source["sha256"])

    def test_partial_queue_stops_without_training_or_analysis(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "validate_preflight"), \
                patch.object(runner, "summarize", return_value=[{"status": "partial"}]), patch.object(runner.queue, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "recovery decision"):
                runner.execute(self.cfg, {}, Path(directory), protocol=OUTER_NORM)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
