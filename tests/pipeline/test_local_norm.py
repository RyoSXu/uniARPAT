"""Internal normalization must change the complete branch and isolate same-shape checkpoints."""
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

from model.periodic_messages import InvariantMessageBlock, PeriodicLocalMessage, _prepare_message_geometry, _unordered_pairs
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_module import exports as module_exports
from tests.pipeline.test_local_structure_controls import predict
from tools.checks import check_local_norm as checker
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval import local_norm_compare as runner
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, write_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import capture_training_rng, comparison_contract, initialize_comparison_model, validate_local_global_config
from utils.local_norm_experiment import LOCAL_NORM_SCHEMA, local_norm_config, validate_local_norm_model
from utils.local_structure_experiment import FROZEN
from utils.paths import file_sha256
from utils.periodic_geometry import build_periodic_records
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash


def direct_block(block, scalar, g):
    """Literal unfactorized messages, independent of the optimized angular affine implementation."""
    values = block.norm(scalar) if block.norm_first else scalar
    flat = values.flatten(0, 1)
    radial, angular = torch.zeros_like(flat), torch.zeros_like(flat)
    for edge in range(len(g.receiver)):
        i, j = g.receiver[edge], g.sender[edge]
        msg = block.edge_mlp(torch.cat((flat[i], flat[j], g.radial[edge]))) * g.cutoff[edge]
        radial = radial.index_add(0, i[None], msg[None])
    left, right = _unordered_pairs(g)
    for e, f in zip(left, right):
        i, j, k = g.receiver[e], g.sender[e], g.sender[f]
        q = (g.unit[e] * g.unit[f]).sum().clamp(-1, 1)
        angular_basis = torch.stack((q, q*q, q**3, q**4))
        forward = block.angle_mlp(torch.cat((flat[i], flat[j], flat[k], g.radial[e], g.radial[f], angular_basis)))
        backward = block.angle_mlp(torch.cat((flat[i], flat[k], flat[j], g.radial[f], g.radial[e], angular_basis)))
        msg = .5 * (forward + backward) * g.cutoff[e] * g.cutoff[f]
        angular = angular.index_add(0, i[None], msg[None])
    radial = radial.view_as(values) / g.degree.clamp(min=1).to(values.dtype).sqrt()[..., None]
    angular = angular.view_as(values) / g.pair_count.clamp(min=1).to(values.dtype).sqrt()[..., None]
    counts = torch.stack((g.degree.to(values.dtype).log1p(), g.effective_count.to(values.dtype)), -1)
    update = block.update(torch.cat((values, radial, angular, counts), -1))
    updated = scalar + update if block.residual_add else update
    return updated if block.norm_first else block.norm(updated)


class TestLocalNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference_cfg = ExperimentConfig(**{**FROZEN, "local_route": "I3", "selection_metric": "balanced_blind_r2", "tag": "_unit_post"})
        cls.cfg = local_norm_config(cls.reference_cfg, "unit_norm")
        cls.params = {route: {**model_params(route), "dropout": .05} for route in ("I3", "LN1")}
        cls.infos = {}
        for route in cls.params:
            model = Transformer(**cls.params[route])
            cls.infos[route] = initialize_comparison_model(model, cls.params[route], 42)

    def create(self, dtype=torch.float32):
        model = Transformer(**self.params["LN1"]).to(dtype)
        initialize_comparison_model(model, self.params["LN1"], 42)
        return model.eval()

    def preflight(self):
        identity = {"evidence_dir": "/synthetic", "reference_commit": runner.REFERENCE_COMMIT,
                    "data_order": {"train_samples": 18704, "valid_samples": 2313}}
        assembled = {"model": {"params": {"sub_model": {"transformer": self.params["LN1"]}}}}
        saved = {"config": runner.restore_post_assembly(assembled), "cli": {"data_contract": {}},
                 "runtime": {"local_global_init": self.infos["I3"], "structure_training_rng": [{"state_sha256": "rng"}]}}
        entry = {"configuration": asdict(self.cfg), "assembled_configuration": assembled,
            "initialization": self.infos["LN1"], "contract": comparison_contract(self.cfg, self.params["LN1"]),
            "data_order": identity["data_order"], "training_start_rng_sha256": "rng",
            "forward_backward": {"all_parameter_gradients_finite": True}, "evaluation_finite": True,
            "symmetry": {"torch.float32": {}, "torch.float64": {}}, "post_norm_checkpoint_rejected": True,
            "checkpoint_roundtrip": {"weights_optimizer_rng_equal": True, "foreign_contracts_rejected": True},
            "costs": [{}, {}, {}]}
        report = {"schema": runner.PREFLIGHT_SCHEMA, "passed": True, "tag_prefix": "unit_norm", "code_sha256": {},
            "runtime": {"gpu": "unit"}, "data_contract": {}, "references": {"A64": identity},
            "post_norm_compatibility": {"passed": True}, "reference_prediction_replay": {"passed": True},
            "learning_rate_35": learning_rate_trace(35, self.cfg.lr), "controls": {"LN1": entry}}
        return report, saved, identity

    def test_pre_and_post_outputs_and_all_gradients_match_direct_complete_messages(self):
        for dtype, tolerance in ((torch.float32, 4e-5), (torch.float64, 2e-11)):
            z, geometry = fixture(dtype)
            records = build_periodic_records(geometry, z.eq(0), 6.)
            g = _prepare_message_geometry(records, angle_pairs=True)
            for norm_first in (False, True):
                block = InvariantMessageBlock(recompute_angles=False, norm_first=norm_first).to(dtype)
                with torch.no_grad():
                    block.norm.weight.uniform_(.5, 1.5)
                    block.norm.bias.uniform_(-.2, .2)
                a = torch.randn((*z.shape, 64), dtype=dtype, requires_grad=True)
                b = a.detach().clone().requires_grad_()
                actual, expected = block(a, g), direct_block(block, b, g)
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                weights = torch.randn_like(actual)
                parameters = tuple(block.parameters())
                ag = torch.autograd.grad((actual * weights).sum(), (a, *parameters))
                bg = torch.autograd.grad((expected * weights).sum(), (b, *parameters))
                for left, right in zip(ag, bg):
                    torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)

    def test_pre_norm_keeps_the_original_identity_state_when_update_is_zero(self):
        z, geometry = fixture()
        g = _prepare_message_geometry(build_periodic_records(geometry, z.eq(0), 6.), angle_pairs=True)
        block = InvariantMessageBlock(norm_first=True)
        with torch.no_grad():
            block.update[-1].weight.zero_()
            block.update[-1].bias.zero_()
        scalar = torch.randn((*z.shape, 64)) * 3 + 2
        torch.testing.assert_close(block(scalar, g), scalar, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            PeriodicLocalMessage("B+", norm_first=True)
        with self.assertRaises(ValueError):
            InvariantMessageBlock(norm_first="pre")

    def test_all_initial_tensors_match_post_norm_without_checkpoint_loading_or_rng_consumption(self):
        models = []
        with patch("torch.load", side_effect=AssertionError("initialization loaded trained weights")):
            for route in ("I3", "LN1"):
                model = Transformer(**self.params[route])
                before = rng_digest(rng_snapshot())
                initialize_comparison_model(model, self.params[route], 42)
                self.assertEqual(before, rng_digest(rng_snapshot()))
                models.append(model)
        for key, value in models[0].state_dict().items():
            other = models[1].state_dict()[key]
            torch.testing.assert_close(value, other, atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), other.data_ptr())
        self.assertEqual(self.infos["LN1"]["post_norm_initialization"], self.infos["I3"])
        self.assertEqual(self.infos["LN1"]["state_sha256"], self.infos["I3"]["state_sha256"])
        for model, first in zip(models, (False, True)):
            self.assertEqual(sum(p.numel() for p in model.parameters()), self.infos["I3"]["parameter_count"])
            self.assertEqual([b.norm_first for layer in model.encoder.local_layers for b in layer.message.blocks], [first]*4)

    def test_full_pre_norm_outputs_keep_e3_permutation_padding_and_finite_gradients(self):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            model = self.create(dtype)
            z, geometry = fixture(dtype)
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

    def test_cli_recipe_contract_and_actual_execution_record(self):
        args = build_arg_parser().parse_args(training_command(self.cfg)[2:])
        self.assertEqual(asdict(ExperimentConfig.from_args(args)), asdict(self.cfg))
        contract = comparison_contract(self.cfg, self.params["LN1"])
        self.assertEqual(contract["schema"], LOCAL_NORM_SCHEMA)
        self.assertEqual(contract["ablation"]["internal_normalization"], "pre-norm")
        post = captured_config(self.reference_cfg)
        pre = captured_config(self.cfg)
        self.assertEqual(runner.restore_post_assembly(pre), post)
        for changes in ({"selection_metric": "balanced_oracle_mae"}, {"seed": 43}, {"epochs": 100},
                        {"batch_size": 64}, {"local_layout": "alternating"}, {"use_amp": True}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(self.cfg, **changes))
        model = self.create()
        validate_local_norm_model(model, self.params["LN1"], require_identity=True)
        model.encoder.local_layers[0].message.blocks[0].norm_first = False
        with self.assertRaises(ValueError):
            validate_local_norm_model(model, self.params["LN1"])
        model.encoder.local_layers[0].message.blocks[0].norm_first = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.yaml"
            path.write_text("{}")
            record = write_local_global_execution(path, model, 0)
            self.assertEqual(record["internal_normalization"], ["pre-norm"]*4)
            self.assertEqual(record["recompute_angles"], [False]*4)
            self.assertEqual(record["pair_chunk"], [65536]*4)

    def test_same_shape_pre_post_checkpoints_are_rejected_in_both_directions(self):
        model = self.create()
        optimizer = torch.optim.AdamW(model.parameters())
        z, geometry = fixture()
        sum(value.square().mean() for value in predict(model, z, geometry).values()).backward()
        optimizer.step()
        before = state_hash(model.state_dict())
        contract = comparison_contract(self.cfg, self.params["LN1"])
        old_contract = comparison_contract(self.reference_cfg, self.params["I3"])
        payload = build_ablation_checkpoint(2, "M1", 42, False, model, optimizer, .5,
            model_contract=contract, training_rng=capture_training_rng())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        with torch.no_grad():
            next(model.parameters()).add_(2)
        restore_ablation_checkpoint(saved, model, optimizer, False, model_contract=contract)
        self.assertEqual(state_hash(model.state_dict()), before)
        for loaded, expected in ((saved, old_contract), ({**saved, "model_contract": old_contract}, contract)):
            with self.assertRaisesRegex(ValueError, "different model contract"):
                restore_ablation_checkpoint(loaded, model, optimizer, False, model_contract=expected)

    def test_preflight_rejects_source_reference_recipe_and_check_drift(self):
        report, saved, identity = self.preflight()
        with patch.object(checker, "code_fingerprints", return_value={}), \
                patch.object(runner, "reference_identity", return_value=(saved, self.reference_cfg, identity)), \
                patch.object(runner.fresh, "runtime_identity", return_value=report["runtime"]):
            self.assertEqual(runner.validate_preflight(report, self.cfg), report)
            for key in ("schema", "code_sha256", "runtime", "data_contract", "post_norm_compatibility", "reference_prediction_replay"):
                changed = deepcopy(report)
                changed[key] = {"drift": True} if key in ("code_sha256", "data_contract") else {}
                with self.assertRaises(ValueError):
                    runner.validate_preflight(changed, self.cfg)
            for key in ("configuration", "contract", "initialization", "training_start_rng_sha256", "data_order",
                        "checkpoint_roundtrip", "post_norm_checkpoint_rejected", "symmetry", "evaluation_finite", "costs"):
                changed = deepcopy(report)
                changed["controls"]["LN1"][key] = {}
                with self.assertRaises(ValueError):
                    runner.validate_preflight(changed, self.cfg)
            changed = deepcopy(report)
            changed["references"]["A64"]["reference_commit"] = "0"*40
            with self.assertRaises(ValueError):
                runner.validate_preflight(changed, self.cfg)

    def test_partial_run_stops_before_any_training_or_analysis(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "validate_preflight"), \
                patch.object(runner, "summarize", return_value=[{"status": "partial"}]), \
                patch.object(runner.queue, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "recovery decision"):
                runner.execute(self.cfg, {}, Path(directory))
            execute.assert_not_called()

    def test_completed_run_requires_the_exact_preflight_assembly_and_runtime_norm(self):
        entry = {"assembled_configuration": {"model": "unit"}, "contract": {"schema": "unit"}}
        saved = {"config": deepcopy(entry["assembled_configuration"]),
                 "runtime": {"local_global_execution": [{"internal_normalization": ["pre-norm"]*4}]}}
        with patch.object(runner.queue, "verify_completed"), \
                patch.object(runner, "load_run", return_value=(saved, {}, entry["contract"])):
            runner.verify_completed(self.cfg, {"controls": {"LN1": entry}})
            saved["config"] = {"model": "different heads"}
            with self.assertRaises(ValueError):
                runner.verify_completed(self.cfg, {"controls": {"LN1": entry}})
            saved["config"] = entry["assembled_configuration"]
            saved["runtime"]["local_global_execution"][0]["internal_normalization"] = ["post-norm"]*4
            with self.assertRaises(ValueError):
                runner.verify_completed(self.cfg, {"controls": {"LN1": entry}})

    def test_training_failure_stops_after_one_attempt_without_analysis(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "validate_preflight"), \
                patch.object(runner, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.queue, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.queue, "run_directory", return_value=Path(directory)/"unused"), \
                patch.object(runner, "analyze") as analyze, \
                patch.object(runner.queue.subprocess, "Popen", return_value=MagicMock(pid=123, wait=lambda:1)) as popen:
            with self.assertRaisesRegex(RuntimeError, "LN1 failed"):
                runner.execute(self.cfg, {}, Path(directory)/"run")
            self.assertEqual(popen.call_count, 1)
            analyze.assert_not_called()
            self.assertEqual(json.loads((Path(directory)/"run/queue.json").read_text())["status"], "failed")

    def test_report_uses_equal_task_blind_selection_and_protects_each_task(self):
        data = module_exports()
        exports = {"LN1": data["I3"], "A64": data["I2"]}
        report = runner.comparison_report(exports, draws=20)
        self.assertTrue(report["decisions"]["pre_norm_conditions_met"])
        self.assertFalse(report["decisions"]["automatic_followup_training"])
        bad = deepcopy(exports)
        bad["LN1"][1].loc[bad["LN1"][1].index[:2], "blind_r2_phdos"] = -.1
        self.assertFalse(runner.comparison_report(bad, draws=20)["decisions"]["pre_norm_conditions_met"])

    def test_analysis_reuses_all_reference_exports_and_infers_only_ln1(self):
        table = pd.DataFrame({"composition": ["x"], "difference_error": [.1]})
        tables = {(group, task): table for group in ("absolute", "reduced") for task in ("edos", "phdos")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = {"exports": {}}
            for suffix in runner.EXPORT_SUFFIXES:
                path = root/f"A64_{suffix}"
                path.write_bytes(b"immutable export")
                reference["exports"][path.name] = {"source": str(path), "sha256": file_sha256(path)}
            preflight = {"references": {"A64": reference}, "code_sha256": {}, "data_contract": {},
                         "controls": {"LN1": {"initialization": {}}}}
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
                    patch.object(runner.fresh, "artifacts", return_value={}), \
                    patch.object(runner, "load_run", return_value=({"config": {}}, {}, {})), \
                    patch.object(runner, "ConfigBuilder", return_value=builder), \
                    patch.object(runner, "export", side_effect=export) as valid_export, \
                    patch.object(runner.bplus, "export_train", side_effect=export_train) as train_export, \
                    patch.object(runner.bplus, "assert_paired_targets"), \
                    patch.object(runner.bplus, "spectrum_pairs", return_value=tables), \
                    patch.object(runner, "load", return_value=({}, pd.DataFrame())), \
                    patch.object(runner, "comparison_report", return_value={}), patch.object(runner, "structural_report", return_value={}):
                result = runner.analyze([self.cfg], preflight, root/"analysis")
            self.assertTrue(result["analysis_complete"])
            self.assertEqual(valid_export.call_count, 1)
            self.assertEqual(train_export.call_count, 1)
            self.assertEqual(train_export.call_args.args[2], "LN1")
            for name, source in reference["exports"].items():
                self.assertEqual(file_sha256(root/"analysis"/name), source["sha256"])
                self.assertEqual(file_sha256(Path(source["source"])), source["sha256"])


if __name__ == "__main__":
    unittest.main()
