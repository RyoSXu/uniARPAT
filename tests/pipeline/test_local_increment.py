"""Incremental readout retains complete A, affine bias, both gradients and isolated contracts."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.nn import functional as F

from model.periodic_messages import PeriodicLocalMessage, prepare_shared_message_geometry
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_module import exports as module_exports
from tests.pipeline.test_local_norm import direct_block
from tests.pipeline.test_local_structure_controls import predict
from tools.checks import check_local_norm as checker
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval import local_norm_compare as runner
from tools.eval.local_global_compare import training_command
from tools.eval.local_increment_compare import INCREMENT_READOUT as PROTOCOL
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, write_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import capture_training_rng, comparison_contract, initialize_comparison_model, validate_local_global_config
from utils.local_increment_experiment import INCREMENT_SCHEMA, increment_config, validate_increment_model
from utils.local_structure_experiment import FROZEN
from utils.periodic_geometry import build_periodic_records
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash


class TestLocalIncrement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = ExperimentConfig(**{**FROZEN, "local_route": "I3", "selection_metric": "balanced_blind_r2", "tag": "_unit_a64"})
        cls.cfg = increment_config(cls.reference, "unit_increment")
        cls.params = {route: {**model_params(route), "dropout": .05} for route in ("I3", "LR1", "LN1", "LN2")}
        cls.infos = {}
        for route in ("I3", "LR1"):
            model = Transformer(**cls.params[route])
            cls.infos[route] = initialize_comparison_model(model, cls.params[route], 42)

    def create(self, route="LR1", dtype=torch.float32):
        model = Transformer(**self.params[route]).to(dtype)
        initialize_comparison_model(model, self.params[route], 42)
        return model.eval()

    def test_direct_complete_message_outputs_and_all_gradients_for_both_readouts(self):
        for dtype, tolerance in ((torch.float32, 4e-5), (torch.float64, 2e-11)):
            z, geometry = fixture(dtype)
            records = build_periodic_records(geometry, z.eq(0), 6.)
            g = prepare_shared_message_geometry(records, "A")
            for delta in (False, True):
                message = PeriodicLocalMessage("A", d_model=16, recompute_angles=False, readout_delta=delta).to(dtype)
                a = torch.randn((*z.shape, 16), dtype=dtype, requires_grad=True)
                b = a.detach().clone().requires_grad_()
                initial = message.local_input(b.masked_fill(z.eq(0)[..., None], 0)).masked_fill(z.eq(0)[..., None], 0)
                scalar = initial
                for block in message.blocks:
                    scalar = direct_block(block, scalar, g).masked_fill(z.eq(0)[..., None], 0)
                expected = message.local_output(scalar-initial if delta else scalar).masked_fill(z.eq(0)[..., None], 0)
                actual = message(a, records, z.eq(0), prepared_geometry=g)
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                parameters = tuple(message.parameters())
                weights = torch.randn_like(actual)
                ag = torch.autograd.grad((actual*weights).sum(), (a, *parameters))
                bg = torch.autograd.grad((expected*weights).sum(), (b, *parameters))
                for left, right in zip(ag, bg):
                    torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)

    def test_internal_states_unchanged_and_difference_removes_only_linear_initial_state(self):
        z, geometry = fixture(torch.float64)
        records = build_periodic_records(geometry, z.eq(0), 6.)
        message = PeriodicLocalMessage("A", d_model=16, recompute_angles=False).double()
        candidate = deepcopy(message)
        candidate.readout_delta = True
        h = torch.randn((*z.shape, 16), dtype=torch.float64)
        full, debug = message(h, records, z.eq(0), return_debug=True)
        delta, other = candidate(h, records, z.eq(0), return_debug=True)
        for left, right in zip(debug["states"], other["states"]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        expected = full-F.linear(debug["states"][0], message.local_output.weight)
        torch.testing.assert_close(delta, expected, atol=1e-12, rtol=1e-12)
        with self.assertRaises(ValueError):
            PeriodicLocalMessage("B+", readout_delta=True)
        with self.assertRaises(ValueError):
            PeriodicLocalMessage("A", readout_delta="increment")

    def test_zero_scalar_increment_keeps_bias_and_its_learning_signal(self):
        z, geometry = fixture()
        records = build_periodic_records(geometry, z.eq(0), 6.)
        message = PeriodicLocalMessage("A", d_model=16, recompute_angles=False, readout_delta=True)
        with torch.no_grad():
            message.local_output.bias.fill_(.7)
        with ExitStack() as stack:
            for block in message.blocks:
                stack.enter_context(patch.object(block, "forward", side_effect=lambda scalar, _: scalar))
            out = message(torch.randn((*z.shape, 16)), records, z.eq(0))
        expected = message.local_output.bias.expand_as(out).masked_fill(z.eq(0)[..., None], 0)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        gradient, = torch.autograd.grad(out.sum(), (message.local_output.bias,))
        torch.testing.assert_close(gradient, torch.full_like(gradient, int(z.ne(0).sum())), atol=0, rtol=0)

    def test_all_initial_tensors_match_reference_without_checkpoint_loading_or_rng_consumption(self):
        with patch("torch.load", side_effect=AssertionError("trained weights used for initialization")):
            models = []
            for route in ("I3", "LR1"):
                model = Transformer(**self.params[route])
                before = rng_digest(rng_snapshot())
                initialize_comparison_model(model, self.params[route], 42)
                self.assertEqual(before, rng_digest(rng_snapshot()))
                models.append(model)
        for key, value in models[0].state_dict().items():
            other = models[1].state_dict()[key]
            torch.testing.assert_close(value, other, atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), other.data_ptr())
        self.assertEqual(self.infos["LR1"]["state_readout_initialization"], self.infos["I3"])
        self.assertEqual(self.infos["LR1"]["state_sha256"], self.infos["I3"]["state_sha256"])

    def test_full_candidate_e3_permutation_padding_empty_rows_and_finite_gradients(self):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            model = self.create(dtype=dtype)
            z, geometry = fixture(dtype)
            expected = predict(model, z, geometry)
            sum(value.square().mean() for value in expected.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertTrue(all(layer.message.local_output.bias.grad.abs().sum() > 0 for layer in model.encoder.local_layers))
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
                outputs = predict(model, torch.zeros_like(z), CrystalGeometry(geometry.cell, torch.full_like(geometry.frac_coords, float("nan"))))
                self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

    def test_cli_recipe_assembly_contract_and_actual_readout_record(self):
        args = build_arg_parser().parse_args(training_command(self.cfg)[2:])
        self.assertEqual(asdict(ExperimentConfig.from_args(args)), asdict(self.cfg))
        self.assertEqual(runner.restore_reference_assembly(captured_config(self.cfg), protocol=PROTOCOL), captured_config(self.reference))
        contract = comparison_contract(self.cfg, self.params["LR1"])
        self.assertEqual(contract["schema"], INCREMENT_SCHEMA)
        self.assertEqual(contract["ablation"]["local_readout"], "W_out(s2-s0)+b_out")
        for changes in ({"selection_metric": "balanced_oracle_mae"}, {"seed": 43}, {"epochs": 100},
                        {"batch_size": 64}, {"local_layout": "alternating"}, {"use_amp": True}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(self.cfg, **changes))
        model = self.create()
        validate_increment_model(model, self.params["LR1"], require_identity=True)
        layer = model.encoder.local_layers[0]
        layer.message.readout_delta = False
        with self.assertRaises(ValueError):
            validate_increment_model(model, self.params["LR1"])
        layer.message.readout_delta = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.yaml"
            path.write_text("{}")
            record = write_local_global_execution(path, model, 0)
            for key, expected in PROTOCOL.execution_fields:
                self.assertEqual(record[key], list(expected))

    def test_same_shape_checkpoint_isolation_from_all_three_reference_readouts(self):
        model = self.create()
        optimizer = torch.optim.AdamW(model.parameters())
        z, geometry = fixture()
        sum(value.square().mean() for value in predict(model, z, geometry).values()).backward()
        optimizer.step()
        contract = comparison_contract(self.cfg, self.params["LR1"])
        payload = build_ablation_checkpoint(2, "M1", 42, False, model, optimizer, .5,
            model_contract=contract, training_rng=capture_training_rng())
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        before = state_hash(model.state_dict())
        restore_ablation_checkpoint(saved, model, optimizer, False, model_contract=contract)
        self.assertEqual(state_hash(model.state_dict()), before)
        for route in ("I3", "LN1", "LN2"):
            other = comparison_contract(replace(self.cfg, local_route=route), self.params[route])
            for loaded, expected in ((saved, other), ({**saved, "model_contract": other}, contract)):
                with self.assertRaisesRegex(ValueError, "different model contract"):
                    restore_ablation_checkpoint(loaded, model, optimizer, False, model_contract=expected)

    def test_preflight_binds_exact_readout_family_and_reference(self):
        identity = {"evidence_dir": "/synthetic", "reference_commit": runner.REFERENCE_COMMIT, "data_order": {}}
        assembled = {"model": {"params": {"sub_model": {"transformer": self.params["LR1"]}}}}
        saved = {"config": runner.restore_reference_assembly(assembled, protocol=PROTOCOL), "cli": {"data_contract": {}},
                 "runtime": {"local_global_init": self.infos["I3"], "structure_training_rng": [{"state_sha256": "rng"}]}}
        entry = {"configuration": asdict(self.cfg), "assembled_configuration": assembled,
            "initialization": self.infos["LR1"], "contract": comparison_contract(self.cfg, self.params["LR1"]),
            "data_order": {}, "training_start_rng_sha256": "rng", "forward_backward": {"all_parameter_gradients_finite": True},
            "evaluation_finite": True, "symmetry": {"torch.float32": {}, "torch.float64": {}}, "state_readout_checkpoint_rejected": True,
            "checkpoint_roundtrip": {"weights_optimizer_rng_equal": True, "foreign_contracts_rejected": True}, "costs": [{}, {}, {}]}
        report = {"schema": "local-increment-preflight-v1", "passed": True, "tag_prefix": "unit_increment", "code_sha256": {},
            "runtime": {"gpu": "unit"}, "data_contract": {}, "references": {"A64": identity}, "reference_compatibility": {"passed": True},
            "reference_prediction_replay": {"passed": True}, "learning_rate_35": learning_rate_trace(35, self.cfg.lr), "controls": {"LR1": entry}}
        with patch.object(checker, "code_fingerprints", return_value={}), \
                patch.object(runner, "reference_identity", return_value=(saved, self.reference, identity)), \
                patch.object(runner.fresh, "runtime_identity", return_value=report["runtime"]):
            runner.validate_preflight(report, self.cfg, protocol=PROTOCOL)
            for key in ("schema", "code_sha256", "runtime", "data_contract", "reference_compatibility", "reference_prediction_replay"):
                changed = deepcopy(report)
                changed[key] = {"drift": True}
                with self.assertRaises(ValueError):
                    runner.validate_preflight(changed, self.cfg, protocol=PROTOCOL)
            with self.assertRaises(ValueError):
                runner.validate_preflight(report, self.cfg)

    def test_completed_candidate_checks_actual_readout_not_just_contract(self):
        entry = {"assembled_configuration": {}, "contract": {}}
        record = {key: list(value) for key, value in PROTOCOL.execution_fields}
        saved = {"config": {}, "runtime": {"local_global_execution": [record]}}
        with patch.object(runner.queue, "verify_completed"), patch.object(runner, "load_run", return_value=(saved, {}, {})):
            runner.verify_completed(self.cfg, {"controls": {"LR1": entry}}, protocol=PROTOCOL)
            for key, _ in PROTOCOL.execution_fields:
                original = record.pop(key)
                with self.assertRaises(ValueError):
                    runner.verify_completed(self.cfg, {"controls": {"LR1": entry}}, protocol=PROTOCOL)
                record[key] = original
            record["local_readout"] = ["state"]*2
            with self.assertRaises(ValueError):
                runner.verify_completed(self.cfg, {"controls": {"LR1": entry}}, protocol=PROTOCOL)

    def test_equal_task_blind_report_and_failure_protection(self):
        data = module_exports()
        exports = {"LR1": data["I3"], "A64": data["I2"]}
        report = runner.comparison_report(exports, draws=20, protocol=PROTOCOL)
        self.assertTrue(report["decisions"]["increment_conditions_met"])
        self.assertFalse(report["decisions"]["automatic_baseline_replacement"])
        bad = deepcopy(exports)
        bad["LR1"][1].loc[bad["LR1"][1].index[:2], "blind_r2_phdos"] = -.1
        self.assertFalse(runner.comparison_report(bad, draws=20, protocol=PROTOCOL)["decisions"]["increment_conditions_met"])

    def test_partial_queue_and_failed_training_never_retry_or_start_analysis(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "validate_preflight"), \
                patch.object(runner, "summarize", return_value=[{"status": "partial"}]), patch.object(runner.queue, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "recovery decision"):
                runner.execute(self.cfg, {}, Path(directory), protocol=PROTOCOL)
            execute.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "validate_preflight"), \
                patch.object(runner, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.queue, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.queue, "run_directory", return_value=Path(directory)/"unused"), \
                patch.object(runner, "analyze") as analyze, \
                patch.object(runner.queue.subprocess, "Popen", return_value=MagicMock(pid=123, wait=lambda:1)) as popen:
            with self.assertRaisesRegex(RuntimeError, "LR1 failed"):
                runner.execute(self.cfg, {}, Path(directory)/"run", protocol=PROTOCOL)
            self.assertEqual(popen.call_count, 1)
            analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
