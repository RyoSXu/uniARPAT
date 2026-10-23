"""Internal-addition control retains complete messages, norms, full-state readout and outer residual."""
from copy import deepcopy
from dataclasses import asdict, replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from model.local_global_encoder import LocalResidualLayer
from model.periodic_messages import InvariantMessageBlock, PeriodicLocalMessage, prepare_shared_message_geometry
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
from tools.eval.local_residual_compare import INTERNAL_RESIDUAL as PROTOCOL
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, write_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import capture_training_rng, comparison_contract, initialize_comparison_model, validate_local_global_config
from utils.local_residual_experiment import RESIDUAL_SCHEMA, residual_config, validate_residual_model
from utils.local_structure_experiment import FROZEN
from utils.periodic_geometry import build_periodic_records
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash


class TestLocalResidual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = ExperimentConfig(**{**FROZEN, "local_route": "I3", "selection_metric": "balanced_blind_r2", "tag": "_unit_a64"})
        cls.cfg = residual_config(cls.reference, "unit_residual")
        cls.params = {route: {**model_params(route), "dropout": .05} for route in ("I3", "LR2", "LR1", "LN1", "LN2")}
        cls.infos = {}
        for route in ("I3", "LR2"):
            model = Transformer(**cls.params[route])
            cls.infos[route] = initialize_comparison_model(model, cls.params[route], 42)

    def create(self, route="LR2", dtype=torch.float32):
        model = Transformer(**self.params[route]).to(dtype)
        initialize_comparison_model(model, self.params[route], 42)
        return model.eval()

    def test_direct_complete_block_formulas_and_all_gradients_with_and_without_addition(self):
        for dtype, tolerance in ((torch.float32, 4e-5), (torch.float64, 2e-11)):
            z, geometry = fixture(dtype)
            g = prepare_shared_message_geometry(build_periodic_records(geometry, z.eq(0), 6.), "A")
            for first in (False, True):
                for add in (True, False):
                    block = InvariantMessageBlock(recompute_angles=False, norm_first=first, residual_add=add).to(dtype)
                    a = torch.randn((*z.shape, 64), dtype=dtype, requires_grad=True)
                    b = a.detach().clone().requires_grad_()
                    actual, expected = block(a, g), direct_block(block, b, g)
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                    parameters = tuple(block.parameters())
                    weights = torch.randn_like(actual)
                    ag = torch.autograd.grad((actual*weights).sum(), (a, *parameters))
                    bg = torch.autograd.grad((expected*weights).sum(), (b, *parameters))
                    for left, right in zip(ag, bg):
                        torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)

    def test_two_block_message_matches_direct_no_addition_and_full_state_readout(self):
        for dtype, tolerance in ((torch.float32, 4e-5), (torch.float64, 2e-11)):
            z, geometry = fixture(dtype)
            records = build_periodic_records(geometry, z.eq(0), 6.)
            g = prepare_shared_message_geometry(records, "A")
            message = PeriodicLocalMessage("A", d_model=16, recompute_angles=False, internal_residual=False).to(dtype)
            a = torch.randn((*z.shape, 16), dtype=dtype, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            scalar = message.local_input(b.masked_fill(z.eq(0)[..., None], 0)).masked_fill(z.eq(0)[..., None], 0)
            states = [scalar]
            for block in message.blocks:
                scalar = direct_block(block, scalar, g).masked_fill(z.eq(0)[..., None], 0)
                states.append(scalar)
            expected = message.local_output(scalar).masked_fill(z.eq(0)[..., None], 0)
            actual, debug = message(a, records, z.eq(0), prepared_geometry=g, return_debug=True)
            for left, right in zip(debug["states"], states):
                torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            parameters = tuple(message.parameters())
            weights = torch.randn_like(actual)
            ag = torch.autograd.grad((actual*weights).sum(), (a, *parameters))
            bg = torch.autograd.grad((expected*weights).sum(), (b, *parameters))
            for left, right in zip(ag, bg):
                torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)

    def test_zero_update_removes_only_the_direct_input_path_and_keeps_post_norm(self):
        z, geometry = fixture()
        g = prepare_shared_message_geometry(build_periodic_records(geometry, z.eq(0), 6.), "A")
        block = InvariantMessageBlock(recompute_angles=False, residual_add=False)
        with torch.no_grad():
            block.update[-1].weight.zero_()
            block.update[-1].bias.zero_()
            block.norm.bias.fill_(.4)
        scalar = torch.randn((*z.shape, 64), requires_grad=True)
        actual = block(scalar, g)
        torch.testing.assert_close(actual, block.norm.bias.expand_as(actual), atol=0, rtol=0)
        gradient, = torch.autograd.grad(actual.square().sum(), (scalar,))
        torch.testing.assert_close(gradient, torch.zeros_like(scalar), atol=0, rtol=0)
        block.residual_add = True
        torch.testing.assert_close(block(scalar, g), block.norm(scalar), atol=0, rtol=0)
        for call in (lambda:InvariantMessageBlock(residual_add="none"),
                     lambda:PeriodicLocalMessage("A", internal_residual="none"),
                     lambda:PeriodicLocalMessage("B+", internal_residual=False),
                     lambda:LocalResidualLayer("C1", 16, internal_residual=False)):
            with self.assertRaises(ValueError):
                call()

    def test_all_initial_tensors_match_without_checkpoint_loading_or_rng_consumption(self):
        with patch("torch.load", side_effect=AssertionError("trained weights used for initialization")):
            models = []
            for route in ("I3", "LR2"):
                model = Transformer(**self.params[route])
                before = rng_digest(rng_snapshot())
                initialize_comparison_model(model, self.params[route], 42)
                self.assertEqual(before, rng_digest(rng_snapshot()))
                models.append(model)
        for key, value in models[0].state_dict().items():
            other = models[1].state_dict()[key]
            torch.testing.assert_close(value, other, atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), other.data_ptr())
        self.assertEqual(self.infos["LR2"]["additive_initialization"], self.infos["I3"])
        self.assertEqual(self.infos["LR2"]["state_sha256"], self.infos["I3"]["state_sha256"])
        self.assertEqual(sum(p.numel() for p in models[1].parameters()), self.infos["I3"]["parameter_count"])
        self.assertEqual([b.residual_add for layer in models[0].encoder.local_layers for b in layer.message.blocks], [True]*4)
        self.assertEqual([b.residual_add for layer in models[1].encoder.local_layers for b in layer.message.blocks], [False]*4)

    def test_full_candidate_e3_permutation_padding_empty_rows_and_finite_gradients(self):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            model = self.create(dtype=dtype)
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
                outputs = predict(model, torch.zeros_like(z), CrystalGeometry(geometry.cell, torch.full_like(geometry.frac_coords, float("nan"))))
                self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

    def test_cli_recipe_assembly_contract_validation_and_actual_execution_record(self):
        args = build_arg_parser().parse_args(training_command(self.cfg)[2:])
        self.assertEqual(asdict(ExperimentConfig.from_args(args)), asdict(self.cfg))
        self.assertEqual(runner.restore_reference_assembly(captured_config(self.cfg), protocol=PROTOCOL), captured_config(self.reference))
        contract = comparison_contract(self.cfg, self.params["LR2"])
        self.assertEqual(contract["schema"], RESIDUAL_SCHEMA)
        self.assertEqual(contract["ablation"]["internal_formula"], "LN64(F(s,R))")
        self.assertEqual(contract["ablation"]["outer_residual"], "h+P_out(s2)")
        for changes in ({"selection_metric": "balanced_oracle_mae"}, {"seed": 43}, {"epochs": 100},
                        {"batch_size": 64}, {"local_layout": "alternating"}, {"use_amp": True}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(self.cfg, **changes))
        model = self.create()
        validate_residual_model(model, self.params["LR2"], require_identity=True)
        layer = model.encoder.local_layers[0]
        for target, name, changed in ((layer.message, "internal_residual", True),
                (layer.message.blocks[0], "residual_add", True), (layer.message, "readout_delta", True),
                (layer, "outer_norm_first", False)):
            old = getattr(target, name)
            setattr(target, name, changed)
            with self.assertRaises(ValueError):
                validate_residual_model(model, self.params["LR2"])
            setattr(target, name, old)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.yaml"
            path.write_text("{}")
            record = write_local_global_execution(path, model, 0)
            for key, value in PROTOCOL.execution_fields:
                self.assertEqual(record[key], list(value))

    def test_same_shape_checkpoints_are_isolated_from_all_existing_candidates(self):
        model = self.create()
        optimizer = torch.optim.AdamW(model.parameters())
        z, geometry = fixture()
        sum(value.square().mean() for value in predict(model, z, geometry).values()).backward()
        optimizer.step()
        contract = comparison_contract(self.cfg, self.params["LR2"])
        payload = build_ablation_checkpoint(2, "M1", 42, False, model, optimizer, .5,
            model_contract=contract, training_rng=capture_training_rng())
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        before = state_hash(model.state_dict())
        restore_ablation_checkpoint(saved, model, optimizer, False, model_contract=contract)
        self.assertEqual(state_hash(model.state_dict()), before)
        for route in ("I3", "LN1", "LN2", "LR1"):
            other = comparison_contract(replace(self.cfg, local_route=route), self.params[route])
            for loaded, expected in ((saved, other), ({**saved, "model_contract": other}, contract)):
                with self.assertRaisesRegex(ValueError, "different model contract"):
                    restore_ablation_checkpoint(loaded, model, optimizer, False, model_contract=expected)

    def test_preflight_binds_candidate_family_reference_sources_and_recipe(self):
        identity = {"evidence_dir": "/synthetic", "reference_commit": runner.REFERENCE_COMMIT, "data_order": {}}
        assembled = {"model": {"params": {"sub_model": {"transformer": self.params["LR2"]}}}}
        saved = {"config": runner.restore_reference_assembly(assembled, protocol=PROTOCOL), "cli": {"data_contract": {}},
                 "runtime": {"local_global_init": self.infos["I3"], "structure_training_rng": [{"state_sha256": "rng"}]}}
        entry = {"configuration": asdict(self.cfg), "assembled_configuration": assembled,
            "initialization": self.infos["LR2"], "contract": comparison_contract(self.cfg, self.params["LR2"]),
            "data_order": {}, "training_start_rng_sha256": "rng", "forward_backward": {"all_parameter_gradients_finite": True},
            "evaluation_finite": True, "symmetry": {"torch.float32": {}, "torch.float64": {}}, "additive_checkpoint_rejected": True,
            "checkpoint_roundtrip": {"weights_optimizer_rng_equal": True, "foreign_contracts_rejected": True}, "costs": [{}, {}, {}]}
        report = {"schema": "local-residual-preflight-v1", "passed": True, "tag_prefix": "unit_residual", "code_sha256": {},
            "runtime": {"gpu": "unit"}, "data_contract": {}, "references": {"A64": identity}, "reference_compatibility": {"passed": True},
            "reference_prediction_replay": {"passed": True}, "learning_rate_35": learning_rate_trace(35, self.cfg.lr), "controls": {"LR2": entry}}
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

    def test_completed_candidate_requires_actual_addition_norm_and_readout_records(self):
        entry = {"assembled_configuration": {}, "contract": {}}
        record = {key: list(value) for key, value in PROTOCOL.execution_fields}
        saved = {"config": {}, "runtime": {"local_global_execution": [record]}}
        with patch.object(runner.queue, "verify_completed"), patch.object(runner, "load_run", return_value=(saved, {}, {})):
            runner.verify_completed(self.cfg, {"controls": {"LR2": entry}}, protocol=PROTOCOL)
            for key, _ in PROTOCOL.execution_fields:
                original = record.pop(key)
                with self.assertRaises(ValueError):
                    runner.verify_completed(self.cfg, {"controls": {"LR2": entry}}, protocol=PROTOCOL)
                record[key] = original
            record["internal_residual_add"] = [True]*4
            with self.assertRaises(ValueError):
                runner.verify_completed(self.cfg, {"controls": {"LR2": entry}}, protocol=PROTOCOL)

    def test_joint_blind_report_and_each_task_protection(self):
        data = module_exports()
        exports = {"LR2": data["I3"], "A64": data["I2"]}
        report = runner.comparison_report(exports, draws=20, protocol=PROTOCOL)
        self.assertTrue(report["decisions"]["no_internal_addition_conditions_met"])
        self.assertFalse(report["decisions"]["automatic_baseline_replacement"])
        bad = deepcopy(exports)
        bad["LR2"][1].loc[bad["LR2"][1].index[:2], "blind_r2_phdos"] = -.1
        self.assertFalse(runner.comparison_report(bad, draws=20, protocol=PROTOCOL)["decisions"]["no_internal_addition_conditions_met"])

    def test_partial_queue_and_training_failure_stop_without_retry_or_analysis(self):
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
            with self.assertRaisesRegex(RuntimeError, "LR2 failed"):
                runner.execute(self.cfg, {}, Path(directory)/"run", protocol=PROTOCOL)
            self.assertEqual(popen.call_count, 1)
            analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
