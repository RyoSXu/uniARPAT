"""Local-state readout preserves I2 at zero and remains a joint invariant model."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture
from tests.pipeline import test_checkpoint_boundary as checkpoint_fixtures
from tests.pipeline.test_checkpoint_selection import samples, summary
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_structure_controls import predict
from tools.eval import local_readout_compare as runner
from tools.eval.lg_structure_diagnosis import pair_errors
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
import tools.train as trainer
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.checkpoint_selection import BLIND_R2_METRIC
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, validate_local_global_config,
)
from utils.local_readout_experiment import (
    GATE_KEY, READOUT_SCHEMA, local_readout_configs, validate_i4_model,
)
from utils.local_structure_experiment import FROZEN
from utils.periodic_geometry import build_periodic_records
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import rng_digest, rng_snapshot


def configurations():
    return local_readout_configs(ExperimentConfig(**{**FROZEN, "local_route": "I1"}), "unit_readout")


class TestLocalReadout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, route="I4", dtype=torch.float32):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            params = {**model_params(route), "dropout": .05}
            model = Transformer(**params)
            before = rng_digest(rng_snapshot())
            info = initialize_comparison_model(model, params, 42)
            self.assertEqual(before, rng_digest(rng_snapshot()))
        finally:
            torch.set_default_dtype(previous)
        return model.eval(), info, params

    def test_candidate_copies_complete_untrained_bplus_and_adds_only_one_zero_scalar(self):
        reference, original, _ = self.create("I2")
        model, info, params = self.create()
        self.assertEqual(info["bplus_initialization"], original)
        self.assertEqual(info["common_state_sha256"], original["state_sha256"])
        self.assertEqual(set(model.state_dict()) - set(reference.state_dict()), {GATE_KEY})
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
            if value.numel():
                self.assertNotEqual(model.state_dict()[name].data_ptr(), value.data_ptr())
        self.assertEqual(info["parameter_count"], original["parameter_count"] + 1)
        validate_i4_model(model, params, require_zero=True, require_identity=True)
        self.assertEqual(float(model.encoder.local_readout_gate), 0.)
        with self.assertRaises(ValueError):
            Transformer(**{**params, "local_layout": "alternating"})

    def test_zero_gate_preserves_training_outputs_all_common_gradients_and_dropout_rng(self):
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
            reference, _, _ = self.create("I2", dtype)
            model, _, _ = self.create(dtype=dtype)
            z, geometry = fixture(dtype)
            captures = []
            for item in (reference, model):
                item.train()
                torch.manual_seed(71)
                outputs = predict(item, z, geometry)
                sum(value.square().mean() for value in outputs.values()).backward()
                captures.append((outputs, rng_digest(rng_snapshot())))
            self.assertEqual(captures[0][1], captures[1][1])
            for key in captures[0][0]:
                torch.testing.assert_close(captures[0][0][key], captures[1][0][key], atol=0, rtol=0)
            actual = dict(model.named_parameters())
            for name, parameter in reference.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                torch.testing.assert_close(parameter.grad, actual[name].grad, atol=tolerance, rtol=tolerance)
            gate_grad = model.encoder.local_readout_gate.grad
            self.assertTrue(torch.isfinite(gate_grad))
            self.assertNotEqual(float(gate_grad), 0.)

    def test_nonzero_gate_matches_manual_residual_with_differentiable_atom_input(self):
        model, _, _ = self.create(dtype=torch.float64)
        z, geometry = fixture(torch.float64)
        mask = z.eq(0)
        records = build_periodic_records(geometry, mask, radius=6.)
        h = model._encode_atoms(z).detach().requires_grad_()
        with torch.no_grad():
            model.encoder.local_readout_gate.fill_(.4)
        captured = {}
        handles = [model.encoder.local_layers[-1].register_forward_hook(
                       lambda _m, _i, output: captured.update(local=output)),
                   model.encoder.global_layers[-1].register_forward_hook(
                       lambda _m, _i, output: captured.update(global_state=output))]
        try:
            actual = model.encoder(h, records, mask)
        finally:
            for handle in handles:
                handle.remove()
        expected = (captured["global_state"] + .4 * (captured["local"] - h.masked_fill(mask[..., None], 0)))
        expected = expected.masked_fill(mask[..., None], 0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        weights = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual) / actual.numel()
        actual_input = torch.autograd.grad((actual * weights).sum(), h, retain_graph=True)[0]
        expected_input = torch.autograd.grad((expected * weights).sum(), h)[0]
        torch.testing.assert_close(actual_input, expected_input, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.isfinite(actual_input).all())
        self.assertEqual(int(actual_input[mask].count_nonzero()), 0)

    def test_nonzero_gate_reaches_both_task_decoders_and_the_h1_head(self):
        model, _, _ = self.create(dtype=torch.float64)
        with torch.no_grad():
            model.encoder.local_readout_gate.fill_(.3)
        captures = {"decoder": []}
        handles = [model.encoder.register_forward_hook(lambda _m, _i, output: captures.update(memory=output)),
                   model.decoder.register_forward_pre_hook(lambda _m, inputs: captures["decoder"].append(inputs[1])),
                   model.eta_head.register_forward_pre_hook(lambda _m, inputs: captures.update(pooled=inputs[0]))]
        z, geometry = fixture(torch.float64)
        try:
            outputs = predict(model, z, geometry)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(captures["decoder"]), 2)
        for memory in captures["decoder"]:
            self.assertIs(memory, captures["memory"])
        valid = z.ne(0).unsqueeze(-1)
        pooled = (captures["memory"] * valid).sum(1) / valid.sum(1)
        torch.testing.assert_close(captures["pooled"], pooled)
        for key in ("edos", "phdos", "eta"):
            gradient = torch.autograd.grad(outputs[key].square().mean(), model.encoder.local_readout_gate,
                                           retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient), key)
            self.assertNotEqual(float(gradient), 0., key)

    def test_nonzero_gate_keeps_e3_and_atom_permutation_with_nan_padding(self):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            model, _, _ = self.create(dtype=dtype)
            with torch.no_grad():
                model.encoder.local_readout_gate.fill_(.25)
                z, geometry = fixture(dtype)
                expected = predict(model, z, geometry)
                expected.update(reconstruct_h1_spectra(expected, [3, 2], [16., 8.]))
                for name, matrix, translation in e3_operations():
                    q, t = torch.as_tensor(matrix, dtype=dtype), torch.as_tensor(translation, dtype=dtype)
                    cell = geometry.cell @ q
                    coords = geometry.frac_coords + torch.bmm(t.expand(2, 1, 3), torch.linalg.inv(cell))
                    actual = predict(model, z, CrystalGeometry(cell, coords))
                    actual.update(reconstruct_h1_spectra(actual, [3, 2], [16., 8.]))
                    for key in expected:
                        with self.subTest(dtype=dtype, operation=name, key=key):
                            torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
                order = torch.tensor([2, 0, 3, 1])
                shuffled = z[:, order]
                coords = geometry.frac_coords[:, order].clone().masked_fill(shuffled.eq(0)[..., None], float("nan"))
                actual = predict(model, shuffled, CrystalGeometry(geometry.cell, coords))
                actual.update(reconstruct_h1_spectra(actual, [3, 2], [16., 8.]))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)

    def test_empty_and_all_padding_states_remain_zero_with_finite_priors(self):
        model, _, _ = self.create()
        model.encoder.local_readout_gate.data.fill_(.7)
        for atoms in (0, 3):
            z = torch.zeros((2, atoms), dtype=torch.long)
            geometry = CrystalGeometry(torch.eye(3).repeat(2, 1, 1) * 10.,
                                       torch.full((2, atoms, 3), float("nan")))
            values = predict(model, z, geometry)
            physical = reconstruct_h1_spectra(values, [0, 0], [0, 0])
            self.assertTrue(all(torch.isfinite(value).all() for value in values.values()))
            self.assertEqual(int(physical["blind_edos"].count_nonzero()), 0)
            self.assertEqual(int(physical["blind_phdos"].count_nonzero()), 0)

    def test_frozen_configs_cli_and_selection_contracts_are_isolated(self):
        controls = configurations()
        self.assertEqual([cfg.local_route for cfg in controls], ["I1", "I2", "I4"])
        self.assertEqual(len({cfg.tag for cfg in controls}), 3)
        for cfg in controls:
            validate_local_global_config(cfg)
            parsed = trainer.build_arg_parser().parse_args(training_command(cfg)[2:])
            self.assertEqual(asdict(cfg), asdict(ExperimentConfig.from_args(parsed)))
            contract = comparison_contract(cfg, model_params(cfg.local_route))
            self.assertEqual(contract["checkpoint_selection"]["mode"], "max")
        for changes in ({"selection_metric": "balanced_oracle_mae"}, {"selection_metric": "balanced_blind_mae"},
                        {"seed": 43}, {"epochs": 100}, {"batch_size": 64}, {"lr": 1e-4},
                        {"global_depth": 8}, {"local_layout": "alternating"}, {"use_amp": True},
                        {"init_ckpt": "old.pth"}, {"lambda_ph": 0.}, {"skip_test_eval": False}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(controls[-1], **changes))

    def test_checkpoint_restores_gate_optimizer_and_rejects_i1_i2_and_old_selection(self):
        model, _, params = self.create()
        cfg = configurations()[-1]
        contract = comparison_contract(cfg, params)
        self.assertEqual(contract["schema"], READOUT_SCHEMA)
        optimizer = torch.optim.AdamW(model.parameters())
        z, geometry = fixture()
        sum(value.square().mean() for value in predict(model, z, geometry).values()).backward()
        optimizer.step()
        gate = model.encoder.local_readout_gate
        self.assertIn(gate, optimizer.state)
        self.assertNotEqual(float(gate), 0.)
        payload = deepcopy(build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, .6,
                           model_contract=contract, training_rng=capture_training_rng()))
        fresh, _, _ = self.create()
        fresh_optimizer = torch.optim.AdamW(fresh.parameters())
        restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=contract)
        torch.testing.assert_close(fresh.encoder.local_readout_gate, gate, atol=0, rtol=0)
        for key, value in optimizer.state[gate].items():
            torch.testing.assert_close(fresh_optimizer.state[fresh.encoder.local_readout_gate][key], value)
        for route in ("I1", "I2"):
            other_cfg = replace(cfg, local_route=route)
            other = comparison_contract(other_cfg, {**params, "local_route": route})
            with self.assertRaisesRegex(ValueError, "different model contract"):
                restore_ablation_checkpoint(payload, fresh, fresh_optimizer, False, model_contract=other)
        with self.assertRaisesRegex(ValueError, "different model contract"):
            restore_ablation_checkpoint({**payload, "model_contract": {**contract,
                "selection_metric": "balanced_oracle_mae"}}, fresh, fresh_optimizer, False, model_contract=contract)

    def test_preflight_rejects_source_config_init_start_rng_and_equivalence_drift(self):
        controls = configurations()
        infos, entries = {}, {}
        for cfg in controls:
            _, info, params = self.create(cfg.local_route)
            infos[cfg.local_route] = info
            entries[cfg.local_route] = {"configuration": asdict(cfg), "initialization": info,
                "assembled_configuration": {"model": {"params": {"sub_model": {"transformer": params}}}},
                "contract": comparison_contract(cfg, params), "training_start_rng_sha256": "rng"}
        saved = {"cli": {"data_contract": {"data": "hash"}}, "runtime": {
            "local_global_init": infos["I1"], "structure_training_rng": [{"state_sha256": "rng"}]}}
        report = {"schema": runner.PREFLIGHT_SCHEMA, "passed": True, "controls": entries,
                  "recipe_reference": {"I1": "identity"}, "data_contract": {"data": "hash"},
                  "code_sha256": {"code": "hash"}, "gate_zero_equivalence": {"passed": True}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "recipe_identity", return_value=(saved, None, report["recipe_reference"])), \
                patch("tools.checks.check_local_readout.code_fingerprints", return_value=report["code_sha256"]):
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(report))
            self.assertEqual(runner.verify_preflight(path, Path(directory), controls), report)
            for key, value in (("code_sha256", {}), ("recipe_reference", {}), ("gate_zero_equivalence", {}),
                               ("passed", False), ("data_contract", {})):
                path.write_text(json.dumps({**report, key: value}))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), controls)
            for field in ("training_start_rng_sha256", "configuration", "initialization"):
                changed = deepcopy(report)
                changed["controls"]["I4"][field] = ("wrong" if field == "training_start_rng_sha256"
                    else {**entries["I4"][field], "seed": 9})
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), controls)

    def test_queue_skips_complete_runs_and_analysis_failure_remains_failed(self):
        for failure in (False, True):
            with tempfile.TemporaryDirectory() as directory, \
                    patch("tools.checks.check_local_readout.code_fingerprints", return_value={}), \
                    patch.object(runner.queue, "summarize", return_value=[{"status": "complete"}]), \
                    patch.object(runner.queue, "verify_completed"), \
                    patch.object(runner.queue.subprocess, "Popen") as process, \
                    patch.object(runner, "analyze", side_effect=RuntimeError("analysis failed") if failure else None):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                        runner.execute(configurations(), {"code_sha256": {}}, Path(directory))
                else:
                    runner.execute(configurations(), {"code_sha256": {}}, Path(directory))
                process.assert_not_called()
                state = json.loads((Path(directory) / "queue.json").read_text())
                self.assertEqual(state["schema"], "local-readout-queue-v1")
                self.assertEqual(state["status"], "failed" if failure else "complete")
                self.assertEqual(state["arms"], {"I1": "complete", "I2": "complete", "I4": "complete"})

    def test_failed_training_stops_before_remaining_arms_without_retry(self):
        failed = MagicMock(pid=12)
        failed.wait.return_value = 1
        with tempfile.TemporaryDirectory() as directory, \
                patch("tools.checks.check_local_readout.code_fingerprints", return_value={}), \
                patch.object(runner.queue, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.queue.subprocess, "Popen", return_value=failed) as process, \
                patch.object(runner, "analyze") as analysis:
            with self.assertRaisesRegex(RuntimeError, "I1 failed"):
                runner.execute(configurations(), {"code_sha256": {}}, Path(directory))
            process.assert_called_once()
            analysis.assert_not_called()
            state = json.loads((Path(directory) / "queue.json").read_text())
            self.assertEqual(state["arms"], {"I1": "running", "I2": "pending", "I4": "pending"})
            self.assertEqual(state["status"], "failed")

    def test_gate_diagnostics_uses_selected_checkpoint_and_rejects_history_drift(self):
        cfg = configurations()[-1]
        params = model_params("I4")
        contract = comparison_contract(cfg, params)
        saved = {"config": {"model": {"params": {"sub_model": {"transformer": params}}}}}
        history = pd.DataFrame({"epoch": [1, 2, 3], "local_readout_gate": [.1, .2, .3]})
        picked = {"history": history, "row": history.iloc[1],
                  "checkpoint": {"model": {GATE_KEY: torch.tensor(.2)}}}
        with patch.object(runner.queue, "run_directory", return_value=Path("unit")), \
                patch.object(runner, "load_run", return_value=(saved, params, contract)), \
                patch.object(runner, "select_best", return_value=picked):
            result = runner.gate_diagnostics(cfg)
            self.assertEqual(result["best_epoch"], 2)
            self.assertAlmostEqual(result["selected_gate"], .2)
            self.assertEqual(result["trajectory"][-1]["local_readout_gate"], .3)
            picked["checkpoint"]["model"][GATE_KEY] = torch.tensor(.3)
            with self.assertRaisesRegex(ValueError, "gate differs"):
                runner.gate_diagnostics(cfg)

    def test_actual_training_control_flow_records_gate_at_joint_best_and_on_resume(self):
        fake = checkpoint_fixtures.TestRunnerRecovery()
        self.addCleanup(fake.doCleanups)
        fake.setUp()
        fake.cfg = replace(configurations()[-1], tag="_recovery_contract")
        transformer = fake.model.model["transformer"]
        transformer.local_route = "I4"
        transformer.encoder = torch.nn.Module()
        transformer.encoder.local_layers = torch.nn.ModuleList()
        transformer.encoder.local_readout_gate = torch.nn.Parameter(torch.tensor(.125))
        fake.evaluate.side_effect = [{"mae_edos_median": .2, "mae_phdos_median": .3,
            "r2_edos_median": .99, "r2_phdos_median": .99,
            "blind_r2_edos_median": e, "blind_r2_phdos_median": p,
            "blind_mae_edos_median": .2, "blind_mae_phdos_median": .3,
            "blind_fail_rate_edos": 2., "blind_fail_rate_phdos": 3.} for e, p in
            [(.2, .4), (.5, .8)] + [(.3, .4)] * 33]
        with patch.object(trainer, "initialize_comparison_model", return_value={"protocol": "unit"}):
            trainer.train_and_eval(fake.cfg)
            trainer.train_and_eval(fake.cfg)
        history = pd.read_csv(fake.history_path)
        self.assertEqual(history.epoch.tolist(), list(range(1, 36)))
        self.assertTrue((history.local_readout_gate == .125).all())
        best = torch.load(fake.best_path, weights_only=True)
        self.assertEqual(best["epoch"], 2)
        self.assertEqual(float(best["model"][GATE_KEY]), .125)
        logs = [str(call) for call in trainer.logger.info.call_args_list]
        self.assertTrue(any("local_readout_gate=0.125" in line and "selection_mode=max" in line for line in logs))

    def test_analysis_reports_joint_primary_and_keeps_train_valid_structure_diagnostic(self):
        ids = ["a", "b", "c", "d", "e"]
        keys = [((14, 1),)] * 3 + [((8, 1),)] * 2
        target = np.array([[.1, .9], [.4, .6], [.6, .4], [.2, .8], [.8, .2]])

        def save_split(out, label, split):
            # Make candidate pair differences deliberately worse while giving
            # its material-level blind metrics a gain. Diagnosis cannot vote.
            prediction = target * (.2 if label == "I4" else .8) + (.4 if label == "I4" else .1)
            np.savez(out / f"{label}_{split}_spectra.npz", material_ids=np.asarray(ids),
                     tgt_shape_edos=target, tgt_shape_phdos=target,
                     pred_shape_edos=prediction, pred_shape_phdos=prediction)
            return {(grouping, task): pair_errors(prediction, target, keys, ids)
                    for grouping in ("absolute", "reduced") for task in ("edos", "phdos")}

        def fake_export(run, out, summary_out, spectra_out):
            label = run.name
            save_split(out.parent, label, "valid")
            frame = samples([.6 if label == "I4" else .5] * 5, [.8 if label == "I4" else .7] * 5)
            frame.index = ids
            frame.reset_index(names="material_id").to_csv(out, index=False)
            summary_out.write_text(json.dumps(summary(label)))

        def fake_train(run, out, label):
            return save_split(out, label, "train")

        with tempfile.TemporaryDirectory() as directory, \
                patch("tools.checks.check_local_readout.code_fingerprints", return_value={}), \
                patch.object(runner.queue, "verify_completed"), \
                patch.object(runner.queue, "run_directory", side_effect=lambda cfg: Path(directory) / cfg.local_route), \
                patch.object(runner, "file_sha256", return_value="hash"), \
                patch.object(runner, "export", side_effect=fake_export), \
                patch.object(runner, "load_run", return_value=({"config": {}}, None, None)), \
                patch.object(runner, "ConfigBuilder") as builder, \
                patch.object(runner, "spectrum_pairs", side_effect=lambda spectrum, samples, dataset:
                             save_split(spectrum.parent, spectrum.name.split('_')[0], "valid")), \
                patch.object(runner, "export_train", side_effect=fake_train), \
                patch.object(runner, "gate_diagnostics", return_value={"diagnostic_only": True}):
            builder.return_value.get_dataset.return_value.__len__.return_value = 2313
            report = runner.analyze(configurations(), {"code_sha256": {}, "data_contract": {}},
                                    Path(directory) / "analysis")
        self.assertTrue(report["analysis_complete"])
        self.assertEqual(report["primary_metric"], BLIND_R2_METRIC)
        self.assertTrue(report["numerical_adoption_conditions"]["both_required"])
        self.assertFalse(report["numerical_adoption_conditions"]["automatic_baseline_replacement"])
        structure = report["structural_diagnosis"]
        self.assertTrue(structure["diagnostic_only"])
        self.assertEqual(len(structure["summaries"]), 24)
        for contrasts in structure["contrasts"].values():
            self.assertEqual(len(contrasts), 8)
            self.assertGreater(contrasts["valid_absolute_edos"]["ci95"][0], 0)


if __name__ == "__main__":
    unittest.main()
