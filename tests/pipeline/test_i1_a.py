"""Full A on the corrected joint backbone: initialization and comparison gates."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from model.periodic_messages import PeriodicLocalMessage
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_structure_controls import predict
from tools.eval import i1_a_compare as runner
from tools.eval.lg_structure_diagnosis import pair_errors
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, write_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.i1_a_experiment import A_SCHEMA, i1_a_config, validate_i3_model
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, deep_reference_state,
    initialize_comparison_model, validate_local_global_config,
)
from utils.local_structure_experiment import FROZEN, is_shared_control_parameter
from utils.paths import file_sha256
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import rng_digest, rng_snapshot


def configuration():
    return i1_a_config(ExperimentConfig(**{**FROZEN, "local_route": "I2"}), "unit_a")


class TestI1A(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, route="I3", dtype=torch.float32):
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

    def test_common_values_match_i2_and_content_matches_canonical_a_without_aliases(self):
        reference, reference_info, _ = self.create("I2")
        model, info, params = self.create()
        canonical, _ = deep_reference_state(42, {**params, "local_route": "A"})
        self.assertEqual(info["i2_initialization"], reference_info)
        self.assertEqual(info["shared_control_state_sha256"], reference_info["shared_control_state_sha256"])
        for name, value in model.state_dict().items():
            source = reference.state_dict()[name] if is_shared_control_parameter(name) else canonical[name]
            torch.testing.assert_close(value, source, atol=0, rtol=0)
            if value.numel():
                self.assertNotEqual(value.data_ptr(), source.data_ptr())
        self.assertEqual(info["parameter_counts"]["shared"], info["reference_parameter_counts"]["shared"])
        self.assertGreater(info["parameter_counts"]["local_content"], info["reference_parameter_counts"]["local_content"])
        self.assertEqual(info["execution_order"], reference_info["execution_order"])
        self.assertEqual(info["update_counts"], reference_info["update_counts"])

    def test_full_a_preserves_approved_settings_and_finite_gradients(self):
        for dtype in (torch.float32, torch.float64):
            model, _, params = self.create(dtype=dtype)
            validate_i3_model(model, params, require_identity=True)
            self.assertEqual(model.encoder.geometry_route, "A")
            for layer in model.encoder.local_layers:
                self.assertIsInstance(layer.message, PeriodicLocalMessage)
                self.assertEqual(layer.message.route, "A")
                self.assertEqual(len(layer.message.blocks), 2)
                self.assertTrue(all(not b.recompute_angles and b.pair_chunk == 65536 for b in layer.message.blocks))
            model.train()
            z, geometry = fixture(dtype)
            with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring")):
                values = predict(model, z, geometry)
                sum(value.square().mean() for value in values.values()).backward()
            for name, parameter in model.named_parameters():
                self.assertTrue(parameter.requires_grad, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_final_spectra_preserve_e3_permutation_and_nan_padding(self):
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            model, _, _ = self.create(dtype=dtype)
            z, geometry = fixture(dtype)
            with torch.no_grad():
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
                coords = geometry.frac_coords[:, order].clone()
                coords[shuffled.eq(0)] = float("nan")
                actual = predict(model, shuffled, CrystalGeometry(geometry.cell, coords))
                actual.update(reconstruct_h1_spectra(actual, [3, 2], [16., 8.]))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)

    def test_bad_global_or_a_execution_settings_are_rejected(self):
        for change in ("norm", "dropout", "freeze", "identity", "recompute", "chunk", "route"):
            model, _, params = self.create()
            attn = model.encoder.global_layers[0].attn
            block = model.encoder.local_layers[0].message.blocks[0]
            if change == "norm":
                model.encoder.global_layers[0].norm_first = True
            elif change == "dropout":
                attn.dropout = .05
            elif change == "freeze":
                attn.in_proj_weight.requires_grad_(False)
            elif change == "recompute":
                block.recompute_angles = True
            elif change == "chunk":
                block.pair_chunk = 4096
            elif change == "route":
                model.encoder.local_layers[0].message.route = "B+"
            else:
                with torch.no_grad():
                    attn.in_proj_bias[0] = .1
            with self.assertRaises(ValueError):
                validate_i3_model(model, params, require_identity=True)

    def test_cli_and_checkpoint_contract_keep_joint_recipe_and_isolate_routes(self):
        cfg = configuration()
        args = build_arg_parser().parse_args(training_command(cfg)[2:])
        self.assertEqual(asdict(cfg), asdict(ExperimentConfig.from_args(args)))
        params = model_params("I3")
        contract = comparison_contract(cfg, params)
        self.assertEqual(contract["schema"], A_SCHEMA)
        self.assertTrue(contract["ablation"]["joint_training"])
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, .5,
                                           model_contract=contract, training_rng=capture_training_rng())
        restore_ablation_checkpoint(payload, model, optimizer, False, model_contract=contract)
        for route in ("I1", "I2", "A", "B+"):
            other = comparison_contract(replace(cfg, local_route=route), {**params, "local_route": route})
            with self.assertRaisesRegex(ValueError, "different model contract"):
                restore_ablation_checkpoint(payload, model, optimizer, False, model_contract=other)
        for changes in ({"seed": 43}, {"epochs": 100}, {"batch_size": 64}, {"local_depth": 3},
                        {"global_depth": 8}, {"local_layout": "alternating"}, {"use_amp": True},
                        {"init_ckpt": "other"}, {"skip_test_eval": False}, {"lambda_ph": 0.}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(cfg, **changes))

    def test_runtime_records_actual_a_execution_settings(self):
        model, _, _ = self.create()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("{}\n")
            entry = write_local_global_execution(path, model, 0)
            self.assertEqual(entry["route"], "I3")
            self.assertEqual(entry["recompute_angles"], [False] * 4)
            self.assertEqual(entry["pair_chunk"], [65536] * 4)

    def test_preflight_rejects_source_reference_recipe_and_rng_drift(self):
        cfg = configuration()
        init = self.create()[1]
        saved = {"cli": {"data_contract": {"manifest": "hash"}}, "runtime": {
            "local_global_init": init["i2_initialization"],
            "structure_training_rng": [{"state_sha256": "rng"}]}}
        report = {"schema": "i1-a-preflight-v1", "passed": True,
                  "code_sha256": {"code": "hash"}, "references": {"I2": "reference"},
                  "data_contract": saved["cli"]["data_contract"],
                  "controls": {"I3": {"configuration": asdict(cfg), "initialization": init,
                                      "training_start_rng_sha256": "rng"}}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "reference_identity", return_value=(saved, None, report["references"])), \
                patch("tools.checks.check_i1_a.code_fingerprints", return_value=report["code_sha256"]):
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(report))
            self.assertEqual(runner.verify_preflight(path, Path(directory), cfg), report)
            for key, value in (("passed", False), ("code_sha256", {}), ("references", {}), ("data_contract", {})):
                path.write_text(json.dumps({**report, key: value}))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), cfg)
            for key in ("training_start_rng_sha256", "configuration"):
                changed = deepcopy(report)
                changed["controls"]["I3"][key] = "different"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), cfg)

    def test_complete_candidate_skips_training_and_analysis_failure_is_reported(self):
        for failure in (False, True):
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(runner.norm, "summarize", return_value=[{"status": "complete"}]), \
                    patch.object(runner.norm, "verify_completed"), \
                    patch.object(runner.norm.subprocess, "Popen") as process, \
                    patch.object(runner, "analyze", side_effect=RuntimeError("analysis failed") if failure else None):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                        runner.execute(configuration(), {}, Path(directory))
                else:
                    runner.execute(configuration(), {}, Path(directory))
                process.assert_not_called()
                queue = json.loads((Path(directory) / "queue.json").read_text())
                self.assertEqual(queue["status"], "failed" if failure else "complete")
                self.assertEqual(queue["arms"], {"I3": "complete"})
                self.assertEqual(queue["schema"], "i1-a-queue-v1")

    def test_training_failure_stops_without_analysis_or_retry(self):
        failed = MagicMock(pid=17)
        failed.wait.return_value = 1
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner.norm, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.norm.subprocess, "Popen", return_value=failed) as start, \
                patch.object(runner, "analyze") as analyze:
            with self.assertRaisesRegex(RuntimeError, "I3 failed"):
                runner.execute(configuration(), {}, Path(directory))
            start.assert_called_once()
            analyze.assert_not_called()
            self.assertEqual(json.loads((Path(directory) / "queue.json").read_text())["status"], "failed")

    def test_analysis_reuses_authenticated_reference_exports_and_infers_i3_only(self):
        ids = ["a", "b", "c", "d", "e"]
        keys = [((14, 1),)] * 3 + [((8, 1),)] * 2
        targets = np.array([[.1, .9], [.4, .6], [.6, .4], [.2, .8], [.8, .2]])
        def save_split(out, label, split):
            strength = {"I3": .8, "I2": .6, "I1": .4}[label]
            prediction = targets * strength + (1. - strength) / 2
            np.savez(out / f"{label}_{split}_spectra.npz", material_ids=np.asarray(ids),
                     tgt_shape_edos=targets, tgt_shape_phdos=targets,
                     pred_shape_edos=prediction, pred_shape_phdos=prediction)
            pd.DataFrame({"material_id": ids}).to_csv(out / f"{label}_{split}_samples.csv", index=False)
            (out / f"{label}_{'train_' if split == 'train' else ''}summary.json").write_text("{}")
            return {(group, task): pair_errors(prediction, targets, keys, ids)
                    for group in ("absolute", "reduced") for task in ("edos", "phdos")}
        def fake_pairs(spectrum, samples, dataset):
            with np.load(spectrum) as values:
                return {(group, task): pair_errors(values["pred_shape_" + task], values["tgt_shape_" + task], keys, ids)
                        for group in ("absolute", "reduced") for task in ("edos", "phdos")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen"
            frozen.mkdir()
            identities = {}
            for label in runner.LABELS:
                run = root / label
                run.mkdir()
                for name in ("config_used.yaml", "checkpoint_best.pth"):
                    (run / name).write_text(label)
                if label != "I3":
                    identities[label] = {"run_dir": str(run)}
                    for split in ("train", "valid"):
                        save_split(frozen, label, split)
            hashes = {name: file_sha256(frozen / name) for name in runner.EXPORT_FILES}
            identities["I2"].update(export_dir=str(frozen), export_files_sha256=hashes)
            info = self.create()[1]
            preflight = {"references": identities, "data_contract": {"data": "hash"},
                         "code_sha256": {"code": "hash"}, "controls": {"I3": {"initialization": info}}}
            saved = {"cli": {"data_contract": preflight["data_contract"]}, "config": {}}
            with patch.object(runner, "reference_identity", return_value=(saved, None, identities)), \
                    patch.object(runner, "verify_completed"), \
                    patch.object(runner, "run_directory", return_value=root / "I3"), \
                    patch.object(runner, "load_run", return_value=(saved, None, None)), \
                    patch.object(runner, "ConfigBuilder"), \
                    patch.object(runner, "export", side_effect=lambda run, samples, summary, **kw:
                                 save_split(samples.parent, "I3", "valid")) as valid_export, \
                    patch.object(runner.bplus, "export_train", side_effect=lambda run, out, label:
                                 save_split(out, label, "train")) as train_export, \
                    patch.object(runner.bplus, "spectrum_pairs", side_effect=fake_pairs), \
                    patch.object(runner.norm, "comparison_report", return_value={"scope": runner.SCOPE}), \
                    patch("tools.checks.check_i1_a.code_fingerprints", return_value=preflight["code_sha256"]):
                report = runner.analyze(configuration(), preflight, root / "analysis")
            valid_export.assert_called_once()
            train_export.assert_called_once()
            self.assertEqual(train_export.call_args.args[2], "I3")
            self.assertTrue(report["analysis_complete"])
            for name, digest in hashes.items():
                self.assertEqual(file_sha256(root / "analysis" / name), digest)
            structure = report["structural_generalization"]
            self.assertEqual(len(structure["summaries"]), 24)
            self.assertEqual(len(list((root / "analysis").glob("*_pairs.csv"))), 24)
            for split in ("train", "valid"):
                entry = structure["summaries"][f"I3_{split}_absolute_edos"]
                self.assertAlmostEqual(entry["error_ratio"], .2)
                self.assertEqual(entry["composition_groups"], 2)
                for label in ("I2", "I1"):
                    contrast = structure["contrasts"][f"I3_minus_{label}"]
                    self.assertEqual(len(contrast), 8)
                    self.assertLess(contrast[f"{split}_absolute_edos"]["ci95"][1], 0)

    def test_analysis_rejects_source_or_reference_drift_before_export(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("tools.checks.check_i1_a.code_fingerprints", return_value={"new": "hash"}), \
                patch.object(runner, "reference_identity") as load:
            with self.assertRaisesRegex(ValueError, "source changed"):
                runner.analyze(configuration(), {"code_sha256": {"old": "hash"}}, Path(directory))
            load.assert_not_called()
        preflight = {"code_sha256": {}, "references": {"I2": {"run_dir": "I2"}}, "data_contract": {}}
        with patch("tools.checks.check_i1_a.code_fingerprints", return_value={}), \
                patch.object(runner, "reference_identity", return_value=({"cli": {"data_contract": {}}}, None, {})), \
                patch.object(runner, "export") as export:
            with self.assertRaisesRegex(ValueError, "references changed"):
                runner.analyze(configuration(), preflight, Path("unused"))
            export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
