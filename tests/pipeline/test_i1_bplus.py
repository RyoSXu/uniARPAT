"""Corrected-backbone B+ refinement preserves the I1 comparison contract."""
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
from tools.eval import i1_bplus_compare as runner
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.i1_bplus_experiment import BPLUS_SCHEMA, i1_bplus_config, validate_i2_model
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, deep_reference_state,
    initialize_comparison_model, validate_local_global_config,
)
from utils.local_structure_experiment import FROZEN, is_shared_control_parameter
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import rng_digest, rng_snapshot


def configuration():
    return i1_bplus_config(ExperimentConfig(**{**FROZEN, "local_route": "I1"}), "unit_bplus")


class TestI1Bplus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, route="I2", dtype=torch.float32):
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

    def test_common_values_match_i1_and_content_matches_untrained_bplus_without_aliases(self):
        reference, reference_info, _ = self.create("I1")
        model, info, params = self.create()
        bplus, _ = deep_reference_state(42, {**params, "local_route": "B+"})
        self.assertEqual(info["i1_initialization"], reference_info)
        for name, value in model.state_dict().items():
            source = reference.state_dict()[name] if is_shared_control_parameter(name) else bplus[name]
            torch.testing.assert_close(value, source, atol=0, rtol=0)
            if value.numel():
                self.assertNotEqual(value.data_ptr(), source.data_ptr())
        self.assertEqual(info["parameter_counts"]["shared"], info["reference_parameter_counts"]["shared"])
        self.assertGreater(info["parameter_counts"]["local_content"], info["reference_parameter_counts"]["local_content"])
        self.assertEqual(info["execution_order"], reference_info["execution_order"])
        self.assertEqual(info["update_counts"], reference_info["update_counts"])

    def test_full_bplus_and_corrected_global_settings_are_active_and_trainable(self):
        model, _, params = self.create()
        validate_i2_model(model, params, require_identity=True)
        self.assertEqual(model.encoder.geometry_route, "B+")
        for layer in model.encoder.local_layers:
            self.assertIsInstance(layer.message, PeriodicLocalMessage)
            self.assertTrue(layer.message.include_high_order)
        for layer in model.encoder.global_layers:
            self.assertFalse(layer.norm_first)
            self.assertEqual(layer.attn.dropout, 0)
            self.assertEqual(layer.dropout_attn.p, .05)
            self.assertEqual(layer.dropout_ffn.p, .05)
            self.assertEqual(layer.ffn[2].p, .05)
        for dtype in (torch.float32, torch.float64):
            model, _, _ = self.create(dtype=dtype)
            model.train()
            z, geometry = fixture(dtype)
            with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring")):
                values = predict(model, z, geometry)
                sum(value.square().mean() for value in values.values()).backward()
            for name, parameter in model.named_parameters():
                self.assertTrue(parameter.requires_grad, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_both_spectra_preserve_e3_and_atom_permutation_with_nan_padding(self):
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

    def test_bad_global_settings_and_frozen_projections_are_rejected(self):
        for change in ("norm", "dropout", "freeze", "identity", "high_order"):
            model, _, params = self.create()
            attn = model.encoder.global_layers[0].attn
            if change == "norm":
                model.encoder.global_layers[0].norm_first = True
            elif change == "dropout":
                attn.dropout = .05
            elif change == "freeze":
                attn.in_proj_weight.requires_grad_(False)
            elif change == "high_order":
                model.encoder.local_layers[0].message.include_high_order = False
            else:
                with torch.no_grad():
                    attn.in_proj_bias[0] = .1
            with self.assertRaises(ValueError):
                validate_i2_model(model, params, require_identity=True)

    def test_cli_recipe_and_checkpoint_contract_isolate_i2_from_i1_and_old_bplus(self):
        cfg = configuration()
        args = build_arg_parser().parse_args(training_command(cfg)[2:])
        self.assertEqual(asdict(cfg), asdict(ExperimentConfig.from_args(args)))
        params = model_params("I2")
        contract = comparison_contract(cfg, params)
        self.assertEqual(contract["schema"], BPLUS_SCHEMA)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, .5,
                                           model_contract=contract, training_rng=capture_training_rng())
        restore_ablation_checkpoint(payload, model, optimizer, False, model_contract=contract)
        for route in ("I1", "B+"):
            other = comparison_contract(replace(cfg, local_route=route), {**params, "local_route": route})
            with self.assertRaisesRegex(ValueError, "different model contract"):
                restore_ablation_checkpoint(payload, model, optimizer, False, model_contract=other)
        for changes in ({"seed": 43}, {"epochs": 100}, {"batch_size": 64}, {"local_depth": 3},
                        {"global_depth": 8}, {"local_layout": "alternating"}, {"use_amp": True},
                        {"init_ckpt": "other"}, {"skip_test_eval": False}, {"lambda_ph": 0.}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(cfg, **changes))

    def test_group_error_ratio_weights_compositions_and_separates_near_zero_targets(self):
        frame = pd.DataFrame({"left_id": ["a", "b", "d", "f"], "right_id": ["b", "c", "e", "g"],
            "composition": ["X", "X", "Y", "Z"], "target_tv": [1., 1., 2., 1e-8],
            "predicted_tv": [1., 1., 2., .5], "difference_error": [.2, .2, 1., .5],
            "skill": [.8, .8, .5, -1e7], "direction_cosine": [1., 1., 1., 0.]})
        summary, filtered = runner.group_summary(frame, "edos")
        self.assertEqual(summary["near_zero_target_pairs"], 1)
        self.assertEqual(summary["composition_groups"], 2)
        self.assertAlmostEqual(summary["group_mean_difference_error"], .6)
        self.assertAlmostEqual(summary["zero_response_group_mean_error"], 1.5)
        self.assertAlmostEqual(summary["error_ratio"], .4)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(summary["model_minus_zero_response"]["groups"], 2)
        self.assertAlmostEqual(summary["model_minus_zero_response"]["mean_group_error_delta"], -.9)

    def test_spectrum_pairing_rejects_targets_or_ids_that_differ(self):
        data = {"material_ids": np.array(["a", "b"]), "tgt_shape_edos": np.ones((2, 3)),
                "tgt_shape_phdos": np.ones((2, 2))}
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left.npz", Path(directory) / "right.npz"
            np.savez(left, **data)
            np.savez(right, **data)
            runner.assert_paired_targets(left, right)
            for key in data:
                changed = deepcopy(data)
                changed[key] = changed[key][::-1] if key == "material_ids" else changed[key] * .9
                np.savez(right, **changed)
                with self.assertRaisesRegex(ValueError, "targets/IDs"):
                    runner.assert_paired_targets(left, right)

    def test_preflight_rejects_drift_in_source_reference_config_and_training_start(self):
        cfg = configuration()
        init = self.create()[1]
        saved = {"cli": {"data_contract": {"manifest": "hash"}}, "runtime": {
            "local_global_init": init["i1_initialization"],
            "structure_training_rng": [{"state_sha256": "rng"}]}}
        report = {"schema": "i1-bplus-preflight-v1", "passed": True,
                  "code_sha256": {"code": "hash"}, "references": {"I1": "reference"},
                  "data_contract": saved["cli"]["data_contract"],
                  "controls": {"I2": {"configuration": asdict(cfg), "initialization": init,
                                      "training_start_rng_sha256": "rng"}}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "reference_identity", return_value=(saved, None, report["references"])), \
                patch("tools.checks.check_i1_bplus.code_fingerprints", return_value=report["code_sha256"]):
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(report))
            self.assertEqual(runner.verify_preflight(path, Path(directory), cfg), report)
            for key, value in (("passed", False), ("code_sha256", {}), ("references", {}), ("data_contract", {})):
                path.write_text(json.dumps({**report, key: value}))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), cfg)
            changed = deepcopy(report)
            changed["controls"]["I2"]["training_start_rng_sha256"] = "different"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                runner.verify_preflight(path, Path(directory), cfg)

    def test_complete_candidate_is_not_retrained_and_analysis_failure_keeps_queue_failed(self):
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
                self.assertEqual(queue["arms"], {"I2": "complete"})
                self.assertEqual(queue["schema"], "i1-bplus-queue-v1")

    def test_training_failure_stops_without_analysis_or_automatic_retry(self):
        failed = MagicMock(pid=17)
        failed.wait.return_value = 1
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner.norm, "summarize", return_value=[{"status": "pending"}]), \
                patch.object(runner.norm.subprocess, "Popen", return_value=failed) as start, \
                patch.object(runner, "analyze") as analyze:
            with self.assertRaisesRegex(RuntimeError, "I2 failed"):
                runner.execute(configuration(), {}, Path(directory))
            start.assert_called_once()
            analyze.assert_not_called()
            queue = json.loads((Path(directory) / "queue.json").read_text())
            self.assertEqual(queue["status"], "failed")

    def test_analysis_reports_both_splits_and_zero_response_with_group_weighting(self):
        ids = ["a", "b", "c", "d", "e"]
        keys = [((14, 1),)] * 3 + [((8, 1),)] * 2
        targets = np.array([[.1, .9], [.4, .6], [.6, .4], [.2, .8], [.8, .2]])
        def save_split(out, label, split):
            prediction = (targets * .8 + .1) if label == "I2" else (targets * .4 + .3)
            np.savez(out / f"{label}_{split}_spectra.npz", material_ids=np.asarray(ids),
                     tgt_shape_edos=targets, tgt_shape_phdos=targets,
                     pred_shape_edos=prediction, pred_shape_phdos=prediction)
            pd.DataFrame({"material_id": ids}).to_csv(out / f"{label}_{split}_samples.csv", index=False)
            result = {}
            for grouping in ("absolute", "reduced"):
                for task in ("edos", "phdos"):
                    frame = runner.pair_errors(prediction, targets, keys, ids)
                    frame.to_csv(out / f"{label}_{split}_{grouping}_{task}_pairs.csv", index=False)
                    result[grouping, task] = frame
            return result
        def fake_valid(cfg, preflight, out, **kwargs):
            out.mkdir()
            for label in ("I2", "I1"):
                save_split(out, label, "valid")
            return {"scope": runner.SCOPE}
        def fake_train(run, out, label):
            return save_split(out, label, "train")
        with tempfile.TemporaryDirectory() as directory:
            identities = {"I1": {"run_dir": str(Path(directory) / "I1")}}
            info = self.create()[1]
            preflight = {"references": identities, "data_contract": {"data": "hash"},
                         "code_sha256": {"code": "hash"}, "controls": {"I2": {"initialization": info}}}
            saved = {"cli": {"data_contract": preflight["data_contract"]}}
            with patch.object(runner, "reference_identity", return_value=(saved, None, identities)), \
                    patch.object(runner.norm, "analyze", side_effect=fake_valid), \
                    patch.object(runner, "export_train", side_effect=fake_train), \
                    patch("tools.checks.check_i1_bplus.code_fingerprints", return_value=preflight["code_sha256"]):
                report = runner.analyze(configuration(), preflight, Path(directory) / "analysis")
            self.assertTrue(report["analysis_complete"])
            structure = report["structural_generalization"]
            self.assertEqual(structure["primary"], "valid_absolute_edos")
            self.assertEqual(len(structure["I2_minus_I1"]), 8)
            for split in ("train", "valid"):
                entry = structure["summaries"][f"I2_{split}_absolute_edos"]
                self.assertAlmostEqual(entry["error_ratio"], .2)
                self.assertEqual(entry["composition_groups"], 2)
                self.assertLess(entry["model_minus_zero_response"]["ci95"][1], 0)
                self.assertLess(structure["I2_minus_I1"][f"{split}_absolute_edos"]["ci95"][1], 0)

    def test_analysis_rejects_changed_source_before_loading_references(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("tools.checks.check_i1_bplus.code_fingerprints", return_value={"new": "hash"}), \
                patch.object(runner, "reference_identity") as load:
            with self.assertRaisesRegex(ValueError, "source changed"):
                runner.analyze(configuration(), {"code_sha256": {"old": "hash"}}, Path(directory))
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
