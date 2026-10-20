"""Migration must rebuild the two capacity arms without historical artifacts."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from model.transformer import Transformer
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_module import exports as module_exports
from tools.checks import check_local_width_fresh as checker
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval import local_width_fresh_compare as runner
from tools.eval.lg_structure_diagnosis import pair_errors
from tools.eval.local_global_compare import training_command
from tools.train import build_arg_parser
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import _is_local_parameter, comparison_contract, initialize_comparison_model
from utils.local_width_experiment import narrow_reference_parameters
from utils.paths import file_sha256
from utils.zproj_alignment import state_hash


class TestFreshWidth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.controls = runner.comparison_configs("unit_fresh")
        wide = {**model_params("W1"), "local_scalar_channels": 128, "local_hidden_channels": 256}
        cls.params = {"I3": narrow_reference_parameters(wide), "W1": wide}
        cls.infos, cls.common = {}, {}
        with patch("torch.load", side_effect=AssertionError("fresh initialization loaded a checkpoint")):
            for route, params in cls.params.items():
                model = Transformer(**params)
                cls.infos[route] = initialize_comparison_model(model, params, 42)
                cls.common[route] = state_hash({key: value for key, value in model.state_dict().items()
                                               if not _is_local_parameter(key)})

    def preflight(self):
        report = {"schema": runner.PREFLIGHT_SCHEMA, "passed": True, "code_sha256": {},
            "runtime": {"gpu": "unit"}, "data_contract": {}, "learning_rate_35": learning_rate_trace(35, 5e-5),
            "controls": {}}
        for cfg in self.controls:
            route = cfg.local_route
            assembled = {"model": {"params": {"sub_model": {"transformer": self.params[route]}}}}
            report["controls"][route] = {"configuration": asdict(cfg), "assembled_configuration": assembled,
                "initialization": self.infos[route], "matched_nonlocal_state_sha256": self.common[route],
                "contract": comparison_contract(cfg, self.params[route]),
                "data_order": {"train_samples": 18704, "valid_samples": 2313}, "training_start_rng_sha256": "rng",
                "forward_backward": {"all_parameter_gradients_finite": True}, "evaluation_finite": True,
                "symmetry": {"torch.float32": {}, "torch.float64": {}},
                "checkpoint_roundtrip": {"weights_optimizer_rng_equal": True, "foreign_contracts_rejected": True},
                "other_width_checkpoint_rejected": True, "costs": [{}, {}, {}]}
        return deepcopy(report)

    def test_frozen_two_commands_and_assembly_need_no_reference_artifacts(self):
        configs = runner.comparison_configs("migr_unit")
        self.assertEqual([cfg.local_route for cfg in configs], ["I3", "W1"])
        for cfg in configs:
            parsed = ExperimentConfig.from_args(build_arg_parser().parse_args(training_command(cfg)[2:]))
            self.assertEqual(parsed, cfg)
            self.assertEqual((cfg.seed, cfg.epochs, cfg.batch_size, cfg.selection_metric),
                             (42, 35, 32, "balanced_blind_r2"))
            self.assertTrue(cfg.skip_test_eval)
            self.assertFalse(cfg.init_ckpt)
        with patch("tools.eval.local_width_compare.reference_identity", side_effect=AssertionError("old reference")), \
                patch("tools.eval.local_module_compare.reference_identity", side_effect=AssertionError("old reference")), \
                patch("sys.argv", ["fresh", "--tag-prefix", "unit_cli"]), redirect_stdout(io.StringIO()) as output:
            runner.main()
        self.assertEqual(len(output.getvalue().strip().splitlines()), 2)
        actual = [captured_config(cfg) for cfg in configs]
        self.assertEqual(runner.restore_narrow_assembly(actual[1]), actual[0])
        for value in ("", "../escape", "two words"):
            with self.assertRaises(ValueError):
                runner.comparison_configs(value)

    def test_reconstructed_initialization_matches_without_loading_weights(self):
        self.assertEqual(self.infos["W1"]["narrow_initialization"], self.infos["I3"])
        self.assertEqual(self.common["I3"], self.common["W1"])
        self.assertEqual(self.common["I3"], self.infos["W1"]["matched_nonlocal_state_sha256"])

    def test_preflight_rejects_old_protocol_and_drift_in_each_bound_identity(self):
        report = self.preflight()
        with patch.object(checker, "code_fingerprints", return_value={}), \
                patch.object(runner, "runtime_identity", return_value=report["runtime"]), \
                patch.object(runner, "data_identity", return_value={}):
            self.assertEqual(runner.validate_preflight(report, self.controls), report)
            for key in ("schema", "runtime", "code_sha256", "data_contract", "learning_rate_35"):
                changed = deepcopy(report)
                changed[key] = "different"
                with self.assertRaises(ValueError):
                    runner.validate_preflight(changed, self.controls)
            for key in ("contract", "configuration", "other_width_checkpoint_rejected", "checkpoint_roundtrip",
                        "symmetry", "evaluation_finite", "forward_backward", "costs"):
                changed = deepcopy(report)
                changed["controls"]["W1"][key] = False if key.endswith("rejected") else {}
                with self.assertRaises(ValueError):
                    runner.validate_preflight(changed, self.controls)
            for key in ("initialization", "matched_nonlocal_state_sha256", "training_start_rng_sha256", "data_order"):
                changed = deepcopy(report)
                changed["controls"]["W1"][key] = {}
                with self.assertRaises(ValueError):
                    runner.validate_preflight(changed, self.controls)

    def test_missing_data_manifest_is_rejected_instead_of_disabling_the_contract(self):
        with patch.object(runner, "crystal_data_contract", return_value=None):
            with self.assertRaisesRegex(ValueError, "complete crystal_q1_v1"):
                runner.data_identity(self.controls[0])

    def test_report_answers_capacity_only_and_enforces_both_task_protections(self):
        values = module_exports()
        data = {"W1": values["I3"], "A64": values["I2"]}
        report = runner.comparison_report(data, draws=20)
        self.assertTrue(report["decisions"]["width_upgrade_conditions_met"])
        self.assertEqual(set(report["runs"]), {"W1", "A64"})
        self.assertEqual(set(report["joint_blind"]), {"W1_minus_A64"})
        self.assertIn("not evaluated", report["decisions"]["I1_baseline_adoption"])
        self.assertFalse(report["decisions"]["automatic_followup_training"])
        bad = deepcopy(data)
        bad["W1"][1].loc[bad["W1"][1].index[:2], "blind_r2_phdos"] = -.1
        self.assertFalse(runner.comparison_report(bad, draws=20)["decisions"]["width_upgrade_conditions_met"])

    def test_partial_second_arm_stops_before_training_the_first(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "validate_preflight"), \
                patch.object(runner, "summarize", return_value=[{"status": "pending"}, {"status": "partial"}]), \
                patch.object(runner.queue, "run_directory", return_value=Path(directory) / "empty"), \
                patch.object(runner.queue, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "automatic resume"):
                runner.execute(self.controls, {}, Path(directory))
            execute.assert_not_called()

    def test_failed_process_stops_queue_and_completed_pair_is_not_retrained(self):
        for complete in (False, True):
            rows = [{"status": "complete" if complete else "pending"}] * 2
            process = MagicMock(pid=17)
            process.wait.return_value = 1
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(runner, "validate_preflight"), patch.object(runner, "summarize", return_value=rows), \
                    patch.object(runner.queue, "summarize", return_value=rows), \
                    patch.object(runner.queue, "run_directory", return_value=Path(directory) / "empty"), \
                    patch.object(runner, "verify_completed"), patch.object(runner.queue, "verify_completed"), \
                    patch.object(runner.queue.subprocess, "Popen", return_value=process) as start, \
                    patch.object(runner, "analyze") as analyze:
                if complete:
                    runner.execute(self.controls, {}, Path(directory))
                    start.assert_not_called()
                    analyze.assert_called_once()
                else:
                    with self.assertRaisesRegex(RuntimeError, "I3 failed"):
                        runner.execute(self.controls, {}, Path(directory))
                    start.assert_called_once()
                    analyze.assert_not_called()
                queue = json.loads((Path(directory) / "queue.json").read_text())
                self.assertEqual(queue["schema"], "local-width-fresh-queue-v1")
                self.assertEqual(queue["status"], "complete" if complete else "failed")

    def test_complete_recipe_is_checked_against_assembled_preflight(self):
        cfg = self.controls[0]
        preflight = self.preflight()
        with patch.object(runner.queue, "verify_completed"), \
                patch.object(runner, "load_run", return_value=({"config": {"different": True}}, {}, {})):
            with self.assertRaisesRegex(ValueError, "assembled preflight"):
                runner.verify_completed(cfg, preflight)

    def test_analysis_infers_both_new_runs_and_never_reports_i1_adoption(self):
        ids = ["a", "b", "c", "d", "e"]
        keys = [((14, 1),)] * 3 + [((8, 1),)] * 2
        targets = np.array([[.1, .9], [.4, .6], [.6, .4], [.2, .8], [.8, .2]])

        def save_split(out, label, split):
            pred = targets * .8 + .1
            np.savez(out / f"{label}_{split}_spectra.npz", material_ids=np.asarray(ids),
                tgt_shape_edos=targets, tgt_shape_phdos=targets, pred_shape_edos=pred, pred_shape_phdos=pred)
            pd.DataFrame({"material_id": ids}).to_csv(out / f"{label}_{split}_samples.csv", index=False)
            (out / f"{label}_{'train_' if split == 'train' else ''}summary.json").write_text("{}")
            return {(group, task): pair_errors(pred, targets, keys, ids)
                    for group in ("absolute", "reduced") for task in ("edos", "phdos")}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {cfg.local_route: root / cfg.local_route for cfg in self.controls}
            histories = {}
            for cfg in self.controls:
                runs[cfg.local_route].mkdir()
                for name in ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth"):
                    (runs[cfg.local_route] / name).write_text(cfg.local_route)
                path = root / (cfg.local_route + "_history.csv")
                path.write_text("complete")
                histories[f"results/reports/history_m1{cfg.tag}.csv"] = path
            builder = MagicMock()
            builder.get_dataset.return_value.__len__.return_value = 2313
            saved = {"config": {}}
            preflight = {"code_sha256": {}, "runtime": {}, "data_contract": {},
                "controls": {cfg.local_route: {"initialization": {}} for cfg in self.controls}}
            tables = save_split(root, "fixture", "valid")
            with patch.object(runner, "validate_preflight"), patch.object(runner, "verify_completed"), \
                    patch.object(runner.queue, "run_directory", side_effect=lambda cfg: runs[cfg.local_route]), \
                    patch.object(runner, "repository_path", side_effect=lambda name: histories[name]), \
                    patch.object(runner, "load_run", return_value=(saved, None, None)), \
                    patch.object(runner, "ConfigBuilder", return_value=builder), \
                    patch.object(runner, "export", side_effect=lambda run, samples, summary, **kw:
                        save_split(samples.parent, runner.ROUTE_LABELS[run.name], "valid")) as valid_export, \
                    patch.object(runner.bplus, "export_train", side_effect=lambda run, out, label:
                        save_split(out, label, "train")) as train_export, \
                    patch.object(runner.bplus, "spectrum_pairs", return_value=tables), \
                    patch.object(runner, "load", return_value=({}, None)), \
                    patch.object(runner, "comparison_report", return_value={}):
                before = {cfg.local_route: file_sha256(runs[cfg.local_route] / "checkpoint_best.pth") for cfg in self.controls}
                report = runner.analyze(self.controls, preflight, root / "analysis")
            self.assertEqual(valid_export.call_count, 2)
            self.assertEqual(train_export.call_count, 2)
            self.assertEqual([call.args[2] for call in train_export.call_args_list], ["A64", "W1"])
            self.assertEqual(set(report["run_artifacts"]), {"A64", "W1"})
            self.assertFalse(report["historical_artifacts_required"])
            self.assertTrue(report["structural_diagnosis"]["diagnostic_only"])
            self.assertEqual(len(report["structural_diagnosis"]["summaries"]), 16)
            self.assertEqual(len(list((root / "analysis").glob("*_pairs.csv"))), 16)
            for cfg in self.controls:
                self.assertEqual(file_sha256(runs[cfg.local_route] / "checkpoint_best.pth"), before[cfg.local_route])


if __name__ == "__main__":
    unittest.main()
