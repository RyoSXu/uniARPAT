"""Matched A/B+ selection must preserve provenance, checkpoints and decisions."""
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

from tests.pipeline.test_global_norm import model_params
from tools.checks import check_local_module as checker
from tools.checks.check_long_budget import learning_rate_trace
from tools.eval import local_module_compare as runner
from tools.eval.lg_structure_diagnosis import pair_errors
from tools.eval.local_global_compare import training_command
from tools.train import build_arg_parser
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.checkpoint_selection import BLIND_R2_METRIC, selection_contract
from utils.experiment_config import ExperimentConfig
from utils.i1_a_experiment import i1_a_config
from utils.local_global_experiment import capture_training_rng, comparison_contract
from utils.local_structure_experiment import FROZEN
from utils.paths import file_sha256


def configuration():
    return runner.candidate_config(ExperimentConfig(**{
        **FROZEN, "local_route": "I2", "selection_metric": BLIND_R2_METRIC}), "unit_local")


def exports(e_values=None, p_values=None):
    result = {}
    for label, value in (("I1", .3), ("I2", .4), ("I3", .5)):
        e = np.full(100, value) if e_values is None else np.asarray(e_values[label])
        p = np.full(100, value) if p_values is None else np.asarray(p_values[label])
        frame = pd.DataFrame({f"{mode}_{metric}_{task}":
            (values if metric == "r2" else np.full(len(values), .1))
            for mode in ("blind", "oracle") for task, values in (("edos", e), ("phdos", p))
            for metric in ("r2", "mae")}, index=[f"m{i}" for i in range(len(e))])
        summary = {"tag": label, "best_epoch": 29, "best_score": value, "epochs": 35,
                   "contract_schema": "unit", "init_protocol": "unit", "epoch_time_s_median": 1.,
                   "epoch_time_s_sum": 35., "peak_vram_mb_median": 1., "peak_vram_mb_max": 1.,
                   "selection_metric": BLIND_R2_METRIC, **selection_contract(BLIND_R2_METRIC)}
        result[label] = summary, frame
    return result


class TestLocalModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_single_a_command_preserves_recipe_and_historical_factory(self):
        cfg = configuration()
        parsed = ExperimentConfig.from_args(build_arg_parser().parse_args(training_command(cfg)[2:]))
        self.assertEqual(asdict(parsed), asdict(cfg))
        self.assertEqual((cfg.local_route, cfg.seed, cfg.epochs, cfg.batch_size), ("I3", 42, 35, 32))
        self.assertEqual(cfg.selection_metric, BLIND_R2_METRIC)
        self.assertTrue(cfg.skip_test_eval)
        old = i1_a_config(ExperimentConfig(**{**FROZEN, "local_route": "I2"}), "unit_local")
        self.assertEqual(old.selection_metric, "balanced_oracle_mae")
        self.assertNotEqual(old.tag, cfg.tag)
        for changes in ({"seed": 43}, {"epochs": 36}, {"use_amp": True},
                        {"selection_metric": "balanced_oracle_mae"}, {"local_route": "I1"}):
            with self.assertRaises(ValueError):
                runner.candidate_config(replace(cfg, **{"local_route": "I2", **changes}), "unit")

    def test_new_a_rejects_oracle_a_and_bplus_checkpoint_contracts(self):
        cfg = configuration()
        params = model_params("I3")
        contract = comparison_contract(cfg, params)
        self.assertEqual(contract["checkpoint_selection"]["mode"], "max")
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, .5,
                                           model_contract=contract, training_rng=capture_training_rng())
        restore_ablation_checkpoint(payload, model, optimizer, False, model_contract=contract)
        for alternate, alternate_params in (
            (replace(cfg, selection_metric="balanced_oracle_mae"), params),
            (replace(cfg, local_route="I2"), {**params, "local_route": "I2"})):
            with self.assertRaisesRegex(ValueError, "different model contract"):
                restore_ablation_checkpoint(payload, model, optimizer, False,
                    model_contract=comparison_contract(alternate, alternate_params))

    def test_route_choice_and_baseline_adoption_are_separate_and_symmetric(self):
        report = runner.comparison_report(exports(), draws=20)
        self.assertEqual(report["decisions"]["route_reference"], "A")
        self.assertTrue(report["decisions"]["A_vs_I1_numerical_adoption_conditions_met"])
        self.assertFalse(report["decisions"]["automatic_baseline_replacement"])
        self.assertEqual(report["bootstrap"]["seed"], runner.BOOTSTRAP_SEED)
        for item in report["joint_blind"].values():
            self.assertTrue(item["bootstrap"]["same_indices_for_tasks_and_models"])
        values = {label: np.full(100, value) for label, value in (("I1", .3), ("I2", .6), ("I3", .5))}
        reverse = runner.comparison_report(exports(values, values), draws=20)
        self.assertEqual(reverse["decisions"]["route_reference"], "B+")
        self.assertTrue(reverse["decisions"]["A_vs_I1_numerical_adoption_conditions_met"])

    def test_failure_rate_blocks_score_winner_and_mixed_selection_is_ineligible(self):
        e = {label: np.full(100, value) for label, value in (("I1", .3), ("I2", .4), ("I3", .5))}
        e["I3"][:2] = -.1
        report = runner.comparison_report(exports(e_values=e), draws=20)
        self.assertGreater(report["joint_blind"]["I3_minus_I2"]["ci95"][0], 0)
        self.assertEqual(report["decisions"]["route_reference"], "undetermined")
        self.assertFalse(report["decisions"]["A_vs_I1_numerical_adoption_conditions_met"])
        mixed = exports()
        mixed["I2"][0]["selection_metric"] = "balanced_oracle_mae"
        mixed_report = runner.comparison_report(mixed, draws=20)
        self.assertEqual(mixed_report["decisions"]["route_reference"], "undetermined")

    def test_no_detected_route_difference_is_not_equivalence_or_automatic_followup(self):
        values = {label: np.full(100, .4) for label in ("I1", "I2", "I3")}
        report = runner.comparison_report(exports(values, values), draws=20)
        self.assertEqual(report["decisions"]["route_reference"], "undetermined")
        self.assertFalse(report["decisions"]["automatic_followup_training"])

    def test_shared_report_default_statistical_seed_is_preserved(self):
        report = runner.norm.comparison_report(exports(), draws=20, labels=runner.LABELS)
        self.assertEqual(report["bootstrap"]["seed"], 20261013)

    def test_checker_supplies_new_protocol_without_reimplementing_a(self):
        with patch.object(checker, "check_a") as check:
            checker.check(Path("new.json"), Path("evidence"), "unit")
        self.assertIs(check.call_args.kwargs["identity_loader"], runner.reference_identity)
        self.assertIs(check.call_args.kwargs["configuration_factory"], runner.candidate_config)
        self.assertIs(check.call_args.kwargs["permutation_probe"], checker.permutation_case)
        self.assertEqual(check.call_args.kwargs["report_schema"], runner.PREFLIGHT_SCHEMA)

    def test_preflight_rejects_source_reference_contract_and_rng_drift(self):
        cfg = configuration()
        assembled = {"model": {"params": {"sub_model": {"transformer": model_params("I3")}}}}
        reference = deepcopy(assembled)
        reference["model"]["params"]["sub_model"]["transformer"]["local_route"] = "I2"
        saved = {"config": reference, "cli": {"data_contract": {"data": "hash"}}, "runtime": {
            "local_global_init": {"state": "init"}, "structure_training_rng": [{"state_sha256": "rng"}]}}
        identities = {"I2": {"data_order": {"samples": "same"}}}
        entry = {"configuration": asdict(cfg), "assembled_configuration": assembled,
                 "contract": comparison_contract(cfg, assembled["model"]["params"]["sub_model"]["transformer"]),
                 "initialization": {"i2_initialization": saved["runtime"]["local_global_init"]},
                 "training_start_rng_sha256": "rng", "data_order": identities["I2"]["data_order"],
                 "checkpoint_roundtrip": True, "i2_checkpoint_rejected": True, "selection_checkpoint_rejected": True,
                 "permutation_invariance": {"torch.float32": {}, "torch.float64": {}}}
        report = {"schema": runner.PREFLIGHT_SCHEMA, "passed": True, "references": identities,
                  "code_sha256": {"code": "hash"}, "data_contract": saved["cli"]["data_contract"],
                  "learning_rate_35": learning_rate_trace(35, cfg.lr), "controls": {"I3": entry}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "reference_identity", return_value=(saved, None, identities)), \
                patch.object(checker, "code_fingerprints", return_value=report["code_sha256"]):
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(report))
            self.assertEqual(runner.verify_preflight(path, Path(directory), cfg), report)
            for key in ("code_sha256", "references", "data_contract", "learning_rate_35"):
                changed = deepcopy(report)
                changed[key] = {}
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), cfg)
            for key in ("training_start_rng_sha256", "contract", "selection_checkpoint_rejected", "permutation_invariance"):
                changed = deepcopy(report)
                changed["controls"]["I3"][key] = {} if key != "selection_checkpoint_rejected" else False
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), cfg)

    def reference_fixture(self, root):
        evidence = root / "evidence"
        (evidence / "analysis").mkdir(parents=True)
        data = root / "data"
        files = {}
        for split in ("train", "valid"):
            path = data / split / "payload.npy"
            path.parent.mkdir(parents=True)
            path.write_text(split)
            files[f"{split}/payload.npy"] = {"sha256": file_sha256(path)}
        files["test/never-open.npy"] = {"sha256": "not-read"}
        (data / "manifest.json").write_text(json.dumps({"schema": "unit", "files": files}))
        data_contract = {"schema": "unit", "manifest_sha256": file_sha256(data / "manifest.json")}
        pre = {"schema": "local-readout-preflight-v1", "passed": True,
               "code_sha256": {"historical": "immutable"}, "data_contract": data_contract, "controls": {}}
        report = {"schema": "local-readout-analysis-v1", "analysis_complete": True,
                  "code_sha256": pre["code_sha256"], "data_contract": data_contract, "run_artifacts": {}}
        saved_runs, histories = {}, {}
        for label in ("I1", "I2"):
            cfg = ExperimentConfig(**{**FROZEN, "local_route": label, "selection_metric": BLIND_R2_METRIC,
                                      "tag": "_reference_" + label.lower()})
            run = root / ("ablation_m1" + cfg.tag)
            run.mkdir()
            for name in ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth"):
                (run / name).write_text(label + name)
            params = model_params(label)
            contract = comparison_contract(cfg, params)
            init = {"state": "i1"} if label == "I1" else {"state": "i2", "i1_initialization": {"state": "i1"}}
            saved = {"config": {"route": label}, "cli": {"data_contract": data_contract}, "runtime": {
                "local_global_init": init, "structure_training_rng": [{"state_sha256": "rng"}]}}
            entry = {"configuration": asdict(cfg), "assembled_configuration": saved["config"],
                     "initialization": init, "contract": contract, "data_order": {"sampler": "same"},
                     "training_start_rng_sha256": "rng"}
            pre["controls"][label] = entry
            history = root / (label + "_history.csv")
            history.write_text("complete")
            histories[f"results/reports/history_m1{cfg.tag}.csv"] = history
            saved_runs[str(run)] = (saved, params, contract)
            report["run_artifacts"][label] = {"run_dir": str(run), **{
                name: file_sha256(run / name) for name in
                ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")},
                "history_path": str(history), "history_sha256": file_sha256(history)}
        for name in runner.historical_a.EXPORT_FILES:
            (evidence / "analysis" / name).write_text(name)
        (evidence / "preflight_final.json").write_text(json.dumps(pre))
        (evidence / "analysis/report.json").write_text(json.dumps(report))
        verified = {"passed": True, "protocol_and_completed_checkpoints_passed": True,
                    "preflight_sha256": file_sha256(evidence / "preflight_final.json"),
                    "input_artifacts_sha256": {name: file_sha256(evidence / "analysis" / name) for name in
                                               (*runner.historical_a.EXPORT_FILES, "report.json")}}
        (evidence / "analysis/verification_codex.json").write_text(json.dumps(verified))
        (evidence / "queue.json").write_text(json.dumps({"schema": "local-readout-queue-v1", "status": "complete",
            "arms": {"I1": "complete", "I2": "complete", "I4": "complete"}}))
        return evidence, data, saved_runs, histories

    def test_reference_authentication_preserves_old_source_and_detects_export_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence, data, runs, histories = self.reference_fixture(Path(directory))
            with patch.object(runner, "load_run", side_effect=lambda p: runs[str(p)]), \
                    patch.object(runner, "verify_completed") as completed, \
                    patch.object(runner, "repository_path", side_effect=lambda p: histories[p]), \
                    patch.object(runner, "input_path", return_value=data):
                _, cfg, identities = runner.reference_identity(evidence)
                self.assertEqual(cfg.local_route, "I2")
                self.assertEqual(completed.call_count, 2)
                self.assertEqual(set(identities), {"I1", "I2"})
                self.assertEqual(len(identities["I2"]["export_files_sha256"]), 12)
                (evidence / "analysis/I2_valid_spectra.npz").write_text("changed")
                with self.assertRaisesRegex(ValueError, "exports differ"):
                    runner.reference_identity(evidence)

    def test_analysis_rejects_source_drift_before_export(self):
        with patch.object(checker, "code_fingerprints", return_value={"new": "hash"}), \
                patch.object(runner, "reference_identity") as reference, patch.object(runner, "export") as export:
            with self.assertRaisesRegex(ValueError, "source changed"):
                runner.analyze(configuration(), {"code_sha256": {}}, Path("unused"))
            reference.assert_not_called()
            export.assert_not_called()

    def test_analysis_infers_only_a_and_preserves_reference_exports(self):
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
                for name in ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth"):
                    (run / name).write_text(label)
                if label != "I3":
                    identities[label] = {"run_dir": str(run)}
                    for split in ("train", "valid"):
                        save_split(frozen, label, split)
            hashes = {name: file_sha256(frozen / name) for name in runner.historical_a.EXPORT_FILES}
            identities["I2"].update(evidence_dir=str(frozen), export_dir=str(frozen), export_files_sha256=hashes)
            preflight = {"references": identities, "data_contract": {"data": "hash"}, "code_sha256": {},
                         "controls": {"I3": {"initialization": {"parameter_counts": {}, "reference_parameter_counts": {}}}}}
            saved = {"cli": {"data_contract": preflight["data_contract"]}, "config": {}}
            history = root / "history.csv"
            history.write_text("complete")
            builder = MagicMock()
            datasets = {split: MagicMock() for split in ("train", "valid")}
            datasets["train"].__len__.return_value = 18704
            datasets["valid"].__len__.return_value = 2313
            builder.get_dataset.side_effect = lambda **kw: datasets[kw["split"]]
            with patch.object(runner, "reference_identity", return_value=(saved, None, identities)), \
                    patch.object(runner, "verify_completed"), \
                    patch.object(runner, "run_directory", return_value=root / "I3"), \
                    patch.object(runner, "repository_path", return_value=history), \
                    patch.object(runner, "load_run", return_value=(saved, None, None)), \
                    patch.object(runner, "ConfigBuilder", return_value=builder), \
                    patch.object(runner, "export", side_effect=lambda run, samples, summary, **kw:
                                 save_split(samples.parent, "I3", "valid")) as valid_export, \
                    patch.object(runner.bplus, "export_train", side_effect=lambda run, out, label:
                                 save_split(out, label, "train")) as train_export, \
                    patch.object(runner.bplus, "spectrum_pairs", side_effect=fake_pairs), \
                    patch.object(runner, "load", return_value=({}, None)), \
                    patch.object(runner, "comparison_report", return_value={}), \
                    patch.object(checker, "code_fingerprints", return_value={}):
                report = runner.analyze(configuration(), preflight, root / "analysis")
            valid_export.assert_called_once()
            train_export.assert_called_once()
            self.assertEqual(train_export.call_args.args[2], "I3")
            self.assertEqual([call.kwargs["split"] for call in builder.get_dataset.call_args_list], ["train", "valid"])
            self.assertTrue(report["analysis_complete"])
            for name, digest in hashes.items():
                self.assertEqual(file_sha256(root / "analysis" / name), digest)
            structure = report["structural_diagnosis"]
            self.assertTrue(structure["diagnostic_only"])
            self.assertNotIn("primary", structure)
            self.assertEqual(len(structure["summaries"]), 24)
            self.assertEqual(len(list((root / "analysis").glob("*_pairs.csv"))), 24)

    def test_completed_a_skips_training_and_failure_stops_without_retry(self):
        for complete in (True, False):
            process = MagicMock(pid=17)
            process.wait.return_value = 1
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(checker, "code_fingerprints", return_value={}), \
                    patch.object(runner, "summarize", return_value=[{"status": "complete" if complete else "pending"}]), \
                    patch.object(runner, "run_directory", return_value=Path(directory) / "new_run"), \
                    patch.object(runner.norm, "summarize", return_value=[{"status": "complete" if complete else "pending"}]), \
                    patch.object(runner.norm, "verify_completed"), \
                    patch.object(runner.norm.subprocess, "Popen", return_value=process) as start, \
                    patch.object(runner, "analyze") as analysis:
                if complete:
                    runner.execute(configuration(), {"code_sha256": {}}, Path(directory))
                    start.assert_not_called()
                    analysis.assert_called_once()
                else:
                    with self.assertRaisesRegex(RuntimeError, "I3 failed"):
                        runner.execute(configuration(), {"code_sha256": {}}, Path(directory))
                    start.assert_called_once()
                    analysis.assert_not_called()
                queue = json.loads((Path(directory) / "queue.json").read_text())
                self.assertEqual(queue["schema"], "local-module-queue-v1")
                self.assertEqual(queue["status"], "complete" if complete else "failed")

    def test_partial_run_cannot_silently_resume(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(checker, "code_fingerprints", return_value={}), \
                patch.object(runner, "summarize", return_value=[{"status": "partial"}]), \
                patch.object(runner.norm, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "recovery decision"):
                runner.execute(configuration(), {"code_sha256": {}}, Path(directory))
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
