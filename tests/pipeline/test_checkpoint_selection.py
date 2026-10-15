"""Joint blind selection through training, recovery, exports and adoption."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from tests.pipeline import test_checkpoint_boundary as checkpoint_fixtures
from tests.pipeline.test_global_norm import model_params
from tools.eval import lg_paired_compare as paired
from tools.eval import lg_valid_export as exporter
from tools.eval.local_global_compare import summarize, training_command
import tools.train as trainer
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.checkpoint_selection import (
    BLIND_R2_METRIC, blind_joint_score, initial_best_score, is_better_score,
    selected_history_row, selection_contract, validation_score,
)
from utils.experiment_config import ExperimentConfig
from utils.lg_baseline_experiment import baseline_contract, validate_baseline_config
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, validate_local_global_config,
)
from utils.local_structure_experiment import FROZEN


def configuration():
    return ExperimentConfig(**{**FROZEN, "local_route": "I1", "selection_metric": BLIND_R2_METRIC,
                               "tag": "_recovery_contract"})


def history():
    return pd.DataFrame({"epoch": [1, 2, 3], "balanced_score": [-.2, .6, .4],
        BLIND_R2_METRIC: [-.2, .6, .4], "blind_r2_edos_median": [-.3, .4, .6],
        "blind_r2_phdos_median": [-.1, .8, .2]})


def summary(tag="unit"):
    return {"tag": tag, "best_epoch": 1, "best_score": .4, "epochs": 35,
            "contract_schema": "unit", "init_protocol": "unit", "epoch_time_s_median": 1.,
            "epoch_time_s_sum": 35., "peak_vram_mb_median": 1., "peak_vram_mb_max": 1.,
            "selection_metric": BLIND_R2_METRIC, **selection_contract(BLIND_R2_METRIC)}


def samples(e, p):
    frame = pd.DataFrame(index=[str(i) for i in range(len(e))])
    for mode in ("blind", "oracle"):
        for task, values in (("edos", e), ("phdos", p)):
            frame[f"{mode}_r2_{task}"] = values
            frame[f"{mode}_mae_{task}"] = 1 - np.asarray(values)
    return frame


class TestCheckpointSelection(unittest.TestCase):
    def test_equal_weights_use_explicit_blind_medians_and_allow_negative_r2(self):
        metrics = {"blind_r2_edos_median": -.6, "blind_r2_phdos_median": .2,
                   "oracle_r2_edos_median": 1., "oracle_r2_phdos_median": 1.,
                   "balanced_oracle_mae": 0., BLIND_R2_METRIC: 999.}
        self.assertAlmostEqual(blind_joint_score(metrics), -.2)
        self.assertAlmostEqual(validation_score(metrics, BLIND_R2_METRIC), -.2)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validation_score({**metrics, "blind_r2_phdos_median": value}, BLIND_R2_METRIC)

    def test_score_direction_preserves_mae_and_ties_keep_previous_best(self):
        self.assertTrue(is_better_score(-.4, initial_best_score(BLIND_R2_METRIC), BLIND_R2_METRIC))
        self.assertTrue(is_better_score(.6, .5, BLIND_R2_METRIC))
        self.assertFalse(is_better_score(.5, .6, BLIND_R2_METRIC))
        self.assertFalse(is_better_score(.6, .6, BLIND_R2_METRIC))
        for metric in (None, "balanced_oracle_mae", "balanced_blind_mae"):
            self.assertTrue(is_better_score(.4, initial_best_score(metric), metric))
            self.assertTrue(is_better_score(.4, .6, metric))
            self.assertFalse(is_better_score(.6, .6, metric))
            self.assertEqual(selection_contract(metric), {})

    def test_history_maximum_formula_validation_and_earliest_tie(self):
        frame = history()
        self.assertEqual(selected_history_row(frame, BLIND_R2_METRIC).epoch, 2)
        frame.loc[2, ["balanced_score", BLIND_R2_METRIC, "blind_r2_edos_median", "blind_r2_phdos_median"]] = [.6, .6, .4, .8]
        self.assertEqual(selected_history_row(frame.iloc[::-1], BLIND_R2_METRIC).epoch, 2)
        for mutate in (lambda f: f.drop(columns="blind_r2_phdos_median"),
                       lambda f: f.assign(balanced_score=.9),
                       lambda f: f.assign(blind_r2_edos_median=float("nan"))):
            with self.assertRaises(ValueError):
                selected_history_row(mutate(frame), BLIND_R2_METRIC)
        legacy = pd.DataFrame({"epoch": [1, 2], "balanced_score": [.4, .2], "balanced_oracle_mae": [.4, .2]})
        self.assertEqual(selected_history_row(legacy, "balanced_oracle_mae").epoch, 2)

    def test_i1_cli_new_contract_and_historical_checkpoint_isolation(self):
        cfg = configuration()
        validate_local_global_config(cfg)
        parsed = trainer.build_arg_parser().parse_args(training_command(cfg)[2:])
        self.assertEqual(asdict(ExperimentConfig.from_args(parsed)), asdict(cfg))
        params = model_params("I1")
        contract = comparison_contract(cfg, params)
        old_cfg = replace(cfg, selection_metric="balanced_oracle_mae")
        old = comparison_contract(old_cfg, params)
        self.assertNotIn("checkpoint_selection", old)
        self.assertEqual(contract["transformer"], old["transformer"])
        self.assertEqual(contract["init_protocol"], old["init_protocol"])
        self.assertEqual(contract["checkpoint_selection"]["mode"], "max")
        self.assertEqual(contract["training"]["selection_metric"], BLIND_R2_METRIC)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, -.2,
            model_contract=contract, training_rng=capture_training_rng())
        restore_ablation_checkpoint(payload, model, optimizer, False, model_contract=contract)
        with self.assertRaisesRegex(ValueError, "different model contract"):
            restore_ablation_checkpoint(payload, model, optimizer, False, model_contract=old)
        for change in ({"seed": 43}, {"epochs": 100}, {"global_depth": 8}, {"lambda_ph": 0.}, {"use_amp": True}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(cfg, **change))

    def test_original_encoder_baseline_can_use_the_same_selection_contract(self):
        cfg = replace(configuration(), local_route="none", local_layout=None, local_depth=None,
                      global_depth=None, lg_baseline=True)
        validate_local_global_config(cfg)
        validate_baseline_config(cfg)
        new = baseline_contract(cfg, {"local_route": "none"})
        old = baseline_contract(replace(cfg, selection_metric="balanced_oracle_mae"), {"local_route": "none"})
        self.assertEqual(new["checkpoint_selection"], selection_contract(BLIND_R2_METRIC)["checkpoint_selection"])
        self.assertNotIn("checkpoint_selection", old)

    def test_real_runner_saves_joint_maximum_and_resume_keeps_score_direction(self):
        fixture = checkpoint_fixtures.TestRunnerRecovery()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.cfg = configuration()
        fixture.model.model["transformer"].local_route = "I1"
        fixture.model.model["transformer"].encoder = SimpleNamespace(local_layers=[])
        metrics = []
        for e, p in [(-.8, -.2), (.4, .8), (.9, .1), (.4, .8)] + [(.1, .1)] * 31:
            metrics.append({"mae_edos_median": .01, "mae_phdos_median": .001,
                "r2_edos_median": .99, "r2_phdos_median": .99,
                "blind_r2_edos_median": e, "blind_r2_phdos_median": p,
                "blind_mae_edos_median": .5, "blind_mae_phdos_median": .1,
                "blind_fail_rate_edos": 1., "blind_fail_rate_phdos": 2.})
        fixture.evaluate.side_effect = deepcopy(metrics)
        checkpoints = []
        original = trainer.build_ablation_checkpoint
        def capture(*args, **kwargs):
            payload = original(*args, **kwargs)
            checkpoints.append(deepcopy(payload))
            return payload
        with patch.object(trainer, "initialize_comparison_model", return_value={"protocol": "unit"}), \
                patch.object(trainer, "build_ablation_checkpoint", side_effect=capture):
            trainer.train_and_eval(fixture.cfg)
            best = torch.load(fixture.best_path, weights_only=True)
            self.assertEqual(best["epoch"], 2)
            self.assertAlmostEqual(best["best_val_score"], .6)
            self.assertEqual(pd.read_csv(fixture.history_path).epoch.tolist(), list(range(1, 36)))
            logs = [str(call) for call in trainer.logger.info.call_args_list]
            self.assertTrue(any("displayed R2=blind" in line and "selection_mode=max" in line for line in logs))
            second = next(payload for payload in checkpoints if payload["epoch"] == 2)
            torch.save(second, fixture.latest_path)
            pd.read_csv(fixture.history_path).iloc[:2].to_csv(fixture.history_path, index=False)
            fixture.evaluate.side_effect = deepcopy(metrics[2:])
            trainer.train_and_eval(fixture.cfg)
        latest = torch.load(fixture.latest_path, weights_only=True)
        self.assertEqual(latest["epoch"], 35)
        self.assertAlmostEqual(latest["best_val_score"], .6)
        self.assertEqual(torch.load(fixture.best_path, weights_only=True)["epoch"], 2)
        self.assertEqual(selected_history_row(pd.read_csv(fixture.history_path), BLIND_R2_METRIC).epoch, 2)

    def test_summary_and_export_choose_same_maximum_and_reject_relabelled_contract(self):
        cfg, params = configuration(), model_params("I1")
        contract = comparison_contract(cfg, params)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / f"results/runs/ablation_m1{cfg.tag}"
            run.mkdir(parents=True)
            report = root / f"results/reports/history_m1{cfg.tag}.csv"
            report.parent.mkdir(parents=True)
            history().to_csv(report, index=False)
            saved = {"cli": {"model": "M1"}, "model_contract": contract,
                     "config": {"model": {"params": {"sub_model": {"transformer": params}}}}}
            (run / "config_used.yaml").write_text(yaml.safe_dump(saved))
            torch.save({"epoch": 2, "best_val_score": .6, "model_contract": contract}, run / "checkpoint_best.pth")
            with patch("tools.eval.local_global_compare.repository_path", side_effect=lambda p: root / p), \
                    patch.object(exporter, "repository_path", side_effect=lambda p: root / p):
                self.assertEqual(summarize([cfg])[0]["best_epoch"], 2)
                self.assertEqual(exporter.select_best(run, BLIND_R2_METRIC, saved)["row"].epoch, 2)
                changed = deepcopy(saved)
                changed["model_contract"]["checkpoint_selection"]["mode"] = "min"
                with self.assertRaisesRegex(ValueError, "selection contract"):
                    exporter.select_best(run, BLIND_R2_METRIC, changed)

    def test_joint_report_shares_material_draws_and_accepts_only_new_matched_protocols(self):
        base = samples([.1, .2, .6, .9], [.8, .2, .3, .4])
        other = samples(base.blind_r2_edos + .1, base.blind_r2_phdos + .1)
        report = paired.compare(summary(), base, [("candidate", summary("new"), other)], 100, 42)
        joint = report["joint_blind"]["candidate"]
        self.assertEqual(report["primary_metric"], BLIND_R2_METRIC)
        self.assertAlmostEqual(joint["difference"], .1)
        self.assertTrue(joint["eligible_for_adoption"])
        self.assertTrue(joint["bootstrap"]["same_indices_for_tasks_and_models"])
        self.assertTrue(all(value["passed"] for value in joint["protections"].values()))
        old = {**summary(), "selection_metric": "balanced_oracle_mae"}
        mixed = paired.compare(old, base, [("candidate", summary(), other)], 100, 42)["joint_blind"]["candidate"]
        self.assertFalse(mixed["eligible_for_adoption"])
        self.assertIn("selection_protocol_mismatch_or_historical_selection", mixed["reasons"])

    def test_joint_gain_cannot_hide_task_r2_or_failure_rate_regression(self):
        base = samples(np.linspace(.2, .7, 20), np.linspace(.3, .8, 20))
        for task, kind in (("phdos", "r2"), ("edos", "failure"), ("phdos", "failure")):
            other = samples(base.blind_r2_edos + .2, base.blind_r2_phdos + .2)
            if kind == "r2":
                other["blind_r2_phdos"] = base.blind_r2_phdos - .05
            else:
                other.loc["0", f"blind_r2_{task}"] = -.1
            joint = paired.compare(summary(), base, [("candidate", summary(), other)], 100, 42)["joint_blind"]["candidate"]
            self.assertGreater(joint["ci95"][0], 0)
            self.assertFalse(joint["eligible_for_adoption"])
            self.assertIn(task + "_protection_failed", joint["reasons"])

    def test_joint_bootstrap_reproduces_two_task_paired_statistic(self):
        base = samples([.1, .4, .5, .9], [.9, .5, .2, .1])
        other = samples([.8, .1, .7, .5], [.2, .7, .6, .3])
        report = paired.compare(summary(), base, [("candidate", summary(), other)], 100, 43)
        generator = np.random.default_rng(43)
        columns = ["blind_r2_edos", "blind_r2_phdos"]
        a, b = base[columns].to_numpy(), other[columns].to_numpy()
        boot = []
        for _ in range(100):
            indices = generator.integers(0, len(base), len(base))
            boot.append(.5 * sum(np.median(b[indices, task]) - np.median(a[indices, task]) for task in (0, 1)))
        np.testing.assert_allclose(report["joint_blind"]["candidate"]["ci95"], np.percentile(boot, [2.5, 97.5]), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
