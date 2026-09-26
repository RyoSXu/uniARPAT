"""E5a checkpoint schema and restore contracts."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import torch
import yaml

import run_ablation_experiments as runner
from utils.experiment_config import ExperimentConfig
from utils.ablation_checkpoint import (
    atomic_torch_save,
    build_ablation_checkpoint,
    restore_ablation_checkpoint,
)


def _parts():
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


class _Scaler:
    def __init__(self, state):
        self.state = dict(state)

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)


def test_e5_fp32_schema_and_old_checkpoint_compatibility():
    model, optimizer = _parts()
    payload = build_ablation_checkpoint(4, "M1", 42, False, model, optimizer, 1.25)
    assert set(payload) == {"epoch", "model_name", "seed", "use_amp", "model", "optimizer", "best_val_score"}
    assert "amp_scaler" not in payload
    old = dict(payload)
    old.pop("use_amp")
    target, target_opt = _parts()
    meta = restore_ablation_checkpoint(old, target, target_opt, False)
    assert meta["epoch"] == 4 and meta["optimizer_restored"]
    for a, b in zip(model.parameters(), target.parameters()):
        assert torch.equal(a, b)


def test_e5_amp_mismatch_rejected_and_atomic_replace():
    model, optimizer = _parts()
    payload = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer, 0.5)
    with unittest.TestCase().assertRaisesRegex(ValueError, "different use_amp"):
        restore_ablation_checkpoint(payload, model, optimizer, True, scaler=object())
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "checkpoint.pth")
        atomic_torch_save(payload, path)
        got = torch.load(path, map_location="cpu")
        assert got["epoch"] == 1 and not os.path.exists(path + ".tmp")


def test_e5_amp_scaler_round_trip():
    model, optimizer = _parts()
    source_scaler = _Scaler({"scale": 1024.0, "growth_tracker": 17})
    payload = build_ablation_checkpoint(3, "M1", 42, True, model, optimizer, 0.25, source_scaler)
    assert "amp_scaler" in payload
    target, target_opt = _parts()
    target_scaler = _Scaler({})
    meta = restore_ablation_checkpoint(payload, target, target_opt, True, target_scaler)
    assert meta["epoch"] == 3 and target_scaler.state == source_scaler.state


def test_e5_slope_calibration_round_trip_and_mismatch_rejection():
    model, optimizer = _parts()
    payload = build_ablation_checkpoint(
        2,
        "M1",
        42,
        False,
        model,
        optimizer,
        0.75,
        edos_slope_ratio=0.1,
        edos_slope_lambda=0.004,
    )
    target, target_opt = _parts()
    meta = restore_ablation_checkpoint(
        payload, target, target_opt, False, edos_slope_ratio=0.1
    )

    assert meta["edos_slope_ratio"] == 0.1
    assert meta["edos_slope_lambda"] == 0.004
    with unittest.TestCase().assertRaisesRegex(ValueError, "different edos_slope_ratio"):
        restore_ablation_checkpoint(
            payload, target, target_opt, False, edos_slope_ratio=0.2
        )
    with unittest.TestCase().assertRaisesRegex(ValueError, "different edos_slope_ratio"):
        restore_ablation_checkpoint(
            payload, target, target_opt, False, edos_slope_ratio=0.0
        )
    with unittest.TestCase().assertRaisesRegex(ValueError, "calibrated lambda"):
        build_ablation_checkpoint(
            2, "M1", 42, False, model, optimizer, 0.75, edos_slope_ratio=0.1
        )


class TestE5CheckpointBoundary(unittest.TestCase):
    def test_fp32_old(self):
        test_e5_fp32_schema_and_old_checkpoint_compatibility()

    def test_mismatch_atomic(self):
        test_e5_amp_mismatch_rejected_and_atomic_replace()

    def test_amp_scaler(self):
        test_e5_amp_scaler_round_trip()

    def test_slope_calibration(self):
        test_e5_slope_calibration_round_trip_and_mismatch_rejection()


class TestRunnerRecovery(unittest.TestCase):
    """Exercise artifact protection through the actual runner control flow."""

    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        default_yaml = (repo / "configs/default.yaml").read_text()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "configs").mkdir()
        (self.root / "configs/default.yaml").write_text(default_yaml)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.run_dir = self.root / "output/ablation_m1_recovery_contract"
        self.run_dir.mkdir(parents=True)
        (self.root / "results").mkdir()
        self.config_path = self.run_dir / "config_used.yaml"
        self.history_path = self.root / "results/history_m1_recovery_contract.csv"
        self.latest_path = self.run_dir / "checkpoint_latest.pth"
        self.best_path = self.run_dir / "checkpoint_best.pth"
        self.cfg = ExperimentConfig(
            epochs=2, tag="_recovery_contract", skip_test_eval=True,
        )
        transformer, optimizer = _parts()
        transformer.use_gated_cross_attn = False
        self.model = SimpleNamespace(
            model={"transformer": transformer}, optimizer={"transformer": optimizer},
            gscaler=_Scaler({}), edos_slope_ratio=0.0, edos_slope_lambda=0.0,
            edos_slope_calibrated=True, edos_slope_calibration=None, to=Mock(),
            train_one_step=Mock(return_value={
                "loss": 1.0, "loss_edos": 0.5, "loss_phdos": 0.5,
            }),
        )
        self.builder = Mock()
        self.builder.get_model.return_value = self.model
        self.builder.get_dataloader.return_value = ["synthetic_batch"]
        self.enterContext(patch.object(runner, "ConfigBuilder", return_value=self.builder))
        self.enterContext(patch.object(runner, "logger", Mock()))
        self.enterContext(patch.object(torch.cuda, "is_available", return_value=False))
        self.scheduler = self.enterContext(patch(
            "utils.builder.build_warmup_cosine_scheduler", return_value=Mock(),
        )).return_value
        self.evaluate = self.enterContext(patch.object(
            runner, "evaluate_split", return_value={
                "mae_edos_median": 0.2, "mae_phdos_median": 0.3,
                "r2_edos_median": 0.5, "r2_phdos_median": 0.7,
            },
        ))

    def saved_run(self, *, use_amp=False, slope_ratio=None):
        source, optimizer = _parts()
        with torch.no_grad():
            source.weight.fill_(0.75)
        source(torch.ones(1, 3)).square().mean().backward()
        optimizer.step()
        payload = build_ablation_checkpoint(
            1, "M1", 42, use_amp, source, optimizer, 0.4,
            _Scaler({"scale": 1024.0, "growth_tracker": 17}) if use_amp else None,
            edos_slope_ratio=slope_ratio,
            edos_slope_lambda=0.004 if slope_ratio else None,
        )
        self.config_path.write_text(
            "original_config_marker: keep_me\nruntime:\n  edos_slope_lambda: 0.004\n"
        )
        self.history_path.write_text("epoch,balanced_score\n1,0.4\n")
        torch.save(payload, self.latest_path)
        torch.save(payload, self.best_path)
        return payload

    def assert_failed_run_preserved(self):
        paths = (self.config_path, self.history_path, self.latest_path, self.best_path)
        before = {path: path.read_bytes() for path in paths}
        with self.assertRaises(Exception):
            runner.train_and_eval(self.cfg)
        self.model.train_one_step.assert_not_called()
        self.evaluate.assert_not_called()
        for path in paths:
            self.assertEqual(before[path], path.read_bytes(), str(path))

    def test_amp_mismatch_preserves_all_artifacts(self):
        self.saved_run(use_amp=True)
        self.assert_failed_run_preserved()

    def test_slope_mismatch_preserves_all_artifacts(self):
        self.saved_run(slope_ratio=0.1)
        self.assert_failed_run_preserved()

    def test_missing_calibration_preserves_all_artifacts(self):
        payload = self.saved_run(slope_ratio=0.1)
        self.cfg.edos_slope_ratio = self.model.edos_slope_ratio = 0.1
        del payload["edos_slope_lambda"]
        torch.save(payload, self.latest_path)
        self.assert_failed_run_preserved()

    def test_unreadable_checkpoint_preserves_all_artifacts(self):
        self.saved_run()
        self.latest_path.write_bytes(b"incomplete checkpoint")
        self.assert_failed_run_preserved()

    def test_incompatible_weights_preserve_all_artifacts(self):
        payload = self.saved_run()
        payload["model"]["weight"] = torch.zeros(9, 9)
        torch.save(payload, self.latest_path)
        self.assert_failed_run_preserved()

    def test_invalid_history_preserves_all_artifacts(self):
        self.saved_run()
        self.history_path.write_text("epoch,wrong_column\n1,0.4\n")
        self.assert_failed_run_preserved()

    def check_compatible_resume(self, *, use_amp=False, slope_ratio=None, old_fp32=False):
        payload = self.saved_run(use_amp=use_amp, slope_ratio=slope_ratio)
        self.cfg.use_amp = use_amp
        self.cfg.edos_slope_ratio = self.model.edos_slope_ratio = slope_ratio or 0.0
        if old_fp32:
            payload.pop("use_amp")
            torch.save(payload, self.latest_path)
        original_config = self.config_path.read_bytes()
        result = runner.train_and_eval(self.cfg)
        self.assertEqual(result, {"test_skipped": True})
        self.model.train_one_step.assert_called_once()
        self.evaluate.assert_called_once()
        self.assertEqual(pd.read_csv(self.history_path).epoch.tolist(), [1, 2])
        self.assertEqual(self.config_path.read_bytes(), original_config)
        saved = torch.load(self.latest_path, map_location="cpu", weights_only=True)
        self.assertEqual(saved["epoch"], 2)
        torch.testing.assert_close(saved["model"]["weight"], payload["model"]["weight"])
        self.assertTrue(saved["optimizer"]["state"])
        return saved

    def test_old_fp32_resume_continues_at_next_epoch(self):
        self.check_compatible_resume(old_fp32=True)

    def test_amp_resume_preserves_scaler(self):
        saved = self.check_compatible_resume(use_amp=True)
        self.assertEqual(saved["amp_scaler"], {"scale": 1024.0, "growth_tracker": 17})

    def test_slope_resume_preserves_calibration(self):
        saved = self.check_compatible_resume(slope_ratio=0.1)
        self.assertEqual(saved["edos_slope_lambda"], 0.004)
        self.assertTrue(self.model.edos_slope_calibrated)

    def test_fresh_run_writes_config_and_history(self):
        self.cfg.epochs = 1
        self.assertEqual(runner.train_and_eval(self.cfg), {"test_skipped": True})
        self.model.train_one_step.assert_called_once()
        self.assertEqual(pd.read_csv(self.history_path).epoch.tolist(), [1])
        config = yaml.safe_load(self.config_path.read_text())
        self.assertEqual(config["cli"]["epochs"], 1)
        self.assertFalse(config["cli"]["use_amp"])
        self.assertTrue(self.latest_path.is_file() and self.best_path.is_file())


if __name__ == "__main__":
    unittest.main()
