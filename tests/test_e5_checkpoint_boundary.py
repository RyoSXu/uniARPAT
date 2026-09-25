"""E5a checkpoint schema and restore contracts."""
import os
import tempfile
import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
