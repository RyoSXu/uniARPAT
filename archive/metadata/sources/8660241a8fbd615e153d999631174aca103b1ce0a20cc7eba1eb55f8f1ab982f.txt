"""Stable checkpoint boundary for ``run_ablation_experiments.py``.

This module intentionally knows nothing about uniARPAT model architecture.  It
only owns the runner's checkpoint schema, atomic persistence, and AMP resume
compatibility rule.
"""
import os

import torch


def build_ablation_checkpoint(epoch, model_name, seed, use_amp, transformer,
                              optimizer, best_val_score, scaler=None,
                              edos_slope_ratio=None, edos_slope_lambda=None):
    """Build the exact runner checkpoint payload for one completed epoch."""
    payload = {
        "epoch": int(epoch),
        "model_name": model_name,
        "seed": int(seed),
        "use_amp": bool(use_amp),
        "model": transformer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_score": float(best_val_score),
    }
    if use_amp:
        if scaler is None:
            raise ValueError("AMP checkpoint needs a GradScaler")
        payload["amp_scaler"] = scaler.state_dict()
    if edos_slope_ratio is not None:
        slope_ratio = float(edos_slope_ratio)
        if not torch.isfinite(torch.tensor(slope_ratio)) or slope_ratio < 0:
            raise ValueError("edos_slope_ratio must be finite and nonnegative")
        if slope_ratio > 0 and edos_slope_lambda is None:
            raise ValueError("slope-loss checkpoints need a calibrated lambda")
        payload["edos_slope_ratio"] = slope_ratio
    if edos_slope_lambda is not None:
        slope_lambda = float(edos_slope_lambda)
        if not torch.isfinite(torch.tensor(slope_lambda)) or slope_lambda <= 0:
            raise ValueError("edos_slope_lambda must be finite and positive")
        payload["edos_slope_lambda"] = slope_lambda
    return payload


def restore_ablation_checkpoint(checkpoint, transformer, optimizer, use_amp,
                                scaler=None, edos_slope_ratio=None):
    """Restore model and best-effort optimizer state, returning metadata.

    Pre-AMP checkpoints have no ``use_amp`` and are FP32 by definition.
    Optimizer mismatch remains deliberately non-fatal, matching the original
    runner behavior for compatible model-only recovery.
    """
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    saved_amp = bool(checkpoint.get("use_amp", False)) if isinstance(checkpoint, dict) else False
    if saved_amp != bool(use_amp):
        raise ValueError("cannot resume a checkpoint with a different use_amp setting")
    saved_slope_ratio = (
        checkpoint.get("edos_slope_ratio") if isinstance(checkpoint, dict) else None
    )
    if edos_slope_ratio is not None:
        if edos_slope_ratio > 0 and saved_slope_ratio is None:
            raise ValueError("cannot resume slope-loss training without calibration metadata")
        if saved_slope_ratio is not None and float(saved_slope_ratio) != float(edos_slope_ratio):
            raise ValueError("cannot resume with a different edos_slope_ratio")
        if edos_slope_ratio > 0 and checkpoint.get("edos_slope_lambda") is None:
            raise ValueError("cannot resume slope-loss training without its calibrated lambda")
        if edos_slope_ratio > 0:
            saved_lambda = float(checkpoint["edos_slope_lambda"])
            if not torch.isfinite(torch.tensor(saved_lambda)) or saved_lambda <= 0:
                raise ValueError("cannot resume slope-loss training with an invalid calibrated lambda")
    transformer.load_state_dict(state)
    if use_amp:
        if scaler is None:
            raise ValueError("AMP resume needs a GradScaler")
        if isinstance(checkpoint, dict) and "amp_scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["amp_scaler"])
    optimizer_restored = False
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_restored = True
        except Exception:
            pass
    return {
        "epoch": int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0,
        "best_val_score": float(checkpoint.get("best_val_score", float("inf")))
        if isinstance(checkpoint, dict) else float("inf"),
        "optimizer_restored": optimizer_restored,
        "edos_slope_ratio": saved_slope_ratio,
        "edos_slope_lambda": (
            checkpoint.get("edos_slope_lambda") if isinstance(checkpoint, dict) else None
        ),
    }


def atomic_torch_save(payload, path):
    """Atomically replace ``path`` with a torch checkpoint payload."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
