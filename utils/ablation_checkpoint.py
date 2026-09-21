"""Stable checkpoint boundary for ``run_ablation_experiments.py``.

This module intentionally knows nothing about uniARPAT model architecture.  It
only owns the runner's checkpoint schema, atomic persistence, and AMP resume
compatibility rule.
"""
import os

import torch


def build_ablation_checkpoint(epoch, model_name, seed, use_amp, transformer,
                              optimizer, best_val_score, scaler=None):
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
    return payload


def restore_ablation_checkpoint(checkpoint, transformer, optimizer, use_amp,
                                scaler=None):
    """Restore model and best-effort optimizer state, returning metadata.

    Pre-AMP checkpoints have no ``use_amp`` and are FP32 by definition.
    Optimizer mismatch remains deliberately non-fatal, matching the original
    runner behavior for compatible model-only recovery.
    """
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    saved_amp = bool(checkpoint.get("use_amp", False)) if isinstance(checkpoint, dict) else False
    if saved_amp != bool(use_amp):
        raise ValueError("cannot resume a checkpoint with a different use_amp setting")
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
    }


def atomic_torch_save(payload, path):
    """Atomically replace ``path`` with a torch checkpoint payload."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
