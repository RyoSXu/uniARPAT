"""Stable checkpoint boundary for ``run_ablation_experiments.py``.

This module intentionally knows nothing about uniARPAT model architecture.  It
only owns the runner's checkpoint schema, atomic persistence, and AMP resume
compatibility rule.
"""
import os

import torch


def build_ablation_checkpoint(epoch, model_name, seed, use_amp, transformer,
                              optimizer, best_val_score, scaler=None,
                              edos_slope_ratio=None, edos_slope_lambda=None,
                              pair_aux_arm=None, pair_ratio=None,
                              pair_lambda=None, pair_plan_hash=None):
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
    pair_values = (pair_aux_arm, pair_ratio, pair_lambda, pair_plan_hash)
    if any(value is not None for value in pair_values):
        if any(value is None for value in pair_values):
            raise ValueError("pair auxiliary checkpoints require complete metadata")
        if pair_aux_arm not in {"control", "candidate"}:
            raise ValueError("pair_aux_arm must be control or candidate")
        ratio = float(pair_ratio)
        weight = float(pair_lambda)
        if not torch.isfinite(torch.tensor(ratio)) or ratio <= 0:
            raise ValueError("pair_ratio must be finite and positive")
        if not torch.isfinite(torch.tensor(weight)) or not 1e-4 <= weight <= 1e4:
            raise ValueError("pair_lambda must be finite and within [1e-4, 1e4]")
        plan_hash = str(pair_plan_hash)
        if len(plan_hash) != 64 or any(char not in "0123456789abcdef" for char in plan_hash):
            raise ValueError("pair_plan_hash must be a lowercase SHA-256 digest")
        payload.update({
            "pair_aux_arm": pair_aux_arm,
            "pair_ratio": ratio,
            "pair_lambda": weight,
            "pair_plan_hash": plan_hash,
        })
    return payload


def restore_ablation_checkpoint(checkpoint, transformer, optimizer, use_amp,
                                scaler=None, edos_slope_ratio=None,
                                pair_aux_arm=None, pair_ratio=None,
                                pair_lambda=None, pair_plan_hash=None):
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
    saved_pair_arm = checkpoint.get("pair_aux_arm") if isinstance(checkpoint, dict) else None
    if pair_aux_arm is not None:
        if pair_aux_arm == "none":
            if saved_pair_arm is not None:
                raise ValueError("cannot resume pair-auxiliary training with pair auxiliary disabled")
        else:
            if pair_aux_arm not in {"control", "candidate"}:
                raise ValueError("pair_aux_arm must be none, control, or candidate")
            if saved_pair_arm != pair_aux_arm:
                raise ValueError("cannot resume with a different pair_aux_arm")
            if checkpoint.get("pair_ratio") is None or pair_ratio is None \
                    or float(checkpoint["pair_ratio"]) != float(pair_ratio):
                raise ValueError("cannot resume with a different pair_ratio")
            if checkpoint.get("pair_plan_hash") != pair_plan_hash:
                raise ValueError("cannot resume with a different pair plan")
            saved_pair_lambda = checkpoint.get("pair_lambda")
            if saved_pair_lambda is None:
                raise ValueError("cannot resume pair auxiliary training without calibrated lambda")
            saved_pair_lambda = float(saved_pair_lambda)
            if not torch.isfinite(torch.tensor(saved_pair_lambda)) \
                    or not 1e-4 <= saved_pair_lambda <= 1e4:
                raise ValueError("cannot resume with an invalid pair_lambda")
            if pair_lambda is None or saved_pair_lambda != float(pair_lambda):
                raise ValueError("cannot resume with a different pair_lambda")
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
        "pair_aux_arm": saved_pair_arm,
        "pair_ratio": checkpoint.get("pair_ratio") if isinstance(checkpoint, dict) else None,
        "pair_lambda": checkpoint.get("pair_lambda") if isinstance(checkpoint, dict) else None,
        "pair_plan_hash": (
            checkpoint.get("pair_plan_hash") if isinstance(checkpoint, dict) else None
        ),
    }


def atomic_torch_save(payload, path):
    """Atomically replace ``path`` with a torch checkpoint payload."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
