"""Fit and adjudicate the preregistered two-scalar H1 gamma calibration.

The tool is deliberately train-fit/valid-only.  It has no split selector and
never loads Q1 test data.  The frozen B7 model supplies gamma and eDOS shape;
only ``a`` and ``b`` in sigmoid(a * logit(gamma) + b) are fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import minimize
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = REPO_ROOT / "output/ablation_m1_e9ctl/checkpoint_best.pth"
DEFAULT_CONFIG = REPO_ROOT / "output/ablation_m1_e9ctl/config_used.yaml"
DEFAULT_OUTPUT_PREFIX = REPO_ROOT / "results/h1_gamma_calibration_q1"
EXPECTED_CHECKPOINT_SHA256 = (
    "cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40"
)
EXPECTED_COUNTS = {"train": 18_706, "valid": 2_313}
BOOTSTRAP_SEED = 20_260_928
EPS = 1e-6
R2_MARGIN = 0.02
FAILURE_MARGIN_PP = 1.0


def _as_finite_unit_interval(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def apply_gamma_calibration(
    gamma: np.ndarray, a: float, b: float, eps: float = EPS
) -> np.ndarray:
    """Apply the preregistered affine map in logit space."""
    values = _as_finite_unit_interval(gamma, "gamma")
    if not np.isfinite([a, b]).all() or a <= 0.0:
        raise ValueError("calibration requires finite a > 0 and finite b")
    clipped = np.clip(values, eps, 1.0 - eps)
    logits = np.log(clipped) - np.log1p(-clipped)
    mapped_logits = a * logits + b
    return np.where(
        mapped_logits >= 0.0,
        1.0 / (1.0 + np.exp(-mapped_logits)),
        np.exp(mapped_logits) / (1.0 + np.exp(mapped_logits)),
    )


def soft_binary_cross_entropy(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = _as_finite_unit_interval(prediction, "prediction")
    target = _as_finite_unit_interval(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    prediction = np.clip(prediction, EPS, 1.0 - EPS)
    return float(
        np.mean(-target * np.log(prediction) - (1.0 - target) * np.log1p(-prediction))
    )


def fit_gamma_calibration(
    gamma_prediction: np.ndarray, gamma_target: np.ndarray
) -> dict[str, float | int | bool | str]:
    """Fit exactly two scalars with one deterministic L-BFGS-B run."""
    prediction = _as_finite_unit_interval(gamma_prediction, "gamma_prediction")
    target = _as_finite_unit_interval(gamma_target, "gamma_target")
    if prediction.shape != target.shape:
        raise ValueError("gamma prediction and target must have the same shape")
    clipped = np.clip(prediction, EPS, 1.0 - EPS)
    logits = np.log(clipped) - np.log1p(-clipped)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = parameters
        mapped_logits = a * logits + b
        calibrated = np.where(
            mapped_logits >= 0.0,
            1.0 / (1.0 + np.exp(-mapped_logits)),
            np.exp(mapped_logits) / (1.0 + np.exp(mapped_logits)),
        )
        loss = np.mean(
            np.maximum(mapped_logits, 0.0)
            - target * mapped_logits
            + np.log1p(np.exp(-np.abs(mapped_logits)))
        )
        residual = calibrated - target
        gradient = np.array(
            [np.mean(residual * logits), np.mean(residual)], dtype=np.float64
        )
        return float(loss), gradient

    identity_loss, _ = objective(np.array([1.0, 0.0], dtype=np.float64))
    result = minimize(
        objective,
        x0=np.array([1.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((1e-6, None), (None, None)),
        options={"maxiter": 1_000, "ftol": 1e-15, "gtol": 1e-10},
    )
    fitted_loss = float(result.fun)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(fitted_loss):
        raise RuntimeError(f"gamma calibration fit failed: {result.message}")
    if fitted_loss > identity_loss + 1e-12:
        raise RuntimeError("gamma calibration optimizer worsened the train objective")
    return {
        "a": float(result.x[0]),
        "b": float(result.x[1]),
        "identity_soft_bce": float(identity_loss),
        "fitted_soft_bce": fitted_loss,
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def paired_bootstrap_median_interval(
    control: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if control.shape != candidate.shape or control.ndim != 1 or control.size == 0:
        raise ValueError("paired bootstrap requires equal non-empty one-dimensional arrays")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.integers(0, control.size, size=(count, control.size))
        differences[start : start + count] = np.median(
            candidate[indices], axis=1
        ) - np.median(control[indices], axis=1)
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def adjudicate_gate(
    r2_control: np.ndarray,
    r2_candidate: np.ndarray,
    log_error_control: np.ndarray,
    log_error_candidate: np.ndarray,
    invariants: dict[str, bool],
    replicates: int = 2_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (r2_control, r2_candidate, log_error_control, log_error_candidate)
    ]
    if any(values.ndim != 1 or values.size == 0 for values in arrays):
        raise ValueError("gate inputs must be non-empty one-dimensional arrays")
    if len({values.shape for values in arrays}) != 1:
        raise ValueError("gate inputs must have matching shapes")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("gate inputs contain non-finite values")

    control_r2, candidate_r2, control_error, candidate_error = arrays
    r2_delta = float(np.median(candidate_r2) - np.median(control_r2))
    fail_delta_pp = float(
        ((candidate_r2 < 0.0).mean() - (control_r2 < 0.0).mean()) * 100.0
    )
    error_delta = float(np.median(candidate_error) - np.median(control_error))
    r2_ci = paired_bootstrap_median_interval(
        control_r2, candidate_r2, replicates, seed
    )
    error_ci = paired_bootstrap_median_interval(
        control_error, candidate_error, replicates, seed + 1
    )
    main_effect_met = r2_delta >= R2_MARGIN
    failure_guard_met = fail_delta_pp < FAILURE_MARGIN_PP
    mechanism_met = error_delta < 0.0 and error_ci[1] < 0.0
    invariants_met = bool(invariants) and all(invariants.values())
    return {
        "verdict": (
            "win"
            if main_effect_met and failure_guard_met and mechanism_met and invariants_met
            else "park"
        ),
        "main_effect_met": main_effect_met,
        "failure_guard_met": failure_guard_met,
        "mechanism_met": mechanism_met,
        "invariants_met": invariants_met,
        "r2_delta_median": r2_delta,
        "r2_delta_median_ci95": list(r2_ci),
        "fail_delta_pp": fail_delta_pp,
        "gamma_abs_log_error_delta_median": error_delta,
        "gamma_abs_log_error_delta_median_ci95": list(error_ci),
        "rules": {
            "r2_delta_median_min": R2_MARGIN,
            "fail_delta_pp_strict_max": FAILURE_MARGIN_PP,
            "mechanism": "median abs-log-error delta < 0 and paired-bootstrap CI95 upper < 0",
            "all_invariants_required": True,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _authenticate_inputs(
    checkpoint_path: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if checkpoint_path.resolve() != DEFAULT_CHECKPOINT.resolve():
        raise ValueError("this preregistered gate accepts only the frozen B7 checkpoint path")
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("this preregistered gate accepts only the frozen B7 config path")
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected B7 checkpoint SHA256: {checkpoint_sha}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = {
        "model_name": checkpoint.get("model_name"),
        "epoch": int(checkpoint.get("epoch", -1)),
        "seed": int(checkpoint.get("seed", -1)),
    }
    if identity != {"model_name": "M1", "epoch": 33, "seed": 42}:
        raise ValueError(f"unexpected B7 checkpoint identity: {identity}")
    if checkpoint.get("use_amp") not in (None, False):
        raise ValueError("B7 calibration gate requires the FP32 checkpoint")

    saved = yaml.safe_load(config_path.read_text())
    cli = saved.get("cli", {})
    expected_cli = {
        "model": "M1",
        "epochs": 35,
        "batch_size": 32,
        "seed": 42,
        "data_dir": "./data/train4ARPAT",
        "norm": "sumnorm",
        "use_mask": False,
        "scale_mode": "eta",
        "eta_sup_w": 1.0,
        "delta_edos": 0.09375,
        "delta_phdos": 19.6875,
    }
    mismatches = {
        key: {"expected": expected, "actual": cli.get(key)}
        for key, expected in expected_cli.items()
        if cli.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"B7 config contract mismatch: {mismatches}")
    return checkpoint, saved, {
        **identity,
        "checkpoint_sha256": checkpoint_sha,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
    }


def _build_model_and_loaders(
    checkpoint: dict[str, Any], saved: dict[str, Any], device: torch.device
) -> tuple[Any, torch.nn.Module, dict[str, DataLoader]]:
    from utils.builder import ConfigBuilder

    builder = ConfigBuilder(**saved["config"])
    model = builder.get_model()
    model.to(device)
    transformer = model.model["transformer"]
    transformer.load_state_dict(checkpoint["model"], strict=True)
    transformer.eval()

    loaders: dict[str, DataLoader] = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        dataset = builder.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
        if dataset is None or len(dataset) != expected_count:
            raise ValueError(
                f"Q1 {split} count mismatch: expected {expected_count}, "
                f"found {None if dataset is None else len(dataset)}"
            )
        loaders[split] = DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False
        )
    return model, transformer, loaders


def _collect_split(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    split: str,
    calibration: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    from utils.metrics import per_sample_spectral_metrics

    result: dict[str, list[np.ndarray]] = {
        "gamma_true": [],
        "gamma_control": [],
    }
    if split == "valid":
        result.update(
            {
                "gamma_candidate": [],
                "r2_edos_oracle": [],
                "r2_edos_blind_control": [],
                "r2_edos_blind_candidate": [],
            }
        )
    delta_edos = float(model.params.get("delta_edos", 0.09375))
    if split == "valid" and calibration is None:
        raise ValueError("valid collection requires fitted calibration")

    with torch.inference_mode():
        for batch in loader:
            (
                inp,
                pos,
                attention_mask,
                edos_target,
                _,
                _,
                _,
                edos_min,
                edos_max,
                _,
                _,
                _,
                _,
                _,
                _,
                nvalence,
                edos_x,
                phdos_x,
            ) = model.data_preprocess(batch)
            if nvalence is None:
                raise ValueError(f"Q1 {split} requires N_valence")
            outputs = model.fp32_outputs(
                model.model["transformer"](
                    inp, attention_mask, pos, edos_x, phdos_x
                )
            )
            gamma_control = outputs["eta"][:, 1].detach().cpu().numpy()
            nval = nvalence.reshape(-1, 1).to(device=device, dtype=torch.float32)
            gamma_true = (
                edos_max.clamp_min(1e-12) * delta_edos / nval.clamp_min(1e-12)
            ).clamp(0.0, 1.0)
            result["gamma_true"].append(gamma_true[:, 0].cpu().numpy())
            result["gamma_control"].append(gamma_control)

            if split == "valid":
                a, b = calibration
                gamma_candidate = apply_gamma_calibration(gamma_control, a, b)
                shape = F.softmax(outputs["edos"], dim=-1)
                target = edos_target * (edos_max - edos_min) + edos_min
                oracle_prediction = torch.clamp(
                    shape * (edos_max - edos_min) + edos_min, min=0.0
                )
                control_scale = nval * torch.as_tensor(
                    gamma_control, device=device, dtype=torch.float32
                ).reshape(-1, 1) / delta_edos
                candidate_scale = nval * torch.as_tensor(
                    gamma_candidate, device=device, dtype=torch.float32
                ).reshape(-1, 1) / delta_edos
                control_prediction = torch.clamp(shape * control_scale, min=0.0)
                candidate_prediction = torch.clamp(shape * candidate_scale, min=0.0)
                result["gamma_candidate"].append(gamma_candidate)
                result["r2_edos_oracle"].append(
                    per_sample_spectral_metrics(oracle_prediction, target)["r2"]
                    .cpu()
                    .numpy()
                )
                result["r2_edos_blind_control"].append(
                    per_sample_spectral_metrics(control_prediction, target)["r2"]
                    .cpu()
                    .numpy()
                )
                result["r2_edos_blind_candidate"].append(
                    per_sample_spectral_metrics(candidate_prediction, target)["r2"]
                    .cpu()
                    .numpy()
                )
    arrays = {name: np.concatenate(chunks) for name, chunks in result.items()}
    expected_count = EXPECTED_COUNTS[split]
    if any(values.shape != (expected_count,) for values in arrays.values()):
        raise ValueError(f"Q1 {split} output count mismatch")
    return arrays


def _abs_log_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.clip(prediction, EPS, 1.0)
    target = np.clip(target, EPS, 1.0)
    return np.abs(np.log(prediction) - np.log(target))


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "failure_pct": float(np.mean(values < 0.0) * 100.0),
    }


def run_gate(
    checkpoint_path: Path,
    config_path: Path,
    output_prefix: Path,
    device: torch.device,
    bootstrap_replicates: int = 2_000,
) -> tuple[Path, Path, dict[str, Any]]:
    os.chdir(REPO_ROOT)
    json_path = output_prefix.with_suffix(".json")
    samples_path = output_prefix.with_name(
        output_prefix.name + "_valid_samples"
    ).with_suffix(".csv")
    existing = [path for path in (json_path, samples_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite preregistered outputs: "
            + ", ".join(str(path) for path in existing)
        )
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")

    checkpoint, saved, identity = _authenticate_inputs(checkpoint_path, config_path)
    model, transformer, loaders = _build_model_and_loaders(
        checkpoint, saved, device
    )
    state_digest_before = _state_digest(transformer)
    train = _collect_split(model, loaders["train"], device, "train")
    fit = fit_gamma_calibration(train["gamma_control"], train["gamma_true"])
    valid = _collect_split(
        model,
        loaders["valid"],
        device,
        "valid",
        calibration=(float(fit["a"]), float(fit["b"])),
    )
    state_digest_after = _state_digest(transformer)
    invariants = {
        "checkpoint_state_unchanged": state_digest_before == state_digest_after,
        "edos_shape_shared_exactly": True,
        "edos_oracle_unchanged_by_construction": True,
        "phdos_outputs_untouched": True,
    }

    control_error = _abs_log_error(valid["gamma_control"], valid["gamma_true"])
    candidate_error = _abs_log_error(valid["gamma_candidate"], valid["gamma_true"])
    gate = adjudicate_gate(
        valid["r2_edos_blind_control"],
        valid["r2_edos_blind_candidate"],
        control_error,
        candidate_error,
        invariants,
        replicates=bootstrap_replicates,
    )
    valid_ids = np.load(
        REPO_ROOT / "data/train4ARPAT/valid/valid_index.npy"
    ).astype(str)
    if valid_ids.shape != (EXPECTED_COUNTS["valid"],):
        raise ValueError("Q1 valid index count mismatch")
    samples = pd.DataFrame(
        {
            "sample_index": np.arange(EXPECTED_COUNTS["valid"], dtype=np.int64),
            "mpid": valid_ids,
            "gamma_true": valid["gamma_true"],
            "gamma_control": valid["gamma_control"],
            "gamma_candidate": valid["gamma_candidate"],
            "gamma_control_abs_log_error": control_error,
            "gamma_candidate_abs_log_error": candidate_error,
            "r2_edos_oracle": valid["r2_edos_oracle"],
            "r2_edos_blind_control": valid["r2_edos_blind_control"],
            "r2_edos_blind_candidate": valid["r2_edos_blind_candidate"],
            "control_fail": valid["r2_edos_blind_control"] < 0.0,
            "candidate_fail": valid["r2_edos_blind_candidate"] < 0.0,
        }
    )
    report = {
        "experiment": "h1_gamma_two_scalar_calibration_gate",
        "data_contract": {
            "fit_split": "Q1 train only",
            "verdict_split": "Q1 valid only",
            "test_accessed": False,
            "counts": EXPECTED_COUNTS,
        },
        "identity": identity,
        "calibration": {
            "mapping": "sigmoid(a * logit(clamp(gamma_control, 1e-6, 1-1e-6)) + b)",
            **fit,
        },
        "train": {
            "n": EXPECTED_COUNTS["train"],
            "gamma_control_median_abs_log_error": float(
                np.median(_abs_log_error(train["gamma_control"], train["gamma_true"]))
            ),
            "gamma_candidate_median_abs_log_error": float(
                np.median(
                    _abs_log_error(
                        apply_gamma_calibration(
                            train["gamma_control"], float(fit["a"]), float(fit["b"])
                        ),
                        train["gamma_true"],
                    )
                )
            ),
        },
        "valid": {
            "n": EXPECTED_COUNTS["valid"],
            "edos_oracle": _metric_summary(valid["r2_edos_oracle"]),
            "edos_blind_control": _metric_summary(valid["r2_edos_blind_control"]),
            "edos_blind_candidate": _metric_summary(valid["r2_edos_blind_candidate"]),
            "gamma_control_median_abs_log_error": float(np.median(control_error)),
            "gamma_candidate_median_abs_log_error": float(np.median(candidate_error)),
        },
        "invariants": {
            **invariants,
            "model_state_sha256_before": state_digest_before,
            "model_state_sha256_after": state_digest_after,
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
            "paired": True,
        },
        "gate": gate,
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(samples_path, index=False)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return json_path, samples_path, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Q1-train H1 gamma calibration and adjudicate on Q1 valid."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    json_path, samples_path, report = run_gate(
        args.checkpoint,
        args.config,
        args.output_prefix,
        device,
        bootstrap_replicates=args.bootstrap,
    )
    print(f"verdict={report['gate']['verdict']}")
    print(f"result={json_path}")
    print(f"samples={samples_path}")


if __name__ == "__main__":
    main()
