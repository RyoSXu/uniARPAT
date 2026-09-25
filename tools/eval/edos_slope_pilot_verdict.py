"""Paired valid-only verdict for the eDOS slope-loss pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval.edos_error_attribution import run_audit


TRAIN_ROUGHNESS_P90 = 0.351600
R2_MARGIN = 0.02
FAILURE_MARGIN_PP = 1.0


def paired_bootstrap_interval(
    control: np.ndarray,
    candidate: np.ndarray,
    statistic: str = "median",
    replicates: int = 2000,
    seed: int = 20260923,
) -> tuple[float, float]:
    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if control.ndim != 1 or candidate.shape != control.shape or control.size == 0:
        raise ValueError("paired values must be matching, nonempty one-dimensional arrays")
    if not np.isfinite(control).all() or not np.isfinite(candidate).all():
        raise ValueError("paired bootstrap values must be finite")
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, control.size, size=(replicates, control.size))
    reducer = np.median if statistic == "median" else np.mean
    differences = reducer(candidate[indices], axis=1) - reducer(control[indices], axis=1)
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def _metric_summary(
    control: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
    seed: int,
    track_failures: bool,
) -> dict[str, float]:
    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    low, high = paired_bootstrap_interval(
        control, candidate, "median", replicates=replicates, seed=seed
    )
    summary = {
        "control_median": float(np.median(control)),
        "candidate_median": float(np.median(candidate)),
        "delta_median": float(np.median(candidate) - np.median(control)),
        "delta_median_ci_low": low,
        "delta_median_ci_high": high,
    }
    if track_failures:
        control_failed = (control < 0.0).astype(np.float64)
        candidate_failed = (candidate < 0.0).astype(np.float64)
        fail_low, fail_high = paired_bootstrap_interval(
            control_failed,
            candidate_failed,
            "mean",
            replicates=replicates,
            seed=seed,
        )
        summary.update(
            {
                "control_fail_pct": float(control_failed.mean() * 100.0),
                "candidate_fail_pct": float(candidate_failed.mean() * 100.0),
                "delta_fail_pp": float(
                    (candidate_failed.mean() - control_failed.mean()) * 100.0
                ),
                "delta_fail_ci_low_pp": fail_low * 100.0,
                "delta_fail_ci_high_pp": fail_high * 100.0,
            }
        )
    return summary


def analyze_paired_samples(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
    replicates: int = 2000,
    seed: int = 20260923,
    roughness_threshold: float = TRAIN_ROUGHNESS_P90,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not control["mpid"].astype(str).is_unique or not candidate["mpid"].astype(str).is_unique:
        raise ValueError("valid sample IDs must be unique in each arm")
    if not np.array_equal(
        control["mpid"].astype(str).to_numpy(), candidate["mpid"].astype(str).to_numpy()
    ):
        raise ValueError("control and candidate valid sample IDs/order do not match")

    roughness_column = "edos_spectral_roughness"
    if roughness_column not in control or roughness_column not in candidate:
        raise ValueError("valid samples are missing eDOS roughness")
    control_roughness = control[roughness_column].to_numpy(dtype=np.float64)
    candidate_roughness = candidate[roughness_column].to_numpy(dtype=np.float64)
    if not np.allclose(control_roughness, candidate_roughness, atol=1e-12, rtol=0):
        raise ValueError("control and candidate roughness targets do not match")
    high_mask = control_roughness >= roughness_threshold
    if not high_mask.any() or high_mask.all():
        raise ValueError("roughness p90 split must contain high and comparison samples")

    metric_columns = {
        "edos_oracle": "r2_edos_oracle_unmasked",
        "edos_blind": "r2_edos_blind_unmasked",
        "phdos_oracle": "r2_phdos_oracle_unmasked",
        "phdos_blind": "r2_phdos_blind_unmasked",
        "roughness_bias": "edos_roughness_bias",
        "slope_error_high_gradient_mae": "edos_slope_error_high_gradient_mae",
    }
    for column in metric_columns.values():
        if column not in control or column not in candidate:
            raise ValueError(f"valid sample rows are missing {column}")

    populations = {"all": np.ones(len(control), dtype=bool), "roughness_train_p90": high_mask}
    metrics = []
    summary_by_key = {}
    for population_name, population_mask in populations.items():
        for metric_name, column in metric_columns.items():
            if population_name == "all" and metric_name in {
                "roughness_bias",
                "slope_error_high_gradient_mae",
            }:
                continue
            if population_name == "roughness_train_p90" and metric_name in {
                "edos_blind",
                "phdos_oracle",
                "phdos_blind",
            }:
                continue
            control_values = control.loc[population_mask, column].to_numpy(dtype=np.float64)
            candidate_values = candidate.loc[population_mask, column].to_numpy(dtype=np.float64)
            track_failures = metric_name in {
                "edos_oracle",
                "edos_blind",
                "phdos_oracle",
                "phdos_blind",
            }
            summary = _metric_summary(
                control_values,
                candidate_values,
                replicates,
                seed,
                track_failures,
            )
            row = {
                "population": population_name,
                "metric": metric_name,
                "n": int(population_mask.sum()),
                **summary,
            }
            metrics.append(row)
            summary_by_key[(population_name, metric_name)] = summary

    overall_oracle = summary_by_key[("all", "edos_oracle")]
    high_oracle = summary_by_key[("roughness_train_p90", "edos_oracle")]
    primary_met = (
        high_oracle["delta_median"] >= R2_MARGIN
        and high_oracle["delta_fail_pp"] <= FAILURE_MARGIN_PP
    )
    overall_guards = [overall_oracle]
    secondary_guards = [
        summary_by_key[("all", "edos_blind")],
        summary_by_key[("all", "phdos_oracle")],
        summary_by_key[("all", "phdos_blind")],
    ]
    guardrails_met = all(
        item["delta_median"] >= -R2_MARGIN
        and item["delta_fail_pp"] <= FAILURE_MARGIN_PP
        for item in overall_guards + secondary_guards
    )
    mechanism = summary_by_key[("roughness_train_p90", "slope_error_high_gradient_mae")]
    mechanism_supported = mechanism["delta_median_ci_high"] < 0.0
    verdict = (
        "reject"
        if not guardrails_met
        else "win"
        if primary_met and mechanism_supported
        else "park"
    )
    decision = {
        "verdict": verdict,
        "primary_met": primary_met,
        "guardrails_met": guardrails_met,
        "mechanism_supported": mechanism_supported,
        "roughness_threshold": float(roughness_threshold),
        "roughness_high_n": int(high_mask.sum()),
        "roughness_other_n": int((~high_mask).sum()),
        "rules": {
            "high_roughness_oracle_delta_median_min": R2_MARGIN,
            "high_roughness_fail_delta_max_pp": FAILURE_MARGIN_PP,
            "overall_oracle_and_blind_phdos_guards": {
                "delta_median_min": -R2_MARGIN,
                "delta_fail_max_pp": FAILURE_MARGIN_PP,
            },
            "mechanism": "high-gradient slope-error median-delta 95% CI upper bound < 0",
        },
    }
    return pd.DataFrame(metrics), decision


def run_valid_pilot(
    control_checkpoint: Path,
    control_config: Path,
    candidate_checkpoint: Path,
    candidate_config: Path,
    output_prefix: Path,
    device: torch.device,
    replicates: int = 2000,
    force: bool = False,
) -> tuple[Path, Path]:
    output_prefix = output_prefix.resolve()
    metrics_path = output_prefix.with_name(f"{output_prefix.name}_paired_metrics.csv")
    verdict_path = output_prefix.with_suffix(".json")
    if not force and (metrics_path.exists() or verdict_path.exists()):
        raise FileExistsError("pilot verdict outputs exist; pass --force to replace them")

    control_prefix = output_prefix.with_name(f"{output_prefix.name}_control")
    candidate_prefix = output_prefix.with_name(f"{output_prefix.name}_candidate")
    control_paths = run_audit(
        control_checkpoint,
        control_config,
        control_prefix,
        device,
        bootstrap_replicates=replicates,
        force=force,
        split="valid",
        expected_epoch=10,
        verify_reference_r2=False,
    )
    candidate_paths = run_audit(
        candidate_checkpoint,
        candidate_config,
        candidate_prefix,
        device,
        bootstrap_replicates=replicates,
        force=force,
        split="valid",
        expected_epoch=10,
        verify_reference_r2=False,
    )
    control_samples = pd.read_csv(control_paths[0])
    candidate_samples = pd.read_csv(candidate_paths[0])
    paired_metrics, decision = analyze_paired_samples(
        control_samples, candidate_samples, replicates=replicates
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paired_metrics.to_csv(metrics_path, index=False)
    verdict_payload = {
        **decision,
        "split": "Q1 valid",
        "checkpoint_epoch": 10,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": 20260923,
        "control_samples": str(control_paths[0]),
        "candidate_samples": str(candidate_paths[0]),
        "paired_metrics": str(metrics_path),
    }
    verdict_path.write_text(json.dumps(verdict_payload, indent=2) + "\n")
    return metrics_path, verdict_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    paths = run_valid_pilot(
        args.control_checkpoint,
        args.control_config,
        args.candidate_checkpoint,
        args.candidate_config,
        args.output_prefix,
        device,
        replicates=args.bootstrap,
        force=args.force,
    )
    print(f"Q1 valid paired metrics: {paths[0]}")
    print(f"Q1 valid pilot verdict: {paths[1]}")


if __name__ == "__main__":
    main()
