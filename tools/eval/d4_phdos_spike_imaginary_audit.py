#!/usr/bin/env python3
"""D4 read-only audit for P0 peak concentration and negative-coordinate mass.

Negative P0 coordinates are a label-shape proxy, not proof of an imaginary
phonon or a data error.  Thresholds are fit on Q1 train only; B7 test R2 is
used solely for the pre-registered failure stratification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 2000
ACTION_MIN_N = 100
ACTION_MIN_DELTA_FAIL_PT = 3.0


def p0_negative_center_count(grid_edges: np.ndarray) -> int:
    edges = np.asarray(grid_edges, dtype=np.float64)
    if edges.shape != (65,):
        raise ValueError(f"expected P0's 65 edges, got {edges.shape}")
    centers = (edges[:-1] + edges[1:]) / 2.0
    return int((centers < 0.0).sum())


def safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return finite zero for degenerate all-zero spectra, never NaN/Inf."""
    result = np.zeros_like(denominator, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def phdos_descriptors(target: np.ndarray, coverage: np.ndarray, negative_bins: int) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=bool)
    if target.ndim != 2 or target.shape[1] != 64 or coverage.shape != target.shape:
        raise ValueError("D4 expects matching [N,64] phDOS target and coverage arrays")
    if negative_bins <= 0 or negative_bins >= target.shape[1]:
        raise ValueError("negative P0 bin count is outside the expected range")
    if not np.isfinite(target).all() or (target < 0.0).any():
        raise ValueError("phDOS targets must be finite and non-negative")
    total = target.sum(axis=1)
    return {
        "total": total,
        "negative_mass_fraction": safe_fraction(target[:, :negative_bins].sum(axis=1), total),
        "peak_share": safe_fraction(target.max(axis=1), total),
        "uncovered_mass_fraction": safe_fraction((target * (~coverage)).sum(axis=1), total),
    }


def train_p90_thresholds(descriptors: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        key: float(np.quantile(descriptors[key], 0.90))
        for key in ("negative_mass_fraction", "peak_share")
    }


def bootstrap_failure_delta(high_fail: np.ndarray, other_fail: np.ndarray, seed: int = BOOTSTRAP_SEED,
                            replicates: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    """Return percentile CI for high-minus-other failure-rate difference, in points."""
    high_fail = np.asarray(high_fail, dtype=np.float64)
    other_fail = np.asarray(other_fail, dtype=np.float64)
    if len(high_fail) == 0 or len(other_fail) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    high = rng.choice(high_fail, size=(replicates, len(high_fail)), replace=True).mean(axis=1)
    other = rng.choice(other_fail, size=(replicates, len(other_fail)), replace=True).mean(axis=1)
    return tuple(float(value * 100.0) for value in np.quantile(high - other, [0.025, 0.975]))


def b7_strata(descriptors: dict[str, np.ndarray], thresholds: dict[str, float], r2_phdos: np.ndarray) -> pd.DataFrame:
    r2 = np.asarray(r2_phdos, dtype=np.float64)
    rows = []
    for metric, threshold in thresholds.items():
        high = descriptors[metric] >= threshold
        for label, group in (("high", high), ("other", ~high)):
            value = r2[group]
            rows.append({
                "metric": metric,
                "threshold_train_p90": threshold,
                "stratum": label,
                "n": int(group.sum()),
                "r2_median": float(np.median(value)) if len(value) else float("nan"),
                "fail_rate_phdos": float((value < 0.0).mean() * 100.0) if len(value) else float("nan"),
            })
        high_fail = r2[high] < 0.0
        other_fail = r2[~high] < 0.0
        low, upper = bootstrap_failure_delta(high_fail, other_fail)
        delta = float(high_fail.mean() * 100.0 - other_fail.mean() * 100.0)
        rows.append({
            "metric": metric,
            "threshold_train_p90": threshold,
            "stratum": "high_minus_other",
            "n": int(high.sum()),
            "r2_median": float("nan"),
            "fail_rate_phdos": delta,
            "delta_fail_ci95_low": low,
            "delta_fail_ci95_high": upper,
            "actionable": bool(high.sum() >= ACTION_MIN_N and delta >= ACTION_MIN_DELTA_FAIL_PT and low > 0.0),
        })
    return pd.DataFrame(rows)


def descriptor_summary(split: str, descriptors: dict[str, np.ndarray], thresholds: dict[str, float]) -> list[dict]:
    rows = []
    for metric in ("negative_mass_fraction", "peak_share", "uncovered_mass_fraction"):
        values = descriptors[metric]
        threshold = thresholds.get(metric)
        high = values >= threshold if threshold is not None else np.zeros(len(values), dtype=bool)
        rows.append({
            "split": split,
            "metric": metric,
            "n": len(values),
            "zero_total_n": int((descriptors["total"] <= 0.0).sum()),
            "q50": float(np.quantile(values, 0.50)),
            "q90": float(np.quantile(values, 0.90)),
            "q99": float(np.quantile(values, 0.99)),
            "threshold_train_p90": threshold,
            "high_n": int(high.sum()) if threshold is not None else None,
            "high_fraction": float(high.mean()) if threshold is not None else None,
        })
    return rows


def _load_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    directory = data_root / split
    target = np.load(directory / f"phdos_tgtdos_{split}.npy", mmap_mode="r")
    coverage = np.load(directory / f"phdos_mask_{split}.npy", mmap_mode="r")
    return target, coverage


def run_audit(data_root: Path, results_root: Path) -> dict:
    with (REPO_ROOT / "data" / "grids_c2b" / "grids.json").open(encoding="utf-8") as handle:
        p0_edges = np.asarray(json.load(handle)["P0"], dtype=np.float64)
    negative_bins = p0_negative_center_count(p0_edges)

    descriptors_by_split = {}
    for split in SPLITS:
        target, coverage = _load_split(data_root, split)
        descriptors_by_split[split] = phdos_descriptors(target, coverage, negative_bins)
    thresholds = train_p90_thresholds(descriptors_by_split["train"])

    descriptor_rows = []
    for split in SPLITS:
        descriptor_rows.extend(descriptor_summary(split, descriptors_by_split[split], thresholds))
    descriptor_df = pd.DataFrame(descriptor_rows)

    b7_samples_path = results_root / "samples_m1_e9ctl_test.csv"
    b7_samples = pd.read_csv(b7_samples_path)
    test_descriptors = descriptors_by_split["test"]
    test_indices = np.load(data_root / "test" / "test_index.npy", allow_pickle=True)
    if len(b7_samples) != len(test_indices) or len(b7_samples) != len(test_descriptors["total"]):
        raise ValueError("B7 test rows, Q1 test index, and phDOS targets must align exactly")
    if "r2_phdos" not in b7_samples:
        raise ValueError("B7 sample table has no r2_phdos column")
    strata_df = b7_strata(test_descriptors, thresholds, b7_samples["r2_phdos"].to_numpy())

    sample_df = pd.DataFrame({
        "sample_index": test_indices.astype(str),
        "r2_phdos": b7_samples["r2_phdos"].to_numpy(),
        "phdos_failed": (b7_samples["r2_phdos"].to_numpy() < 0.0),
        **{key: value for key, value in test_descriptors.items() if key != "total"},
        **{f"high_{key}": test_descriptors[key] >= threshold for key, threshold in thresholds.items()},
    })

    results_root.mkdir(parents=True, exist_ok=True)
    descriptor_path = results_root / "d4_phdos_descriptor_summary.csv"
    strata_path = results_root / "d4_phdos_b7_strata.csv"
    samples_path = results_root / "d4_phdos_b7_test_samples.csv"
    summary_path = results_root / "d4_phdos_audit_summary.json"
    descriptor_df.to_csv(descriptor_path, index=False)
    strata_df.to_csv(strata_path, index=False)
    sample_df.to_csv(samples_path, index=False)

    actionable = strata_df.loc[strata_df["stratum"] == "high_minus_other"]
    summary = {
        "audit": "D4 phDOS spike and negative-coordinate quality audit",
        "data_root": str(data_root),
        "negative_center_bins": negative_bins,
        "negative_center_range_cm-1": [float(((p0_edges[:-1] + p0_edges[1:]) / 2.0)[0]),
                                          float(((p0_edges[:-1] + p0_edges[1:]) / 2.0)[negative_bins - 1])],
        "threshold_source": "Q1 train p90 only",
        "thresholds": thresholds,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "action_rule": {"min_n": ACTION_MIN_N, "min_delta_failure_points": ACTION_MIN_DELTA_FAIL_PT,
                        "ci_lower_gt_zero": True},
        "actionable_metrics": actionable.loc[actionable["actionable"] == True, "metric"].tolist(),
        "outputs": {"descriptor_summary": str(descriptor_path), "b7_strata": str(strata_path), "b7_test_samples": str(samples_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    summary = run_audit(REPO_ROOT / "data" / "train4ARPAT", REPO_ROOT / "results")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
