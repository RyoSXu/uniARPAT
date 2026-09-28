#!/usr/bin/env python3
"""Pre-registered Q1-valid-only verdict for the eDOS pair auxiliary pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.edos_error_attribution import run_audit, validate_edos_shape_sumnorm
from tools.eval.edos_slope_pilot_verdict import paired_bootstrap_interval
from tools.eval.g2_structure_path_audit import contrast_metrics


EXPECTED_VALID_COUNT = 2313
EXPECTED_EPOCH = 10
BOOTSTRAP_SEED = 20260928
METRIC_COLUMNS = {
    "edos_oracle": "r2_edos_oracle_unmasked",
    "edos_blind": "r2_edos_blind_unmasked",
    "phdos_oracle": "r2_phdos_oracle_unmasked",
    "phdos_blind": "r2_phdos_blind_unmasked",
}


def metric_summary(control, candidate, replicates=2000, seed=BOOTSTRAP_SEED):
    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if control.ndim != 1 or candidate.shape != control.shape or control.size == 0:
        raise ValueError("paired metrics must be matching nonempty vectors")
    median_low, median_high = paired_bootstrap_interval(
        control, candidate, "median", replicates, seed,
    )
    control_failed = (control < 0).astype(np.float64)
    candidate_failed = (candidate < 0).astype(np.float64)
    fail_low, fail_high = paired_bootstrap_interval(
        control_failed, candidate_failed, "mean", replicates, seed,
    )
    return {
        "n": int(control.size),
        "control_median": float(np.median(control)),
        "candidate_median": float(np.median(candidate)),
        "delta_median": float(np.median(candidate) - np.median(control)),
        "delta_median_ci": [median_low, median_high],
        "control_fail_pct": float(100 * control_failed.mean()),
        "candidate_fail_pct": float(100 * candidate_failed.mean()),
        "delta_fail_pp": float(100 * (candidate_failed.mean() - control_failed.mean())),
        "delta_fail_ci_pp": [100 * fail_low, 100 * fail_high],
    }


def mechanism_summary(pair_table, replicates=2000, seed=BOOTSTRAP_SEED):
    held = pair_table.loc[pair_table["primary_valid"].astype(bool)]
    groups = {}
    for arm in ("control", "candidate"):
        frame = held.loc[held["arm"] == arm]
        groups[arm] = frame.groupby("reduced_group", sort=True)[
            "contrast_error_tv"
        ].median()
    if not groups["control"].index.equals(groups["candidate"].index):
        raise ValueError("held-out reduced-composition groups do not align")
    control = groups["control"].to_numpy(dtype=np.float64)
    candidate = groups["candidate"].to_numpy(dtype=np.float64)
    if control.size != 117 or len(held.loc[held["arm"] == "control"]) != 163:
        raise ValueError("held-out mechanism population is not the frozen 117 groups / 163 pairs")
    control_mean = float(control.mean())
    relative_reduction = float((control_mean - candidate.mean()) / control_mean)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, control.size, size=(replicates, control.size))
    control_boot = control[indices].mean(axis=1)
    candidate_boot = candidate[indices].mean(axis=1)
    reductions = (control_boot - candidate_boot) / control_boot
    low, high = np.quantile(reductions, [0.025, 0.975])
    return {
        "pairs": 163,
        "groups": 117,
        "control_group_equal_error": control_mean,
        "candidate_group_equal_error": float(candidate.mean()),
        "relative_reduction": relative_reduction,
        "relative_reduction_ci": [float(low), float(high)],
    }


def evaluate_verdict(overall, paired_subset, mechanism):
    overall_main = (
        overall["edos_blind"]["delta_median"] >= 0.02
        and overall["edos_blind"]["delta_fail_pp"] < 1.0
    )
    shape_guard = (
        overall["edos_oracle"]["delta_median"] >= 0.0
        and overall["edos_oracle"]["delta_fail_pp"] < 1.0
    )
    shared_guard = all(
        overall[name]["delta_median"] >= -0.02
        and overall[name]["delta_fail_pp"] < 1.0
        for name in ("phdos_oracle", "phdos_blind")
    )
    mechanism_met = (
        mechanism["relative_reduction"] >= 0.05
        and mechanism["relative_reduction_ci"][0] > 0.0
    )
    paired_guard = all(
        paired_subset[name]["delta_median"] >= -0.02
        and paired_subset[name]["delta_fail_pp"] < 1.0
        for name in ("edos_oracle", "edos_blind")
    )
    gates = {
        "overall_main": overall_main,
        "shape_guard": shape_guard,
        "shared_task_guard": shared_guard,
        "mechanism": mechanism_met,
        "paired_material_guard": paired_guard,
    }
    return {"verdict": "win" if all(gates.values()) else "park", "gates": gates}


def validate_artifacts(
    checkpoint, config, arm, expected_plan_hash=None, expected_pair_lambda=None
):
    if Path(checkpoint).name != "checkpoint_latest.pth":
        raise ValueError(f"{arm} verdict requires checkpoint_latest.pth")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("epoch"), payload.get("model_name"), payload.get("seed")) \
            != (EXPECTED_EPOCH, "M1", 42) or payload.get("use_amp", False):
        raise ValueError(f"{arm} must use the epoch-10 M1 latest checkpoint")
    if payload.get("pair_aux_arm") != arm or payload.get("pair_ratio") != 0.10:
        raise ValueError(f"{arm} checkpoint pair metadata mismatch")
    if not 1e-4 <= float(payload.get("pair_lambda", 0.0)) <= 1e4:
        raise ValueError(f"{arm} checkpoint pair lambda is invalid")
    if expected_plan_hash is not None and payload.get("pair_plan_hash") != expected_plan_hash:
        raise ValueError("control and candidate pair plans differ")
    if expected_pair_lambda is not None \
            and payload.get("pair_lambda") != expected_pair_lambda:
        raise ValueError("control and candidate calibrated pair lambdas differ")
    import yaml
    saved = yaml.safe_load(Path(config).read_text())
    cli = saved.get("cli", {})
    runtime = saved.get("runtime", {})
    expected_cli = {
        "model": "M1", "epochs": 10, "batch_size": 32, "lr": 5e-5,
        "seed": 42, "norm": "sumnorm", "scale_mode": "eta",
        "pair_aux_arm": arm, "pair_ratio": 0.10,
        "skip_test_eval": True, "use_amp": False, "use_bucket_batch": False,
        "reset_rng_after_init": True,
    }
    mismatches = {
        key: (cli.get(key), value) for key, value in expected_cli.items()
        if cli.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{arm} config differs from the frozen pilot: {mismatches}")
    if runtime.get("pair_lambda") != payload.get("pair_lambda") \
            or runtime.get("pair_plan_hash") != payload.get("pair_plan_hash"):
        raise ValueError(f"{arm} config and checkpoint calibration metadata differ")
    return payload["pair_plan_hash"], payload["pair_lambda"]


def build_pair_table(shape_paths, sample_frames):
    reference = pd.read_csv(ROOT / "results/g2_encoder_adaptation_q1_pairs.csv")
    reference = reference.loc[
        (reference["arm"] == "original") & (reference["split"] == "valid")
    ].reset_index(drop=True)
    if len(reference) != 320:
        raise ValueError("frozen valid pair reference must contain 320 pairs")
    valid_ids = np.load(ROOT / "data/train4ARPAT/valid/valid_index.npy").astype(str)
    sample_a = reference["sample_index_a"].to_numpy(dtype=np.int64)
    sample_b = reference["sample_index_b"].to_numpy(dtype=np.int64)
    if len({(min(a, b), max(a, b)) for a, b in zip(sample_a, sample_b)}) != 320:
        raise ValueError("frozen valid pair reference contains duplicate pairs")
    if not np.array_equal(valid_ids[sample_a], reference["mpid_a"].astype(str)) \
            or not np.array_equal(valid_ids[sample_b], reference["mpid_b"].astype(str)):
        raise ValueError("frozen pair reference IDs differ from the Q1 valid index")
    raw_targets = np.load(ROOT / "data/train4ARPAT/valid/edos_tgtdos_valid.npy").astype(np.float64)
    targets = raw_targets / raw_targets.sum(axis=1, keepdims=True)
    shapes = {
        arm: validate_edos_shape_sumnorm(np.load(path), EXPECTED_VALID_COUNT)
        for arm, path in shape_paths.items()
    }
    rows = []
    for arm in ("control", "candidate"):
        if not np.array_equal(sample_frames[arm]["mpid"].astype(str), valid_ids):
            raise ValueError(f"{arm} valid sample order differs from frozen Q1 index")
        for pair in reference.itertuples(index=False):
            a, b = int(pair.sample_index_a), int(pair.sample_index_b)
            metrics = contrast_metrics(shapes[arm][a], shapes[arm][b], targets[a], targets[b])
            if abs(metrics["target_tv"] - float(pair.target_tv)) > 2e-6:
                raise ValueError("frozen pair target TV differs from Q1 valid targets")
            rows.append({
                "arm": arm,
                "sample_index_a": a,
                "sample_index_b": b,
                "mpid_a": valid_ids[a],
                "mpid_b": valid_ids[b],
                "reduced_group": pair.reduced_group,
                "primary_valid": bool(pair.primary_valid),
                "contrast_error_tv": metrics["contrast_error_tv"],
                "target_tv": metrics["target_tv"],
                "predicted_tv": metrics["predicted_tv"],
            })
    return pd.DataFrame(rows)


def analyze_frames(frames, pair_table, replicates=2000):
    ids = frames["control"]["mpid"].astype(str).to_numpy()
    if len(ids) != EXPECTED_VALID_COUNT or not np.array_equal(
            ids, frames["candidate"]["mpid"].astype(str).to_numpy()):
        raise ValueError("control and candidate valid rows are not aligned")
    overall = {
        name: metric_summary(
            frames["control"][column], frames["candidate"][column], replicates,
        )
        for name, column in METRIC_COLUMNS.items()
    }
    held = pair_table.loc[
        (pair_table["arm"] == "control") & pair_table["primary_valid"].astype(bool)
    ]
    held_ids = set(held["mpid_a"].astype(str)) | set(held["mpid_b"].astype(str))
    if len(held_ids) != 256:
        raise ValueError("paired protection population is not the frozen 256 materials")
    mask = np.asarray([sample_id in held_ids for sample_id in ids])
    paired_subset = {
        name: metric_summary(
            frames["control"].loc[mask, column],
            frames["candidate"].loc[mask, column],
            replicates,
        )
        for name, column in METRIC_COLUMNS.items()
        if name.startswith("edos_")
    }
    mechanism = mechanism_summary(pair_table, replicates)
    return overall, paired_subset, mechanism, evaluate_verdict(
        overall, paired_subset, mechanism,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("control", "candidate"):
        parser.add_argument(f"--{arm}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{arm}-config", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    plan_hash, pair_lambda = validate_artifacts(
        args.control_checkpoint, args.control_config, "control",
    )
    validate_artifacts(
        args.candidate_checkpoint, args.candidate_config, "candidate",
        plan_hash, pair_lambda,
    )
    verdict_path = args.output_prefix.with_suffix(".json")
    pairs_path = args.output_prefix.with_name(args.output_prefix.name + "_pairs.csv")
    if not args.force and (verdict_path.exists() or pairs_path.exists()):
        raise FileExistsError("verdict outputs exist; pass --force to replace them")
    frames, shape_paths = {}, {}
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        for arm in ("control", "candidate"):
            prefix = staging / arm
            shape_paths[arm] = staging / f"{arm}_shapes.npy"
            paths = run_audit(
                getattr(args, f"{arm}_checkpoint"),
                getattr(args, f"{arm}_config"),
                prefix,
                device,
                bootstrap_replicates=args.bootstrap,
                force=False,
                split="valid",
                expected_epoch=EXPECTED_EPOCH,
                verify_reference_r2=False,
                edos_shape_sumnorm_path=shape_paths[arm],
            )
            frames[arm] = pd.read_csv(paths[0])
        pair_table = build_pair_table(shape_paths, frames)
        overall, paired_subset, mechanism, decision = analyze_frames(
            frames, pair_table, args.bootstrap,
        )
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pair_table.to_csv(pairs_path, index=False)
    payload = {
        **decision,
        "split": "Q1 valid",
        "checkpoint_epoch": EXPECTED_EPOCH,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "overall": overall,
        "paired_256_materials": paired_subset,
        "mechanism_117_groups_163_pairs": mechanism,
    }
    verdict_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
