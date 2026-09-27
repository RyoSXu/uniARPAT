"""Q1 valid-only verdict for the candidate-1 joint-content three-arm pilot.

This tool consumes the three epoch-10 ``checkpoint_latest.pth`` files and the
matching ``config_used.yaml`` files of the ``control`` / ``radial`` / ``joint``
arms and evaluates them on the frozen Q1 valid split only. It never accepts a
split or test argument, and it never reads the test split.

Reuse contract (no new metric semantics):
  * model reconstruction, valid forward passes and per-sample oracle/blind R2
    come from ``tools/eval/edos_error_attribution.py::run_audit`` with
    ``split="valid"`` and ``expected_epoch=10``;
  * paired bootstrap intervals come from
    ``tools/eval/edos_slope_pilot_verdict.py::paired_bootstrap_interval``;
  * the frozen roughness threshold ``TRAIN_ROUGHNESS_P90`` is reused as is.

Verdict contract (design ``docs/design/design-joint-content-pilot.md``):
  * only the ``joint vs control`` point estimates decide ``win``;
  * bootstrap intervals are reported but never enter the verdict;
  * the pure decision covers ``win``, ``tie``,
    ``degraded_or_guard_failed`` and ``technical_incomplete``;
  * ``only_beats_radial`` is an auxiliary classification that never changes
    the primary verdict.

All formal files are staged in a temporary directory and moved into place only
after the whole set has been produced. A preflight failure (bad checkpoint,
missing columns, wrong order, mismatched configuration) therefore leaves no
formal result behind.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval.edos_error_attribution import (  # noqa: E402
    EXPECTED_EDOS_BINS,
    run_audit,
    validate_edos_shape_sumnorm,
)
from tools.eval.edos_slope_pilot_verdict import (  # noqa: E402
    TRAIN_ROUGHNESS_P90,
    paired_bootstrap_interval,
)
from tools.eval.g2_structure_path_audit import contrast_metrics  # noqa: E402
from tools.eval.joint_content_resource_gate import (  # noqa: E402
    ARMS,
    B7_CKPT_SHA256,
    expected_g2_keys,
    file_sha256,
)

G2_PREFIX = "encoder.g2_msgs."
ARM_ORDER = tuple(ARMS)
EXPECTED_EPOCH = 10
EXPECTED_VALID_COUNT = 2313
B7_INIT_CKPT = "output/ablation_m1_e9ctl/checkpoint_best.pth"
BOOTSTRAP_SEED = 20260923

R2_WIN_MARGIN = 0.02
R2_GUARD_MARGIN = 0.02
# Failure-rate margins are fractions of samples: 0.01 == 1 percentage point.
FAILURE_MARGIN = 0.01

METRIC_COLUMNS = {
    "edos_oracle": "r2_edos_oracle_unmasked",
    "edos_blind": "r2_edos_blind_unmasked",
    "phdos_oracle": "r2_phdos_oracle_unmasked",
    "phdos_blind": "r2_phdos_blind_unmasked",
}
METRICS = tuple(METRIC_COLUMNS)
# Fixed comparison roles (candidate minus reference).
COMPARISONS = (
    ("joint_vs_control", "control", "joint"),
    ("joint_vs_radial", "radial", "joint"),
    ("radial_vs_control", "control", "radial"),
)
ARM_OUTPUT_DIRS = {
    "control": "ablation_m1_jcctl",
    "radial": "ablation_m1_jcrad",
    "joint": "ablation_m1_jcjoint",
}

# Composition-pair auxiliary readout: frozen 320-pair table and SumNorm targets.
COMPOSITION_PAIRS_REFERENCE = "results/edos_spectral_support_q1_valid_pairs.csv"
VALID_INDEX_PATH = "data/train4ARPAT/valid/valid_index.npy"
VALID_TARGET_PATH = "data/train4ARPAT/valid/edos_tgtdos_valid.npy"
COMPOSITION_TARGET_ATOL = 2e-6
COMPOSITION_PAIR_METRICS = ("predicted_tv", "contrast_error_tv", "contrast_cosine")
COMPOSITION_PAIR_COLUMNS = (
    "arm",
    "sample_index_a",
    "sample_index_b",
    "mpid_a",
    "mpid_b",
    "target_tv",
    "predicted_tv",
    "contrast_error_tv",
    "contrast_cosine",
)


# ---------------------------------------------------------------------------
# Preflight helpers (pure, CPU-testable)
# ---------------------------------------------------------------------------

def normalize_path(value, repo_root=REPO_ROOT):
    """Return an absolute, normalized path string for identity comparison."""
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(repo_root) / path
    return os.path.normpath(os.path.abspath(str(path)))


def load_arm_config(path):
    """Load one ``config_used.yaml`` mapping."""
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    return config


def _arm_expected(arm):
    if arm not in ARM_ORDER:
        raise ValueError(f"arm must be one of {ARM_ORDER}, got {arm!r}")
    use_g2 = arm != "control"
    content_mode = "joint" if arm == "joint" else "radial"
    return use_g2, content_mode


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_arm_config(arm, config, *, b7_init_path=B7_INIT_CKPT, repo_root=REPO_ROOT):
    """Validate one arm's frozen pilot contract and return its ``cli`` mapping."""
    expected_use_g2, expected_mode = _arm_expected(arm)
    cli = config.get("cli")
    _require(isinstance(cli, dict), f"{arm} config must contain a cli mapping")
    effective = config.get("config")
    _require(isinstance(effective, dict), f"{arm} config must contain a config mapping")

    _require(cli.get("model") == "M1", f"{arm} pilot must be M1")
    _require(cli.get("epochs") == EXPECTED_EPOCH,
             f"{arm} pilot config epochs must be {EXPECTED_EPOCH}")
    _require(cli.get("batch_size") == 32, f"{arm} pilot batch_size must be 32")
    _require(cli.get("seed") == 42, f"{arm} pilot seed must be 42")
    _require(cli.get("norm") == "sumnorm", f"{arm} pilot norm must be sumnorm")
    _require(cli.get("scale_mode") == "eta", f"{arm} pilot scale_mode must be eta")
    _require(cli.get("use_g2") is expected_use_g2,
             f"{arm} use_g2 must be {expected_use_g2}, got {cli.get('use_g2')!r}")
    _require(cli.get("g2_content_mode") == expected_mode,
             f"{arm} g2_content_mode must be {expected_mode!r}, "
             f"got {cli.get('g2_content_mode')!r}")
    _require(cli.get("skip_test_eval") is True, f"{arm} must set skip_test_eval=true")
    _require(cli.get("use_amp") is False, f"{arm} must set use_amp=false")
    _require(cli.get("use_bucket_batch") is False,
             f"{arm} must set use_bucket_batch=false")
    _require(cli.get("reset_rng_after_init") is True,
             f"{arm} must set reset_rng_after_init=true")
    init_ckpt = cli.get("init_ckpt")
    _require(bool(init_ckpt), f"{arm} must record init_ckpt")
    _require(normalize_path(init_ckpt, repo_root)
             == normalize_path(b7_init_path, repo_root),
             f"{arm} init_ckpt must normalize to the B7 checkpoint {b7_init_path}")

    transformer = (effective.get("model", {}).get("params", {})
                   .get("sub_model", {}).get("transformer", {}))
    _require(transformer.get("use_g2") is expected_use_g2,
             f"{arm} effective transformer use_g2 disagrees with cli")
    _require(transformer.get("g2_content_mode") == expected_mode,
             f"{arm} effective transformer g2_content_mode disagrees with cli")
    _require(effective.get("model", {}).get("params", {}).get("use_amp", False) is False,
             f"{arm} effective config must set use_amp=false")
    return cli


def strip_arm_specific_keys(config):
    """Copy a config with only the per-arm keys removed for equality checking."""
    stripped = copy.deepcopy(config)
    cli = stripped.get("cli")
    if isinstance(cli, dict):
        cli.pop("use_g2", None)
        cli.pop("g2_content_mode", None)
    transformer = (stripped.get("config", {}).get("model", {}).get("params", {})
                   .get("sub_model", {}).get("transformer"))
    if isinstance(transformer, dict):
        transformer.pop("use_g2", None)
        transformer.pop("g2_content_mode", None)
    return stripped


def validate_cross_arm_configs(configs):
    """Require the three arm configs to match beyond use_g2/g2_content_mode.

    ``tag`` is intentionally not compared: the runner never writes it into
    ``config_used.yaml``. This is recorded as a documented gap in the summary.
    """
    _require(set(configs) == set(ARM_ORDER), "configs must cover control/radial/joint")
    baseline = strip_arm_specific_keys(configs["control"])
    for arm in ARM_ORDER:
        if strip_arm_specific_keys(configs[arm]) != baseline:
            raise ValueError(
                f"{arm} config differs from control beyond use_g2/g2_content_mode")


def validate_checkpoint(payload, arm, config):
    """Check the epoch-10 checkpoint identity and G2 key set for one arm."""
    expected_use_g2, expected_mode = _arm_expected(arm)
    _require(isinstance(payload, dict) and "model" in payload,
             f"{arm} checkpoint payload must be a dict with a model state dict")
    _require(payload.get("epoch") == EXPECTED_EPOCH,
             f"{arm} checkpoint epoch must be {EXPECTED_EPOCH}, "
             f"got {payload.get('epoch')!r}")
    _require(payload.get("model_name") == "M1", f"{arm} checkpoint model must be M1")
    _require(payload.get("use_amp", False) is False,
             f"{arm} checkpoint must be FP32 (use_amp false)")
    state = payload["model"]
    _require(isinstance(state, dict), f"{arm} checkpoint model entry must be a state dict")

    transformer = (config.get("config", {}).get("model", {}).get("params", {})
                   .get("sub_model", {}).get("transformer", {}))
    num_layers = int(transformer.get("num_encoder_layers", 6))
    mode = expected_mode if expected_use_g2 else "control"
    expected_keys = expected_g2_keys(num_layers, mode)
    actual_keys = {key for key in state if key.startswith(G2_PREFIX)}
    _require(actual_keys == expected_keys,
             f"{arm} checkpoint G2 key set mismatch: "
             f"missing={sorted(expected_keys - actual_keys)[:5]} "
             f"unexpected={sorted(actual_keys - expected_keys)[:5]}")
    return state


def validate_arm_artifact_paths(arm, checkpoint_path, config_path):
    """Require the frozen unique tag directory and its sibling config file."""
    _arm_expected(arm)
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)
    expected_dir = ARM_OUTPUT_DIRS[arm]
    _require(checkpoint_path.name == "checkpoint_latest.pth",
             f"{arm} verdict checkpoint must be checkpoint_latest.pth")
    _require(checkpoint_path.parent.name == expected_dir,
             f"{arm} checkpoint must be under {expected_dir}")
    _require(config_path.name == "config_used.yaml",
             f"{arm} config must be config_used.yaml")
    _require(config_path.parent.resolve() == checkpoint_path.parent.resolve(),
             f"{arm} checkpoint and config must be sibling artifacts")


def load_and_validate_checkpoint(arm, path, config):
    """Load one checkpoint for preflight validation; never mutates any model.

    The formal verdict is frozen to epoch-10 ``checkpoint_latest.pth``; using a
    ``checkpoint_best.pth`` (or any other file name) is rejected here.
    """
    _require(Path(path).name == "checkpoint_latest.pth",
             f"{arm} verdict checkpoint must be checkpoint_latest.pth, got {Path(path).name}")
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    try:
        return validate_checkpoint(payload, arm, config)
    finally:
        del payload


def validate_sample_frames(frames, *, expected_count=EXPECTED_VALID_COUNT,
                           expected_ids=None):
    """Require equal-count, uniquely identified, same-order valid arms."""
    _require(set(frames) == set(ARM_ORDER),
             "sample frames must cover control/radial/joint")
    reference_ids = None
    for arm in ARM_ORDER:
        frame = frames[arm]
        if len(frame) != expected_count:
            raise ValueError(
                f"{arm} valid samples n={len(frame)} != {expected_count}")
        for column in METRIC_COLUMNS.values():
            if column not in frame.columns:
                raise ValueError(f"{arm} valid samples are missing {column}")
        ids = frame["mpid"].astype(str).to_numpy()
        if len(set(ids)) != len(ids):
            raise ValueError(f"{arm} valid sample IDs are not unique")
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(ids, reference_ids):
            raise ValueError("valid sample IDs/order do not match across arms")
    if expected_ids is not None:
        expected = np.asarray(expected_ids, dtype=str)
        if not np.array_equal(reference_ids, expected):
            raise ValueError("valid sample IDs/order do not match the frozen index")
    return reference_ids


# ---------------------------------------------------------------------------
# Paired metrics and pure verdict
# ---------------------------------------------------------------------------

def paired_metric_summary(reference, candidate, metric, *, replicates=2000,
                          seed=BOOTSTRAP_SEED):
    """Paired reference/candidate median R2 and failure-rate summary.

    Failure is ``R2 < 0`` (project definition). Bootstrap intervals are only
    reported; they are never used by the verdict. Non-finite inputs yield NaN
    deltas so the verdict can report ``technical_incomplete`` instead of
    inventing a number.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or candidate.shape != reference.shape or reference.size == 0:
        raise ValueError("paired metric values must be matching nonempty 1D arrays")

    reference_failed = (reference < 0.0).astype(np.float64)
    candidate_failed = (candidate < 0.0).astype(np.float64)
    summary = {
        "metric": metric,
        "n": int(reference.size),
        "reference_median": float(np.median(reference)),
        "candidate_median": float(np.median(candidate)),
        "delta_median": float(np.median(candidate) - np.median(reference)),
        "reference_fail": float(reference_failed.mean()),
        "candidate_fail": float(candidate_failed.mean()),
        "delta_fail": float(candidate_failed.mean() - reference_failed.mean()),
    }
    finite = np.isfinite(reference).all() and np.isfinite(candidate).all()
    if finite:
        low, high = paired_bootstrap_interval(
            reference, candidate, "median", replicates=replicates, seed=seed)
        fail_low, fail_high = paired_bootstrap_interval(
            reference_failed, candidate_failed, "mean", replicates=replicates, seed=seed)
        summary.update({
            "delta_median_ci_low": low,
            "delta_median_ci_high": high,
            "delta_fail_ci_low": fail_low,
            "delta_fail_ci_high": fail_high,
        })
    else:
        summary.update({
            "delta_median_ci_low": float("nan"),
            "delta_median_ci_high": float("nan"),
            "delta_fail_ci_low": float("nan"),
            "delta_fail_ci_high": float("nan"),
        })
    return summary


def build_comparison(reference_frame, candidate_frame, *, replicates=2000,
                     seed=BOOTSTRAP_SEED):
    """Per-metric paired comparison of one candidate against its reference arm."""
    return {
        metric: paired_metric_summary(
            reference_frame[column].to_numpy(dtype=np.float64),
            candidate_frame[column].to_numpy(dtype=np.float64),
            metric,
            replicates=replicates,
            seed=seed,
        )
        for metric, column in METRIC_COLUMNS.items()
    }


def _win_conditions(comparison):
    """The seven pre-registered win conditions, boundaries exactly as frozen."""
    return {
        "blind_edos_median_ge_0.02":
            comparison["edos_blind"]["delta_median"] >= R2_WIN_MARGIN,
        "blind_edos_fail_lt_0.01":
            comparison["edos_blind"]["delta_fail"] < FAILURE_MARGIN,
        "oracle_edos_median_ge_0":
            comparison["edos_oracle"]["delta_median"] >= 0.0,
        "oracle_phdos_median_ge_-0.02":
            comparison["phdos_oracle"]["delta_median"] >= -R2_WIN_MARGIN,
        "oracle_phdos_fail_lt_0.01":
            comparison["phdos_oracle"]["delta_fail"] < FAILURE_MARGIN,
        "blind_phdos_median_ge_-0.02":
            comparison["phdos_blind"]["delta_median"] >= -R2_WIN_MARGIN,
        "blind_phdos_fail_lt_0.01":
            comparison["phdos_blind"]["delta_fail"] < FAILURE_MARGIN,
    }


def evaluate_joint_verdict(comparison, *, technical_incomplete=False):
    """Pure verdict over ``joint vs control`` point estimates only.

    No bootstrap interval, no ``joint vs radial`` value and no auxiliary readout
    participates here. Missing or non-finite deltas are reported as
    ``technical_incomplete``.
    """
    deltas = {
        metric: {
            "delta_median": float(comparison[metric]["delta_median"]),
            "delta_fail": float(comparison[metric]["delta_fail"]),
        }
        for metric in METRICS
    }
    non_finite = any(
        not math.isfinite(values[key])
        for values in deltas.values()
        for key in ("delta_median", "delta_fail")
    )
    if technical_incomplete or non_finite:
        return {
            "verdict": "technical_incomplete",
            "win_conditions": {},
            "tie_conditions": {},
            "guard_breaches": {},
            "deltas": deltas,
            "reason": "missing or non-finite joint-vs-control point estimates",
        }

    win_conditions = _win_conditions(comparison)
    tie_conditions = {
        metric: (abs(deltas[metric]["delta_median"]) < R2_WIN_MARGIN
                 and abs(deltas[metric]["delta_fail"]) < FAILURE_MARGIN)
        for metric in METRICS
    }
    guard_breaches = {}
    for metric in METRICS:
        guard_breaches[f"{metric}_median_le_-0.02"] = (
            deltas[metric]["delta_median"] <= -R2_GUARD_MARGIN)
        guard_breaches[f"{metric}_fail_ge_0.01"] = (
            deltas[metric]["delta_fail"] >= FAILURE_MARGIN)
    if all(win_conditions.values()):
        verdict = "win"
    elif all(tie_conditions.values()):
        verdict = "tie"
    elif any(guard_breaches.values()):
        verdict = "degraded_or_guard_failed"
    else:
        verdict = "primary_insufficient"
    return {
        "verdict": verdict,
        "win_conditions": win_conditions,
        "tie_conditions": tie_conditions,
        "guard_breaches": guard_breaches,
        "deltas": deltas,
    }


def win_conditions_met(decision):
    """True only when a full, non-empty win-condition set is satisfied."""
    conditions = decision.get("win_conditions") or {}
    return bool(conditions) and all(conditions.values())


def auxiliary_radial_classification(control_decision, radial_decision):
    """Auxiliary joint-vs-radial classification; never affects the verdict."""
    joint_wins_vs_radial = win_conditions_met(radial_decision)
    return {
        "joint_wins_vs_radial": bool(joint_wins_vs_radial),
        "only_beats_radial": bool(
            joint_wins_vs_radial and control_decision["verdict"] != "win"),
    }


# ---------------------------------------------------------------------------
# CSV / auxiliary readouts
# ---------------------------------------------------------------------------

def comparison_frame(name, comparison, reference_arm, candidate_arm):
    """Tabular form of one paired comparison, failure rates in percentage points."""
    rows = []
    for metric in METRICS:
        summary = comparison[metric]
        rows.append({
            "comparison": name,
            "metric": metric,
            "reference_arm": reference_arm,
            "candidate_arm": candidate_arm,
            "n": summary["n"],
            "reference_median": summary["reference_median"],
            "candidate_median": summary["candidate_median"],
            "delta_median": summary["delta_median"],
            "delta_median_ci_low": summary["delta_median_ci_low"],
            "delta_median_ci_high": summary["delta_median_ci_high"],
            "reference_fail_pct": summary["reference_fail"] * 100.0,
            "candidate_fail_pct": summary["candidate_fail"] * 100.0,
            "delta_fail_pp": summary["delta_fail"] * 100.0,
            "delta_fail_ci_low_pp": summary["delta_fail_ci_low"] * 100.0,
            "delta_fail_ci_high_pp": summary["delta_fail_ci_high"] * 100.0,
        })
    return pd.DataFrame(rows)


def _group_readout(kind, name, metric, population, mask, frames, reference_arm,
                   candidate_arm):
    column = METRIC_COLUMNS[metric]
    reference = frames[reference_arm].loc[mask, column].to_numpy(dtype=np.float64)
    candidate = frames[candidate_arm].loc[mask, column].to_numpy(dtype=np.float64)
    if reference.size == 0:
        return None
    return {
        "kind": kind,
        "comparison": name,
        "population": population,
        "metric": metric,
        "n": int(reference.size),
        "reference_median": float(np.median(reference)),
        "candidate_median": float(np.median(candidate)),
        "delta_median": float(np.median(candidate) - np.median(reference)),
        "reference_fail_pct": float(100.0 * (reference < 0).mean()),
        "candidate_fail_pct": float(100.0 * (candidate < 0).mean()),
        "delta_fail_pp": float(100.0 * ((candidate < 0).mean() - (reference < 0).mean())),
    }


def load_sumnorm_targets(repo_root, target_path=VALID_TARGET_PATH):
    """Load and SumNorm-normalize the frozen valid eDOS target shapes."""
    raw = np.load(Path(repo_root) / target_path).astype(np.float64)
    if raw.ndim != 2 or raw.shape[1] != EXPECTED_EDOS_BINS:
        raise ValueError("valid target spectra do not match the frozen Q1 valid shape")
    totals = raw.sum(axis=1, keepdims=True)
    if (totals <= 0.0).any():
        raise ValueError("valid target spectra must have positive total mass")
    return raw / totals


def build_composition_pair_table(
    shape_paths,
    frames,
    *,
    repo_root=REPO_ROOT,
    reference_path=COMPOSITION_PAIRS_REFERENCE,
    valid_index_path=VALID_INDEX_PATH,
    target_path=VALID_TARGET_PATH,
    expected_count=EXPECTED_VALID_COUNT,
    target_atol=COMPOSITION_TARGET_ATOL,
):
    """Per-arm predicted contrasts on the frozen 320 same-composition pairs.

    Reuses the frozen pair table and the formula in
    ``tools/eval/g2_structure_path_audit.py::contrast_metrics`` verbatim; the
    SumNorm target shapes must reproduce the frozen ``oracle_target_tv``.
    """
    repo_root = Path(repo_root)
    reference = pd.read_csv(repo_root / reference_path)
    required = {"sample_index_a", "sample_index_b", "mpid_a", "mpid_b",
                "oracle_target_tv"}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"reference pair table is missing columns {sorted(missing)}")
    if reference.empty:
        raise ValueError("reference pair table is empty")

    sample_a = reference["sample_index_a"].to_numpy(dtype=np.int64)
    sample_b = reference["sample_index_b"].to_numpy(dtype=np.int64)
    if (sample_a < 0).any() or (sample_b < 0).any() \
            or (sample_a >= expected_count).any() or (sample_b >= expected_count).any():
        raise ValueError("reference pair indices fall outside the valid split")
    if np.any(sample_a == sample_b):
        raise ValueError("reference pairs must join two distinct samples")
    ordered = set(zip(sample_a.tolist(), sample_b.tolist()))
    unordered = {(min(x, y), max(x, y)) for x, y in ordered}
    if len(ordered) != len(sample_a) or len(unordered) != len(sample_a):
        raise ValueError("reference pair table contains duplicate material pairs")

    valid_index = np.load(repo_root / valid_index_path).astype(str)
    if valid_index.shape[0] != expected_count:
        raise ValueError("valid index length does not match the frozen split")
    if not np.array_equal(
            valid_index[sample_a], reference["mpid_a"].astype(str).to_numpy()):
        raise ValueError("reference mpid_a does not match the valid index")
    if not np.array_equal(
            valid_index[sample_b], reference["mpid_b"].astype(str).to_numpy()):
        raise ValueError("reference mpid_b does not match the valid index")

    target = load_sumnorm_targets(repo_root, target_path)
    if target.shape[0] != expected_count:
        raise ValueError("valid target spectra do not match the frozen split length")

    shapes = {}
    for arm in ARM_ORDER:
        if arm not in shape_paths:
            raise ValueError(f"missing predicted SumNorm shapes for {arm}")
        shapes[arm] = validate_edos_shape_sumnorm(
            np.load(Path(shape_paths[arm])), expected_count)
        frame = frames.get(arm)
        if frame is None:
            continue
        if len(frame) != expected_count:
            raise ValueError(f"{arm} valid samples do not match the frozen split")
        if not np.array_equal(frame["mpid"].astype(str).to_numpy(), valid_index):
            raise ValueError(f"{arm} valid sample order does not match the frozen index")

    oracle_tv = reference["oracle_target_tv"].to_numpy(dtype=np.float64)
    rows = []
    for position in range(len(reference)):
        a = int(sample_a[position])
        b = int(sample_b[position])
        target_a, target_b = target[a], target[b]
        pair_rows = []
        for arm in ARM_ORDER:
            metrics = contrast_metrics(
                shapes[arm][a], shapes[arm][b], target_a, target_b)
            pair_rows.append({
                "arm": arm,
                "sample_index_a": a,
                "sample_index_b": b,
                "mpid_a": valid_index[a],
                "mpid_b": valid_index[b],
                "target_tv": metrics["target_tv"],
                "predicted_tv": metrics["predicted_tv"],
                "contrast_error_tv": metrics["contrast_error_tv"],
                "contrast_cosine": metrics["contrast_cosine"],
            })
        if abs(pair_rows[0]["target_tv"] - float(oracle_tv[position])) > target_atol:
            raise ValueError(
                "SumNorm target shape does not reproduce oracle_target_tv for pair "
                f"{a}-{b}: {pair_rows[0]['target_tv']} != {oracle_tv[position]}")
        rows.extend(pair_rows)
    return pd.DataFrame(rows, columns=list(COMPOSITION_PAIR_COLUMNS))


def composition_pair_readout_rows(pairs_frame):
    """Summary rows for the composition-pair table; explanatory only.

    Reports reference/candidate medians and the candidate-minus-reference
    delta for each fixed comparison and pair metric. Failure columns are left
    NaN because a pair contrast has no R2/failure semantics.
    """
    rows = []
    by_arm = {arm: pairs_frame.loc[pairs_frame["arm"] == arm] for arm in ARM_ORDER}
    for name, reference_arm, candidate_arm in COMPARISONS:
        reference = by_arm[reference_arm]
        candidate = by_arm[candidate_arm]
        if not np.array_equal(reference["sample_index_a"].to_numpy(),
                              candidate["sample_index_a"].to_numpy()) \
                or not np.array_equal(reference["sample_index_b"].to_numpy(),
                                      candidate["sample_index_b"].to_numpy()):
            raise ValueError("composition pair order does not match across arms")
        for metric in COMPOSITION_PAIR_METRICS:
            reference_values = reference[metric].to_numpy(dtype=np.float64)
            candidate_values = candidate[metric].to_numpy(dtype=np.float64)
            rows.append({
                "kind": "composition_pair_table",
                "comparison": name,
                "population": "all",
                "metric": metric,
                "n": int(reference_values.size),
                "reference_median": float(np.median(reference_values)),
                "candidate_median": float(np.median(candidate_values)),
                "delta_median": float(
                    np.median(candidate_values) - np.median(reference_values)),
                "reference_fail_pct": float("nan"),
                "candidate_fail_pct": float("nan"),
                "delta_fail_pp": float("nan"),
            })
    return rows


def build_auxiliary_readouts(frames, comparisons, repo_root=REPO_ROOT,
                             composition_pairs=None):
    """Frozen-definition explanatory readouts; never touches the verdict.

    Returns ``(frame, gaps)``. A readout that cannot be reproduced without
    inventing a new definition is recorded as a gap instead of being guessed.
    The composition-pair table is supplied by the caller and summarised here;
    it never contributes to the verdict.
    """
    rows = []
    gaps = []

    reference_frame = frames["control"]
    for arm in ARM_ORDER:
        if not np.array_equal(
            frames[arm]["mpid"].astype(str).to_numpy(),
            reference_frame["mpid"].astype(str).to_numpy(),
        ):
            gaps.append({
                "readout": "auxiliary_populations",
                "status": "unavailable",
                "reason": f"{arm} valid order differs from control",
            })
            return pd.DataFrame(rows), gaps

    if "edos_spectral_roughness" not in reference_frame.columns:
        gaps.append({
            "readout": "roughness_train_p90",
            "status": "unavailable",
            "reason": "valid samples are missing edos_spectral_roughness",
        })
    else:
        roughness = reference_frame["edos_spectral_roughness"].to_numpy(dtype=np.float64)
        populations = (
            ("roughness_train_p90", roughness >= TRAIN_ROUGHNESS_P90),
            ("roughness_other", roughness < TRAIN_ROUGHNESS_P90),
        )
        for name, reference_arm, candidate_arm in COMPARISONS:
            for metric in METRICS:
                for population, mask in populations:
                    row = _group_readout(
                        "roughness_train_p90", name, metric, population, mask,
                        frames, reference_arm, candidate_arm)
                    if row is not None:
                        rows.append(row)

    support_path = Path(repo_root) / "results/edos_spectral_support_q1_train_valid_samples.csv"
    try:
        support = pd.read_csv(support_path)
        valid = support.loc[support["split"] == "valid"].sort_values(
            "sample_index", kind="stable")
        if len(valid) != len(reference_frame):
            raise ValueError(
                f"support table has {len(valid)} valid rows, expected {len(reference_frame)}")
        if not np.array_equal(
            valid["mpid"].astype(str).to_numpy(),
            reference_frame["mpid"].astype(str).to_numpy(),
        ):
            raise ValueError("support table valid order does not match the arm order")
        quartile = valid["train_support_quartile"].to_numpy()
        for name, reference_arm, candidate_arm in COMPARISONS:
            for metric in METRICS:
                for value in sorted(set(quartile.tolist())):
                    mask = quartile == value
                    row = _group_readout(
                        "spectral_support_quartile", name, metric,
                        f"q{int(value)}", mask, frames, reference_arm, candidate_arm)
                    if row is not None:
                        rows.append(row)
    except Exception as exc:  # auxiliary only: record a gap, never fatal
        gaps.append({
            "readout": "spectral_support_quartile",
            "status": "unavailable",
            "reason": str(exc),
        })

    if composition_pairs is not None:
        rows.extend(composition_pair_readout_rows(composition_pairs))
    return pd.DataFrame(rows), gaps


# ---------------------------------------------------------------------------
# Orchestration and atomic writing
# ---------------------------------------------------------------------------

def formal_paths(output_prefix):
    base = output_prefix.name
    return {
        "control_samples": output_prefix.with_name(f"{base}_control_samples.csv"),
        "radial_samples": output_prefix.with_name(f"{base}_radial_samples.csv"),
        "joint_samples": output_prefix.with_name(f"{base}_joint_samples.csv"),
        "joint_vs_control": output_prefix.with_name(f"{base}_joint_vs_control.csv"),
        "joint_vs_radial": output_prefix.with_name(f"{base}_joint_vs_radial.csv"),
        "radial_vs_control": output_prefix.with_name(f"{base}_radial_vs_control.csv"),
        "auxiliary": output_prefix.with_name(f"{base}_auxiliary_readouts.csv"),
        "composition_pairs": output_prefix.with_name(f"{base}_composition_pairs.csv"),
        "summary": output_prefix.with_suffix(".json"),
    }


def run_verdict(arm_checkpoints, arm_configs, output_prefix, device, replicates=2000):
    """Evaluate the three valid-only arms and write the verdict atomically."""
    output_prefix = Path(output_prefix).resolve()
    paths = formal_paths(output_prefix)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"verdict outputs already exist: {existing}")

    arm_checkpoints = {arm: Path(arm_checkpoints[arm]).resolve() for arm in ARM_ORDER}
    arm_configs = {arm: Path(arm_configs[arm]).resolve() for arm in ARM_ORDER}
    for arm in ARM_ORDER:
        validate_arm_artifact_paths(arm, arm_checkpoints[arm], arm_configs[arm])
    b7_path = (REPO_ROOT / B7_INIT_CKPT).resolve()
    _require(file_sha256(b7_path) == B7_CKPT_SHA256,
             "frozen B7 initialization checkpoint SHA256 mismatch")

    configs = {arm: load_arm_config(arm_configs[arm]) for arm in ARM_ORDER}
    for arm in ARM_ORDER:
        validate_arm_config(arm, configs[arm], b7_init_path=B7_INIT_CKPT,
                            repo_root=REPO_ROOT)
    validate_cross_arm_configs(configs)
    for arm in ARM_ORDER:
        load_and_validate_checkpoint(arm, arm_checkpoints[arm], configs[arm])

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_prefix.name}_", dir=output_prefix.parent))
    try:
        frames = {}
        shape_paths = {}
        for arm in ARM_ORDER:
            staged_prefix = staging / arm
            shape_paths[arm] = staging / f"{arm}_edos_shape_sumnorm.npy"
            run_audit(
                Path(arm_checkpoints[arm]),
                Path(arm_configs[arm]),
                staged_prefix,
                device,
                bootstrap_replicates=replicates,
                force=False,
                split="valid",
                expected_epoch=EXPECTED_EPOCH,
                verify_reference_r2=False,
                edos_shape_sumnorm_path=shape_paths[arm],
            )
            sample_path = staged_prefix.with_name(staged_prefix.name + "_samples.csv")
            frames[arm] = pd.read_csv(sample_path)

        valid_index_path = REPO_ROOT / VALID_INDEX_PATH
        expected_ids = np.load(valid_index_path).astype(str) if valid_index_path.exists() else None
        validate_sample_frames(
            frames, expected_count=EXPECTED_VALID_COUNT, expected_ids=expected_ids)

        comparisons = {}
        comparison_frames = {}
        for name, reference_arm, candidate_arm in COMPARISONS:
            comparisons[name] = build_comparison(
                frames[reference_arm], frames[candidate_arm], replicates=replicates)
            comparison_frames[name] = comparison_frame(
                name, comparisons[name], reference_arm, candidate_arm)

        decision = evaluate_joint_verdict(comparisons["joint_vs_control"])
        radial_decision = evaluate_joint_verdict(comparisons["joint_vs_radial"])
        radial_classification = auxiliary_radial_classification(
            decision, radial_decision)
        decision["joint_wins_vs_radial"] = radial_classification["joint_wins_vs_radial"]
        decision["only_beats_radial"] = radial_classification["only_beats_radial"]
        decision["only_beats_radial_note"] = (
            "auxiliary classification only; it never changes the primary verdict "
            "and never starts another route")

        composition_frame = build_composition_pair_table(
            shape_paths, frames, repo_root=REPO_ROOT)
        auxiliary_frame, auxiliary_gaps = build_auxiliary_readouts(
            frames, comparisons, REPO_ROOT, composition_pairs=composition_frame)

        staged = {}
        for arm in ARM_ORDER:
            key = f"{arm}_samples"
            staged[key] = staging / paths[key].name
            frames[arm].to_csv(staged[key], index=False)
        for name, _, _ in COMPARISONS:
            staged[name] = staging / paths[name].name
            comparison_frames[name].to_csv(staged[name], index=False)
        staged["auxiliary"] = staging / paths["auxiliary"].name
        auxiliary_frame.to_csv(staged["auxiliary"], index=False)
        staged["composition_pairs"] = staging / paths["composition_pairs"].name
        composition_frame.to_csv(staged["composition_pairs"], index=False)

        payload = {
            "tool": "joint_content_pilot_verdict",
            "split": "Q1 valid",
            "checkpoint_epoch": EXPECTED_EPOCH,
            "expected_valid_count": EXPECTED_VALID_COUNT,
            "bootstrap_replicates": replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_note": "paired intervals are reported only; the verdict "
                              "never uses them",
            "device": str(device),
            "arms": {
                arm: {
                    "checkpoint": str(Path(arm_checkpoints[arm])),
                    "config": str(Path(arm_configs[arm])),
                    "use_g2": arm != "control",
                    "g2_content_mode": "joint" if arm == "joint" else "radial",
                    "samples_csv": str(paths[f"{arm}_samples"]),
                }
                for arm in ARM_ORDER
            },
            "comparisons": {
                name: {
                    "reference_arm": reference_arm,
                    "candidate_arm": candidate_arm,
                    "csv": str(paths[name]),
                    "metrics": comparisons[name],
                }
                for name, reference_arm, candidate_arm in COMPARISONS
            },
            "verdict": decision,
            "auxiliary": {
                "csv": str(paths["auxiliary"]),
                "composition_pairs_csv": str(paths["composition_pairs"]),
                "gaps": auxiliary_gaps,
            },
            "rules": {
                "win": {
                    "blind_edos_delta_median_min": R2_WIN_MARGIN,
                    "blind_edos_fail_delta_max": FAILURE_MARGIN,
                    "oracle_edos_delta_median_min": 0.0,
                    "phdos_oracle_delta_median_min": -R2_WIN_MARGIN,
                    "phdos_oracle_fail_delta_max": FAILURE_MARGIN,
                    "phdos_blind_delta_median_min": -R2_WIN_MARGIN,
                    "phdos_blind_fail_delta_max": FAILURE_MARGIN,
                },
                "tie": {
                    "abs_delta_median_lt": R2_WIN_MARGIN,
                    "abs_delta_fail_lt": FAILURE_MARGIN,
                },
                "note": "point estimates decide; bootstrap intervals are "
                        "reported only; tag is not recorded in config_used.yaml "
                        "so it is not compared",
            },
        }
        staged["summary"] = staging / paths["summary"].name
        staged["summary"].write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

        for key, final_path in paths.items():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[key], final_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return paths, decision


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARM_ORDER:
        parser.add_argument(f"--{arm}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{arm}-config", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoints = {arm: getattr(args, f"{arm}_checkpoint") for arm in ARM_ORDER}
    configs = {arm: getattr(args, f"{arm}_config") for arm in ARM_ORDER}
    paths, decision = run_verdict(
        checkpoints, configs, args.output_prefix, device, replicates=args.bootstrap)
    print(f"Q1 valid verdict: {decision['verdict']}")
    print(f"Summary: {paths['summary']}")


if __name__ == "__main__":
    main()
