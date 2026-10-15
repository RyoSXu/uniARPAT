"""Checkpoint scores and directions; legacy MAE contracts stay unchanged."""
from __future__ import annotations

import math

import numpy as np

BLIND_R2_METRIC = "balanced_blind_r2"
SELECTION_METRICS = ("balanced_oracle_mae", "balanced_blind_mae", BLIND_R2_METRIC)


def selection_mode(metric):
    if metric == BLIND_R2_METRIC:
        return "max"
    if metric is None or metric in SELECTION_METRICS:
        return "min"
    raise ValueError(f"unknown checkpoint selection metric: {metric}")


def selection_contract(metric):
    """Only new trials receive metadata; old contract dictionaries are exact."""
    selection_mode(metric)
    if metric != BLIND_R2_METRIC:
        return {}
    return {"checkpoint_selection": {
        "schema": "joint-blind-r2-selection-v1", "metric": metric, "mode": "max", "split": "valid",
        "formula": "0.5*blind_r2_edos_median + 0.5*blind_r2_phdos_median",
        "task_weights": {"edos": .5, "phdos": .5}, "tie_break": "earliest_epoch",
        "diagnostics_only": ["oracle_mae", "same_composition_spectral_difference"],
    }}


def blind_joint_score(metrics):
    values = [float(metrics[f"blind_r2_{task}_median"]) for task in ("edos", "phdos")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("blind checkpoint selection needs finite R2 for both tasks")
    return .5 * (values[0] + values[1])


def validation_score(metrics, metric):
    selection_mode(metric)
    if metric == BLIND_R2_METRIC:
        value = blind_joint_score(metrics)
    elif metric is None:
        value = .5 * metrics["mae_edos_median"] + .5 * metrics["mae_phdos_median"]
    else:
        value = metrics[metric]
    if not math.isfinite(float(value)):
        raise ValueError("non-finite checkpoint selection score")
    return float(value)


def initial_best_score(metric):
    return -math.inf if selection_mode(metric) == "max" else math.inf


def is_better_score(value, best, metric):
    if not math.isfinite(float(value)):
        raise ValueError("non-finite checkpoint selection score")
    return value > best if selection_mode(metric) == "max" else value < best


def selected_history_row(history, metric):
    """Validate the recorded score before choosing one epoch for every reader."""
    mode = selection_mode(metric)
    required = {"epoch", "balanced_score"} | ({metric} if metric is not None else set())
    if history.empty or not required.issubset(history.columns):
        raise ValueError("incomplete checkpoint selection history")
    score = history["balanced_score"]
    if not np.isfinite(score.to_numpy(float)).all():
        raise ValueError("non-finite checkpoint selection history")
    if metric is not None and not (score == history[metric]).all():
        raise ValueError("history selection metric mismatch")
    if metric == BLIND_R2_METRIC:
        columns = [f"blind_r2_{task}_median" for task in ("edos", "phdos")]
        if not set(columns).issubset(history.columns):
            raise ValueError("blind selection history requires both task R2 columns")
        expected = .5 * (history[columns[0]] + history[columns[1]])
        if not np.isfinite(expected).all() or not np.allclose(score, expected, rtol=1e-12, atol=1e-12):
            raise ValueError("recorded score differs from equal-weight blind R2")
        history = history.sort_values("epoch", kind="stable")
        score = history["balanced_score"]
    index = score.idxmax() if mode == "max" else score.idxmin()
    return history.loc[index]
