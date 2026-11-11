"""Project-comparable scoring of any model's valid spectra (cell units, frozen E0/P0 grids).

The per-material math is imported from the project (``utils.metrics.per_sample_spectral_metrics``
and ``utils.spectral_balance.normalized_spectral_error``) and the aggregation mirrors
``tools.train.evaluate_local_global_split`` so numbers are comparable with existing exports.

Modes
  blind  : spectrum predicted from structure only (shape and total).
  oracle : predicted SumNorm shape multiplied by the label's bin sum (shape-only diagnostic).
  shape  : KL / W1 / TV between SumNorm shapes, materials with a non-zero target only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.metrics import per_sample_spectral_metrics
from utils.spectral_balance import normalized_spectral_error, validate_balance_protocol

TASKS = ("edos", "phdos")
DEFAULT_PROTOCOL = "results/runs/phdos_head_20261106/protocol.json"


def load_protocol(path=DEFAULT_PROTOCOL):
    """Frozen train-fitted task energies C_t and bin widths (same file the h2 working point uses)."""
    return validate_balance_protocol(json.loads(Path(path).read_text()))


def sumnorm(spectra):
    """Row-wise SumNorm; all-zero rows stay zero (no artificial shape)."""
    spectra = np.asarray(spectra, dtype=np.float64)
    total = spectra.sum(-1, keepdims=True)
    return spectra / np.where(total > 0, total, 1.0)


def oracle_reconstruction(pred_shape, target):
    """Predicted shape times the label bin sum, exactly the project's oracle denormalisation."""
    return (sumnorm(pred_shape) * np.asarray(target, np.float64).sum(-1, keepdims=True)).astype(np.float32)


def material_scores(prediction, target, n_atoms, task_normalizer):
    """Per-material mae/mse/r2/energy in FP32, as the project's evaluator computes them."""
    p = torch.as_tensor(np.asarray(prediction, np.float32))
    t = torch.as_tensor(np.asarray(target, np.float32))
    atoms = torch.as_tensor(np.asarray(n_atoms, np.float32))
    out = {key: value.numpy() for key, value in per_sample_spectral_metrics(p, t).items()}
    out["energy"] = normalized_spectral_error(p, t, atoms, task_normalizer).numpy()
    return out


def shape_scores(pred_shape, target):
    """KL(target||pred), W1 (mean |CDF difference| over bins) and TV on SumNorm shapes."""
    t = sumnorm(target)
    p = sumnorm(pred_shape)
    nonzero = np.asarray(target).sum(-1) > 0
    kl = (t * (np.log(np.clip(t, 1e-12, None)) - np.log(np.clip(p, 1e-12, None)))).sum(-1)
    w1 = np.abs(np.cumsum(p, -1) - np.cumsum(t, -1)).mean(-1)
    tv = .5 * np.abs(p - t).sum(-1)
    return {name: np.where(nonzero, value, np.nan) for name, value in
            (("kl", kl), ("w1", w1), ("tv", tv))}


def score_task(task, target, n_atoms, protocol, *, pred_shape, pred_blind=None, material_ids=None):
    """Return (per-material frame, summary) for one task; key names follow the project exports."""
    normalizer = protocol["tasks"][task]
    predictions = {"oracle": oracle_reconstruction(pred_shape, target)}
    if pred_blind is not None:
        predictions["blind"] = np.asarray(pred_blind, np.float32)
    columns = {}
    for mode, prediction in predictions.items():
        for metric, values in material_scores(prediction, target, n_atoms, normalizer).items():
            columns[f"{mode}_{metric}_{task}"] = values
    for metric, values in shape_scores(pred_shape, target).items():
        columns[f"shape_{metric}_{task}"] = values
    frame = pd.DataFrame(columns)
    if material_ids is not None:
        frame["material_id"] = list(material_ids)
    return frame, summarize_task(frame, task)


def summarize_task(frame, task):
    summary = {}
    for mode in ("blind", "oracle"):
        if f"{mode}_r2_{task}" not in frame:
            continue
        for metric in ("mae", "mse", "r2"):
            values = frame[f"{mode}_{metric}_{task}"]
            for aggregate in ("mean", "median", "std"):
                summary[f"{mode}_{metric}_{task}_{aggregate}"] = float(getattr(values, aggregate)())
        r2 = frame[f"{mode}_r2_{task}"]
        summary[f"{mode}_fail_rate_{task}"] = float((r2 < 0).mean() * 100)
        summary[f"{mode}_failures_{task}"] = int((r2 < 0).sum())
        summary[f"{mode}_r2_{task}_p10"] = float(r2.quantile(.1))
        energy = frame[f"{mode}_energy_{task}"]
        summary[f"{mode}_energy_{task}_mean"] = float(energy.mean())
        summary[f"{mode}_energy_{task}_median"] = float(energy.median())
        summary[f"{mode}_energy_{task}_p90"] = float(energy.quantile(.9))
    for metric in ("kl", "w1", "tv"):
        values = frame[f"shape_{metric}_{task}"].dropna()
        summary[f"shape_{metric}_{task}_mean"] = float(values.mean())
        summary[f"shape_{metric}_{task}_median"] = float(values.median())
    summary[f"shape_samples_{task}"] = int(frame[f"shape_kl_{task}"].notna().sum())
    summary["samples"] = int(len(frame))
    return summary


def paired_difference(candidate, reference, seed=0, resamples=10000):
    """Mean(candidate - reference) over materials with a percentile bootstrap 95% interval."""
    delta = np.asarray(candidate, np.float64) - np.asarray(reference, np.float64)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(delta.mean()), "ci95": [float(low), float(high)], "n": int(len(delta))}


def valid_atom_counts(material_ids, data_root="data/crystal_q1_v1"):
    """Atom counts aligned to ``material_ids`` from the frozen valid split (never test)."""
    root = Path(data_root) / "valid"
    ids = np.load(root / "valid_index.npy", allow_pickle=False)
    counts = np.load(root / "n_atoms_valid.npy", allow_pickle=False)
    order = pd.Index(ids).get_indexer(list(material_ids))
    if (order < 0).any():
        raise ValueError("material ids are not all in the frozen valid split")
    return counts[order]


def selfcheck_reference_export(spectra_path, summary_path, protocol, data_root="data/crystal_q1_v1"):
    """Re-score an existing project export with this module and compare with its stored summary.

    Returns a list of (key, recomputed, stored, abs difference, passed). Tolerances cover FP32
    export rounding; failure counts must match exactly.
    """
    with np.load(spectra_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    stored = json.loads(Path(summary_path).read_text())["recomputed_valid"]
    atoms = valid_atom_counts(values["material_ids"], data_root)
    rows = []
    for task in TASKS:
        target = values[f"tgt_{task}"]
        frame, summary = score_task(task, target, atoms, protocol,
                                    pred_shape=values[f"pred_shape_{task}"],
                                    pred_blind=values[f"pred_blind_{task}"])
        oracle_gap = float(np.abs(oracle_reconstruction(values[f"pred_shape_{task}"], target)
                                  - values[f"pred_oracle_{task}"]).max())
        rows.append((f"max|oracle rebuilt - exported| {task}", oracle_gap, 0.0, oracle_gap,
                     oracle_gap <= 1e-4 * max(1.0, float(target.max()))))
        for mode in ("blind", "oracle"):
            for key, tolerance in ((f"{mode}_energy_{task}_mean", 2e-6), (f"{mode}_energy_{task}_median", 2e-6),
                                   (f"{mode}_energy_{task}_p90", 2e-6), (f"{mode}_r2_{task}_median", 2e-6),
                                   (f"{mode}_r2_{task}_p10", 2e-6), (f"{mode}_fail_rate_{task}", 1e-9),
                                   (f"{mode}_mae_{task}_mean", 2e-6)):
                gap = abs(summary[key] - stored[key])
                rows.append((key, summary[key], stored[key], gap, gap <= tolerance * max(1.0, abs(stored[key]))))
    joint = .5 * (rows_value(rows, "blind_energy_edos_mean") + rows_value(rows, "blind_energy_phdos_mean"))
    gap = abs(joint - stored["balanced_blind_energy"])
    rows.append(("balanced_blind_energy", joint, stored["balanced_blind_energy"], gap, gap <= 2e-6))
    return rows


def rows_value(rows, key):
    return next(row[1] for row in rows if row[0] == key)
