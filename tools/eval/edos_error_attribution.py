"""Read-only Q1 B7 eDOS error attribution audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Frozen Q1 eDOS grid width; SumNorm shapes are per-sample softmax vectors.
EXPECTED_EDOS_BINS = 128


def spectral_descriptors(spectra: np.ndarray, edge_bins: int = 8) -> dict[str, np.ndarray]:
    spectra = np.asarray(spectra, dtype=np.float64)
    if spectra.ndim != 2 or spectra.shape[1] < 2:
        raise ValueError("spectra must be a two-dimensional array with at least two bins")
    if not np.isfinite(spectra).all() or (spectra < 0).any():
        raise ValueError("spectra must contain finite, nonnegative values")
    if edge_bins < 0 or 2 * edge_bins > spectra.shape[1]:
        raise ValueError("edge_bins must be nonnegative and cover no more than half the bins")

    total = spectra.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("each spectrum must have positive total mass")
    shape = spectra / total
    entropy = -(shape * np.log(np.maximum(shape, 1e-30))).sum(axis=1)
    edge_mass = np.zeros(len(shape), dtype=np.float64)
    if edge_bins:
        edge_mass = shape[:, :edge_bins].sum(axis=1) + shape[:, -edge_bins:].sum(axis=1)
    return {
        "roughness": 0.5 * np.abs(np.diff(shape, axis=1)).sum(axis=1),
        "entropy_norm": entropy / np.log(shape.shape[1]),
        "peak_share": shape.max(axis=1),
        "edge_mass": edge_mass,
    }


def shape_error_descriptors(
    prediction: np.ndarray,
    target: np.ndarray,
    high_gradient_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be matching [samples, bins] arrays")
    if prediction.shape[1] < 3:
        raise ValueError("shape error diagnostics require at least three spectral bins")
    if not 0.0 < high_gradient_fraction < 1.0:
        raise ValueError("high_gradient_fraction must be between zero and one")

    prediction_descriptors = spectral_descriptors(prediction, edge_bins=0)
    target_descriptors = spectral_descriptors(target, edge_bins=0)
    prediction_shape = prediction / prediction.sum(axis=1, keepdims=True)
    target_shape = target / target.sum(axis=1, keepdims=True)
    prediction_gradient = np.diff(prediction_shape, axis=1)
    target_gradient = np.diff(target_shape, axis=1)
    gradient_error = np.abs(prediction_gradient - target_gradient)
    target_gradient_magnitude = np.abs(target_gradient)

    edge_count = gradient_error.shape[1]
    high_edge_count = max(1, int(np.ceil(edge_count * high_gradient_fraction)))
    high_gradient_indices = np.argsort(
        -target_gradient_magnitude, axis=1, kind="stable"
    )[:, :high_edge_count]
    high_gradient_mask = np.zeros_like(gradient_error, dtype=bool)
    np.put_along_axis(high_gradient_mask, high_gradient_indices, True, axis=1)
    other_gradient_mask = ~high_gradient_mask
    high_gradient_error = np.where(high_gradient_mask, gradient_error, 0.0)
    other_gradient_error = np.where(other_gradient_mask, gradient_error, 0.0)
    total_gradient_error = gradient_error.sum(axis=1)

    prediction_peak_bin = prediction_shape.argmax(axis=1)
    target_peak_bin = target_shape.argmax(axis=1)
    peak_bin_shift = prediction_peak_bin - target_peak_bin
    return {
        "predicted_roughness": prediction_descriptors["roughness"],
        "roughness_bias": (
            prediction_descriptors["roughness"] - target_descriptors["roughness"]
        ),
        "predicted_peak_share": prediction_descriptors["peak_share"],
        "peak_share_bias": (
            prediction_descriptors["peak_share"] - target_descriptors["peak_share"]
        ),
        "peak_bin_shift": peak_bin_shift.astype(np.float64),
        "peak_bin_shift_abs": np.abs(peak_bin_shift).astype(np.float64),
        "slope_error_mae": gradient_error.mean(axis=1),
        "slope_error_high_gradient_mae": (
            high_gradient_error.sum(axis=1) / high_gradient_mask.sum(axis=1)
        ),
        "slope_error_other_gradient_mae": (
            other_gradient_error.sum(axis=1) / other_gradient_mask.sum(axis=1)
        ),
        "slope_error_high_gradient_share": np.divide(
            high_gradient_error.sum(axis=1),
            total_gradient_error,
            out=np.zeros_like(total_gradient_error),
            where=total_gradient_error > 0.0,
        ),
    }


def masked_spectral_r2(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.ndim != 2:
        raise ValueError("prediction, target, and mask must be matching [samples, bins] arrays")

    weights = mask.astype(np.float64)
    counts = weights.sum(axis=1)
    result = np.full(len(prediction), np.nan, dtype=np.float64)
    valid = counts >= 2
    if not valid.any():
        return result
    mean = np.divide(
        (target * weights).sum(axis=1),
        counts,
        out=np.zeros_like(counts),
        where=counts > 0,
    )
    residual = ((target - prediction) ** 2 * weights).sum(axis=1)
    total = ((target - mean[:, None]) ** 2 * weights).sum(axis=1)
    result[valid] = 1.0 - residual[valid] / (total[valid] + eps)
    return result


def structural_descriptors(
    elements: np.ndarray,
    positions: np.ndarray,
    reference_elements: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    elements = np.asarray(elements)
    positions = np.asarray(positions, dtype=np.float64)
    if elements.ndim != 2 or elements.shape[1] < 3:
        raise ValueError("elements must contain two special slots followed by atom tokens")
    if positions.size != len(elements) * 82 * 3:
        raise ValueError("positions must contain 82 lattice/coordinate rows per sample")

    atom_tokens = elements[:, 2:]
    active = atom_tokens > 0
    atom_count = active.sum(axis=1)
    unique_count = np.array(
        [np.unique(row[row > 0]).size for row in atom_tokens], dtype=np.float64
    )

    reference = elements if reference_elements is None else np.asarray(reference_elements)
    reference_atoms = reference[:, 2:]
    atomic_counts = np.bincount(reference_atoms[reference_atoms > 0].astype(np.int64))
    probabilities = atomic_counts / atomic_counts.sum()
    rarity_by_token = np.zeros(len(probabilities), dtype=np.float64)
    present = probabilities > 0
    rarity_by_token[present] = -np.log(probabilities[present])
    max_token = int(atom_tokens.max(initial=0))
    if max_token >= len(rarity_by_token):
        rarity_by_token = np.pad(rarity_by_token, (0, max_token + 1 - len(rarity_by_token)))
        rarity_by_token[len(probabilities):] = -np.log(1.0 / max(atomic_counts.sum(), 1))
    rarity_sum = np.where(active, rarity_by_token[atom_tokens.clip(min=0)], 0.0).sum(axis=1)
    mean_rarity = rarity_sum / np.maximum(atom_count, 1)

    pos = positions.reshape(len(elements), 82, 3)
    a = pos[:, 0, 0]
    b = pos[:, 0, 1]
    c = 1.0 / np.maximum(pos[:, 0, 2], 1e-4)
    alpha, beta, gamma = np.deg2rad(pos[:, 1, :]).T
    cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
    volume_factor = np.maximum(
        0.0,
        1.0
        + 2.0 * cos_alpha * cos_beta * cos_gamma
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2,
    )
    volume = a * b * c * np.sqrt(volume_factor)
    aspect_ratio = np.maximum.reduce([a, b, c]) / np.maximum(
        np.minimum.reduce([a, b, c]), 1e-8
    )
    return {
        "natoms": atom_count.astype(np.float64),
        "n_unique_elements": unique_count,
        "volume_per_atom": volume / np.maximum(atom_count, 1),
        "cell_aspect_ratio": aspect_ratio,
        "mean_element_rarity": mean_rarity,
    }


def bootstrap_difference_ci(
    high: np.ndarray,
    other: np.ndarray,
    statistic: str,
    replicates: int = 2000,
    seed: int = 20260923,
) -> tuple[float, float]:
    high = np.asarray(high, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if high.size == 0 or other.size == 0:
        return float("nan"), float("nan")
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    rng = np.random.default_rng(seed)
    high_draws = rng.choice(high, size=(replicates, high.size), replace=True)
    other_draws = rng.choice(other, size=(replicates, other.size), replace=True)
    reducer = np.median if statistic == "median" else np.mean
    differences = reducer(high_draws, axis=1) - reducer(other_draws, axis=1)
    low, upper = np.quantile(differences, [0.025, 0.975])
    return float(low), float(upper)


def bootstrap_median_ci(
    values: np.ndarray,
    replicates: int = 2000,
    seed: int = 20260923,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, values.size), replace=True)
    low, upper = np.quantile(np.median(draws, axis=1), [0.025, 0.975])
    return float(low), float(upper)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2:
        return float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def validate_edos_shape_sumnorm(
    shapes: np.ndarray,
    expected_count: int,
    row_sum_atol: float = 1e-4,
) -> np.ndarray:
    """Validate per-sample SumNorm eDOS shapes before they are persisted.

    Shapes must be a two-dimensional ``[samples, bins]`` array with exactly
    ``expected_count`` rows and ``EXPECTED_EDOS_BINS`` columns, finite,
    nonnegative, and each row summing to one.
    """
    shapes = np.asarray(shapes, dtype=np.float64)
    if shapes.ndim != 2:
        raise ValueError("SumNorm eDOS shapes must be a two-dimensional [samples, bins] array")
    if shapes.shape[0] != expected_count:
        raise ValueError(
            f"SumNorm eDOS shapes have {shapes.shape[0]} rows, expected {expected_count}")
    if shapes.shape[1] != EXPECTED_EDOS_BINS:
        raise ValueError(
            f"SumNorm eDOS shapes have {shapes.shape[1]} bins, expected {EXPECTED_EDOS_BINS}")
    if not np.isfinite(shapes).all():
        raise ValueError("SumNorm eDOS shapes must be finite")
    if (shapes < 0.0).any():
        raise ValueError("SumNorm eDOS shapes must be nonnegative")
    if not np.allclose(shapes.sum(axis=1), 1.0, atol=row_sum_atol, rtol=0):
        raise ValueError("SumNorm eDOS shape rows must sum to one")
    return shapes


def finalize_evaluation_arrays(result: dict[str, list]) -> dict[str, np.ndarray]:
    """Convert scalar lists and variable-size shape batches to stable arrays."""
    arrays = {}
    for name, values in result.items():
        if name == "edos_shape_sumnorm":
            arrays[name] = (
                np.concatenate(values, axis=0).astype(np.float64, copy=False)
                if values
                else np.empty((0, EXPECTED_EDOS_BINS), dtype=np.float64)
            )
        else:
            arrays[name] = np.asarray(values, dtype=np.float64)
    return arrays


def evaluate_b7(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
    split: str = "test",
    expected_epoch: int | None = 33,
    collect_edos_shape_sumnorm: bool = False,
) -> dict[str, np.ndarray]:
    from utils.builder import ConfigBuilder
    from utils.metrics import per_sample_spectral_metrics

    if split not in {"valid", "test"}:
        raise ValueError("split must be 'valid' or 'test'")
    full_config = yaml.safe_load(config_path.read_text())
    builder = ConfigBuilder(**full_config["config"])
    loader = builder.get_dataloader(
        split=split, dos_minmax=True, batch_size=32, dos_sumnorm=True
    )
    model = builder.get_model()
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = int(checkpoint.get("epoch", -1))
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError(f"expected checkpoint epoch {expected_epoch}, found epoch {epoch}")
    model.model["transformer"].load_state_dict(checkpoint["model"])
    model.model["transformer"].eval()

    result: dict[str, list[float]] = {
        "r2_oracle_unmasked": [],
        "r2_blind_unmasked": [],
        "r2_oracle_masked": [],
        "r2_blind_masked": [],
        "r2_phdos_oracle_unmasked": [],
        "r2_phdos_blind_unmasked": [],
        "gamma_true": [],
        "gamma_pred": [],
        "coverage_fraction": [],
        "target_mass_outside_mask": [],
        "predicted_roughness": [],
        "roughness_bias": [],
        "predicted_peak_share": [],
        "peak_share_bias": [],
        "peak_bin_shift": [],
        "peak_bin_shift_abs": [],
        "slope_error_mae": [],
        "slope_error_high_gradient_mae": [],
        "slope_error_other_gradient_mae": [],
        "slope_error_high_gradient_share": [],
    }
    if collect_edos_shape_sumnorm:
        result["edos_shape_sumnorm"] = []
    delta_edos = float(model.params.get("delta_edos", 0.09375))
    with torch.no_grad():
        for batch in loader:
            (
                inp,
                pos,
                attention_mask,
                edos_target,
                phdos_target,
                _,
                _,
                edos_min,
                edos_max,
                _,
                _,
                phdos_min,
                phdos_max,
                edos_coverage,
                phdos_coverage,
                nvalence,
                edos_x,
                phdos_x,
            ) = model.data_preprocess(batch)
            if edos_coverage is None or nvalence is None:
                raise ValueError(f"Q1 {split} requires eDOS coverage masks and N_valence")

            outputs = model.fp32_outputs(
                model.model["transformer"](inp, attention_mask, pos, edos_x, phdos_x)
            )
            shape = F.softmax(outputs["edos"], dim=-1)
            if collect_edos_shape_sumnorm:
                result["edos_shape_sumnorm"].append(shape.detach().cpu().numpy())
            phdos_shape = F.softmax(outputs["phdos"], dim=-1)
            target = edos_target * (edos_max - edos_min) + edos_min
            oracle_prediction = torch.clamp(shape * (edos_max - edos_min) + edos_min, min=0.0)
            phdos_target_physical = (
                phdos_target * (phdos_max - phdos_min) + phdos_min
            )
            phdos_oracle_prediction = torch.clamp(
                phdos_shape * (phdos_max - phdos_min) + phdos_min, min=0.0
            )

            nval = nvalence.reshape(-1, 1).to(device=device, dtype=torch.float32)
            gamma_prediction = outputs["eta"][:, 1:2]
            gamma_true = (
                edos_max.clamp_min(1e-12) * delta_edos / nval.clamp_min(1e-12)
            ).clamp(0.0, 1.0)
            blind_scale = nval * gamma_prediction / delta_edos
            blind_prediction = torch.clamp(shape * blind_scale, min=0.0)
            natoms = (~attention_mask[:, 2:]).sum(dim=-1, keepdim=True).float().clamp_min(1.0)
            eta_prediction = outputs["eta"][:, 0:1]
            delta_phdos = float(model.params.get("delta_phdos", 19.6875))
            phdos_blind_scale = 3.0 * natoms * eta_prediction / delta_phdos
            phdos_blind_prediction = torch.clamp(phdos_shape * phdos_blind_scale, min=0.0)
            coverage = edos_coverage.to(device=device, dtype=torch.bool)

            oracle_r2 = per_sample_spectral_metrics(oracle_prediction, target)["r2"]
            blind_r2 = per_sample_spectral_metrics(blind_prediction, target)["r2"]
            phdos_oracle_r2 = per_sample_spectral_metrics(
                phdos_oracle_prediction, phdos_target_physical
            )["r2"]
            phdos_blind_r2 = per_sample_spectral_metrics(
                phdos_blind_prediction, phdos_target_physical
            )["r2"]
            shape_diagnostics = shape_error_descriptors(
                shape.detach().cpu().numpy(), target.detach().cpu().numpy()
            )
            arrays = {
                "r2_oracle_unmasked": oracle_r2,
                "r2_blind_unmasked": blind_r2,
                "r2_oracle_masked": masked_spectral_r2(
                    oracle_prediction.cpu().numpy(), target.cpu().numpy(), coverage.cpu().numpy()
                ),
                "r2_blind_masked": masked_spectral_r2(
                    blind_prediction.cpu().numpy(), target.cpu().numpy(), coverage.cpu().numpy()
                ),
                "r2_phdos_oracle_unmasked": phdos_oracle_r2,
                "r2_phdos_blind_unmasked": phdos_blind_r2,
                "gamma_true": gamma_true.squeeze(-1),
                "gamma_pred": gamma_prediction.squeeze(-1),
                "coverage_fraction": coverage.float().mean(dim=-1),
                "target_mass_outside_mask": (
                    (target * (~coverage)).sum(dim=-1) / target.sum(dim=-1).clamp_min(1e-12)
                ),
                **shape_diagnostics,
            }
            for name, values in arrays.items():
                value_array = values.detach().cpu().numpy() if torch.is_tensor(values) else values
                result[name].extend(np.asarray(value_array).reshape(-1).tolist())

    return finalize_evaluation_arrays(result)


def ordered_valid_reference_r2(reference: pd.DataFrame, expected_count: int) -> np.ndarray:
    edos_reference = reference.loc[reference["task"] == "edos"].sort_values(
        "sample_index", kind="stable"
    )
    indices = edos_reference["sample_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(expected_count)):
        raise ValueError("C2.1b valid eDOS rows do not match the Q1 valid sample order")
    return edos_reference["normalized_r2"].to_numpy(dtype=np.float64)


def build_strata_summary(
    train_features: dict[str, np.ndarray],
    samples: pd.DataFrame,
    replicates: int,
) -> pd.DataFrame:
    specs = [
        ("roughness_train_p90", "edos_spectral_roughness", "p90", "all"),
        ("entropy_train_p90", "edos_entropy_norm", "p90", "all"),
        ("peak_share_train_p90", "edos_peak_share", "p90", "all"),
        ("edge_mass_train_p90", "edos_edge_mass_16_bins", "p90", "all"),
        ("natoms_train_p90", "natoms", "p90", "all"),
        ("n_species_train_p90", "n_unique_elements", "p90", "all"),
        ("volume_per_atom_train_p90", "volume_per_atom", "p90", "all"),
        ("cell_aspect_ratio_train_p90", "cell_aspect_ratio", "p90", "all"),
        ("element_rarity_train_p90", "mean_element_rarity", "p90", "all"),
        ("partial_coverage", "edos_coverage_fraction", "partial", "all"),
        ("roughness_train_p90_full_coverage", "edos_spectral_roughness", "p90", "full"),
        ("entropy_train_p90_full_coverage", "edos_entropy_norm", "p90", "full"),
        ("peak_share_train_p90_full_coverage", "edos_peak_share", "p90", "full"),
        ("edge_mass_train_p90_full_coverage", "edos_edge_mass_16_bins", "p90", "full"),
    ]
    rows = []
    for index, (name, column, rule, population_rule) in enumerate(specs):
        population_mask = (
            samples["edos_coverage_fraction"].to_numpy(dtype=np.float64) == 1.0
            if population_rule == "full"
            else np.ones(len(samples), dtype=bool)
        )
        if rule == "p90":
            threshold = float(np.quantile(train_features[column], 0.90))
            feature_high = samples[column].to_numpy(dtype=np.float64) >= threshold
            threshold_text = f"train p90 >= {threshold:.8g}"
        else:
            threshold = 1.0
            feature_high = samples[column].to_numpy(dtype=np.float64) < 1.0
            threshold_text = "coverage fraction < 1"
        high_mask = population_mask & feature_high
        other_mask = population_mask & ~feature_high
        if population_rule == "full":
            threshold_text += "; full coverage only"
        if not high_mask.any() or not other_mask.any():
            raise ValueError(f"stratum {name} has an empty comparison group")
        row: dict[str, float | int | str] = {
            "stratum": name,
            "feature": column,
            "threshold_or_rule": threshold_text,
            "threshold": threshold,
            "n_high": int(high_mask.sum()),
            "n_other": int(other_mask.sum()),
        }
        feature_values = samples[column].to_numpy(dtype=np.float64)
        gap = samples["edos_blind_gap"].to_numpy(dtype=np.float64)
        gamma_error = samples["gamma_abs_error"].to_numpy(dtype=np.float64)
        oracle = samples["r2_edos_oracle_unmasked"].to_numpy(dtype=np.float64)
        row["spearman_feature_oracle_r2"] = _spearman(
            feature_values[population_mask], oracle[population_mask]
        )
        row["spearman_feature_blind_gap"] = _spearman(
            feature_values[population_mask], gap[population_mask]
        )
        row["spearman_feature_gamma_abs_error"] = _spearman(
            feature_values[population_mask], gamma_error[population_mask]
        )

        if name in {
            "roughness_train_p90",
            "entropy_train_p90",
            "roughness_train_p90_full_coverage",
            "entropy_train_p90_full_coverage",
        }:
            shape_columns = (
                "edos_roughness_bias",
                "edos_peak_share_bias",
                "edos_peak_bin_shift_abs",
                "edos_slope_error_mae",
                "edos_slope_error_high_gradient_mae",
                "edos_slope_error_other_gradient_mae",
                "edos_slope_error_high_gradient_share",
            )
            for shape_index, shape_column in enumerate(shape_columns):
                shape_values = samples[shape_column].to_numpy(dtype=np.float64)
                shape_high = shape_values[high_mask]
                shape_other = shape_values[other_mask]
                shape_ci = bootstrap_difference_ci(
                    shape_high,
                    shape_other,
                    "median",
                    replicates=replicates,
                    seed=20263923 + index * 100 + shape_index,
                )
                column_prefix = shape_column.removeprefix("edos_")
                row[f"{column_prefix}_median_high"] = float(np.median(shape_high))
                row[f"{column_prefix}_median_other"] = float(np.median(shape_other))
                row[f"{column_prefix}_median_delta"] = float(
                    np.median(shape_high) - np.median(shape_other)
                )
                row[f"{column_prefix}_median_delta_ci95_low"] = shape_ci[0]
                row[f"{column_prefix}_median_delta_ci95_high"] = shape_ci[1]
                if shape_column == "edos_roughness_bias":
                    high_ci = bootstrap_median_ci(
                        shape_high,
                        replicates=replicates,
                        seed=20264923 + index,
                    )
                    row[f"{column_prefix}_median_high_ci95_low"] = high_ci[0]
                    row[f"{column_prefix}_median_high_ci95_high"] = high_ci[1]

        for mode in ("oracle", "blind"):
            metric = samples[f"r2_edos_{mode}_unmasked"].to_numpy(dtype=np.float64)
            high_values = metric[high_mask]
            other_values = metric[other_mask]
            median_delta = float(np.median(high_values) - np.median(other_values))
            median_ci = bootstrap_difference_ci(
                high_values,
                other_values,
                "median",
                replicates=replicates,
                seed=20260923 + index * 10 + (mode == "blind"),
            )
            fail_high = (high_values < 0).astype(np.float64)
            fail_other = (other_values < 0).astype(np.float64)
            fail_delta = float((fail_high.mean() - fail_other.mean()) * 100.0)
            fail_ci = bootstrap_difference_ci(
                fail_high,
                fail_other,
                "mean",
                replicates=replicates,
                seed=20261923 + index * 10 + (mode == "blind"),
            )
            row.update(
                {
                    f"{mode}_median_high": float(np.median(high_values)),
                    f"{mode}_median_other": float(np.median(other_values)),
                    f"{mode}_median_delta": median_delta,
                    f"{mode}_median_delta_ci95_low": median_ci[0],
                    f"{mode}_median_delta_ci95_high": median_ci[1],
                    f"{mode}_fail_pct_high": float(fail_high.mean() * 100.0),
                    f"{mode}_fail_pct_other": float(fail_other.mean() * 100.0),
                    f"{mode}_fail_delta_pp": fail_delta,
                    f"{mode}_fail_delta_ci95_low_pp": fail_ci[0] * 100.0,
                    f"{mode}_fail_delta_ci95_high_pp": fail_ci[1] * 100.0,
                }
            )
        gap_high = gap[high_mask]
        gap_other = gap[other_mask]
        row["blind_gap_median_high"] = float(np.median(gap_high))
        row["blind_gap_median_other"] = float(np.median(gap_other))
        row["blind_gap_median_delta"] = float(np.median(gap_high) - np.median(gap_other))
        rows.append(row)
    return pd.DataFrame(rows)


def run_audit(
    checkpoint_path: Path,
    config_path: Path,
    output_prefix: Path,
    device: torch.device,
    bootstrap_replicates: int = 2000,
    force: bool = False,
    split: str = "test",
    expected_epoch: int | None = 33,
    verify_reference_r2: bool = True,
    edos_shape_sumnorm_path: Path | None = None,
) -> tuple[Path, Path]:
    os.chdir(REPO_ROOT)
    if split not in {"valid", "test"}:
        raise ValueError("split must be 'valid' or 'test'")
    if not verify_reference_r2 and split != "valid":
        raise ValueError("reference R2 checks may only be skipped for valid candidate evaluation")
    output_prefix = output_prefix if output_prefix.is_absolute() else REPO_ROOT / output_prefix
    sample_path = output_prefix.with_name(f"{output_prefix.name}_samples.csv")
    strata_path = output_prefix.with_name(f"{output_prefix.name}_strata.csv")
    if not force and (sample_path.exists() or strata_path.exists()):
        raise FileExistsError("audit outputs already exist; pass --force to replace them")

    manifest = json.loads(Path("data/train4ARPAT/manifest.json").read_text())
    expected_count = {"valid": 2313, "test": 2287}[split]
    if manifest["splits"][split]["n"] != expected_count:
        raise ValueError(f"expected the frozen Q1 {split} split with {expected_count:,} samples")

    ids = np.load(f"data/train4ARPAT/{split}/{split}_index.npy").astype(str)
    if len(ids) != expected_count:
        raise ValueError(f"{split}_index length does not match the Q1 manifest")

    evaluated = evaluate_b7(
        checkpoint_path,
        config_path,
        device,
        split=split,
        expected_epoch=expected_epoch,
        collect_edos_shape_sumnorm=edos_shape_sumnorm_path is not None,
    )
    if len(evaluated["r2_oracle_unmasked"]) != len(ids):
        raise ValueError(f"B7 inference output does not match the Q1 {split} split length")
    if split == "test":
        official = pd.read_csv("results/samples_m1_e9ctl_test.csv")
        blind_reference = pd.read_csv("results/samples_m1_e9ctl_blind_test.csv")
        phdos_sidecar = pd.read_csv("results/d4_phdos_b7_test_samples.csv")
        if len(official) != len(ids) or len(blind_reference) != len(ids):
            raise ValueError("B7 sample results do not match the Q1 test split length")
        if not np.array_equal(ids, phdos_sidecar["sample_index"].astype(str).to_numpy()):
            raise ValueError("B7 test row order does not match the ID-anchored D4 sidecar")
        if not np.allclose(
            official["r2_phdos"], phdos_sidecar["r2_phdos"], atol=2e-5, rtol=0
        ):
            raise ValueError("B7 sample rows do not match the ID-anchored D4 phDOS metrics")
        if not np.allclose(
            evaluated["r2_oracle_unmasked"], official["r2_edos"], atol=2e-5, rtol=0
        ):
            raise ValueError("recomputed B7 oracle R2 does not reproduce the official sample results")
        if not np.allclose(
            evaluated["r2_oracle_unmasked"],
            blind_reference["r2_edos_oracle"],
            atol=2e-5,
            rtol=0,
        ):
            raise ValueError("recomputed B7 oracle R2 does not reproduce the blind sidecar")
        if not np.allclose(
            evaluated["r2_blind_unmasked"], blind_reference["r2_edos_blind"], atol=2e-5, rtol=0
        ):
            raise ValueError("recomputed B7 blind R2 does not reproduce the blind sidecar")
    elif verify_reference_r2:
        valid_reference = pd.read_csv("results/c2_1b_valid_samples.csv")
        reference_r2 = ordered_valid_reference_r2(valid_reference, expected_count)
        if not np.allclose(
            evaluated["r2_oracle_unmasked"], reference_r2, atol=2e-5, rtol=0
        ):
            raise ValueError("recomputed B7 valid oracle R2 does not reproduce C2.1b samples")

    if edos_shape_sumnorm_path is not None:
        shape_path = Path(edos_shape_sumnorm_path)
        if not shape_path.is_absolute():
            shape_path = REPO_ROOT / shape_path
        shapes = validate_edos_shape_sumnorm(
            evaluated["edos_shape_sumnorm"], expected_count)
        shape_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(shape_path, shapes.astype(np.float32))

    train_spectra = np.load("data/train4ARPAT/train/edos_tgtdos_train.npy")
    split_spectra = np.load(f"data/train4ARPAT/{split}/edos_tgtdos_{split}.npy")
    train_spectral = spectral_descriptors(train_spectra)
    split_spectral = spectral_descriptors(split_spectra)
    train_elements = np.load("data/train4ARPAT/train/elements_train.npy")
    split_elements = np.load(f"data/train4ARPAT/{split}/elements_{split}.npy")
    train_positions = np.load("data/train4ARPAT/train/positions_train.npy")
    split_positions = np.load(f"data/train4ARPAT/{split}/positions_{split}.npy")
    train_structure = structural_descriptors(train_elements, train_positions)
    split_structure = structural_descriptors(
        split_elements, split_positions, reference_elements=train_elements
    )

    samples = pd.DataFrame(
        {
            "mpid": ids,
            "r2_edos_oracle_unmasked": evaluated["r2_oracle_unmasked"],
            "r2_edos_blind_unmasked": evaluated["r2_blind_unmasked"],
            "r2_edos_oracle_masked": evaluated["r2_oracle_masked"],
            "r2_edos_blind_masked": evaluated["r2_blind_masked"],
            "r2_phdos_oracle_unmasked": evaluated["r2_phdos_oracle_unmasked"],
            "r2_phdos_blind_unmasked": evaluated["r2_phdos_blind_unmasked"],
            "edos_coverage_fraction": evaluated["coverage_fraction"],
            "edos_target_mass_outside_mask": evaluated["target_mass_outside_mask"],
            "gamma_true": evaluated["gamma_true"],
            "gamma_pred": evaluated["gamma_pred"],
            "gamma_abs_error": np.abs(evaluated["gamma_pred"] - evaluated["gamma_true"]),
            "edos_blind_gap": evaluated["r2_oracle_unmasked"] - evaluated["r2_blind_unmasked"],
        }
    )
    shape_columns = (
        "predicted_roughness",
        "roughness_bias",
        "predicted_peak_share",
        "peak_share_bias",
        "peak_bin_shift",
        "peak_bin_shift_abs",
        "slope_error_mae",
        "slope_error_high_gradient_mae",
        "slope_error_other_gradient_mae",
        "slope_error_high_gradient_share",
    )
    for column in shape_columns:
        samples[f"edos_{column}"] = evaluated[column]
    samples["gamma_abs_log_ratio_error"] = np.abs(
        np.log(np.maximum(evaluated["gamma_pred"], 1e-12))
        - np.log(np.maximum(evaluated["gamma_true"], 1e-12))
    )
    spectral_columns = {
        "roughness": "edos_spectral_roughness",
        "entropy_norm": "edos_entropy_norm",
        "peak_share": "edos_peak_share",
        "edge_mass": "edos_edge_mass_16_bins",
    }
    for feature, column in spectral_columns.items():
        samples[column] = split_spectral[feature]
    for feature, values in split_structure.items():
        samples[feature] = values

    train_features = {
        "edos_spectral_roughness": train_spectral["roughness"],
        "edos_entropy_norm": train_spectral["entropy_norm"],
        "edos_peak_share": train_spectral["peak_share"],
        "edos_edge_mass_16_bins": train_spectral["edge_mass"],
        **train_structure,
    }
    strata = build_strata_summary(train_features, samples, bootstrap_replicates)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(sample_path, index=False)
    strata.to_csv(strata_path, index=False)
    return sample_path, strata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("output/ablation_m1_e9ctl/checkpoint_best.pth")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("output/ablation_m1_e9ctl/config_used.yaml")
    )
    parser.add_argument(
        "--output-prefix", type=Path
    )
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--expected-epoch", type=int, default=33)
    parser.add_argument("--skip-reference-r2-check", action="store_true")
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
    paths = run_audit(
        args.checkpoint,
        args.config,
        args.output_prefix
        or Path(
            "results/edos_error_attribution_q1"
            if args.split == "test"
            else "results/edos_error_attribution_q1_valid"
        ),
        device,
        bootstrap_replicates=args.bootstrap,
        force=args.force,
        split=args.split,
        expected_epoch=args.expected_epoch,
        verify_reference_r2=not args.skip_reference_r2_check,
    )
    print(f"Q1 {args.split} eDOS attribution samples: {paths[0]}")
    print(f"Q1 {args.split} eDOS attribution strata: {paths[1]}")


if __name__ == "__main__":
    main()
