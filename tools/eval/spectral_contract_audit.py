"""Audit frozen train/valid spectral contracts without models or test arrays.

Covered-bin scores, optimal scales and alternative epochs are diagnostics only.
The audit never changes labels, checkpoints or the official selection rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.checkpoint_selection import BLIND_R2_METRIC, selected_history_row
from utils.paths import file_sha256, input_path, repository_path

SPLITS = ("train", "valid")


def validate_spectra(target, prediction=None, mask=None):
    target = np.asarray(target, dtype=float)
    if target.ndim != 2 or target.shape[1] < 2 or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("finite nonnegative [materials, bins] targets required")
    if prediction is not None and (prediction.shape != target.shape or not np.isfinite(prediction).all()
                                   or (prediction < 0).any()):
        raise ValueError("finite nonnegative predictions must match targets")
    if mask is not None and (mask.shape != target.shape or mask.dtype != np.bool_):
        raise ValueError("boolean coverage mask must match targets")
    return target


def spectral_r2(prediction, target):
    target = validate_spectra(target, prediction)
    residual = np.square(prediction - target).sum(-1)
    centered = np.square(target - target.mean(-1, keepdims=True)).sum(-1)
    return 1 - residual / (centered + 1e-8)


def energy_window_ratios(target, edges, n_valence):
    """Integrate the stored density on either side of an exact zero bin edge."""
    target = validate_spectra(target)
    edges, n_valence = np.asarray(edges, float), np.asarray(n_valence, float)
    if (edges.shape != (target.shape[1] + 1,) or not np.isfinite(edges).all()
            or not (np.diff(edges) > 0).all() or n_valence.shape != (len(target),)
            or not np.isfinite(n_valence).all() or (n_valence <= 0).any()):
        raise ValueError("aligned increasing bin edges and positive valence counts required")
    if not ((edges == 0).any() and edges[0] < 0 < edges[-1]):
        raise ValueError("zero must be an exact edge inside the energy window")
    masses = target * np.diff(edges)
    below = masses[:, edges[1:] <= 0].sum(-1) / n_valence
    above = masses[:, edges[:-1] >= 0].sum(-1) / n_valence
    return below, above


def coverage_layout(target, mask):
    target = validate_spectra(target, mask=mask)
    if (target[~mask] != 0).any():
        raise ValueError("this padding diagnostic requires zero-filled uncovered targets")
    indices = np.arange(target.shape[1])
    first = np.where(mask, indices, target.shape[1]).min(-1)
    last = np.where(mask, indices, -1).max(-1)
    inside = (indices >= first[:, None]) & (indices <= last[:, None])
    return {"covered_bins": mask.sum(-1), "internal_holes": ((~mask) & inside).sum(-1),
            "exterior_uncovered": ((~mask) & (~inside)).sum(-1),
            "zero_target": (target.sum(-1) == 0)}


def padding_decomposition(prediction, target, mask):
    """Separate the changed mean-reference denominator from outside error.

    False-mask bins are not declared physical zeros. They are the zero-filled
    values in the current cache, and are varied only for this sensitivity check.
    """
    target = validate_spectra(target, prediction, mask)
    layout = coverage_layout(target, mask)
    count = layout["covered_bins"]
    total = target.sum(-1)
    energy = np.square(target).sum(-1)
    mean = np.divide(total, count, out=np.zeros(len(target)), where=count > 0)
    covered_variance = (np.square(target - mean[:, None]) * mask).sum(-1)
    full_variance = np.square(target - target.mean(-1, keepdims=True)).sum(-1)
    growth = total**2 * (np.divide(1., count, out=np.zeros(len(target)), where=count > 0)
                        - 1 / target.shape[1])
    covered_error = (np.square(prediction - target) * mask).sum(-1)
    outside_error = (np.square(prediction) * (~mask)).sum(-1)
    usable = (count >= 2) & (covered_variance > 1e-12 * energy)
    full = spectral_r2(prediction, target)
    covered = 1 - covered_error / (covered_variance + 1e-8)
    ideal_padding = 1 - covered_error / (full_variance + 1e-8)
    identity_error = np.abs(full_variance - covered_variance - growth)
    if np.any(identity_error > 1e-10 * np.maximum(energy, 1e-20)):
        raise AssertionError("zero-padding variance identity failed")
    denominator_effect = ideal_padding - covered
    outside_penalty = ideal_padding - full
    if not np.allclose(full - covered, denominator_effect - outside_penalty, atol=1e-10, rtol=1e-10):
        raise AssertionError("R2 padding decomposition failed")
    return {"full": full, "covered": covered, "usable": usable,
            "denominator_effect": denominator_effect, "outside_penalty": outside_penalty,
            "outside_squared_error": outside_error, "covered_squared_error": covered_error,
            "variance_identity_error": identity_error, "zero_target": layout["zero_target"]}


def fixed_shape_optimum(shape, target, cap):
    """Nonnegative least-squares scale for a frozen shape, with/without cap."""
    target = validate_spectra(target, shape)
    cap = np.asarray(cap, float)
    if cap.shape != (len(target),) or not np.isfinite(cap).all() or (cap <= 0).any():
        raise ValueError("positive finite per-material caps required")
    if (shape.sum(-1) <= 0).any():
        raise ValueError("predicted shapes must have positive mass")
    q = shape / shape.sum(-1, keepdims=True)
    optimal = (q * target).sum(-1) / np.square(q).sum(-1)
    return {"optimal_total": optimal,
            "unbounded_r2": spectral_r2(q * optimal[:, None], target),
            "bounded_r2": spectral_r2(q * np.minimum(optimal, cap)[:, None], target)}


def describe(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite diagnostic values")
    if not len(values):
        return {"n": 0, "median": None, "p10": None, "p90": None}
    return {"n": len(values), "median": float(np.median(values)),
            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))}


def score_summary(values):
    values = np.asarray(values)
    return {**describe(values), "negative_count": int((values < 0).sum()),
            "negative_pct": float((values < 0).mean() * 100) if len(values) else None}


def audit(data_dir, analysis_dir, baseline, candidate, out):
    started = time.perf_counter()
    data_dir, analysis_dir, out = Path(data_dir), Path(analysis_dir), Path(out)
    if out.exists():
        raise FileExistsError(f"use a new audit output directory: {out}")
    hashes, tables, metadata = {}, {}, {}

    def track(path):
        path = Path(path).resolve()
        hashes[str(path)] = file_sha256(path)
        return path

    def array(path):
        path = track(path)
        relative = path.relative_to(data_dir.resolve()).as_posix()
        if hashes[str(path)] != manifest["files"][relative]["sha256"]:
            raise ValueError(f"data manifest mismatch: {relative}")
        return np.load(path, allow_pickle=False)

    manifest = json.loads(track(data_dir / "manifest.json").read_text())
    if manifest.get("schema") != "crystal-q1-v1":
        raise ValueError("this audit requires the frozen crystal-q1-v1 schema")
    grids = json.loads(track(data_dir.parent / "reference/grids.json").read_text())
    source = json.loads(track(analysis_dir / "report.json").read_text())
    if source["data_contract"]["manifest_sha256"] != file_sha256(data_dir / "manifest.json"):
        raise ValueError("export report and current data manifest differ")
    for name in ("model/heads.py", "model/model.py", "model/losses.py", "utils/dataset.py",
                 "utils/metrics.py", "utils/spectral_inference.py", "utils/checkpoint_selection.py"):
        path = track(ROOT / name)
        if hashes[str(path)] != source["code_sha256"][name]:
            raise ValueError(f"current contract implementation differs from the bound experiment: {name}")
    report = {"schema": "spectral-contract-audit-v1", "diagnostic_only": True,
              "scope": "existing selected checkpoints; train/valid only; no forward, fitting or updates",
              "input_sha256": hashes, "labels": {}, "predictions": {}, "histories": {},
              "covered_variance_relative_threshold": 1e-12}
    # The same element reference is used by the existing migration/inference
    # paths. Its sum is a sensitivity reference, not per-task POTCAR ground truth.
    from pymatgen.core import Element

    element_table = json.loads(track(data_dir.parent / "reference/valence_by_element.json").read_text())
    zval = np.full(119, np.nan)
    for symbol, entry in element_table.items():
        zval[Element(symbol).Z] = entry["zval"]
    for split in SPLITS:
        directory = data_dir / split
        ids = array(directory / f"{split}_index.npy")
        keep = ~np.isin(ids, manifest.get("training_exclusions", [])) if split == "train" else np.ones(len(ids), bool)
        ids = ids[keep]
        if len(set(ids)) != len(ids) or len(ids) != manifest["splits"][split]["training_n"]:
            raise ValueError("effective material IDs do not match manifest view")
        nval = array(directory / f"nvalence_{split}.npy").astype(float)[keep]
        natoms = array(directory / f"n_atoms_{split}.npy").astype(float)[keep]
        numbers = array(directory / f"atomic_numbers_{split}.npy")
        offsets = array(directory / f"atom_offsets_{split}.npy")
        if (offsets.shape != (len(keep) + 1,) or offsets[0] != 0 or offsets[-1] != len(numbers)
                or (np.diff(offsets) <= 0).any() or not np.isfinite(zval[numbers]).all()):
            raise ValueError("complete nonempty atom packing and known valence references required")
        composition_nval = np.add.reduceat(zval[numbers], offsets[:-1])[keep]
        metadata[split] = {}
        rows = pd.DataFrame({"material_id": ids, "stored_n_valence": nval,
                             "composition_reference_n_valence": composition_nval,
                             "valence_reference_relative_difference": (nval - composition_nval) / composition_nval})
        report["labels"][split] = {}
        for task, grid, counts in (("edos", "E0", nval), ("phdos", "P0", 3 * natoms)):
            target = array(directory / f"{task}_tgtdos_{split}.npy").astype(float)[keep]
            mask = array(directory / f"{task}_mask_{split}.npy")[keep]
            edges = np.asarray(grids[grid], float)
            widths = np.diff(edges)
            if len(widths) != target.shape[1] or not np.allclose(widths, widths[0], atol=1e-12, rtol=0):
                raise ValueError("current H1 audit requires its uniform production grids")
            layout = coverage_layout(target, mask)
            cap = counts / widths[0]
            ratio = target.sum(-1) / cap
            above = ratio > 1 + 1e-6
            metadata[split][task] = {"ids": ids, "target": target, "mask": mask, "cap": cap}
            rows[f"{task}_true_scale"] = ratio
            rows[f"{task}_covered_bins"] = layout["covered_bins"]
            rows[f"{task}_internal_holes"] = layout["internal_holes"]
            item = {"true_scale": describe(ratio), "above_one": int(above.sum()),
                    "above_one_pct": float(above.mean() * 100),
                    "coverage_fraction": describe(mask.mean(-1)),
                    "materials_with_internal_holes": int((layout["internal_holes"] > 0).sum()),
                    "internal_hole_bins": int(layout["internal_holes"].sum()),
                    "exterior_uncovered_bins": int(layout["exterior_uncovered"].sum()),
                    "zero_target_ids": ids[layout["zero_target"]].tolist(),
                    "zero_target_covered_bins": layout["covered_bins"][layout["zero_target"]].tolist()}
            if task == "edos":
                below, above_zero = energy_window_ratios(target, edges, nval)
                rows["edos_negative_energy_ratio"] = below
                rows["edos_positive_energy_ratio"] = above_zero
                rows["edos_total_ratio_composition_reference"] = ratio * nval / composition_nval
                item.update({"above_total_with_negative_part_at_most_one": int((above & (below <= 1 + 1e-6)).sum()),
                             "negative_part_above_one": int((below > 1 + 1e-6).sum()),
                             "negative_part_above_one_ids": ids[below > 1 + 1e-6].tolist()})
            report["labels"][split][task] = item
        tables[f"{split}_labels.csv"] = rows

    # Frozen valid exports are sufficient for physical scores and padding effects.
    checked = 0
    for name in (baseline, candidate):
        with np.load(track(analysis_dir / f"{name}_valid_spectra.npz"), allow_pickle=False) as values:
            saved = {key: values[key] for key in values.files}
        frame = pd.read_csv(track(analysis_dir / f"{name}_valid_samples.csv"))
        if frame.material_id.duplicated().any() or frame.material_id.tolist() != saved["material_ids"].tolist():
            raise ValueError("export spectra and sample IDs are not aligned")
        report["predictions"][name] = {}
        diag = pd.DataFrame({"material_id": saved["material_ids"]})
        for task in ("edos", "phdos"):
            info = metadata["valid"][task]
            order = pd.Index(info["ids"]).get_indexer(saved["material_ids"])
            if len(order) != len(info["ids"]) or (order < 0).any():
                raise ValueError("complete valid material IDs required")
            target = saved[f"tgt_{task}"].astype(float)
            pred = saved[f"pred_blind_{task}"].astype(float)
            if not np.allclose(target, info["target"][order], rtol=3e-6, atol=2e-6):
                raise ValueError("saved targets differ from current frozen data")
            padding = padding_decomposition(pred, target, info["mask"][order])
            for mode in ("blind", "oracle"):
                score = spectral_r2(saved[f"pred_{mode}_{task}"].astype(float), target)
                recorded = frame[f"{mode}_r2_{task}"].to_numpy()
                if not np.array_equal(score < 0, recorded < 0) or abs(np.median(score) - np.median(recorded)) > 3e-6:
                    raise ValueError("saved per-material metric/failure mismatch")
                checked += len(score)
            optimal = fixed_shape_optimum(saved[f"pred_shape_{task}"].astype(float), target, info["cap"][order])
            usable = padding["usable"]
            item = {"official_full": score_summary(frame[f"blind_r2_{task}"].to_numpy()),
                    "full_same_usable_cohort": score_summary(padding["full"][usable]),
                    "covered_only": score_summary(padding["covered"][usable]),
                    "paired_full_minus_covered": describe((padding["full"] - padding["covered"])[usable]),
                    "padding_denominator_effect": describe(padding["denominator_effect"][usable]),
                    "outside_prediction_penalty": describe(padding["outside_penalty"][usable]),
                    "fixed_shape_best_unbounded": score_summary(optimal["unbounded_r2"]),
                    "fixed_shape_best_current_cap": score_summary(optimal["bounded_r2"])}
            report["predictions"][name][task] = item
            for key in ("full", "covered", "usable", "denominator_effect", "outside_penalty"):
                diag[f"{task}_{key}"] = padding[key]
            diag[f"{task}_fixed_shape_best"] = optimal["unbounded_r2"]
        tables[f"{name}_valid_diagnostics.csv"] = diag
        artifact = source["run_artifacts"][name]
        history_path = track(repository_path(artifact["history_path"]))
        if hashes[str(history_path)] != artifact["history_sha256"]:
            raise ValueError("training history differs from the bound experiment")
        history = pd.read_csv(history_path)
        if source["runs"][name]["selection_metric"] != BLIND_R2_METRIC:
            raise ValueError("this audit requires the joint blind checkpoint contract")
        selected = selected_history_row(history, BLIND_R2_METRIC)
        if int(selected.epoch) != source["runs"][name]["best_epoch"]:
            raise ValueError("official selected history epoch differs from source report")
        report["histories"][name] = {"official_epoch": int(selected.epoch),
                                     "official_score": float(selected.balanced_score),
                                     "retrospective_only": True}
        feasible = np.ones(len(history), bool)
        for task in ("edos", "phdos"):
            base_score = report["predictions"][baseline][task]["official_full"]
            feasible &= history[f"blind_fail_rate_{task}"].to_numpy() <= base_score["negative_pct"] + 1e-10
            feasible &= history[f"blind_r2_{task}_median"].to_numpy() >= base_score["median"] - .02
        fields = ["epoch", "balanced_blind_r2", "blind_r2_edos_median", "blind_r2_phdos_median",
                  "blind_fail_rate_edos", "blind_fail_rate_phdos"]
        alternatives = history.loc[feasible].sort_values(["balanced_blind_r2", "epoch"], ascending=[False, True])
        report["histories"][name]["no_failure_increase_diagnostic"] = {
            "failure_increase_limit_pp": 0., "r2_drop_limit": .02,
            "row": alternatives.iloc[0][fields].to_dict() if len(alternatives) else None,
            "not_a_revised_selection_rule": True,
            "cannot_restore_unsaved_weights_from_history": True}
        tables[f"{name}_history.csv"] = history[["epoch", "balanced_blind_r2", "blind_r2_edos_median",
                                                 "blind_r2_phdos_median", "blind_fail_rate_edos",
                                                 "blind_fail_rate_phdos"]]

    audit_table = data_dir.parent / "reference/edos_valence_by_material.parquet"
    valence = pd.read_parquet(track(audit_table)).set_index("mpid")
    if not valence.index.is_unique:
        raise ValueError("historical valence audit IDs must be unique")
    for split in SPLITS:
        rows = tables[f"{split}_labels.csv"]
        selected = rows.loc[rows.edos_negative_energy_ratio > 1 + 1e-6]
        joined = selected.merge(valence, left_on="material_id", right_index=True, how="left", validate="1:1")
        tables[f"{split}_negative_energy_review.csv"] = joined
    hashes[str(Path(__file__).resolve())] = file_sha256(__file__)
    for helper in (ROOT / "utils/paths.py", ROOT / "utils/checkpoint_selection.py"):
        track(helper)
    if any(file_sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("an input changed during the audit")
    report["verification"] = {"metric_failure_flags_checked": checked, "inputs_unchanged": True,
                               "cpu_elapsed_seconds": time.perf_counter() - started}
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source_snapshot"
    for path in (Path(__file__).resolve(), ROOT / "utils/paths.py", ROOT / "utils/checkpoint_selection.py"):
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    report["source_snapshot"] = {
        str(path.relative_to(out)): file_sha256(path) for path in snapshot.rglob("*.py")}
    for name, frame in tables.items():
        frame.to_csv(out / name, index=False)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/crystal_q1_v1")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--baseline", default="A64")
    parser.add_argument("--candidate", default="LR2")
    parser.add_argument("--out", required=True, help="new output directory; existing outputs are refused")
    args = parser.parse_args()
    if any(not name.isalnum() for name in (args.baseline, args.candidate)) or args.baseline == args.candidate:
        parser.error("distinct alphanumeric model names required")
    result = audit(input_path(args.data_dir), input_path(args.analysis_dir), args.baseline, args.candidate,
                   input_path(args.out))
    print(json.dumps({"labels": result["labels"], "verification": result["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
