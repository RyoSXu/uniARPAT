"""Read-only C0/C1/C2 diagnosis: response, coverage strata, frozen linear probes.

Uses existing MAE-selected checkpoints (seed 42), fixed train subset and full valid.
No parameter updates on the saved models; train-only fitting applies only to diagnostic
coefficients (global magnitude scale and ridge probes). Train/valid only; no test access.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import torch

from tools.eval.lg_structure_diagnosis import (
    InformationProbe,
    dump,
    feature_pairs,
    loader,
    pair_errors,
    paired_group_interval,
    strata,
    summarize_pairs,
)
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position, spectral_summary
from tools.eval.lg_valid_export import load_run, select_best
from tools.eval.local_structure_compare import verify_completed_control, verify_preflight
from tools.train import evaluate_split
from utils.builder import ConfigBuilder
from utils.local_structure_experiment import (
    DEFAULT_REFERENCE,
    load_control_reference,
    structure_control_configs,
)
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import state_hash

ATTRIBUTION = ROOT / "results/runs/local_structure_attribution_20261009"
PRIOR_GEOMETRY = ROOT / "results/runs/lg_structure_diagnosis_20261009"
PRIOR_SUBSET_REPORT = PRIOR_GEOMETRY / "report.json"
LABELS = ("C0", "C1", "C2")
GROUPINGS = ("absolute", "reduced")
PROBE_STAGES = ("local[1].scalar", "global[5]", "decoder.edos")
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0)
TASK = "edos"


def run_directory(label, reference_dir, controls, _preflight):
    if label == "C2":
        return reference_dir
    cfg = next(c for c in controls if c.local_route == label)
    return repository_path(f"results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}")


def load_export_tables(analysis_dir, label, split):
    pairs = {}
    for grouping in ("absolute", "reduced"):
        path = analysis_dir / f"{label}_{split}_{grouping}_{TASK}_pairs.csv"
        pairs[grouping] = pd.read_csv(path)
    with np.load(analysis_dir / f"{label}_{split}_spectra.npz") as spectra:
        ids = spectra["material_ids"].tolist()
        arrays = {key: spectra[key] for key in spectra.files if key != "material_ids"}
    samples = pd.read_csv(analysis_dir / f"{label}_{split}_samples.csv")
    return pairs, arrays, ids, samples


def fit_global_scale(pred_delta, target_delta):
    pred_delta = np.asarray(pred_delta, dtype=np.float64)
    target_delta = np.asarray(target_delta, dtype=np.float64)
    denom = float((pred_delta * pred_delta).sum())
    if denom < 1e-18:
        return 0.0
    return float((pred_delta * target_delta).sum() / denom)


def pair_delta_vectors(spectra, pairs, space_left, space_right):
    left = pairs.i.to_numpy(int)
    right = pairs.j.to_numpy(int)
    pred = spectra[space_left][left] - spectra[space_left][right]
    target = spectra[space_right][left] - spectra[space_right][right]
    return pred.astype(np.float64), target.astype(np.float64)


def calibrated_errors(pred, target, scale):
    pred = np.atleast_2d(np.asarray(pred, dtype=np.float64))
    target = np.atleast_2d(np.asarray(target, dtype=np.float64))
    scaled = scale * pred
    err = 0.5 * np.abs(scaled - target).sum(-1)
    amp = 0.5 * np.abs(target).sum(-1)
    skill = np.where(amp > 1e-10, 1.0 - err / amp, np.nan)
    denom = np.linalg.norm(target, axis=1) * np.linalg.norm(scaled, axis=1)
    cosine = np.where(denom > 1e-12, (scaled * target).sum(-1) / denom, np.nan)
    ratio = np.where(amp > 1e-10, 0.5 * np.abs(scaled).sum(-1) / amp, np.nan)
    return dict(difference_error=err, skill=skill, direction_cosine=cosine, amplitude_ratio=ratio)


def summarize_vector_metrics(errors):
    out = {}
    for key, values in errors.items():
        finite = values[np.isfinite(values)]
        out[key + "_median"] = float(np.median(finite)) if len(finite) else None
    out["pairs"] = len(errors["difference_error"])
    return out


def grouped_mean_frame(frame, value_col):
    grouped = frame.groupby("composition")[value_col].mean()
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(grouped.to_numpy(), len(grouped), replace=True).mean()
                     for _ in range(2000)])
    return {"groups": len(grouped), "mean_group": float(grouped.mean()),
            "ci95": np.percentile(boot, [2.5, 97.5]).tolist()}


def response_block(analysis_dir, labels, grouping):
    block = {"grouping": grouping, "task": TASK, "space": "pred_shape_vs_tgt_shape",
             "zero_reference": "C0", "calibration": {}, "versus_zero": {}, "models": {}}
    for split in ("train", "valid"):
        c0_pairs = pd.read_csv(analysis_dir / f"C0_{split}_{grouping}_{TASK}_pairs.csv")
        block["models"]["C0_" + split] = summarize_pairs(c0_pairs)
        finite = c0_pairs.target_tv > 1e-10
        block["models"]["C0_" + split]["amplitude_ratio_median"] = (
            float((c0_pairs.loc[finite, "predicted_tv"] / c0_pairs.loc[finite, "target_tv"]).median())
            if finite.any() else None)
    for label in labels:
        for split in ("train", "valid"):
            pairs, _, _, _ = load_export_tables(analysis_dir, label, split)
            p = pairs[grouping]
            block["models"][f"{label}_{split}"] = summarize_pairs(p)
            finite = p.target_tv > 1e-10
            block["models"][f"{label}_{split}"]["amplitude_ratio_median"] = (
                float((p.loc[finite, "predicted_tv"] / p.loc[finite, "target_tv"]).median())
                if finite.any() else None)
            block["models"][f"{label}_{split}"]["direction_cosine_median"] = (
                float(p.direction_cosine.dropna().median()) if p.direction_cosine.notna().any()
                else None)
        train_pairs, train_spectra, _, _ = load_export_tables(analysis_dir, label, "train")
        valid_pairs, valid_spectra, _, _ = load_export_tables(analysis_dir, label, "valid")
        pred_tr, tgt_tr = pair_delta_vectors(
            train_spectra, train_pairs[grouping], f"pred_shape_{TASK}", f"tgt_shape_{TASK}")
        pred_va, tgt_va = pair_delta_vectors(
            valid_spectra, valid_pairs[grouping], f"pred_shape_{TASK}", f"tgt_shape_{TASK}")
        scale = fit_global_scale(pred_tr, tgt_tr)
        block["calibration"][label] = {
            "train_global_scale": scale,
            "fit_scope": f"all {grouping}-composition train pairs; blind, locked before valid",
            "eval": {}}
        for split, pairs_g, pred, target in (("train", train_pairs[grouping], pred_tr, tgt_tr),
                                             ("valid", valid_pairs[grouping], pred_va, tgt_va)):
            err = calibrated_errors(pred, target, scale)["difference_error"]
            raw = pairs_g.difference_error.to_numpy(float)
            block["calibration"][label]["eval"][split] = {
                **summarize_vector_metrics(calibrated_errors(pred, target, scale)),
                "raw_group_mean": grouped_mean_frame(pairs_g.assign(delta=raw), "delta"),
                "group_mean": grouped_mean_frame(
                    pairs_g.assign(calibrated_difference_error=err), "calibrated_difference_error"),
                "raw_minus_calibrated_group_mean": grouped_mean_frame(
                    pairs_g.assign(delta=raw - err), "delta"),
            }
        oracle = fit_global_scale(pred_va, tgt_va)
        err_oracle = calibrated_errors(pred_va, tgt_va, oracle)["difference_error"]
        raw_va = valid_pairs[grouping].difference_error.to_numpy(float)
        block["calibration"][label]["valid_oracle_scale"] = {
            "scale": oracle,
            "mode": "oracle upper bound: scalar fitted on valid; never used for model selection",
            "group_mean": grouped_mean_frame(
                valid_pairs[grouping].assign(calibrated_difference_error=err_oracle),
                "calibrated_difference_error"),
            "raw_minus_oracle_group_mean": grouped_mean_frame(
                valid_pairs[grouping].assign(delta=raw_va - err_oracle), "delta")}
    for label in ("C1", "C2"):
        block["versus_zero"][label] = {}
        for split in ("train", "valid"):
            c0 = pd.read_csv(analysis_dir / f"C0_{split}_{grouping}_{TASK}_pairs.csv")
            cur = pd.read_csv(analysis_dir / f"{label}_{split}_{grouping}_{TASK}_pairs.csv")
            block["versus_zero"][label][split] = {
                "paired_group_error_delta": paired_group_interval(c0, cur),
                "skill_gain_vs_zero_median": float(
                    (cur.skill.fillna(0.) - c0.skill.fillna(0.)).median()),
            }
    return block


def pair_strata(metadata, pairs, thresholds):
    meta = metadata.set_index("material_id")
    tags = {}
    for row in pairs.itertuples():
        left = meta.loc[row.left_id]
        right = meta.loc[row.right_id]
        tags[row.Index] = {
            "composition_seen": left.same_composition_train_count > 0,
            "composition_unseen": left.same_composition_train_count == 0,
            "radial_near": (np.isfinite(left.radial_support_distance)
                            and np.isfinite(right.radial_support_distance)
                            and left.radial_support_distance <= thresholds["radial_q25"]
                            and right.radial_support_distance <= thresholds["radial_q25"]),
            "radial_far": (np.isfinite(left.radial_support_distance)
                           and np.isfinite(right.radial_support_distance)
                           and left.radial_support_distance > thresholds["radial_q75"]
                           and right.radial_support_distance > thresholds["radial_q75"]),
        }
    frame = pairs.copy()
    for name in ("composition_seen", "composition_unseen", "radial_near", "radial_far"):
        frame[name] = [tags[i][name] for i in frame.index]
    return frame


def support_count_strata(frame, metadata):
    """Same-composition train support bins; counts come from the full-train geometry table."""
    counts = metadata.same_composition_train_count.to_numpy()
    conditions = {
        "composition_count_0": counts == 0,
        "composition_count_1": counts == 1,
        "composition_count_ge2": counts >= 2,
    }
    return {name: {"n": int(selected.sum()), **spectral_summary(frame.loc[selected])}
            for name, selected in conditions.items() if selected.any()}


def strict_match_report(verification_path):
    """Prior strict-match evidence, including concrete close-structure/large-gap cases."""
    if not verification_path.exists():
        return {}
    strict = json.loads(verification_path.read_text()).get("strict_matches", {})
    report = {}
    for split in ("train", "valid"):
        value = strict.get(split, {})
        cases = [{key: row.get(key) for key in
                  ("left_id", "right_id", "formula", "normalized_rms",
                   "volume_per_atom_ratio", "edos_target_tv", "phdos_target_tv")}
                 for row in value.get("pairs", []) if row.get("strict_match")]
        report[split] = {
            "candidate_pairs": value.get("candidate_pairs"),
            "strict_count": value.get("strict_count"),
            "settings": value.get("settings"),
            "close_pairs_with_large_edos_gap": sorted(
                cases, key=lambda row: -(row["edos_target_tv"] or 0.0)),
        }
    return report


def coverage_block(analysis_dir, geometry_dir, prior_report, labels, groupings):
    train_geo = pd.read_csv(geometry_dir / "geometry_train.csv")
    valid_geo = pd.read_csv(geometry_dir / "geometry_valid.csv")
    thresholds = prior_report["strata_thresholds_train_only"]
    block = {"geometry_source": str(geometry_dir),
             "thresholds_train_only": thresholds,
             "per_sample_strata_C1": {}, "pair_strata": {g: {} for g in groupings},
             "structure_matches": {},
             "proxy_limits": prior_report.get("radial_proxy", "")}
    train_ids = prior_report["splits"]["train"]["material_ids"]
    for split, ids, geo in (("train", train_ids, train_geo), ("valid", None, valid_geo)):
        if split == "train":
            frame = (pd.read_csv(analysis_dir / "C1_train_samples.csv")
                     .set_index("material_id").loc[train_ids].reset_index())
            meta = train_geo.set_index("material_id").loc[train_ids].reset_index()
            target_shape = np.load(analysis_dir / "C1_train_spectra.npz")["tgt_shape_edos"]
        else:
            frame = pd.read_csv(analysis_dir / "C1_valid_samples.csv")
            meta = valid_geo
            target_shape = np.load(analysis_dir / "C1_valid_spectra.npz")["tgt_shape_edos"]
        strata_report = strata(frame, meta, target_shape, thresholds)
        strata_report.update(support_count_strata(frame, meta))
        block["per_sample_strata_C1"][split] = strata_report
        for grouping in groupings:
            tagged = pair_strata(
                geo, pd.read_csv(analysis_dir / f"C1_{split}_{grouping}_{TASK}_pairs.csv"), thresholds)
            summary = {}
            for label in labels:
                other = pd.read_csv(analysis_dir / f"{label}_{split}_{grouping}_{TASK}_pairs.csv")
                keyed = other.set_index(["left_id", "right_id", "composition"])
                for name in ("composition_seen", "composition_unseen", "radial_near", "radial_far"):
                    selected = tagged[tagged[name]]
                    if not len(selected):
                        continue
                    sub = keyed.loc[selected.set_index(["left_id", "right_id", "composition"]).index]
                    sub = sub.reset_index()
                    summary[f"{label}_{name}"] = {
                        "pairs": len(sub),
                        "summarize": summarize_pairs(sub),
                        "group_error": grouped_mean_frame(sub, "difference_error"),
                    }
            block["pair_strata"][grouping][split] = summary
    strict_report = strict_match_report(geometry_dir / "verification_reproducible.json")
    block["structure_matches"] = {
        "tolerant_valid_edos_pairs_prior_baseline35":
        prior_report["groups"]["baseline35"]["splits"]["valid"]["structure_matches"],
        "strict": strict_report,
        "note": "geometry proxy omits magnetism, smearing and calculation settings; "
                "close structures with large spectral gaps are provenance questions, not label errors",
    }
    return block


def stack_probe_features(probe_features, n_samples):
    out = {}
    for stage in PROBE_STAGES:
        chunks = probe_features.get(stage, [])
        if not chunks:
            raise ValueError(f"missing probe stage: {stage}")
        values = np.concatenate(chunks, axis=0)
        if values.shape[0] != n_samples:
            raise ValueError(f"probe rows {values.shape[0]} != {n_samples} for {stage}")
        out[stage] = values.astype(np.float64)
    return out


def apply_ridge_probe(model, features, pairs, target_shape):
    left, right = pairs.i.to_numpy(int), pairs.j.to_numpy(int)
    pred = model.predict(features[left] - features[right])
    actual = target_shape[left] - target_shape[right]
    err = 0.5 * np.abs(pred - actual).sum(-1)
    return pred, err


def ridge_probe_cv(features, pairs, target_shape, compositions, alphas=RIDGE_ALPHAS, n_splits=5):
    left = pairs.i.to_numpy(int)
    right = pairs.j.to_numpy(int)
    x = features[left] - features[right]
    y = target_shape[left] - target_shape[right]
    groups = compositions
    unique = np.unique(groups)
    if len(unique) < n_splits:
        n_splits = max(2, len(unique))
    splitter = GroupKFold(n_splits=n_splits)
    trace, best_alpha, best_score = {}, alphas[0], np.inf
    for alpha in alphas:
        scores = []
        for train_idx, test_idx in splitter.split(x, y, groups):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            err = 0.5 * np.abs(pred - y[test_idx]).sum(-1)
            scores.append(float(np.median(err)))
        trace[str(alpha)] = float(np.mean(scores))
        if trace[str(alpha)] < best_score:
            best_alpha, best_score = alpha, trace[str(alpha)]
    final = Ridge(alpha=best_alpha, fit_intercept=True)
    final.fit(x, y)
    info = {"cv_median_pair_error": trace,
            "selected_at_grid_edge": bool(best_alpha in (alphas[0], alphas[-1])),
            "groups": int(len(unique)), "n_splits": int(n_splits)}
    return final, best_alpha, best_score, info


def probe_block(out_dir, labels, reference_dir, controls, preflight, prior_report, device, groupings):
    rows = {split: prior_report["splits"][split]["rows"] for split in ("train", "valid")}
    block = {"stages": PROBE_STAGES, "task": TASK, "mode": "shape target deltas only",
             "fit": "ridge on train pairs; alpha via composition-group CV on train; locked on valid",
             "alphas": list(RIDGE_ALPHAS), "models": {}, "comparisons": {}}
    frames = {}
    for label in labels:
        run = run_directory(label, reference_dir, controls, preflight)
        if label in ("C0", "C1"):
            saved, picked = verify_completed_control(
                next(c for c in controls if c.local_route == label), preflight)
        else:
            saved, _, contract = load_run(run)
            picked = select_best(run, contract["selection_metric"], saved)
        build = ConfigBuilder(**saved["config"])
        model = build.get_model()
        transformer = model.model["transformer"]
        transformer.load_state_dict(picked["checkpoint"]["model"], strict=True)
        del picked["checkpoint"]
        model.to(device)
        transformer.eval()
        digest_before = state_hash(transformer.state_dict())
        ckpt_digest = file_sha256(run / "checkpoint_best.pth")
        entry = {"run": str(run.relative_to(ROOT)), "checkpoint_sha256": ckpt_digest,
                 "groupings": {}}
        split_data = {}
        for split in ("train", "valid"):
            dataset = build.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
            ids = prior_report["splits"][split]["material_ids"]
            data = loader(dataset, rows[split])
            keys = composition_keys_by_position(dataset, rows[split])
            probe = InformationProbe(transformer)
            tick = time.monotonic()
            try:
                measured = evaluate_split(model, data, sumnorm=True, return_sample_level=True,
                                          return_spectra=True, sample_ids=ids)
            finally:
                probe.close()
            spectra = measured["sample_spectra"]
            n = len(ids)
            features = stack_probe_features(probe.features, n)
            path = out_dir / f"probe_features_{label}_{split}.npz"
            np.savez(path, material_ids=np.asarray(ids), **{k: v for k, v in features.items()})
            pairs = {grouping: pair_errors(
                spectra[f"pred_shape_{TASK}"], spectra[f"tgt_shape_{TASK}"], keys[grouping], ids)
                for grouping in groupings}
            split_data[split] = (features, pairs, spectra, time.monotonic() - tick, path.name)
            del dataset, measured, probe
        for grouping in groupings:
            train_pairs = split_data["train"][1][grouping]
            target = split_data["train"][2][f"tgt_shape_{TASK}"]
            fitted = {stage: ridge_probe_cv(split_data["train"][0][stage], train_pairs, target,
                                            train_pairs.composition.to_numpy())
                      for stage in PROBE_STAGES}
            group_entry = {"train_pairs": len(train_pairs), "splits": {}}
            for split in ("train", "valid"):
                features, pairs_by_grouping, spectra, seconds, feature_name = split_data[split]
                pairs = pairs_by_grouping[grouping]
                pair_frame = pairs.copy()
                shape = spectra[f"tgt_shape_{TASK}"]
                stage_reports = {}
                for stage in PROBE_STAGES:
                    model_final, alpha, cv_median, trace = fitted[stage]
                    _, err = apply_ridge_probe(model_final, features[stage], pairs, shape)
                    pair_frame[f"probe_{stage}_error"] = err
                    stage_reports[stage] = {
                        "train_cv_alpha": alpha,
                        "train_cv_median_pair_error": cv_median,
                        "train_cv_trace": trace,
                        "locked_median_pair_error": float(np.median(err)),
                        "locked_group_mean_error": grouped_mean_frame(
                            pair_frame, f"probe_{stage}_error"),
                        "model_median_pair_error": float(pairs.difference_error.median()),
                        "model_group_mean_error": grouped_mean_frame(pair_frame, "difference_error"),
                        "feature_pair_geometry": feature_pairs({stage: [features[stage]]}, pairs)[stage],
                    }
                group_entry["splits"][split] = {
                    "pairs": len(pairs), "seconds": seconds, "features": feature_name,
                    "stages": stage_reports,
                    "lossy_feature_warning": "stages use pooled mean/RMS summaries only"}
                frames[label, grouping, split] = pair_frame
            entry["groupings"][grouping] = group_entry
        if state_hash(transformer.state_dict()) != digest_before or file_sha256(
                run / "checkpoint_best.pth") != ckpt_digest:
            raise ValueError("checkpoint or model state changed during probe collection")
        entry["state_unchanged"] = True
        block["models"][label] = entry
        del model, transformer, build
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for grouping in groupings:
        block["comparisons"][grouping] = {}
        for stage in PROBE_STAGES:
            block["comparisons"][grouping][stage] = {}
            for split in ("train", "valid"):
                c0 = frames["C0", grouping, split]
                c1 = frames["C1", grouping, split]
                c2 = frames["C2", grouping, split]
                col = f"probe_{stage}_error"
                block["comparisons"][grouping][stage][split] = {
                    "C1_minus_C0_probe_group_mean": grouped_mean_frame(
                        c1.assign(delta=c1[col] - c0[col]), "delta"),
                    "C1_minus_C2_probe_group_mean": grouped_mean_frame(
                        c1.assign(delta=c1[col] - c2[col]), "delta"),
                    "C1_model_minus_probe_group_mean": grouped_mean_frame(
                        c1.assign(delta=c1.difference_error - c1[col]), "delta"),
                }
    return block


def diagnose(out_dir, smoke=False):
    if (out_dir / "report.json").exists():
        raise FileExistsError(f"refusing completed output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analysis_dir = ATTRIBUTION / "analysis"
    preflight_path = ATTRIBUTION / "preflight_r1.json"
    reference_dir = repository_path(DEFAULT_REFERENCE)
    _, reference_cfg = load_control_reference(reference_dir)
    controls = structure_control_configs(reference_cfg, "attrib20261009")
    preflight = verify_preflight(preflight_path, reference_dir, controls)
    prior_report = json.loads(PRIOR_SUBSET_REPORT.read_text())
    if smoke:
        raise ValueError("smoke mode not implemented; use check tests for plumbing")

    def progress(stage):
        print(json.dumps({"stage": stage, "seconds": round(time.monotonic() - start, 3)}), flush=True)

    inputs = [f"{label}_{split}_{grouping}_{TASK}_pairs.csv"
              for label in LABELS for split in ("train", "valid") for grouping in GROUPINGS]
    inputs += [f"{label}_{split}_{name}.{ext}" for label in LABELS
               for split in ("train", "valid") for name, ext in (("spectra", "npz"), ("samples", "csv"))]
    manifest = {
        "scope": "C0/C1/C2 read-only diagnosis; seed 42; balanced_oracle_mae checkpoints; "
                 "train/valid only; blind uses model scale, oracle not re-fit here",
        "groupings": list(GROUPINGS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "attribution_analysis": str(analysis_dir),
        "preflight_sha256": file_sha256(preflight_path),
        "prior_report_sha256": file_sha256(PRIOR_SUBSET_REPORT),
        "prior_verification_sha256": file_sha256(PRIOR_GEOMETRY / "verification_reproducible.json"),
        "analysis_inputs_sha256": {name: file_sha256(analysis_dir / name) for name in sorted(inputs)},
        "checkpoints": {label: file_sha256(run_directory(label, reference_dir, controls, preflight)
                                           / "checkpoint_best.pth") for label in LABELS},
        "source_sha256": file_sha256(Path(__file__)),
    }
    dump(out_dir / "manifest.json", manifest)
    progress("manifest")

    report = {
        "scope": manifest["scope"],
        "seed": 42,
        "device": str(device),
        "response_magnitude_direction": {grouping: response_block(analysis_dir, LABELS, grouping)
                                         for grouping in GROUPINGS},
        "coverage_identifiability": coverage_block(
            analysis_dir, PRIOR_GEOMETRY, prior_report, LABELS, GROUPINGS),
    }
    dump(out_dir / "report.partial.json", report)
    progress("response_and_coverage")
    report["representation_probes"] = probe_block(
        out_dir, LABELS, reference_dir, controls, preflight, prior_report, device, GROUPINGS)
    progress("probes")
    report["seconds"] = time.monotonic() - start
    report["peak_vram_mb"] = (torch.cuda.max_memory_allocated() / 2**20
                              if torch.cuda.is_available() else 0.0)
    report["artifact_sha256"] = {path.name: file_sha256(path)
                                 for path in sorted(out_dir.glob("probe_features_*.npz"))}
    report["limitations"] = [
        "Single training seed; diagnostic fits are not model parameters.",
        "Global magnitude scale fits one scalar on all train pairs per grouping; the valid-oracle "
        "scale is an upper bound only and is never used for model selection.",
        "Linear probes read pooled mean/RMS features; negative probe results do not prove "
        "full representations lack information.",
        "Probe pair counts are small (train pairs per grouping recorded per model); intervals "
        "are composition-group resamples and do not cover training-seed variation.",
        "Geometry strata use train-only thresholds and a radial support proxy without angles, "
        "magnetism, smearing or calculation settings.",
        "Reuses prior tolerant structure matches; strict match lists come from the prior "
        "verification and matched structures may still differ geometrically.",
    ]
    dump(out_dir / "report.json", report)
    progress("report")
    print(json.dumps({"complete": str(out_dir), "seconds": report["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    diagnose(args.out_dir, args.smoke)


if __name__ == "__main__":
    main()
