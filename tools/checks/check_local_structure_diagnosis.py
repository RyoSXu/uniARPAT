"""Independently recompute the C0/C1/C2 diagnosis numbers from the raw exports.

Verifies manifest fingerprints (source, preflight, prior evidence, analysis inputs,
checkpoints), material ordering between exports and probe features, and replays the
reported pair, calibration and probe statistics from the saved arrays. No model is
loaded and no test split is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.eval.local_structure_diagnosis import (
    GROUPINGS,
    LABELS,
    PROBE_STAGES,
    TASK,
    apply_ridge_probe,
    fit_global_scale,
    grouped_mean_frame,
    ridge_probe_cv,
)
from utils.paths import file_sha256

ATTRIBUTION = ROOT / "results/runs/local_structure_attribution_20261009"
PRIOR = ROOT / "results/runs/lg_structure_diagnosis_20261009"
TOL = 1e-9


def check(condition, message, result, checks):
    checks.append({"name": message, "passed": bool(condition)})
    if not condition:
        raise ValueError(message)


def assert_close(message, actual, expected, checks, tol=TOL):
    difference = abs(float(actual) - float(expected))
    checks.append({"name": message, "passed": difference <= tol, "abs_diff": difference})
    if difference > tol:
        raise ValueError(f"{message}: {difference} > {tol}")


def pair_rows(spectra, pairs):
    """Independent restatement of the signed pair-difference formula.

    The subtraction follows the export formula order (cast then subtract), so the
    replay is bit-identical to the saved columns instead of merely close.
    """
    left, right = pairs.i.to_numpy(int), pairs.j.to_numpy(int)
    target = spectra["tgt_shape_edos"].astype(np.float64)
    pred = spectra["pred_shape_edos"].astype(np.float64)
    delta_t = target[left] - target[right]
    delta_p = pred[left] - pred[right]
    amplitude = .5 * np.abs(delta_t).sum(-1)
    error = .5 * np.abs(delta_p - delta_t).sum(-1)
    denom = np.linalg.norm(delta_t, axis=-1) * np.linalg.norm(delta_p, axis=-1)
    cosine = np.divide((delta_p * delta_t).sum(-1), denom, out=np.full(len(pairs), np.nan),
                       where=denom > 1e-12)
    return amplitude, .5 * np.abs(delta_p).sum(-1), error, cosine


def delta_vectors(spectra, pairs):
    """Match the diagnosis order: subtract in the stored dtype, then cast to float64."""
    left, right = pairs.i.to_numpy(int), pairs.j.to_numpy(int)
    pred = (spectra["pred_shape_edos"][left] - spectra["pred_shape_edos"][right]).astype(np.float64)
    target = (spectra["tgt_shape_edos"][left] - spectra["tgt_shape_edos"][right]).astype(np.float64)
    return pred, target


def verify(diagnosis_dir: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"refusing existing verification: {output}")
    report = json.loads((diagnosis_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((diagnosis_dir / "manifest.json").read_text(encoding="utf-8"))
    checks = []
    result = {"source_sha256": file_sha256(Path(__file__)),
              "report_sha256": file_sha256(diagnosis_dir / "report.json"),
              "checks": checks, "max_abs": {}}

    # 1. Fingerprints: nothing used by the diagnosis changed after it ran.
    check(manifest["source_sha256"] == file_sha256(ROOT / "tools/eval/local_structure_diagnosis.py"),
          "diagnosis source unchanged since the run", result, checks)
    check(manifest["preflight_sha256"] == file_sha256(ATTRIBUTION / "preflight_r1.json"),
          "preflight fingerprint unchanged", result, checks)
    check(manifest["prior_report_sha256"] == file_sha256(PRIOR / "report.json"),
          "prior diagnosis report unchanged", result, checks)
    check(manifest["prior_verification_sha256"] == file_sha256(PRIOR / "verification_reproducible.json"),
          "prior verification unchanged", result, checks)
    analysis = Path(manifest["attribution_analysis"])
    for name, digest in manifest["analysis_inputs_sha256"].items():
        check(digest == file_sha256(analysis / name), f"analysis input unchanged: {name}",
              result, checks)
    for label in LABELS:
        entry = report["representation_probes"]["models"][label]
        run = ROOT / entry["run"]
        digest = file_sha256(run / "checkpoint_best.pth")
        check(digest == manifest["checkpoints"][label] == entry["checkpoint_sha256"],
              f"checkpoint unchanged: {label}", result, checks)
        check(entry["state_unchanged"] is True, f"model state restored after probes: {label}",
              result, checks)
    for name, digest in report["artifact_sha256"].items():
        check(digest == file_sha256(diagnosis_dir / name), f"probe features unchanged: {name}",
              result, checks)

    # 2. Material ordering between the analysis exports and the probe features.
    for label in LABELS:
        for split in ("train", "valid"):
            exported = np.load(analysis / f"{label}_{split}_spectra.npz")["material_ids"]
            probe = np.load(diagnosis_dir / f"probe_features_{label}_{split}.npz")["material_ids"]
            check(bool((exported == probe).all()),
                  f"material order matches exports: {label}_{split}", result, checks)

    # 3. Pair statistics replayed from the raw arrays.
    for grouping in GROUPINGS:
        block = report["response_magnitude_direction"][grouping]["models"]
        for label in LABELS:
            for split in ("train", "valid"):
                pairs = pd.read_csv(analysis / f"{label}_{split}_{grouping}_{TASK}_pairs.csv")
                spectra = np.load(analysis / f"{label}_{split}_spectra.npz")
                amplitude, predicted, error, cosine = pair_rows(spectra, pairs)
                key = f"{label}_{split}"
                assert_close(f"target_tv_median {key} {grouping}", np.median(amplitude),
                             block[key]["target_tv_median"], checks)
                assert_close(f"predicted_tv_median {key} {grouping}", np.median(predicted),
                             block[key]["predicted_tv_median"], checks)
                assert_close(f"difference_error_median {key} {grouping}", np.median(error),
                             block[key]["difference_error_median"], checks)
                assert_close(f"direction_cosine_median {key} {grouping}", np.nanmedian(cosine),
                             block[key]["direction_cosine_median"], checks)
                skill = 1 - error / np.where(amplitude > 1e-10, amplitude, np.nan)
                assert_close(f"skill_median {key} {grouping}", np.nanmedian(skill),
                             block[key]["skill_median"], checks)
                result["max_abs"][f"pair_{grouping}_{key}"] = float(np.max(np.abs(error - pairs.difference_error.to_numpy(float))))

    # 4. Global calibration scale and its locked evaluation.
    for grouping in GROUPINGS:
        calibration = report["response_magnitude_direction"][grouping]["calibration"]
        for label in LABELS:
            train_pairs = pd.read_csv(analysis / f"{label}_train_{grouping}_{TASK}_pairs.csv")
            valid_pairs = pd.read_csv(analysis / f"{label}_valid_{grouping}_{TASK}_pairs.csv")
            train_spectra = np.load(analysis / f"{label}_train_spectra.npz")
            valid_spectra = np.load(analysis / f"{label}_valid_spectra.npz")
            pred_tr, target_tr = delta_vectors(train_spectra, train_pairs)
            scale = fit_global_scale(pred_tr, target_tr)
            assert_close(f"train_global_scale {label} {grouping}", scale,
                         calibration[label]["train_global_scale"], checks, tol=1e-12)
            for split, frame, spectra in (("train", train_pairs, train_spectra),
                                          ("valid", valid_pairs, valid_spectra)):
                delta_p, delta_t = delta_vectors(spectra, frame)
                err = .5 * np.abs(scale * delta_p - delta_t).sum(-1)
                raw = frame.difference_error.to_numpy(float)
                expected = grouped_mean_frame(frame.assign(delta=raw - err), "delta")
                saved = calibration[label]["eval"][split]["raw_minus_calibrated_group_mean"]
                assert_close(f"raw_minus_calibrated {label} {split} {grouping}",
                             expected["mean_group"], saved["mean_group"], checks)
                for lo, hi in zip(expected["ci95"], saved["ci95"]):
                    assert_close(f"raw_minus_calibrated_ci {label} {split} {grouping}", lo, hi, checks)
            oracle = calibration[label]["valid_oracle_scale"]
            pred_va, target_va = delta_vectors(valid_spectra, valid_pairs)
            assert_close(f"valid_oracle_scale {label} {grouping}",
                         fit_global_scale(pred_va, target_va), oracle["scale"], checks, tol=1e-12)

    # 5. Frozen-probe replay from the saved features.
    for grouping in GROUPINGS:
        for label in LABELS:
            entry = report["representation_probes"]["models"][label]["groupings"][grouping]
            saved_features, saved_pairs = {}, {}
            for split in ("train", "valid"):
                archive = np.load(diagnosis_dir / f"probe_features_{label}_{split}.npz")
                saved_features[split] = {stage: archive[stage].astype(np.float64)
                                         for stage in PROBE_STAGES}
                saved_pairs[split] = pd.read_csv(
                    analysis / f"{label}_{split}_{grouping}_{TASK}_pairs.csv")
            spectra = np.load(analysis / f"{label}_train_spectra.npz")
            saved_target = spectra["tgt_shape_edos"]
            for stage in PROBE_STAGES:
                model, alpha, _, info = ridge_probe_cv(
                    saved_features["train"][stage], saved_pairs["train"], saved_target,
                    saved_pairs["train"].composition.to_numpy())
                reported = entry["splits"]["valid"]["stages"][stage]
                check(alpha == reported["train_cv_alpha"],
                      f"probe alpha replay {label} {grouping} {stage}", result, checks)
                check(info["selected_at_grid_edge"] == reported["train_cv_trace"]["selected_at_grid_edge"],
                      f"probe grid-edge flag replay {label} {grouping} {stage}", result, checks)
                for split in ("train", "valid"):
                    shape = np.load(analysis / f"{label}_{split}_spectra.npz")["tgt_shape_edos"]
                    _, err = apply_ridge_probe(model, saved_features[split][stage],
                                               saved_pairs[split], shape)
                    reported = entry["splits"][split]["stages"][stage]
                    assert_close(f"probe locked median {label} {split} {grouping} {stage}",
                                 float(np.median(err)), reported["locked_median_pair_error"], checks)
                    group = grouped_mean_frame(saved_pairs[split].assign(delta=err), "delta")
                    assert_close(f"probe group mean {label} {split} {grouping} {stage}",
                                 group["mean_group"], reported["locked_group_mean_error"]["mean_group"], checks)

    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verified": str(output), "checks": len(checks), "all_passed": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.diagnosis_dir, args.out)
