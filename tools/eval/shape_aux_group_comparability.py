"""Compare structure and label gaps of train-success groups with all valid groups.

Read-only. Uses the finished shape-auxiliary stage readout and stored input
geometry. No forward pass, no parameter update, and no test arrays.
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
import torch

from tools.eval.lg_structure_diagnosis import geometry_table
from tools.eval.lg_valid_export import load_run
from utils.builder import ConfigBuilder
from utils.paths import repository_path

READOUT = "results/runs/shape_aux_stage_readout_20261027"
RUN = "results/runs/ablation_m1_sbal_shape20261025_shape_a64_s42_e35"
TASKS = ("edos", "phdos")
PRIMARY = "edos"
NEAR_ZERO_TV = 1e-6
MEDIAN_RATIO = 0.8
DRAWS = 2000
SEED = 42
STRUCTURE_METRICS = ("volume_gap", "radial_gap")
LABEL_METRICS = ("target_tv",)
FLAGGED = ("mp-aaaaadad", "mp-aaaaarbn", "mp-aaacioqq", "mp-aaacpjlg", "mp-aaacrkqc")


def relative_gap(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim == 1:
        gap = np.abs(left - right)
        denom = 0.5 * (np.abs(left) + np.abs(right))
    else:
        gap = np.linalg.norm(left - right, axis=1)
        denom = 0.5 * (np.linalg.norm(left, axis=1) + np.linalg.norm(right, axis=1))
    return gap / np.maximum(denom, 1e-12)


def group_table(frame):
    """One row per composition. Skill and label TV ignore near-zero targets."""
    rows = []
    for composition, part in frame.groupby("composition", sort=True):
        main = part[part.target_tv >= NEAR_ZERO_TV]
        skill = float(main.skill.mean()) if len(main) else None
        target = float(main.target_tv.mean()) if len(main) else None
        rows.append({
            "composition": composition,
            "pairs": int(len(part)),
            "main_pairs": int(len(main)),
            "skill": skill,
            "target_tv": target,
            "volume_gap": float(part.volume_gap.mean()),
            "radial_gap": float(part.radial_gap.mean()),
            "flagged_pair": bool(part.flagged.any()),
        })
    return pd.DataFrame(rows)


def mean_difference_interval(valid, train, seed=SEED, draws=DRAWS):
    valid = np.asarray(valid, dtype=np.float64)
    train = np.asarray(train, dtype=np.float64)
    if len(valid) == 0 or len(train) == 0:
        return {"valid_groups": int(len(valid)), "train_groups": int(len(train)),
                "mean_difference": None, "ci95": None}
    rng = np.random.default_rng(seed)
    draws_v = rng.choice(valid, (draws, len(valid)), replace=True).mean(1)
    draws_t = rng.choice(train, (draws, len(train)), replace=True).mean(1)
    diff = draws_v - draws_t
    return {"valid_groups": int(len(valid)), "train_groups": int(len(train)),
            "valid_mean": float(valid.mean()), "train_mean": float(train.mean()),
            "valid_median": float(np.median(valid)), "train_median": float(np.median(train)),
            "mean_difference": float(valid.mean() - train.mean()),
            "ci95": np.percentile(diff, [2.5, 97.5]).tolist()}


def clearly_smaller(valid, train, interval, ratio=MEDIAN_RATIO):
    """Valid is a smaller population only when both the interval and the median say so."""
    ci = interval.get("ci95")
    if ci is None or ci[1] >= 0:
        return False
    train_median = float(np.median(train))
    if not np.isfinite(train_median) or train_median <= 0:
        return False
    return float(np.median(valid)) <= ratio * train_median


def classify(metric_flags, train_success_groups):
    if train_success_groups < 2:
        return "unresolved"
    if any(metric_flags.values()):
        return "data_difference"
    return "same_class"


def _lookup_vectors(geometry):
    volume = dict(zip(geometry.material_id, geometry.volume_per_atom.astype(float)))
    rbf_columns = [f"rbf{i}" for i in range(16)]
    values = geometry[rbf_columns].to_numpy(float)
    radial = {mid: values[i] for i, mid in enumerate(geometry.material_id)}
    return volume, radial


def attach_gaps(frame, volume, radial):
    left_v = frame.left_id.map(volume)
    right_v = frame.right_id.map(volume)
    if left_v.isna().any() or right_v.isna().any():
        raise ValueError("pair material is missing input volume")
    left_r = np.stack([radial[mid] for mid in frame.left_id])
    right_r = np.stack([radial[mid] for mid in frame.right_id])
    out = frame.copy()
    out["volume_gap"] = relative_gap(left_v.to_numpy(), right_v.to_numpy())
    out["radial_gap"] = relative_gap(left_r, right_r)
    out["flagged"] = frame.left_id.isin(FLAGGED) | frame.right_id.isin(FLAGGED)
    return out


def _paired_ids(readout):
    ids = set()
    for task in TASKS:
        for split in ("train", "valid"):
            frame = pd.read_csv(readout / f"pairs_{task}_{split}_model.csv")
            ids.update(frame.left_id)
            ids.update(frame.right_id)
    return ids


def geometry_for_pairs(saved, needed_ids, device):
    builder = ConfigBuilder(**saved["config"])
    frames = []
    for split in ("train", "valid"):
        dataset = builder.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
        rows = [row for row, mid in enumerate(dataset.material_ids) if str(mid) in needed_ids]
        if not rows:
            continue
        frames.append(geometry_table(dataset, rows, device))
    geometry = pd.concat(frames, ignore_index=True)
    if geometry.material_id.duplicated().any():
        raise ValueError("material ids collide across train and valid")
    missing = needed_ids.difference(geometry.material_id)
    if missing:
        raise ValueError(f"paired materials missing from stored geometry: {len(missing)}")
    return geometry


def evaluate(readout, geometry):
    volume, radial = _lookup_vectors(geometry)
    report = {"primary_task": PRIMARY, "median_ratio": MEDIAN_RATIO, "tasks": {}}
    for task in TASKS:
        task_report = {"splits": {}}
        grouped = {}
        for split in ("train", "valid"):
            frame = pd.read_csv(readout / f"pairs_{task}_{split}_model.csv")
            frame = attach_gaps(frame, volume, radial)
            groups = group_table(frame)
            grouped[split] = groups
            finite = groups[groups.skill.notna()]
            task_report["splits"][split] = {
                "groups": int(len(groups)),
                "success_groups": int((finite.skill > 0).sum()),
                "success_fraction": float((finite.skill > 0).mean()) if len(finite) else None,
                "flagged_groups": int(groups.flagged_pair.sum()),
            }
        success = grouped["train"]
        success = success[success.skill.notna() & (success.skill > 0)]
        valid = grouped["valid"]
        flags = {}
        metrics = {}
        for name in (*STRUCTURE_METRICS, *LABEL_METRICS):
            valid_values = valid[name].to_numpy(float)
            train_values = success[name].dropna().to_numpy(float)
            if name == "target_tv":
                valid_values = valid.loc[valid.target_tv.notna(), name].to_numpy(float)
            interval = mean_difference_interval(valid_values, train_values)
            flag = clearly_smaller(valid_values, train_values, interval)
            flags[name] = flag
            metrics[name] = {**interval, "clearly_smaller": flag}
        verdict = classify(flags, int(len(success)))
        task_report["train_success_groups"] = int(len(success))
        task_report["metrics"] = metrics
        task_report["verdict"] = verdict
        report["tasks"][task] = task_report
    meanings = {
        "same_class": "验证组与训练成功组是同一类结构和标签差；当前配方预测不了验证集谱差，归因结束",
        "data_difference": "验证组的结构差或标签差明显更小；记为数据差异，不改网络，归因结束",
        "unresolved": "训练成功组不足，只保留数字",
    }
    verdict = report["tasks"][PRIMARY]["verdict"]
    report["verdict"] = verdict
    report["meaning"] = meanings[verdict]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--readout-dir", type=Path, default=None)
    args = parser.parse_args()
    if (args.out_dir / "report.json").exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    readout = repository_path(args.readout_dir or READOUT)
    saved, _params, contract = load_run(repository_path(RUN))
    if contract.get("schema") != "joint-spectral-balance-shape-model-v1":
        raise ValueError("comparability is locked to the shape-auxiliary checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = geometry_for_pairs(saved, _paired_ids(readout), device)
    geometry.to_csv(args.out_dir / "paired_geometry.csv", index=False)
    report = evaluate(readout, geometry)
    report["device"] = str(device)
    report["readout"] = str(readout.relative_to(ROOT))
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "meaning": report["meaning"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
