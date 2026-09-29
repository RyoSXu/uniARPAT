"""Paired Q1 valid-only blind/oracle verdict for the periodic many-body encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets.dataset import Dos_Dataset
from model.transformer import Transformer
from utils.metrics import per_sample_spectral_metrics


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/ablation_m1_e9ctl"
CANDIDATE = ROOT / "output/ablation_m1_pmb1"
RESULTS = ROOT / "results"


def _load_model(directory, expected_epoch, candidate, device):
    with (directory / "config_used.yaml").open() as stream:
        config = yaml.safe_load(stream)
    cli = config["cli"]
    expected_recipe = {
        "model": "M1", "epochs": 35, "batch_size": 32, "lr": 5e-5,
        "seed": 42, "norm": "sumnorm", "scale_mode": "eta",
    }
    for key, expected in expected_recipe.items():
        if cli.get(key) != expected:
            raise ValueError(f"unexpected {key} in checkpoint recipe")
    if candidate and (
        not cli.get("skip_test_eval") or cli.get("use_amp")
        or cli.get("init_ckpt")):
        raise ValueError("candidate run departed from valid-only FP32 scratch recipe")
    params = config["config"]["model"]["params"]["sub_model"]["transformer"]
    if bool(params.get("use_periodic_manybody", False)) != candidate:
        raise ValueError("checkpoint config does not match verdict arm")
    checkpoint = torch.load(directory / "checkpoint_best.pth", map_location="cpu", weights_only=True)
    if (checkpoint.get("model_name"), checkpoint.get("seed")) != ("M1", 42):
        raise ValueError("unexpected checkpoint model or seed")
    if expected_epoch is not None and checkpoint.get("epoch") != expected_epoch:
        raise ValueError("unexpected checkpoint epoch")
    if candidate and not 1 <= checkpoint.get("epoch", 0) <= 35:
        raise ValueError("candidate checkpoint is outside the 35-epoch run")
    model = Transformer(**params)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, int(checkpoint["epoch"])


def _predict(model, loader, device):
    task_values = {name: [] for name in (
        "r2_edos_oracle", "r2_edos_blind", "r2_phdos_oracle", "r2_phdos_blind")}
    e_shapes = []
    target_shapes = []
    with torch.inference_mode():
        for batch in loader:
            src, pos = batch[0].to(device), batch[1].to(device)
            e_target, p_target = batch[2].to(device), batch[3].to(device)
            e_min, e_max = batch[6].to(device).reshape(-1, 1), batch[7].to(device).reshape(-1, 1)
            p_min, p_max = batch[10].to(device).reshape(-1, 1), batch[11].to(device).reshape(-1, 1)
            nval = batch[14].to(device).reshape(-1, 1)
            out = model(src, src.eq(0), pos, batch[15].to(device), batch[16].to(device))
            e_shape = F.softmax(out["edos"], dim=-1)
            p_shape = F.softmax(out["phdos"], dim=-1)
            gamma = out["eta"][:, 1:2]
            eta = out["eta"][:, 0:1]
            natoms = src[:, 2:].ne(0).sum(dim=1, keepdim=True)
            true_e = e_target * (e_max - e_min) + e_min
            true_p = p_target * (p_max - p_min) + p_min
            preds = {
                "r2_edos_oracle": (e_shape * (e_max - e_min) + e_min).clamp_min(0),
                "r2_edos_blind": e_shape * (nval * gamma / 0.09375),
                "r2_phdos_oracle": (p_shape * (p_max - p_min) + p_min).clamp_min(0),
                "r2_phdos_blind": p_shape * (3.0 * natoms * eta / 19.6875),
            }
            for name, pred in preds.items():
                true = true_e if "edos" in name else true_p
                r2 = per_sample_spectral_metrics(pred, true)["r2"]
                if not torch.isfinite(r2).all():
                    raise FloatingPointError(f"nonfinite {name}")
                task_values[name].append(r2.cpu().numpy())
            e_shapes.append(e_shape.cpu().numpy())
            target_shapes.append(e_target.cpu().numpy())
    return ({name: np.concatenate(chunks) for name, chunks in task_values.items()},
            np.concatenate(e_shapes), np.concatenate(target_shapes))


def _comparison(base, candidate, rng):
    n = len(base)
    samples = rng.integers(0, n, size=(2000, n))
    delta_boot = np.median(candidate[samples], axis=1) - np.median(base[samples], axis=1)
    fail_boot = 100 * (
        np.mean(candidate[samples] < 0, axis=1)
        - np.mean(base[samples] < 0, axis=1))
    return {
        "b7_median": float(np.median(base)),
        "candidate_median": float(np.median(candidate)),
        "delta_median": float(np.median(candidate) - np.median(base)),
        "delta_median_ci95": np.quantile(delta_boot, [0.025, 0.975]).tolist(),
        "b7_fail_percent": float(100 * np.mean(base < 0)),
        "candidate_fail_percent": float(100 * np.mean(candidate < 0)),
        "delta_fail_pp": float(100 * (np.mean(candidate < 0) - np.mean(base < 0))),
        "delta_fail_pp_ci95": np.quantile(fail_boot, [0.025, 0.975]).tolist(),
    }


def _pair_readout(base_shape, candidate_shape, target_shape, ids):
    pairs = pd.read_csv(RESULTS / "edos_spectral_support_q1_valid_pairs.csv")
    rows = []
    for row in pairs.itertuples(index=False):
        a, b = int(row.sample_index_a), int(row.sample_index_b)
        if (ids[a], ids[b]) != (row.mpid_a, row.mpid_b):
            raise ValueError("composition pair indices do not match Q1 valid")
        target_delta = target_shape[a] - target_shape[b]
        target_tv = 0.5 * float(np.abs(target_delta).sum())
        if abs(target_tv - float(row.oracle_target_tv)) > 2e-5:
            raise ValueError("composition pair target TV mismatch")
        base_delta = base_shape[a] - base_shape[b]
        candidate_delta = candidate_shape[a] - candidate_shape[b]
        base_tv = 0.5 * float(np.abs(base_delta).sum())
        if abs(base_tv - float(row.b7_predicted_tv)) > 2e-4:
            raise ValueError("composition pair B7 predicted TV mismatch")
        rows.append({
            "mpid_a": row.mpid_a, "mpid_b": row.mpid_b,
            "target_tv": target_tv,
            "b7_predicted_tv": base_tv,
            "candidate_predicted_tv": 0.5 * float(np.abs(candidate_delta).sum()),
            "b7_contrast_error_tv": 0.5 * float(np.abs(base_delta - target_delta).sum()),
            "candidate_contrast_error_tv": 0.5 * float(np.abs(candidate_delta - target_delta).sum()),
        })
    return pd.DataFrame(rows)


def main():
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dos_Dataset(data_dir=str(ROOT / "data/train4ARPAT"), split="valid",
                          dos_minmax=True, dos_sumnorm=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    ids = np.load(ROOT / "data/train4ARPAT/valid/valid_index.npy")
    if len(dataset) != 2313 or len(ids) != len(dataset):
        raise ValueError("Q1 valid sample count mismatch")

    base_model, base_epoch = _load_model(BASE, 33, False, device)
    base, base_shape, target_shape = _predict(base_model, loader, device)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    historical = pd.read_csv(RESULTS / "edos_error_attribution_q1_valid_samples.csv")
    if not np.array_equal(historical.mpid.to_numpy(), ids):
        raise ValueError("frozen B7 valid sample order mismatch")
    for name, column in (("r2_edos_oracle", "r2_edos_oracle_unmasked"),
                         ("r2_edos_blind", "r2_edos_blind_unmasked")):
        max_error = np.max(np.abs(base[name] - historical[column].to_numpy()))
        if max_error > 2e-5:
            raise ValueError(f"frozen B7 {name} mismatch: {max_error}")

    candidate_model, candidate_epoch = _load_model(CANDIDATE, None, True, device)
    candidate, candidate_shape, target_again = _predict(candidate_model, loader, device)
    if not np.array_equal(target_shape, target_again):
        raise ValueError("target arrays changed between paired runs")

    comparisons = {
        name: _comparison(base[name], candidate[name], np.random.default_rng(20260929))
        for name in base}
    e_oracle = comparisons["r2_edos_oracle"]
    e_blind = comparisons["r2_edos_blind"]
    ph_safe = all(
        comparisons[name]["delta_median"] > -0.02
        and comparisons[name]["delta_fail_pp"] < 1.0
        for name in ("r2_phdos_oracle", "r2_phdos_blind"))
    science_pass = e_oracle["delta_median"] >= 0.02 and e_oracle["delta_fail_pp"] < 1.0
    deployment_pass = e_blind["delta_median"] >= 0.02 and e_blind["delta_fail_pp"] < 1.0
    pairs = _pair_readout(base_shape, candidate_shape, target_shape, ids)
    pair_summary = {
        name: float(pairs[name].median()) for name in (
            "target_tv", "b7_predicted_tv", "candidate_predicted_tv",
            "b7_contrast_error_tv", "candidate_contrast_error_tv")}
    verdict = {
        "split": "Q1 valid", "n": len(dataset), "model": "M1", "seed": 42,
        "b7_epoch": base_epoch, "candidate_epoch": candidate_epoch,
        "comparisons": comparisons, "composition_pairs": len(pairs),
        "pair_medians": pair_summary,
        "science_pass": science_pass, "deployment_pass": deployment_pass,
        "phdos_protected": ph_safe,
        "adoptable_after_reproduction": science_pass and deployment_pass and ph_safe,
        "test_used": False,
    }
    sample_frame = pd.DataFrame({"mpid": ids})
    for name in base:
        sample_frame[f"b7_{name}"] = base[name]
        sample_frame[f"candidate_{name}"] = candidate[name]
        sample_frame[f"delta_{name}"] = candidate[name] - base[name]
    sample_frame.to_csv(RESULTS / "periodic_manybody_q1_valid_samples.csv", index=False)
    pairs.to_csv(RESULTS / "periodic_manybody_q1_valid_pairs.csv", index=False)
    (RESULTS / "periodic_manybody_q1_valid_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
