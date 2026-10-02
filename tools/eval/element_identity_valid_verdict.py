"""Element-identity control experiment: Q1 valid-only paired verdict.

对应设计：``docs/design/design-element-identity-control.md`` 第 6 节。
复用 ``tools/eval/periodic_manybody_valid_verdict.py`` 的逐样本 oracle/blind
计算与配对 bootstrap 逻辑，但去掉其中固定的 B7 路径、epoch 33、
``use_periodic_manybody`` 校验和「A 臂预测须等于历史 B7」断言：本工具判读的
A 臂是重新训练的 100 轮对照，不是冻结 B7 checkpoint。

判读前逐臂核对 manifest（模式、表指纹、常数值、配置哈希、checkpoint SHA-256）、
``config_used.yaml`` 配方与 checkpoint 身份，核对通过才加载权重。
旧 B7 只按相同 valid 口径作历史对照，单独加载、单独标注。
test 不参与本工具的任何计算。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset import Dos_Dataset  # noqa: E402
from model.transformer import Transformer  # noqa: E402
from utils.atom_feature import PeriodicTable, legacy3_const_vector  # noqa: E402
from utils.metrics import per_sample_spectral_metrics  # noqa: E402

RESULTS = ROOT / "results"
DATA_DIR = ROOT / "data/train4ARPAT"
TABLE_CSV = ROOT / "utils/periodic_table_v2.csv"
PAIRS_CSV = RESULTS / "edos_spectral_support_q1_valid_pairs.csv"
B7_SAMPLES_CSV = RESULTS / "edos_error_attribution_q1_valid_samples.csv"
B7_DIR = ROOT / "output/ablation_m1_e9ctl"

METRICS = ("r2_edos_oracle", "r2_edos_blind", "r2_phdos_oracle", "r2_phdos_blind")
DELTA_EDOS = 0.09375
DELTA_PHDOS = 19.6875
BOOTSTRAP_REPLICATES = 2000
RNG_SEED_AB = 20260930      # A100 vs B100
RNG_SEED_AB7 = 20260931     # A100 vs 历史 B7
RARE_COUNT_THRESHOLD = 500
RARE_EXPECTED = (239, 2074)  # 文档第 6 节固定的元素频次分层样本数


# ---------------------------------------------------------------------------
# 载入与身份核对
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm(directory: Path, manifest_path: Path, atom_feat_mode: str,
             checkpoint_name: str, device) -> tuple[Transformer, dict]:
    """核对 manifest/配方/checkpoint 身份后载入一臂；任何不匹配都拒绝载入。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("atom_feat_mode") != atom_feat_mode:
        raise ValueError(f"manifest mode {manifest.get('atom_feat_mode')!r} != {atom_feat_mode!r}")
    if manifest.get("hashes", {}).get("table_sha256") != sha256_file(TABLE_CSV):
        raise ValueError("periodic table SHA-256 changed since preflight")

    config_path = directory / "config_used.yaml"
    if manifest.get("config_used_sha256") != sha256_file(config_path):
        raise ValueError("config_used.yaml does not match the manifest record")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    cli = config["cli"]
    expected_recipe = {
        "model": "M1", "epochs": 100, "batch_size": 32, "lr": 5e-5, "seed": 42,
        "norm": "sumnorm", "scale_mode": "eta", "atom_feat": atom_feat_mode,
        "skip_test_eval": True, "init_ckpt": "", "pair_aux_arm": "none",
    }
    bad = {k: (cli.get(k), v) for k, v in expected_recipe.items() if cli.get(k) != v}
    if bad:
        raise ValueError(f"unexpected run recipe: {bad}")
    if cli.get("use_amp"):
        raise ValueError("run used AMP; the frozen recipe is FP32")
    params = config["config"]["model"]["params"]["sub_model"]["transformer"]
    if params.get("atom_feat_mode") != atom_feat_mode:
        raise ValueError("effective atom_feat_mode mismatch")

    ckpt_path = directory / checkpoint_name
    record = manifest.get("checkpoints", {}).get(checkpoint_name, {})
    if not record.get("sha256"):
        raise ValueError(f"manifest has no recorded SHA-256 for {checkpoint_name}; finalize first")
    if record["sha256"] != sha256_file(ckpt_path):
        raise ValueError(f"{checkpoint_name} SHA-256 mismatch against manifest")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    identity = (checkpoint.get("model_name"), checkpoint.get("seed"))
    if identity != ("M1", 42):
        raise ValueError(f"unexpected checkpoint identity: {identity}")
    epoch = int(checkpoint.get("epoch", 0))
    if not 1 <= epoch <= 100 or epoch != int(record.get("epoch", -1)):
        raise ValueError(f"unexpected checkpoint epoch: {epoch}")

    model = Transformer(**params)
    model.load_state_dict(checkpoint["model"], strict=True)
    # feature_map 不在 state_dict 中，须独立核对数值分支的表指纹与常数。
    feature_map = model.num_emb_encoder.feature_map
    table = PeriodicTable().atom_feature_map()
    if atom_feat_mode == "legacy3":
        if not torch.equal(feature_map, table):
            raise ValueError("A-arm feature_map does not equal the current property table")
    else:
        c = legacy3_const_vector()
        if not torch.equal(feature_map[0], table[0]) or not torch.equal(
                feature_map[1:], c.expand_as(feature_map[1:])):
            raise ValueError("B-arm feature_map does not match the preregistered constant table")
        if not np.allclose(c.numpy(), manifest["fingerprint"]["const_vector_float32"], atol=0):
            raise ValueError("B-arm constant vector differs from the manifest record")
    model.to(device).eval()
    return model, {"epoch": epoch, "checkpoint": checkpoint_name}


# ---------------------------------------------------------------------------
# 逐样本 oracle / blind 预测（与冻结工具同式）
# ---------------------------------------------------------------------------

def predict(model, loader, device) -> tuple[dict, dict]:
    task_values = {name: [] for name in METRICS}
    extras = {name: [] for name in (
        "gamma_pred", "gamma_true", "eta_pred", "eta_true")}
    e_shapes, target_shapes = [], []
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
                "r2_edos_blind": e_shape * (nval * gamma / DELTA_EDOS),
                "r2_phdos_oracle": (p_shape * (p_max - p_min) + p_min).clamp_min(0),
                "r2_phdos_blind": p_shape * (3.0 * natoms * eta / DELTA_PHDOS),
            }
            for name, pred in preds.items():
                true = true_e if "edos" in name else true_p
                r2 = per_sample_spectral_metrics(pred, true)["r2"]
                if not torch.isfinite(r2).all():
                    raise FloatingPointError(f"nonfinite {name}")
                task_values[name].append(r2.cpu().numpy())
            # H1 尺度真值口径与 model.py / edos_error_attribution.py 一致。
            extras["gamma_true"].append(
                (e_max.clamp_min(1e-12) * DELTA_EDOS / nval.clamp_min(1e-12)).clamp(0, 1).cpu().numpy())
            extras["gamma_pred"].append(gamma.cpu().numpy())
            extras["eta_true"].append(
                (p_max.clamp_min(1e-12) * DELTA_PHDOS / (3.0 * natoms)).clamp(0, 1).cpu().numpy())
            extras["eta_pred"].append(eta.cpu().numpy())
            e_shapes.append(e_shape.cpu().numpy())
            target_shapes.append(e_target.cpu().numpy())
    metrics = {name: np.concatenate(chunks) for name, chunks in task_values.items()}
    extras = {name: np.concatenate(chunks).reshape(-1) for name, chunks in extras.items()}
    return metrics, {**extras, "e_shape": np.concatenate(e_shapes),
                     "target_shape": np.concatenate(target_shapes)}


# ---------------------------------------------------------------------------
# 配对 bootstrap、分层与同组成 pair
# ---------------------------------------------------------------------------

def paired_comparison(values_a: np.ndarray, values_b: np.ndarray, rng) -> dict:
    n = len(values_a)
    samples = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    delta_boot = np.median(values_b[samples], axis=1) - np.median(values_a[samples], axis=1)
    fail_boot = 100 * (np.mean(values_b[samples] < 0, axis=1) - np.mean(values_a[samples] < 0, axis=1))
    return {
        "a_median": float(np.median(values_a)),
        "b_median": float(np.median(values_b)),
        "delta_median": float(np.median(values_b) - np.median(values_a)),
        "delta_median_ci95": np.quantile(delta_boot, [0.025, 0.975]).tolist(),
        "a_fail_percent": float(100 * np.mean(values_a < 0)),
        "b_fail_percent": float(100 * np.mean(values_b < 0)),
        "delta_fail_pp": float(100 * (np.mean(values_b < 0) - np.mean(values_a < 0))),
        "delta_fail_pp_ci95": np.quantile(fail_boot, [0.025, 0.975]).tolist(),
    }


def scale_error_summary(extras: dict) -> dict:
    out = {}
    for name in ("gamma", "eta"):
        pred, true = extras[f"{name}_pred"], extras[f"{name}_true"]
        out[f"{name}_abs_error_median"] = float(np.median(np.abs(pred - true)))
        out[f"{name}_abs_log_ratio_error_median"] = float(np.median(
            np.abs(np.log(np.maximum(pred, 1e-12)) - np.log(np.maximum(true, 1e-12)))))
    return out


def element_frequency_strata() -> np.ndarray:
    """样本所含最稀有元素在 train 原子槽的出现次数；<500 为稀有组。"""
    train_atoms = np.load(DATA_DIR / "train/elements_train.npy")[:, 2:]
    valid_atoms = np.load(DATA_DIR / "valid/elements_valid.npy")[:, 2:]
    counts = np.bincount(train_atoms.ravel(), minlength=119)
    rarest = np.array([counts[row[row > 0]].min() for row in valid_atoms])
    strata = rarest < RARE_COUNT_THRESHOLD
    if (int(strata.sum()), int((~strata).sum())) != RARE_EXPECTED:
        raise ValueError("element-frequency strata no longer match the frozen 239/2074 split")
    return rarest, strata


def strata_readout(strata: np.ndarray, metrics_a: dict, metrics_b: dict) -> list[dict]:
    rows = []
    for group, mask in (("rarest_lt_500", strata), ("other", ~strata)):
        row = {"stratum": group, "n": int(mask.sum())}
        for name in METRICS:
            a, b = metrics_a[name][mask], metrics_b[name][mask]
            row[f"{name}_a_median"] = float(np.median(a))
            row[f"{name}_b_median"] = float(np.median(b))
            row[f"{name}_delta_median"] = float(np.median(b) - np.median(a))
        rows.append(row)
    return rows


def pair_readout(shapes: dict, target_shape: np.ndarray, ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    pairs = pd.read_csv(PAIRS_CSV)
    rows = []
    for row in pairs.itertuples(index=False):
        a, b = int(row.sample_index_a), int(row.sample_index_b)
        if (ids[a], ids[b]) != (row.mpid_a, row.mpid_b):
            raise ValueError("composition pair indices do not match Q1 valid")
        target_delta = target_shape[a] - target_shape[b]
        target_tv = 0.5 * float(np.abs(target_delta).sum())
        if abs(target_tv - float(row.oracle_target_tv)) > 2e-5:
            raise ValueError("composition pair target TV mismatch")
        record = {"mpid_a": row.mpid_a, "mpid_b": row.mpid_b, "target_tv": target_tv}
        for arm, shape in shapes.items():
            delta = shape[a] - shape[b]
            record[f"{arm}_predicted_tv"] = 0.5 * float(np.abs(delta).sum())
            record[f"{arm}_contrast_error_tv"] = 0.5 * float(np.abs(delta - target_delta).sum())
        rows.append(record)
    frame = pd.DataFrame(rows)
    summary = {
        "pairs": len(frame),
        "target_tv_median": float(frame["target_tv"].median()),
        **{f"{arm}_predicted_tv_median": float(frame[f"{arm}_predicted_tv"].median())
           for arm in shapes},
        **{f"{arm}_contrast_error_tv_median": float(frame[f"{arm}_contrast_error_tv"].median())
           for arm in shapes},
    }
    return frame, summary


# ---------------------------------------------------------------------------
# 历史 B7 对照（独立标注；训练长度与学习率调度均不同于 A100）
# ---------------------------------------------------------------------------

def load_b7(device) -> tuple[Transformer, dict]:
    with (B7_DIR / "config_used.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    cli = config["cli"]
    expected = {"model": "M1", "epochs": 35, "batch_size": 32, "lr": 5e-5,
                "seed": 42, "norm": "sumnorm", "scale_mode": "eta", "atom_feat": "legacy3"}
    bad = {k: (cli.get(k), v) for k, v in expected.items() if cli.get(k) != v}
    if bad:
        raise ValueError(f"unexpected historical B7 recipe: {bad}")
    checkpoint = torch.load(B7_DIR / "checkpoint_best.pth", map_location="cpu", weights_only=True)
    identity = (checkpoint.get("epoch"), checkpoint.get("model_name"), checkpoint.get("seed"))
    if identity != (33, "M1", 42):
        raise ValueError(f"unexpected historical B7 checkpoint identity: {identity}")
    params = config["config"]["model"]["params"]["sub_model"]["transformer"]
    model = Transformer(**params)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, {"epoch": 33, "checkpoint": "checkpoint_best.pth",
                   "epochs_trained": 35, "lr_schedule": "warmup+cosine over 35 epochs"}


def crosscheck_b7(metrics: dict) -> dict:
    historical = pd.read_csv(B7_SAMPLES_CSV)
    ids = np.load(DATA_DIR / "valid/valid_index.npy")
    if not np.array_equal(historical.mpid.to_numpy(), ids):
        raise ValueError("historical B7 valid sample order mismatch")
    max_error = {}
    for name, column in (("r2_edos_oracle", "r2_edos_oracle_unmasked"),
                         ("r2_edos_blind", "r2_edos_blind_unmasked")):
        err = float(np.max(np.abs(metrics[name] - historical[column].to_numpy())))
        if err > 2e-5:
            raise ValueError(f"historical B7 {name} mismatch: {err}")
        max_error[name] = err
    return max_error


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def arm_record(info: dict, history_path: Path, metrics: dict, extras: dict) -> dict:
    history = pd.read_csv(history_path)
    best_row = history.loc[history["balanced_score"].idxmin()]
    last_row = history.loc[history["epoch"].idxmax()]
    return {
        "checkpoint": info,
        "selection": {
            "rule": "min valid balanced_score = 0.5*MAE_edos_median + 0.5*MAE_phdos_median",
            "best_epoch": int(best_row["epoch"]),
            "best_balanced_score": float(best_row["balanced_score"]),
        },
        "epoch100_runner_valid": {
            "r2_edos_median": float(last_row["r2_edos_median"]),
            "r2_phdos_median": float(last_row["r2_phdos_median"]),
            "mae_edos_median": float(last_row["mae_edos_median"]),
            "mae_phdos_median": float(last_row["mae_phdos_median"]),
            "balanced_score": float(last_row["balanced_score"]),
            "train_loss": float(last_row["train_loss"]),
            "epoch_time_s": float(last_row["epoch_time_s"]),
            "peak_vram_mb": float(last_row["peak_vram_mb"]),
        },
        "per_sample": {name: float(np.median(metrics[name])) for name in METRICS},
        "fail_percent": {name: float(100 * np.mean(metrics[name] < 0)) for name in METRICS},
        "scale_error": scale_error_summary(extras),
        "history_csv": str(history_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="_eidprop100_s42", help="A-arm (property) tag")
    parser.add_argument("--tag_const", default="_eidconst100_s42", help="B-arm (constant) tag")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dos_Dataset(data_dir=str(DATA_DIR), split="valid", dos_minmax=True, dos_sumnorm=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    ids = np.load(DATA_DIR / "valid/valid_index.npy")
    if len(dataset) != 2313 or len(ids) != len(dataset):
        raise ValueError("Q1 valid sample count mismatch")

    dir_a = ROOT / f"output/ablation_m1{args.tag}"
    dir_b = ROOT / f"output/ablation_m1{args.tag_const}"
    man_a = RESULTS / f"{args.tag.lstrip('_')}_manifest.json"
    man_b = RESULTS / f"{args.tag_const.lstrip('_')}_manifest.json"

    # 两臂 best（选点结果）与 latest（第 100 轮）都评估。
    outputs = {}
    for arm, directory, manifest, mode in (
            ("A100", dir_a, man_a, "legacy3"), ("B100", dir_b, man_b, "legacy3_const")):
        for which, checkpoint_name in (("best", "checkpoint_best.pth"), ("epoch100", "checkpoint_latest.pth")):
            model, info = load_arm(directory, manifest, mode, checkpoint_name, device)
            metrics, extras = predict(model, loader, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            outputs[(arm, which)] = (info, metrics, extras)

    for _, (_, _, extras) in outputs.items():
        if not np.array_equal(extras["target_shape"], outputs[("A100", "best")][2]["target_shape"]):
            raise ValueError("target arrays changed between paired runs")

    comparisons = {}
    for which in ("best", "epoch100"):
        rng = np.random.default_rng(RNG_SEED_AB)
        comparisons[which] = {
            name: paired_comparison(outputs[("A100", which)][1][name],
                                    outputs[("B100", which)][1][name], rng)
            for name in METRICS}

    # 历史 B7 同口径对照（只评 best，epoch 33）。
    b7_model, b7_info = load_b7(device)
    b7_metrics, b7_extras = predict(b7_model, loader, device)
    del b7_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    crosscheck = crosscheck_b7(b7_metrics)
    rng_b7 = np.random.default_rng(RNG_SEED_AB7)
    comparison_b7 = {
        name: paired_comparison(b7_metrics[name], outputs[("A100", "best")][1][name], rng_b7)
        for name in METRICS}

    rarest, strata = element_frequency_strata()
    strata_rows = strata_readout(strata, outputs[("A100", "best")][1], outputs[("B100", "best")][1])
    shapes = {"A100": outputs[("A100", "best")][2]["e_shape"],
              "B100": outputs[("B100", "best")][2]["e_shape"],
              "B7": b7_extras["e_shape"]}
    pairs_frame, pairs_summary = pair_readout(shapes, outputs[("A100", "best")][2]["target_shape"], ids)

    # 部署保护：B 臂 eDOS blind 与 phDOS oracle/blind 不得越线。
    protected = ("r2_edos_blind", "r2_phdos_oracle", "r2_phdos_blind")
    deployment_flags = {
        name: comparisons["best"][name]["delta_median"] > -0.02
        and comparisons["best"][name]["delta_fail_pp"] < 1.0
        for name in protected}

    verdict = {
        "experiment": "element-identity control (design-element-identity-control.md)",
        "split": "Q1 valid", "n": len(dataset), "seed": args.seed,
        "seeds_run": [args.seed],
        "claim_level": "single-seed preliminary evidence only",
        "test_used": False,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES,
                      "rng_seed_a_vs_b": RNG_SEED_AB, "rng_seed_a_vs_b7": RNG_SEED_AB7},
        "arms": {
            "A100": arm_record(outputs[("A100", "best")][0],
                               RESULTS / f"history_m1{args.tag}.csv",
                               outputs[("A100", "best")][1], outputs[("A100", "best")][2]),
            "B100": arm_record(outputs[("B100", "best")][0],
                               RESULTS / f"history_m1{args.tag_const}.csv",
                               outputs[("B100", "best")][1], outputs[("B100", "best")][2]),
        },
        "epoch100_oracle_blind": {
            arm: {"per_sample": {name: float(np.median(outputs[(arm, "epoch100")][1][name]))
                                 for name in METRICS},
                  "fail_percent": {name: float(100 * np.mean(outputs[(arm, "epoch100")][1][name] < 0))
                                   for name in METRICS}}
            for arm in ("A100", "B100")},
        "a100_vs_b100_best": comparisons["best"],
        "a100_vs_b100_epoch100": comparisons["epoch100"],
        "a100_vs_b100_scale_error": {
            "A100": scale_error_summary(outputs[("A100", "best")][2]),
            "B100": scale_error_summary(outputs[("B100", "best")][2])},
        "a100_vs_historical_b7": {
            "note": "B7 为 35 轮、warmup+cosine 35 轮调度的冻结 checkpoint；A100 为 100 轮、"
                    "100 轮调度从零训练。差异混合了训练长度与学习率调度，不能单独归因于多训练 65 轮。",
            "b7_info": b7_info,
            "crosscheck_max_abs_error": crosscheck,
            "comparisons": comparison_b7,
            "scale_error": {"A100": scale_error_summary(outputs[("A100", "best")][2]),
                            "B7": scale_error_summary(b7_extras)},
        },
        "element_frequency_strata": strata_rows,
        "composition_pairs": pairs_summary,
        "deployment_protection": {
            "rule": "B 臂 eDOS blind 与 phDOS oracle/blind 中位 R² 下降 <0.02 且失败率上升 <1pp",
            "per_metric_pass": deployment_flags,
            "pass": all(deployment_flags.values())},
        "outputs": {
            "samples_csv": "results/eid_s42_valid_samples.csv",
            "pairs_csv": "results/eid_s42_valid_pairs.csv",
            "verdict_json": "results/eid_s42_valid_verdict.json"},
    }

    sample_frame = pd.DataFrame({"mpid": ids, "rarest_element_train_count": rarest,
                                 "rare_stratum": strata})
    for arm in ("A100", "B100"):
        for name in METRICS:
            sample_frame[f"{arm}_{name}"] = outputs[(arm, "best")][1][name]
            sample_frame[f"{arm}_{name}_epoch100"] = outputs[(arm, "epoch100")][1][name]
    for name in METRICS:
        sample_frame[f"delta_best_{name}"] = (sample_frame[f"B100_{name}"]
                                              - sample_frame[f"A100_{name}"])
        sample_frame[f"b7_{name}"] = b7_metrics[name]
    for arm, extras in (("A100", outputs[("A100", "best")][2]), ("B100", outputs[("B100", "best")][2]),
                        ("B7", b7_extras)):
        for key in ("gamma_pred", "gamma_true", "eta_pred", "eta_true"):
            sample_frame[f"{arm}_{key}"] = extras[key]
    sample_frame.to_csv(RESULTS / "eid_s42_valid_samples.csv", index=False)
    pairs_frame.to_csv(RESULTS / "eid_s42_valid_pairs.csv", index=False)
    (RESULTS / "eid_s42_valid_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
