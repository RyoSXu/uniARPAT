"""Z100 (z_only) 单种子临时 baseline：Q1 valid-only 判读。

对应设计：``docs/design/design-element-identity-zonly.md`` 第 6 节。
复用 ``tools/eval/element_identity_valid_verdict.py`` 的逐样本 oracle/blind 计算、
配对 bootstrap、尺度误差、元素频次分层与同组成 pair 逻辑；Z100 权重在 manifest、
配方与 checkpoint 身份核对通过后加载。A100/B100 的逐样本结果在身份、样本 ID、
配置与指纹核对通过后复用 ``results/eid_s42_valid_samples.csv`` /
``results/eid_s42_valid_pairs.csv``，不覆盖任何 A100/B100 产物；所有新产物使用
``eid_zonly_s42_*`` 独立文件名。test 不参与本工具的任何计算。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import element_identity_valid_verdict as base  # noqa: E402
from element_identity_preflight import sha256_file  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"
DATA_DIR = ROOT / "data/train4ARPAT"

Z_TAG = "_eidzonly100_s42"
A_TAG = "_eidprop100_s42"
B_TAG = "_eidconst100_s42"
REUSED_SAMPLES = RESULTS / "eid_s42_valid_samples.csv"
REUSED_PAIRS = RESULTS / "eid_s42_valid_pairs.csv"
REUSED_VERDICT = RESULTS / "eid_s42_valid_verdict.json"

METRICS = base.METRICS
BOOTSTRAP_REPLICATES = 2000
RNG_SEED_ZA = 20260933      # Z100 vs A100
RNG_SEED_ZB = 20260934      # Z100 vs B100（辅助）
CURVE_EPOCHS = (25, 35, 50, 75, 100)
EXPECTED_A100_PARAMS = 71_152_964
EXPECTED_Z100_PARAMS = 70_625_092

# 工程采用标准（单种子临时 baseline）：相对 A100 四项中位 R² 下降均 <0.02、
# 四项失败率上升均 <1 个百分点，且实现/配置/数值检查全部通过。
DROP_TOL = 0.02
FAIL_RISE_TOL_PP = 1.0


def load_z100(directory: Path, checkpoint_name: str, device):
    """核对 manifest/配方/checkpoint 身份后载入 Z100；任何不匹配都拒绝载入。"""
    from model.transformer import Transformer

    manifest = json.loads((RESULTS / "eidzonly100_s42_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("atom_feat_mode") != "z_only":
        raise ValueError("manifest is not a z_only run")
    for key, name in (("table_sha256", "utils/periodic_table_v2.csv"),
                      ("split_sha256", "index/split_v2.yaml"),
                      ("data_manifest_sha256", "data/train4ARPAT/manifest.json")):
        if manifest["hashes"][key] != sha256_file(ROOT / name):
            raise ValueError(f"{name} changed since preflight")
    config_path = directory / "config_used.yaml"
    if manifest.get("config_used_sha256") != sha256_file(config_path):
        raise ValueError("config_used.yaml does not match the manifest record")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    cli = config["cli"]
    expected_recipe = {"model": "M1", "epochs": 100, "batch_size": 32, "lr": 5e-5, "seed": 42,
                       "norm": "sumnorm", "scale_mode": "eta", "atom_feat": "z_only",
                       "skip_test_eval": True, "init_ckpt": "", "pair_aux_arm": "none"}
    bad = {k: (cli.get(k), v) for k, v in expected_recipe.items() if cli.get(k) != v}
    if bad:
        raise ValueError(f"unexpected run recipe: {bad}")
    if cli.get("use_amp"):
        raise ValueError("run used AMP; the frozen recipe is FP32")
    params = config["config"]["model"]["params"]["sub_model"]["transformer"]
    if params.get("atom_feat_mode") != "z_only":
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
    if not (model.num_emb_encoder is None and model.num_norm is None and model.fuse_proj is None):
        raise ValueError("loaded model is not the z_only architecture")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, {"epoch": epoch, "checkpoint": checkpoint_name}


def verify_reused_arm(tag: str, atom_feat_mode: str) -> dict:
    """复用前核对 A100/B100 的 manifest、配置哈希与 checkpoint 哈希。"""
    manifest = json.loads(
        (RESULTS / f"{tag.lstrip('_')}_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("atom_feat_mode") != atom_feat_mode:
        raise ValueError(f"{tag} manifest mode mismatch")
    if manifest["hashes"]["table_sha256"] != sha256_file(ROOT / "utils/periodic_table_v2.csv"):
        raise ValueError(f"{tag} property table hash changed")
    if manifest["hashes"]["split_sha256"] != sha256_file(ROOT / "index/split_v2.yaml"):
        raise ValueError(f"{tag} split hash changed")
    save_dir = OUTPUT / f"ablation_m1{tag}"
    if manifest["config_used_sha256"] != sha256_file(save_dir / "config_used.yaml"):
        raise ValueError(f"{tag} config_used.yaml mismatch")
    ckpt = save_dir / "checkpoint_best.pth"
    if manifest["checkpoints"]["checkpoint_best.pth"]["sha256"] != sha256_file(ckpt):
        raise ValueError(f"{tag} checkpoint_best hash mismatch")
    return {"tag": tag, "atom_feat_mode": atom_feat_mode,
            "best_epoch": manifest["checkpoints"]["checkpoint_best.pth"]["epoch"],
            "checkpoint_sha256": manifest["checkpoints"]["checkpoint_best.pth"]["sha256"],
            "config_used_sha256": manifest["config_used_sha256"]}


def reuse_per_sample() -> tuple[pd.DataFrame, dict]:
    """载入并核对 A100/B100 逐样本结果；与 eid_s42_valid_verdict.json 交叉核对。"""
    frame = pd.read_csv(REUSED_SAMPLES)
    ids = np.load(DATA_DIR / "valid/valid_index.npy")
    if not np.array_equal(frame["mpid"].to_numpy(), ids):
        raise ValueError("reused sample IDs do not match Q1 valid order")
    if len(frame) != 2313:
        raise ValueError("reused sample count mismatch")
    reused_verdict = json.loads(REUSED_VERDICT.read_text(encoding="utf-8"))
    crosscheck = {}
    for arm in ("A100", "B100"):
        for name in METRICS:
            column = f"{arm}_{name}"
            recorded = reused_verdict["arms"][arm]["per_sample"][name]
            computed = float(np.median(frame[column].to_numpy()))
            err = abs(recorded - computed)
            # CSV 以 float32 十进制写出、读回 float64，往返误差量级 ~1e-8；1e-6 远小于
            # 任何有意义的指标差异，足够判定口径一致。
            if err > 1e-6:
                raise ValueError(f"reused {column} median mismatch vs verdict json: {err}")
            crosscheck[column] = err
    return frame, {"crosscheck_max_abs_error": max(crosscheck.values()),
                   "samples_csv_sha256": sha256_file(REUSED_SAMPLES),
                   "verdict_json_sha256": sha256_file(REUSED_VERDICT)}


def history_row(path: Path, epoch: int) -> dict:
    history = pd.read_csv(path)
    row = history.loc[history["epoch"] == epoch]
    if row.empty:
        raise ValueError(f"{path.name} has no epoch {epoch}")
    row = row.iloc[0]
    return {"epoch": epoch,
            "balanced_score": float(row["balanced_score"]),
            "r2_edos_median": float(row["r2_edos_median"]),
            "r2_phdos_median": float(row["r2_phdos_median"]),
            "fail_rate_edos": float(row["fail_rate_edos"]),
            "fail_rate_phdos": float(row["fail_rate_phdos"])}


def learning_curves() -> dict:
    out = {}
    for arm, tag in (("Z100", Z_TAG), ("A100", A_TAG), ("B100", B_TAG)):
        path = RESULTS / f"history_m1{tag}.csv"
        out[arm] = [history_row(path, epoch) for epoch in CURVE_EPOCHS]
    return out


def arm_summary(tag: str, metrics: dict, params: int, cost: dict) -> dict:
    history = pd.read_csv(RESULTS / f"history_m1{tag}.csv")
    best_row = history.loc[history["balanced_score"].idxmin()]
    last_row = history.loc[history["epoch"].idxmax()]
    return {
        "best_epoch": int(best_row["epoch"]),
        "best_balanced_score": float(best_row["balanced_score"]),
        "selection_rule": "min valid balanced_score = 0.5*MAE_edos_median + 0.5*MAE_phdos_median",
        "epoch100_runner_valid": {
            "r2_edos_median": float(last_row["r2_edos_median"]),
            "r2_phdos_median": float(last_row["r2_phdos_median"]),
            "balanced_score": float(last_row["balanced_score"]),
            "train_loss": float(last_row["train_loss"]),
        },
        "per_sample_median_r2": {name: float(np.median(metrics[name])) for name in METRICS},
        "fail_percent": {name: float(100 * np.mean(metrics[name] < 0)) for name in METRICS},
        "trainable_params": params,
        "cost": cost,
        "history_csv": f"results/history_m1{tag}.csv",
    }


def pair_table(z_shape: np.ndarray, target_shape: np.ndarray, ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Z100 的同组成 pair TV 与 A100/B100 复用列合并。"""
    frame_z, summary_z = base.pair_readout({"Z100": z_shape}, target_shape, ids)
    reused = pd.read_csv(REUSED_PAIRS)
    if not np.array_equal(reused[["mpid_a", "mpid_b"]].to_numpy(),
                          frame_z[["mpid_a", "mpid_b"]].to_numpy()):
        raise ValueError("reused pair IDs do not match the fixed 320-pair list")
    if not np.allclose(reused["target_tv"].to_numpy(), frame_z["target_tv"].to_numpy(), atol=1e-9):
        raise ValueError("reused pair target TV mismatch")
    merged = frame_z.copy()
    for arm in ("A100", "B100"):
        for suffix in ("predicted_tv", "contrast_error_tv"):
            merged[f"{arm}_{suffix}"] = reused[f"{arm}_{suffix}"].to_numpy()
    summary = {
        "pairs": summary_z["pairs"],
        "target_tv_median": summary_z["target_tv_median"],
        "Z100_predicted_tv_median": summary_z["Z100_predicted_tv_median"],
        "Z100_contrast_error_tv_median": summary_z["Z100_contrast_error_tv_median"],
        "A100_predicted_tv_median": float(merged["A100_predicted_tv"].median()),
        "A100_contrast_error_tv_median": float(merged["A100_contrast_error_tv"].median()),
        "B100_predicted_tv_median": float(merged["B100_predicted_tv"].median()),
        "B100_contrast_error_tv_median": float(merged["B100_contrast_error_tv"].median()),
    }
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torch.utils.data import DataLoader
    from datasets.dataset import Dos_Dataset

    dataset = Dos_Dataset(data_dir=str(DATA_DIR), split="valid", dos_minmax=True, dos_sumnorm=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    ids = np.load(DATA_DIR / "valid/valid_index.npy")
    if len(dataset) != 2313 or len(ids) != len(dataset):
        raise ValueError("Q1 valid sample count mismatch")

    dir_z = OUTPUT / f"ablation_m1{Z_TAG}"
    z_outputs = {}
    for which, checkpoint_name in (("best", "checkpoint_best.pth"),
                                   ("epoch100", "checkpoint_latest.pth")):
        model, info = load_z100(dir_z, checkpoint_name, device)
        metrics, extras = base.predict(model, loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        z_outputs[which] = (info, metrics, extras)

    reused_arm_info = {arm: verify_reused_arm(tag, mode) for arm, tag, mode in
                       (("A100", A_TAG, "legacy3"), ("B100", B_TAG, "legacy3_const"))}
    reused_frame, reused_info = reuse_per_sample()
    if not np.array_equal(z_outputs["best"][2]["target_shape"],
                          z_outputs["epoch100"][2]["target_shape"]):
        raise ValueError("target arrays changed between paired runs")

    def reused_metrics(arm: str, which: str) -> dict:
        suffix = "" if which == "best" else "_epoch100"
        return {name: reused_frame[f"{arm}_{name}{suffix}"].to_numpy() for name in METRICS}

    comparisons = {}
    for which in ("best", "epoch100"):
        rng = np.random.default_rng(RNG_SEED_ZA)
        comparisons[which] = {
            "z100_minus_a100": {
                name: base.paired_comparison(reused_metrics("A100", which)[name],
                                             z_outputs[which][1][name], rng)
                for name in METRICS},
            "z100_minus_b100": {
                name: base.paired_comparison(reused_metrics("B100", which)[name],
                                             z_outputs[which][1][name],
                                             np.random.default_rng(RNG_SEED_ZB))
                for name in METRICS},
        }

    pairs_frame, pairs_summary = pair_table(
        z_outputs["best"][2]["e_shape"], z_outputs["best"][2]["target_shape"], ids)

    z_history = pd.read_csv(RESULTS / f"history_m1{Z_TAG}.csv")
    cost = {"total_epoch_time_s": float(z_history["epoch_time_s"].sum()),
            "mean_epoch_time_s": float(z_history["epoch_time_s"].mean()),
            "peak_vram_mb": float(z_history["peak_vram_mb"].max())}

    arms = {
        "Z100": arm_summary(Z_TAG, z_outputs["best"][1], EXPECTED_Z100_PARAMS, cost),
        "A100": arm_summary(A_TAG, reused_metrics("A100", "best"), EXPECTED_A100_PARAMS,
                            {"note": "复用 A100 历史记录"}),
        "B100": arm_summary(B_TAG, reused_metrics("B100", "best"), EXPECTED_A100_PARAMS,
                            {"note": "复用 B100 历史记录"}),
    }
    arms["Z100"]["checkpoint"] = z_outputs["best"][0]
    arms["Z100"]["scale_error"] = base.scale_error_summary(z_outputs["best"][2])

    adoption_flags = {}
    for name in METRICS:
        comp = comparisons["best"]["z100_minus_a100"][name]
        adoption_flags[name] = {
            "delta_median": comp["delta_median"],
            "median_drop_within_0.02": comp["delta_median"] > -DROP_TOL,
            "delta_fail_pp": comp["delta_fail_pp"],
            "fail_rise_within_1pp": comp["delta_fail_pp"] < FAIL_RISE_TOL_PP,
        }
    metric_gate = all(all(v[k] for k in ("median_drop_within_0.02", "fail_rise_within_1pp"))
                      for v in adoption_flags.values())

    verdict = {
        "experiment": "Z100 z_only simplified element representation "
                      "(design-element-identity-zonly.md)",
        "split": "Q1 valid", "n": len(dataset), "seed": args.seed,
        "seeds_run": [args.seed],
        "claim_level": "single-seed temporary baseline candidate; not strict equivalence",
        "test_used": False,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES,
                      "rng_seed_z_vs_a": RNG_SEED_ZA, "rng_seed_z_vs_b": RNG_SEED_ZB},
        "selection_rule_note": "选点仍为 valid balanced_score = 0.5*MAE_edos_median + "
                               "0.5*MAE_phdos_median；该分数受原始 MAE 尺度影响，本轮记录该限制，未改规则。",
        "arms": arms,
        "epoch100_oracle_blind": {
            "Z100": {"per_sample": {name: float(np.median(z_outputs["epoch100"][1][name]))
                                    for name in METRICS},
                     "fail_percent": {name: float(100 * np.mean(z_outputs["epoch100"][1][name] < 0))
                                      for name in METRICS}},
            "A100": {"per_sample": {name: float(np.median(reused_metrics("A100", "epoch100")[name]))
                                    for name in METRICS},
                     "fail_percent": {name: float(100 * np.mean(reused_metrics("A100", "epoch100")[name] < 0))
                                      for name in METRICS}},
            "B100": {"per_sample": {name: float(np.median(reused_metrics("B100", "epoch100")[name]))
                                    for name in METRICS},
                     "fail_percent": {name: float(100 * np.mean(reused_metrics("B100", "epoch100")[name] < 0))
                                      for name in METRICS}},
        },
        "comparisons_best": comparisons["best"],
        "comparisons_epoch100": comparisons["epoch100"],
        "learning_curves": learning_curves(),
        "composition_pairs": pairs_summary,
        "reused_artifacts": {"arms": reused_arm_info, **reused_info},
        "adoption_criterion": {
            "rule": "相对 A100 四项中位 R² 下降均 <0.02 且四项失败率上升均 <1 个百分点，"
                    "且实现、配置与数值检查全部通过（preflight 36/36、verify-run 8/8）",
            "per_metric": adoption_flags,
            "metric_gate_pass": metric_gate,
        },
        "limitations": [
            "bootstrap 区间只覆盖固定 checkpoint 下的 valid 样本重采样，不含训练 seed 波动与选点不确定性。",
            "本实验只跑了 seed 42；差异含初始化随机流差异（删除模块改变随机数消耗，同 seed 参数初值不同），"
            "不能把差异单独归因于三项性质。",
            "balanced_score 受原始 MAE 尺度影响；选点规则保持一致，未改用 R²。",
            "达标不表示严格等价或稳定非劣；未检验未见元素泛化。",
        ],
        "outputs": {
            "samples_csv": "results/eid_zonly_s42_valid_samples.csv",
            "pairs_csv": "results/eid_zonly_s42_valid_pairs.csv",
            "verdict_json": "results/eid_zonly_s42_valid_verdict.json"},
    }

    sample_frame = pd.DataFrame({"mpid": ids})
    for name in METRICS:
        sample_frame[f"Z100_{name}"] = z_outputs["best"][1][name]
        sample_frame[f"Z100_{name}_epoch100"] = z_outputs["epoch100"][1][name]
        sample_frame[f"A100_{name}"] = reused_metrics("A100", "best")[name]
        sample_frame[f"A100_{name}_epoch100"] = reused_metrics("A100", "epoch100")[name]
        sample_frame[f"B100_{name}"] = reused_metrics("B100", "best")[name]
        sample_frame[f"B100_{name}_epoch100"] = reused_metrics("B100", "epoch100")[name]
        sample_frame[f"delta_best_{name}"] = (sample_frame[f"Z100_{name}"]
                                              - sample_frame[f"A100_{name}"])
    for key in ("gamma_pred", "gamma_true", "eta_pred", "eta_true"):
        sample_frame[f"Z100_{key}"] = z_outputs["best"][2][key]
    sample_frame.to_csv(RESULTS / "eid_zonly_s42_valid_samples.csv", index=False)
    pairs_frame.to_csv(RESULTS / "eid_zonly_s42_valid_pairs.csv", index=False)
    (RESULTS / "eid_zonly_s42_valid_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
