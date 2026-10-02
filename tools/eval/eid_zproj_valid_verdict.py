"""ZP100 (z_only_proj) 对齐初始化候选：Q1 valid-only 判读。

对应设计：``docs/design/design-element-identity-zonlyproj.md`` 第 6 节。
复用 ``tools/eval/element_identity_valid_verdict.py`` 的逐样本 oracle/blind 计算、
配对 bootstrap、尺度误差与同组成 pair 逻辑；ZP100 权重在 manifest、配方、对齐记录与
checkpoint 身份核对通过后加载。A100/B100/Z100 三臂逐样本结果在身份、样本 ID、配置与
指纹核对通过后复用：

* A100/B100：``results/eid_s42_valid_samples.csv``（原始产物）；
* Z100：``results/eid_zonly_s42_valid_samples.csv`` / ``..._pairs.csv``，并与
  ``results/eid_zonly_s42_valid_verdict.json`` 的中位数/失败率交叉核对，同时核对
  Z100 manifest 与 checkpoint SHA-256 未漂移。

不覆盖任何 A100/B100/Z100 产物；所有新产物使用 ``eid_zproj_s42_*`` 独立文件名。
test 不参与本工具的任何计算。
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

P_TAG = "_eidzproj100_s42"
Z_TAG = "_eidzonly100_s42"
A_TAG = "_eidprop100_s42"
B_TAG = "_eidconst100_s42"
REUSED_SAMPLES = RESULTS / "eid_zonly_s42_valid_samples.csv"
REUSED_PAIRS = RESULTS / "eid_zonly_s42_valid_pairs.csv"
REUSED_Z_VERDICT = RESULTS / "eid_zonly_s42_valid_verdict.json"
REUSED_AB_SAMPLES = RESULTS / "eid_s42_valid_samples.csv"

METRICS = base.METRICS
BOOTSTRAP_REPLICATES = 2000
RNG_SEED_PA = 20260940      # ZP100 vs A100
RNG_SEED_PZ = 20260941      # ZP100 vs Z100
RNG_SEED_PB = 20260942      # ZP100 vs B100（辅助）
CURVE_EPOCHS = (25, 35, 50, 75, 100)
EXPECTED_A100_PARAMS = 71_152_964
EXPECTED_Z100_PARAMS = 70_625_092
EXPECTED_ZP100_PARAMS = 70_887_748

# 工程采用标准（单种子临时 baseline）：相对 A100 四项中位 R² 下降均 <0.02、
# 四项失败率上升均 <1 个百分点，且实现/配置/数值检查全部通过。
DROP_TOL = 0.02
FAIL_RISE_TOL_PP = 1.0


def load_zp100(directory: Path, checkpoint_name: str, device):
    """核对 manifest/配方/对齐记录/checkpoint 身份后载入 ZP100；不匹配即拒绝载入。"""
    from model.transformer import Transformer

    manifest = json.loads((RESULTS / "eidzproj100_s42_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("atom_feat_mode") != "z_only_proj":
        raise ValueError("manifest is not a z_only_proj run")
    for key, name in (("table_sha256", "utils/periodic_table_v2.csv"),
                      ("split_sha256", "index/split_v2.yaml"),
                      ("data_manifest_sha256", "data/train4ARPAT/manifest.json")):
        if manifest["hashes"][key] != sha256_file(ROOT / name):
            raise ValueError(f"{name} changed since preflight")
    config_path = directory / "config_used.yaml"
    if manifest.get("config_used_sha256") != sha256_file(config_path):
        raise ValueError("config_used.yaml does not match the manifest record")
    align_path = directory / "zproj_align.json"
    if manifest.get("align_record_sha256") != sha256_file(align_path):
        raise ValueError("zproj_align.json does not match the manifest record")
    align = json.loads(align_path.read_text(encoding="utf-8"))
    if not (align["reference"]["fingerprint_match"] is True
            and all(item["pass"] for item in align["checks"])):
        raise ValueError("in-run alignment record is not a clean Z100-aligned initialization")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    cli = config["cli"]
    expected_recipe = {"model": "M1", "epochs": 100, "batch_size": 32, "lr": 5e-5, "seed": 42,
                       "norm": "sumnorm", "scale_mode": "eta", "atom_feat": "z_only_proj",
                       "skip_test_eval": True, "init_ckpt": "", "pair_aux_arm": "none"}
    bad = {k: (cli.get(k), v) for k, v in expected_recipe.items() if cli.get(k) != v}
    if bad:
        raise ValueError(f"unexpected run recipe: {bad}")
    if cli.get("use_amp"):
        raise ValueError("run used AMP; the frozen recipe is FP32")
    params = config["config"]["model"]["params"]["sub_model"]["transformer"]
    if params.get("atom_feat_mode") != "z_only_proj":
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
    if not (isinstance(model.atom_proj, torch.nn.Linear)
            and model.num_emb_encoder is None and model.num_norm is None
            and model.fuse_proj is None):
        raise ValueError("loaded model is not the z_only_proj architecture")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, {"epoch": epoch, "checkpoint": checkpoint_name,
                   "state_sha256": align["candidate_init"]["state_sha256"]}


def verify_reused_arm(tag: str, atom_feat_mode: str) -> dict:
    """复用前核对 A100/B100/Z100 的 manifest、配置哈希与 checkpoint 哈希。"""
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
    out = {}
    for name in ("checkpoint_best.pth", "checkpoint_latest.pth"):
        ckpt = save_dir / name
        if manifest["checkpoints"][name]["sha256"] != sha256_file(ckpt):
            raise ValueError(f"{tag} {name} hash mismatch")
        out[name] = {"sha256": manifest["checkpoints"][name]["sha256"],
                     "epoch": manifest["checkpoints"][name]["epoch"]}
    return {"tag": tag, "atom_feat_mode": atom_feat_mode,
            "best_epoch": out["checkpoint_best.pth"]["epoch"],
            "config_used_sha256": manifest["config_used_sha256"], "checkpoints": out}


def reuse_per_sample() -> tuple[pd.DataFrame, dict]:
    """载入并核对三臂逐样本结果；Z100 与其 verdict json 交叉核对，A/B 与原始 CSV 对齐。"""
    frame = pd.read_csv(REUSED_SAMPLES)
    ids = np.load(DATA_DIR / "valid/valid_index.npy")
    if not np.array_equal(frame["mpid"].to_numpy(), ids):
        raise ValueError("reused sample IDs do not match Q1 valid order")
    if len(frame) != 2313:
        raise ValueError("reused sample count mismatch")

    # A100/B100 列必须与原始产物逐值一致（Z100 判读 CSV 是其复制品）。
    source = pd.read_csv(REUSED_AB_SAMPLES)
    crosscheck = {}
    for arm in ("A100", "B100"):
        for name in METRICS:
            for suffix in ("", "_epoch100"):
                column = f"{arm}_{name}{suffix}"
                err = float(np.max(np.abs(frame[column].to_numpy()
                                          - source[column].to_numpy())))
                if err > 1e-6:
                    raise ValueError(f"{column} differs from the original A/B sample CSV: {err}")
                crosscheck[column] = err

    # Z100 列与 Z100 判读 json 的中位数/失败率交叉核对。
    reused_verdict = json.loads(REUSED_Z_VERDICT.read_text(encoding="utf-8"))
    for which, section in (("best", reused_verdict["arms"]["Z100"]["per_sample_median_r2"]),
                           ("epoch100", reused_verdict["epoch100_oracle_blind"]["Z100"]["per_sample"])):
        suffix = "" if which == "best" else "_epoch100"
        for name in METRICS:
            column = f"Z100_{name}{suffix}"
            err = abs(float(section[name]) - float(np.median(frame[column].to_numpy())))
            if err > 1e-6:
                raise ValueError(f"reused {column} median mismatch vs verdict json: {err}")
            crosscheck[column] = err
    for which, section in (("best", reused_verdict["arms"]["Z100"]["fail_percent"]),
                           ("epoch100", reused_verdict["epoch100_oracle_blind"]["Z100"]["fail_percent"])):
        suffix = "" if which == "best" else "_epoch100"
        for name in METRICS:
            column = f"Z100_{name}{suffix}"
            err = abs(float(section[name])
                      - float(100 * np.mean(frame[column].to_numpy() < 0)))
            if err > 1e-6:
                raise ValueError(f"reused {column} fail rate mismatch vs verdict json: {err}")
            crosscheck[column] = err
    return frame, {"crosscheck_max_abs_error": max(crosscheck.values()),
                   "samples_csv_sha256": sha256_file(REUSED_SAMPLES),
                   "ab_source_samples_csv_sha256": sha256_file(REUSED_AB_SAMPLES),
                   "z100_verdict_json_sha256": sha256_file(REUSED_Z_VERDICT)}


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
    for arm, tag in (("ZP100", P_TAG), ("Z100", Z_TAG), ("A100", A_TAG), ("B100", B_TAG)):
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


def pair_table(p_shape: np.ndarray, target_shape: np.ndarray, ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """ZP100 的同组成 pair TV 与 Z100/A100/B100 复用列合并。"""
    frame_p, summary_p = base.pair_readout({"ZP100": p_shape}, target_shape, ids)
    reused = pd.read_csv(REUSED_PAIRS)
    if not np.array_equal(reused[["mpid_a", "mpid_b"]].to_numpy(),
                          frame_p[["mpid_a", "mpid_b"]].to_numpy()):
        raise ValueError("reused pair IDs do not match the fixed 320-pair list")
    if not np.allclose(reused["target_tv"].to_numpy(), frame_p["target_tv"].to_numpy(), atol=1e-9):
        raise ValueError("reused pair target TV mismatch")
    merged = frame_p.copy()
    for arm in ("Z100", "A100", "B100"):
        for suffix in ("predicted_tv", "contrast_error_tv"):
            merged[f"{arm}_{suffix}"] = reused[f"{arm}_{suffix}"].to_numpy()
    summary = {
        "pairs": summary_p["pairs"],
        "target_tv_median": summary_p["target_tv_median"],
        "ZP100_predicted_tv_median": summary_p["ZP100_predicted_tv_median"],
        "ZP100_contrast_error_tv_median": summary_p["ZP100_contrast_error_tv_median"],
        **{f"{arm}_predicted_tv_median": float(merged[f"{arm}_predicted_tv"].median())
           for arm in ("Z100", "A100", "B100")},
        **{f"{arm}_contrast_error_tv_median": float(merged[f"{arm}_contrast_error_tv"].median())
           for arm in ("Z100", "A100", "B100")},
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

    manifest = json.loads((RESULTS / "eidzproj100_s42_manifest.json").read_text(encoding="utf-8"))
    dir_p = OUTPUT / f"ablation_m1{P_TAG}"
    p_outputs = {}
    for which, checkpoint_name in (("best", "checkpoint_best.pth"),
                                   ("epoch100", "checkpoint_latest.pth")):
        model, info = load_zp100(dir_p, checkpoint_name, device)
        metrics, extras = base.predict(model, loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        p_outputs[which] = (info, metrics, extras)

    reused_arm_info = {arm: verify_reused_arm(tag, mode) for arm, tag, mode in
                       (("A100", A_TAG, "legacy3"), ("B100", B_TAG, "legacy3_const"),
                        ("Z100", Z_TAG, "z_only"))}
    reused_frame, reused_info = reuse_per_sample()
    if not np.array_equal(p_outputs["best"][2]["target_shape"],
                          p_outputs["epoch100"][2]["target_shape"]):
        raise ValueError("target arrays changed between paired runs")

    def reused_metrics(arm: str, which: str) -> dict:
        suffix = "" if which == "best" else "_epoch100"
        return {name: reused_frame[f"{arm}_{name}{suffix}"].to_numpy() for name in METRICS}

    comparisons = {}
    for which in ("best", "epoch100"):
        comparisons[which] = {
            "zp100_minus_a100": {
                name: base.paired_comparison(reused_metrics("A100", which)[name],
                                             p_outputs[which][1][name],
                                             np.random.default_rng(RNG_SEED_PA))
                for name in METRICS},
            "zp100_minus_z100": {
                name: base.paired_comparison(reused_metrics("Z100", which)[name],
                                             p_outputs[which][1][name],
                                             np.random.default_rng(RNG_SEED_PZ))
                for name in METRICS},
            "zp100_minus_b100": {
                name: base.paired_comparison(reused_metrics("B100", which)[name],
                                             p_outputs[which][1][name],
                                             np.random.default_rng(RNG_SEED_PB))
                for name in METRICS},
        }

    pairs_frame, pairs_summary = pair_table(
        p_outputs["best"][2]["e_shape"], p_outputs["best"][2]["target_shape"], ids)

    p_history = pd.read_csv(RESULTS / f"history_m1{P_TAG}.csv")
    cost = {"total_epoch_time_s": float(p_history["epoch_time_s"].sum()),
            "mean_epoch_time_s": float(p_history["epoch_time_s"].mean()),
            "peak_vram_mb": float(p_history["peak_vram_mb"].max())}

    arms = {
        "ZP100": arm_summary(P_TAG, p_outputs["best"][1], EXPECTED_ZP100_PARAMS, cost),
        "Z100": arm_summary(Z_TAG, reused_metrics("Z100", "best"), EXPECTED_Z100_PARAMS,
                            {"note": "复用 Z100 历史记录，不重训"}),
        "A100": arm_summary(A_TAG, reused_metrics("A100", "best"), EXPECTED_A100_PARAMS,
                            {"note": "复用 A100 历史记录，不重训"}),
        "B100": arm_summary(B_TAG, reused_metrics("B100", "best"), EXPECTED_A100_PARAMS,
                            {"note": "复用 B100 历史记录，不重训"}),
    }
    arms["ZP100"]["checkpoint"] = p_outputs["best"][0]
    arms["ZP100"]["scale_error"] = base.scale_error_summary(p_outputs["best"][2])
    arms["ZP100"]["init_alignment"] = {
        "reference": "Z100 untrained init (z_only, seed 42)",
        "init_state_sha256": manifest["alignment"]["reference"]["init_state_sha256"],
        "training_start_rng_sha256": manifest["alignment"]["reference"][
            "training_start_rng_sha256"],
        "atom_proj_init": "weight = I(512), bias = 0",
    }

    adoption_flags = {}
    for name in METRICS:
        comp = comparisons["best"]["zp100_minus_a100"][name]
        adoption_flags[name] = {
            "delta_median": comp["delta_median"],
            "median_drop_within_0.02": comp["delta_median"] > -DROP_TOL,
            "delta_fail_pp": comp["delta_fail_pp"],
            "fail_rise_within_1pp": comp["delta_fail_pp"] < FAIL_RISE_TOL_PP,
        }
    metric_gate = all(all(v[k] for k in ("median_drop_within_0.02", "fail_rise_within_1pp"))
                      for v in adoption_flags.values())

    z100_flags = {}
    for name in METRICS:
        comp = comparisons["best"]["zp100_minus_z100"][name]
        z100_flags[name] = {
            "delta_median": comp["delta_median"],
            "delta_median_ci95": comp["delta_median_ci95"],
            "delta_fail_pp": comp["delta_fail_pp"],
            "delta_fail_pp_ci95": comp["delta_fail_pp_ci95"],
        }

    verdict = {
        "experiment": "ZP100 z_only + learnable atom projection with Z100-aligned "
                      "initialization (design-element-identity-zonlyproj.md)",
        "split": "Q1 valid", "n": len(dataset), "seed": args.seed,
        "seeds_run": [args.seed],
        "claim_level": "single-seed temporary baseline candidate; conclusions limited to this "
                       "recipe and initialization; not cross-seed stability",
        "test_used": False,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "rng_seed_p_vs_a": RNG_SEED_PA,
                      "rng_seed_p_vs_z": RNG_SEED_PZ, "rng_seed_p_vs_b": RNG_SEED_PB},
        "selection_rule_note": "选点仍为 valid balanced_score = 0.5*MAE_edos_median + "
                               "0.5*MAE_phdos_median；该分数受原始 MAE 尺度影响，本轮记录该限制，未改规则。",
        "arms": arms,
        "epoch100_oracle_blind": {
            arm: {"per_sample": {name: float(np.median(metrics[name])) for name in METRICS},
                  "fail_percent": {name: float(100 * np.mean(metrics[name] < 0))
                                   for name in METRICS}}
            for arm, metrics in (("ZP100", p_outputs["epoch100"][1]),
                                 ("Z100", reused_metrics("Z100", "epoch100")),
                                 ("A100", reused_metrics("A100", "epoch100")),
                                 ("B100", reused_metrics("B100", "epoch100")))
        },
        "comparisons_best": comparisons["best"],
        "comparisons_epoch100": comparisons["epoch100"],
        "z100_effect_summary_best": z100_flags,
        "learning_curves": learning_curves(),
        "composition_pairs": pairs_summary,
        "reused_artifacts": {"arms": reused_arm_info, **reused_info},
        "implementation_checks": {
            "preflight_passed": sum(1 for c in manifest["checks"] if c["pass"]),
            "preflight_total": len(manifest["checks"]),
            "align_record_checks": len(json.loads(
                (dir_p / "zproj_align.json").read_text(encoding="utf-8"))["checks"]),
            "verified_utc": manifest.get("verified_utc"),
            "finalized_utc": manifest.get("finalized_utc"),
        },
        "adoption_criterion": {
            "rule": "相对 A100 四项中位 R² 下降均 <0.02 且四项失败率上升均 <1 个百分点，"
                    "且实现、配置与数值检查全部通过",
            "per_metric": adoption_flags,
            "metric_gate_pass": metric_gate,
        },
        "limitations": [
            "bootstrap 区间只覆盖固定 checkpoint 下的 valid 样本重采样，不含训练 seed 波动与选点不确定性。",
            "只跑了 seed 42；结论限于本次配方与 Z100 对齐初始化，不能声称跨种子稳定。",
            "balanced_score 受原始 MAE 尺度影响；选点规则保持一致，未改用 R²。",
            "对齐只保证共享初值、随机流与数据顺序一致；投影改变了参数量与优化器状态数量。",
            "不检验未见元素泛化；不涉及几何 encoder 路径。",
        ],
        "outputs": {
            "samples_csv": "results/eid_zproj_s42_valid_samples.csv",
            "pairs_csv": "results/eid_zproj_s42_valid_pairs.csv",
            "verdict_json": "results/eid_zproj_s42_valid_verdict.json"},
    }

    sample_frame = pd.DataFrame({"mpid": ids})
    for name in METRICS:
        sample_frame[f"ZP100_{name}"] = p_outputs["best"][1][name]
        sample_frame[f"ZP100_{name}_epoch100"] = p_outputs["epoch100"][1][name]
        for arm in ("Z100", "A100", "B100"):
            sample_frame[f"{arm}_{name}"] = reused_metrics(arm, "best")[name]
            sample_frame[f"{arm}_{name}_epoch100"] = reused_metrics(arm, "epoch100")[name]
        sample_frame[f"delta_best_vs_z100_{name}"] = (sample_frame[f"ZP100_{name}"]
                                                      - sample_frame[f"Z100_{name}"])
        sample_frame[f"delta_best_vs_a100_{name}"] = (sample_frame[f"ZP100_{name}"]
                                                      - sample_frame[f"A100_{name}"])
    for key in ("gamma_pred", "gamma_true", "eta_pred", "eta_true"):
        sample_frame[f"ZP100_{key}"] = p_outputs["best"][2][key]
    sample_frame.to_csv(RESULTS / "eid_zproj_s42_valid_samples.csv", index=False)
    pairs_frame.to_csv(RESULTS / "eid_zproj_s42_valid_pairs.csv", index=False)
    (RESULTS / "eid_zproj_s42_valid_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
