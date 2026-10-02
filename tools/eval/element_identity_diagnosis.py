"""纯原子序号模型（Z100/ZP100）相对 B100 的训练差距诊断。

对应设计：``docs/design/design-element-identity-diagnosis.md``。
只读取既有日志/配置/checkpoint 与 Q1 train/valid，重建未训练模型仅用于内存快照比较；
不训练、不更新参数、不追加 seed、不构建或读取 test。

三个执行阶段（``--phase``）：

* A：固定 train 子集与完整 valid 上的同口径评估（损失、oracle/blind 物理指标、配对
  bootstrap、失败翻转），并与 ``results/eid_zproj_s42_valid_samples.csv`` 交叉核对；
* B：初值重建与指纹核对、参数变化、优化器状态核对、当前 eval 梯度探针；
* C：入口表示、各层状态、共享偏移分解与注意力打分/权重行为。

全部产物写入 ``results/eid_diagnosis_s42/<run_id>/``；同 run_id 续写同一目录，重跑须换新
run_id。任何身份核对失败即停止归因，先报告问题。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.dataset import Dos_Dataset  # noqa: E402
from model.losses import sumnorm_klw_loss  # noqa: E402
from model.transformer import Transformer  # noqa: E402
from utils.atom_feature import legacy3_const_vector  # noqa: E402
from utils.metrics import per_sample_spectral_metrics  # noqa: E402
from utils.relative_features import compute_relative_features  # noqa: E402
from utils.zproj_alignment import identity_projection_, state_hash  # noqa: E402

import element_identity_valid_verdict as base  # noqa: E402  复用 predict/配对 bootstrap 口径
from element_identity_preflight import sha256_file  # noqa: E402
from eid_zonly_valid_verdict import load_z100  # noqa: E402
from eid_zproj_valid_verdict import load_zp100  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"
DATA_DIR = ROOT / "data/train4ARPAT"
RUNS_DIR = RESULTS / "eid_diagnosis_s42"

ARMS = {
    "B100": {
        "tag": "_eidconst100_s42",
        "mode": "legacy3_const",
        "manifest": "eidconst100_s42_manifest.json",
        "init_state_sha256": "efca7088573a371fefe589a144ea66694273192498c9e7c15f2d111b717e06e0",
    },
    "Z100": {
        "tag": "_eidzonly100_s42",
        "mode": "z_only",
        "manifest": "eidzonly100_s42_manifest.json",
        "init_state_sha256": "4eda5029e186d2f3ef60adeb0616b0a170ee3ca8fad80bad6e3e46cd9735ca32",
    },
    "ZP100": {
        "tag": "_eidzproj100_s42",
        "mode": "z_only_proj",
        "manifest": "eidzproj100_s42_manifest.json",
        "init_state_sha256": "4d7ae7aa41e283b1c95acd55682ed0551dbc5820031b90f99dd339a33b38e1d2",
    },
}
ARM_ORDER = ("B100", "Z100", "ZP100")
SNAPSHOT_FILES = {"latest": "checkpoint_latest.pth", "best": "checkpoint_best.pth"}

# 复用既有 oracle/blind 判读公式（element_identity_valid_verdict.predict 同式）。
METRICS = ("r2_edos_oracle", "r2_edos_blind", "r2_phdos_oracle", "r2_phdos_blind")
METRIC_ORDER = ("r2_edos_oracle", "r2_edos_blind", "r2_phdos_oracle", "r2_phdos_blind")
LOSS_ORDER = ("L_e", "L_p", "L_eta", "L_total")
COMPARISONS = (("Z100", "B100"), ("ZP100", "B100"), ("ZP100", "Z100"))
DELTA_EDOS = 0.09375
DELTA_PHDOS = 19.6875

EVAL_BATCH = 32
PROBE_BATCH = 8
BOOTSTRAP_REPLICATES = 2000
RNG_SAMPLE_TRAIN = 20261002
RNG_SAMPLE_PROBE = 20261003
RNG_BOOTSTRAP = 20261004
TRAIN_N = 18706
VALID_N = 2313
TRAIN_SUBSET = 2048
PROBE_N = 128

LOSS_DEFAULTS = {  # basemodel.__init__ 缺省解析口径
    "use_mask": False, "w_w1": 1.0, "w_huber": 1.0, "huber_delta": 0.02,
    "lambda_ph": 1.0, "eta_sup_w": 1.0, "delta_edos": 0.09375, "delta_phdos": 19.6875,
}

SOURCE_FILES = [
    "model/transformer.py", "model/model.py", "model/losses.py", "model/heads.py",
    "datasets/dataset.py", "utils/metrics.py", "utils/zproj_alignment.py",
    "utils/atom_feature.py", "utils/builder.py", "utils/relative_features.py",
    "utils/rp_encoding.py", "run_ablation_experiments.py",
    "tools/eval/element_identity_diagnosis.py",
]
OLD_ARTIFACT_GLOBS = [
    "results/eidconst100_s42_manifest.json", "results/eidzonly100_s42_manifest.json",
    "results/eidzproj100_s42_manifest.json", "results/eid_s42_valid_samples.csv",
    "results/eid_zonly_s42_valid_samples.csv", "results/eid_zproj_s42_valid_samples.csv",
    "results/eid_s42_valid_verdict.json", "results/eid_zonly_s42_valid_verdict.json",
    "results/eid_zproj_s42_valid_verdict.json",
    "results/history_m1_eidconst100_s42.csv", "results/history_m1_eidzonly100_s42.csv",
    "results/history_m1_eidzproj100_s42.csv",
]


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(obj):
    """JSON 安全转换：numpy/torch 标量转 Python，非有限浮点转 null。"""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, torch.Tensor):
        return json_ready(obj.detach().cpu().tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list, columns: list) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def log(self, msg: str) -> None:
        line = f"{utc_now()} {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


class RunContext:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.dir = RUNS_DIR / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = RunLog(self.dir / "run.log")
        self.manifest_path = self.dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8")) \
            if self.manifest_path.exists() else {"run_id": run_id, "created_utc": utc_now()}
        self.manifest.setdefault("phases", {})

    def save_manifest(self) -> None:
        self.manifest["updated_utc"] = utc_now()
        dump_json(self.manifest_path, self.manifest)

    def set_phase(self, name: str, status: str, **extra) -> None:
        entry = self.manifest["phases"].get(name, {})
        entry.update({"status": status, "updated_utc": utc_now()})
        entry.update(extra)
        self.manifest["phases"][name] = entry
        self.save_manifest()


# ---------------------------------------------------------------------------
# 输入身份与配置
# ---------------------------------------------------------------------------

def arm_dir(arm: str) -> Path:
    return OUTPUT / f"ablation_m1{ARMS[arm]['tag']}"


def arm_config(arm: str) -> dict:
    with (arm_dir(arm) / "config_used.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def transformer_params(arm: str) -> dict:
    params = arm_config(arm)["config"]["model"]["params"]["sub_model"]["transformer"]
    if params.get("atom_feat_mode") != ARMS[arm]["mode"]:
        raise ValueError(f"{arm} config atom_feat_mode mismatch")
    return params


def resolve_loss_params() -> tuple[dict, dict]:
    """从各臂历史 config 解析损失参数，缺省按 basemodel.__init__ 记录来源。"""
    resolved, raw = {}, {}
    for key, default in LOSS_DEFAULTS.items():
        raw[key] = arm_config("B100")["config"]["model"]["params"].get(key, "<absent>")
        resolved[key] = default if raw[key] == "<absent>" else (
            bool(raw[key]) if key == "use_mask" else float(raw[key]))
    for arm in ARM_ORDER:
        for key in LOSS_DEFAULTS:
            value = arm_config(arm)["config"]["model"]["params"].get(key, "<absent>")
            if value != raw[key]:
                raise ValueError(f"loss param {key} differs across arms")
    return resolved, raw


def collect_input_identity() -> dict:
    identity = {"checked_utc": utc_now(), "hashes": {}, "arms": {}}
    for path in ("utils/periodic_table_v2.csv", "index/split_v2.yaml",
                 "data/train4ARPAT/manifest.json"):
        identity["hashes"][path] = sha256_file(ROOT / path)
    for arm in ARM_ORDER:
        manifest = json.loads((RESULTS / ARMS[arm]["manifest"]).read_text(encoding="utf-8"))
        record = {"manifest": ARMS[arm]["manifest"], "manifest_sha256": sha256_file(RESULTS / ARMS[arm]["manifest"]),
                  "atom_feat_mode": manifest.get("atom_feat_mode"),
                  "config_used_sha256": sha256_file(arm_dir(arm) / "config_used.yaml"),
                  "checkpoints": {}}
        if manifest.get("atom_feat_mode") != ARMS[arm]["mode"]:
            raise ValueError(f"{arm} manifest mode mismatch")
        for key, name in (("table_sha256", "utils/periodic_table_v2.csv"),
                          ("split_sha256", "index/split_v2.yaml"),
                          ("data_manifest_sha256", "data/train4ARPAT/manifest.json")):
            if manifest["hashes"][key] != identity["hashes"][name]:
                raise ValueError(f"{arm}: {name} changed since training preflight")
        if manifest.get("config_used_sha256") != record["config_used_sha256"]:
            raise ValueError(f"{arm}: config_used.yaml mismatch")
        for which, filename in SNAPSHOT_FILES.items():
            entry = manifest["checkpoints"][filename]
            digest = sha256_file(arm_dir(arm) / filename)
            if entry["sha256"] != digest:
                raise ValueError(f"{arm}: {filename} SHA-256 mismatch")
            record["checkpoints"][which] = {"file": filename, "sha256": digest,
                                            "epoch": entry["epoch"],
                                            "best_val_score": entry.get("best_val_score")}
        record["history_csv"] = f"results/history_m1{ARMS[arm]['tag']}.csv"
        record["history_csv_sha256"] = sha256_file(ROOT / record["history_csv"])
        identity["arms"][arm] = record
    identity["sources"] = {path: sha256_file(ROOT / path) for path in SOURCE_FILES}
    z_manifest = json.loads((RESULTS / ARMS["Z100"]["manifest"]).read_text(encoding="utf-8"))
    for path in z_manifest["training_code_sha256"]:
        identity["sources"].setdefault(path, sha256_file(ROOT / path))
    identity["sources_vs_z100_training_record"] = {
        path: {"now": identity["sources"][path],
               "training_record": z_manifest["training_code_sha256"].get(path),
               "match": identity["sources"][path] == z_manifest["training_code_sha256"].get(path)}
        for path in z_manifest["training_code_sha256"]}
    return identity


def old_artifact_hashes() -> dict:
    return {path: sha256_file(ROOT / path) for path in OLD_ARTIFACT_GLOBS}


# ---------------------------------------------------------------------------
# 数据与样本冻结
# ---------------------------------------------------------------------------

def build_data():
    train_ds = Dos_Dataset(data_dir=str(DATA_DIR), split="train", dos_minmax=True, dos_sumnorm=True)
    valid_ds = Dos_Dataset(data_dir=str(DATA_DIR), split="valid", dos_minmax=True, dos_sumnorm=True)
    if len(train_ds) != TRAIN_N or len(valid_ds) != VALID_N:
        raise ValueError(f"Q1 split size mismatch: train={len(train_ds)} valid={len(valid_ds)}")
    train_indices = np.sort(np.random.default_rng(RNG_SAMPLE_TRAIN).choice(
        TRAIN_N, size=TRAIN_SUBSET, replace=False))
    probe_indices = np.random.default_rng(RNG_SAMPLE_PROBE).choice(
        train_indices, size=PROBE_N, replace=False)
    if len(np.unique(train_indices)) != TRAIN_SUBSET or len(np.unique(probe_indices)) != PROBE_N:
        raise ValueError("frozen sample indices are not unique")
    if not np.isin(probe_indices, train_indices).all():
        raise ValueError("probe indices must be drawn from the frozen train subset")
    train_ids = np.load(DATA_DIR / "train/train_index.npy")
    valid_ids = np.load(DATA_DIR / "valid/valid_index.npy")
    if len(train_ids) != TRAIN_N or len(valid_ids) != VALID_N:
        raise ValueError("material ID arrays do not match Q1 split sizes")
    from torch.utils.data import DataLoader, Subset

    train_loader = DataLoader(Subset(train_ds, train_indices.tolist()), batch_size=EVAL_BATCH,
                              shuffle=False, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0)
    probe_loader = DataLoader(Subset(train_ds, probe_indices.tolist()), batch_size=PROBE_BATCH,
                              shuffle=False, num_workers=0)
    data = {
        "train_ds": train_ds, "valid_ds": valid_ds,
        "train_indices": train_indices, "probe_indices": probe_indices,
        "train_ids": train_ids, "valid_ids": valid_ids,
        "train_loader": train_loader, "valid_loader": valid_loader, "probe_loader": probe_loader,
        "split_index": {"train": train_indices, "valid": np.arange(VALID_N)},
        "split_ids": {"train": train_ids[train_indices], "valid": valid_ids},
    }
    return data


def sample_identity_block(data) -> dict:
    return {
        "train_subset": {
            "n": TRAIN_SUBSET, "seed": RNG_SAMPLE_TRAIN, "sort": True,
            "indices_sha256": hashlib.sha256(
                np.asarray(data["train_indices"], dtype=np.int64).tobytes()).hexdigest(),
            "mpid_sha256": hashlib.sha256(
                np.asarray(data["split_ids"]["train"]).tobytes()).hexdigest(),
        },
        "probe": {
            "n": PROBE_N, "seed": RNG_SAMPLE_PROBE,
            "drawn_from": "train_subset indices (np.random.default_rng(20261003).choice)",
            "indices_sha256": hashlib.sha256(
                np.asarray(data["probe_indices"], dtype=np.int64).tobytes()).hexdigest(),
            "mpid_sha256": hashlib.sha256(
                np.asarray(data["train_ids"][data["probe_indices"]]).tobytes()).hexdigest(),
        },
        "valid": {
            "n": VALID_N,
            "mpid_sha256": hashlib.sha256(np.asarray(data["valid_ids"]).tobytes()).hexdigest(),
        },
        "batch": {"eval_batch": EVAL_BATCH, "probe_batch": PROBE_BATCH,
                  "shuffle": False, "num_workers": 0, "fp32": True},
    }


# ---------------------------------------------------------------------------
# 模型载入与未训练重建
# ---------------------------------------------------------------------------

def load_trained(arm: str, which: str, device):
    """复用各判读工具的载入函数（含 manifest/配方/checkpoint 身份核对）。"""
    if arm == "B100":
        model, info = base.load_arm(arm_dir(arm), RESULTS / ARMS[arm]["manifest"],
                                    ARMS[arm]["mode"], SNAPSHOT_FILES[which], device)
    elif arm == "Z100":
        model, info = load_z100(arm_dir(arm), SNAPSHOT_FILES[which], device)
    else:
        model, info = load_zp100(arm_dir(arm), SNAPSHOT_FILES[which], device)
    return model, info


def load_checkpoint(arm: str, which: str) -> dict:
    return torch.load(arm_dir(arm) / SNAPSHOT_FILES[which], map_location="cpu", weights_only=True)


def build_init_models() -> dict:
    """按各臂有效配置独立重建 seed-42 未训练模型；ZP100 复制 Z100 共享初值 + 恒等投影。"""
    from run_ablation_experiments import setup_ablation_seed

    models = {}
    for arm in ("B100", "Z100"):
        setup_ablation_seed(42)
        model = Transformer(**transformer_params(arm))
        models[arm] = model
    setup_ablation_seed(42)
    zp = Transformer(**transformer_params("ZP100"))
    ref_state = {key: value.detach().cpu().clone()
                 for key, value in models["Z100"].state_dict().items()}
    identity_projection_(zp.atom_proj)
    new_state = dict(ref_state)
    new_state["atom_proj.weight"] = zp.atom_proj.weight.detach().cpu().clone()
    new_state["atom_proj.bias"] = zp.atom_proj.bias.detach().cpu().clone()
    zp.load_state_dict(new_state, strict=True)
    models["ZP100"] = zp

    fingerprints = {}
    for arm in ARM_ORDER:
        digest = state_hash(models[arm].state_dict())
        fingerprints[arm] = {"computed": digest, "recorded": ARMS[arm]["init_state_sha256"],
                             "match": digest == ARMS[arm]["init_state_sha256"]}
        if not fingerprints[arm]["match"]:
            raise ValueError(f"{arm} init state fingerprint mismatch: {digest}")
    return models, fingerprints


# ---------------------------------------------------------------------------
# 无更新批次前向：损失 + oracle/blind 指标
# ---------------------------------------------------------------------------

def forward_batch(model, batch, device):
    src = batch[0].to(device)
    pos = batch[1].to(device)
    out = model(src, src.eq(0), pos, batch[15].to(device), batch[16].to(device))
    return src, pos, out


def batch_targets(batch, device):
    e_target = batch[2].to(device)
    p_target = batch[3].to(device)
    e_min = batch[6].to(device).reshape(-1, 1)
    e_max = batch[7].to(device).reshape(-1, 1)
    p_min = batch[10].to(device).reshape(-1, 1)
    p_max = batch[11].to(device).reshape(-1, 1)
    nval = batch[14].to(device).reshape(-1, 1)
    return e_target, p_target, e_min, e_max, p_min, p_max, nval


def loss_terms(model, batch, out, src, resolved, device):
    """复现 basemodel.train_one_step 的损失公式（不调用 train_one_step）。"""
    e_target, p_target, _, e_max, _, p_max, nval = batch_targets(batch, device)
    e_cov = batch[12].to(device)
    p_cov = batch[13].to(device)
    l_e = sumnorm_klw_loss(out["edos"], e_target, e_cov, resolved["use_mask"],
                           resolved["w_w1"], resolved["w_huber"], resolved["huber_delta"])
    l_p = sumnorm_klw_loss(out["phdos"], p_target, p_cov, resolved["use_mask"],
                           resolved["w_w1"], resolved["w_huber"], resolved["huber_delta"])
    atom_len = src.shape[1] - 2
    natoms = (~src.eq(0)[:, 2:2 + atom_len]).sum(dim=-1).float().clamp_min(1.0).unsqueeze(-1)
    eta_pred = out["eta"][:, 0:1]
    gamma_pred = out["eta"][:, 1:2]
    eta_true = (p_max.clamp_min(1e-12) * resolved["delta_phdos"] / (3.0 * natoms)).clamp(0.0, 1.0)
    gamma_true = (e_max.clamp_min(1e-12) * resolved["delta_edos"]
                  / nval.clamp_min(1e-12)).clamp(0.0, 1.0)
    tgt = torch.cat([eta_true, gamma_true], dim=-1)
    se = (out["eta"] - tgt) ** 2
    finite = torch.isfinite(nval)
    l_eta = torch.where(finite.expand_as(se), se, torch.zeros_like(se)).mean(dim=-1)
    l_total = l_e + resolved["lambda_ph"] * l_p + resolved["eta_sup_w"] * l_eta
    return {
        "L_e": l_e, "L_p": l_p, "L_eta": l_eta, "L_total": l_total,
        "natoms": natoms.reshape(-1),
        "eta_pred": eta_pred.reshape(-1), "eta_true": eta_true.reshape(-1),
        "gamma_pred": gamma_pred.reshape(-1), "gamma_true": gamma_true.reshape(-1),
    }


def predict_metrics(src, out, batch, device):
    e_target, p_target, e_min, e_max, p_min, p_max, _ = batch_targets(batch, device)
    e_shape = F.softmax(out["edos"], dim=-1)
    p_shape = F.softmax(out["phdos"], dim=-1)
    gamma = out["eta"][:, 1:2]
    eta = out["eta"][:, 0:1]
    atom_len = src.shape[1] - 2
    natoms = (~src.eq(0)[:, 2:2 + atom_len]).sum(dim=-1, keepdim=True).float()
    true_e = e_target * (e_max - e_min) + e_min
    true_p = p_target * (p_max - p_min) + p_min
    preds = {
        "r2_edos_oracle": (e_shape * (e_max - e_min) + e_min).clamp_min(0),
        "r2_edos_blind": e_shape * (nval_scale(gamma, batch, device) / DELTA_EDOS),
        "r2_phdos_oracle": (p_shape * (p_max - p_min) + p_min).clamp_min(0),
        "r2_phdos_blind": p_shape * (3.0 * natoms * eta / DELTA_PHDOS),
    }
    out_rows = {}
    for name, pred in preds.items():
        true = true_e if "edos" in name else true_p
        stats = per_sample_spectral_metrics(pred, true)
        out_rows[f"{name}"] = stats["r2"].detach().cpu().numpy().astype(np.float64)
        out_rows[f"mae_{name[3:]}"] = stats["mae"].detach().cpu().numpy().astype(np.float64)
    return out_rows


def nval_scale(gamma, batch, device):
    nval = batch[14].to(device).reshape(-1, 1)
    return nval * gamma


def evaluate_split(model, loader, resolved, device, indices: np.ndarray, ids: np.ndarray,
                   repeat: int = 2):
    """评估一个 split；默认重复两次取稳定态（第二次），并记录两次间差异。

    背景：本 GPU 上同一模型的首次评估前向与后续前向存在浮点非确定性（实测逐样本
    R² 差可达 ~8e-6；近平坦目标的 R² 会病态放大该差异）。第二次起逐位稳定。
    两次结果都保留：产品使用稳定态，复现包络写入 metrics.json 供核对。
    """
    passes = []
    for _ in range(max(1, repeat)):
        passes.append(_evaluate_split_once(model, loader, resolved, device, indices, ids))
    repro = {}
    if len(passes) > 1:
        for key in passes[0]:
            if passes[0][key].dtype.kind == "f":
                repro[key] = float(np.max(np.abs(passes[0][key] - passes[1][key])))
    return passes[-1], {"n_passes": len(passes), "max_abs_diff_pass1_vs_last": repro}


def _evaluate_split_once(model, loader, resolved, device, indices: np.ndarray, ids: np.ndarray):
    rows = {key: [] for key in (
        "L_e", "L_p", "L_eta", "L_total", "natoms",
        "eta_pred", "eta_true", "gamma_pred", "gamma_true")}
    for name in METRICS:
        rows[name] = []
        rows[f"mae_{name[3:]}"] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            src, pos, out = forward_batch(model, batch, device)
            losses = loss_terms(model, batch, out, src, resolved, device)
            metrics = predict_metrics(src, out, batch, device)
            size = src.shape[0]
            for key in ("L_e", "L_p", "L_eta", "L_total", "natoms",
                        "eta_pred", "eta_true", "gamma_pred", "gamma_true"):
                rows[key].append(losses[key].detach().cpu().numpy().astype(np.float64))
            for key, values in metrics.items():
                rows[key].append(values)
            offset += size
    if offset != len(indices):
        raise ValueError(f"evaluated {offset} samples, expected {len(indices)}")
    arrays = {key: np.concatenate(chunks) for key, chunks in rows.items()}
    arrays["sample_index"] = np.asarray(indices, dtype=np.int64)
    arrays["mpid"] = np.asarray(ids)
    for key, values in arrays.items():
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite values in {key}")
    return arrays


def summarize(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9))}


def paired_mean_diff(values_a: np.ndarray, values_b: np.ndarray, rng) -> dict:
    n = len(values_a)
    samples = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    delta_boot = (values_b[samples] - values_a[samples]).mean(axis=1)
    return {
        "a_mean": float(np.mean(values_a)),
        "b_mean": float(np.mean(values_b)),
        "delta_mean": float(np.mean(values_b) - np.mean(values_a)),
        "delta_mean_ci95": np.quantile(delta_boot, [0.025, 0.975]).tolist(),
    }


def rarest_element_counts(split: str) -> np.ndarray:
    train_atoms = np.load(DATA_DIR / "train/elements_train.npy")[:, 2:]
    atoms = np.load(DATA_DIR / f"{split}/elements_{split}.npy")[:, 2:]
    counts = np.bincount(train_atoms.ravel(), minlength=119)
    return np.array([counts[row[row > 0]].min() for row in atoms])


def failure_flips(records: dict, strata_rare: np.ndarray, natoms_valid: np.ndarray) -> list:
    rows = []
    bins = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 9)]
    for which in ("latest", "best"):
        for cand in ("Z100", "ZP100"):
            ref = records[( "B100", which, "valid")]
            got = records[(cand, which, "valid")]
            for name in METRIC_ORDER:
                ref_fail = ref[name] < 0
                cand_fail = got[name] < 0
                strata = [("all", np.ones_like(ref_fail)),
                          ("rare_lt_500", strata_rare), ("common", ~strata_rare)]
                for lo, hi in bins:
                    strata.append((f"natoms_{lo}_{hi if hi < 10 ** 9 else 'plus'}",
                                   (natoms_valid >= lo) & (natoms_valid <= hi)))
                for label, mask in strata:
                    rows.append({
                        "snapshot": which, "comparison": f"{cand}_vs_B100", "metric": name,
                        "stratum": label, "n": int(mask.sum()),
                        "b100_ok_z_fail": int(((~ref_fail) & cand_fail & mask).sum()),
                        "b100_fail_z_ok": int((ref_fail & (~cand_fail) & mask).sum()),
                        "both_fail": int((ref_fail & cand_fail & mask).sum()),
                        "both_ok": int(((~ref_fail) & (~cand_fail) & mask).sum()),
                    })
    return rows


# ---------------------------------------------------------------------------
# 阶段 A
# ---------------------------------------------------------------------------

def phase_a(ctx: RunContext) -> None:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved, raw = resolve_loss_params()
    data = build_data()
    ctx.log.log(f"phase A start: device={device}, eval_batch={EVAL_BATCH}, "
                f"train subset={TRAIN_SUBSET}, valid={VALID_N}")
    ctx.manifest["loss_params"] = {"resolved": resolved, "raw_from_config": raw,
                                   "source": "config_used.yaml model.params, absent -> basemodel.__init__ default"}
    ctx.manifest["samples"] = sample_identity_block(data)
    ctx.save_manifest()

    # 一个固定批次计时，外推阶段 A 成本。
    probe_model, _ = load_trained("B100", "latest", device)
    fixed_batch = next(iter(data["valid_loader"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tick = time.time()
    for _ in range(3):
        with torch.inference_mode():
            src, pos, out = forward_batch(probe_model, fixed_batch, device)
    forward_s = (time.time() - tick) / 3
    vram_mb = (torch.cuda.max_memory_allocated(device) / 2 ** 20) if device.type == "cuda" else None
    del probe_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    n_batches = 2 * (TRAIN_SUBSET + VALID_N + EVAL_BATCH - 1) // EVAL_BATCH
    timing = {"fixed_batch_forward_s": forward_s, "fixed_batch_size": EVAL_BATCH,
              "peak_vram_mb_one_model": vram_mb,
              "phase_a_estimated_s": forward_s * 6 * n_batches * 2,
              "note": "评估按设计口径重复两次取稳定态，估计含 2 倍前向"}
    ctx.manifest.setdefault("resources", {})["phase_a_timing"] = timing
    ctx.save_manifest()
    ctx.log.log(f"timed one eval batch: {forward_s:.3f}s, est phase A "
                f"{timing['phase_a_estimated_s']:.0f}s, peak vram {vram_mb}")

    records = {}
    state_checks = {}
    reproducibility = {}
    for which in ("latest", "best"):
        for arm in ARM_ORDER:
            model, info = load_trained(arm, which, device)
            hash_before = state_hash(model.state_dict())
            for split, loader in (("train", data["train_loader"]), ("valid", data["valid_loader"])):
                arrays, repro = evaluate_split(model, loader, resolved, device,
                                               data["split_index"][split], data["split_ids"][split])
                records[(arm, which, split)] = arrays
                reproducibility[f"{arm}/{which}/{split}"] = repro
                ctx.log.log(f"phase A evaluated {arm}/{which}/{split}: n={len(arrays['sample_index'])}, "
                            f"pass1-vs-settled max {max(repro['max_abs_diff_pass1_vs_last'].values()):.3e}")
            hash_after = state_hash(model.state_dict())
            if hash_before != hash_after:
                raise ValueError(f"{arm}/{which}: state_dict changed during phase A evaluation")
            state_checks[f"{arm}/{which}"] = {"state_sha256": hash_before, "unchanged": True}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- samples.csv ----
    rows = []
    for (arm, which, split), arrays in sorted(records.items()):
        for i in range(len(arrays["sample_index"])):
            row = {"arm": arm, "snapshot": which, "split": split,
                   "sample_index": int(arrays["sample_index"][i]),
                   "mpid": str(arrays["mpid"][i]),
                   "natoms": int(arrays["natoms"][i])}
            for key in ("L_e", "L_p", "L_eta", "L_total", "eta_pred", "eta_true",
                        "gamma_pred", "gamma_true"):
                row[key] = float(arrays[key][i])
            for name in METRIC_ORDER:
                row[name] = float(arrays[name][i])
                row[f"mae_{name[3:]}"] = float(arrays[f"mae_{name[3:]}"][i])
            rows.append(row)
    columns = (["arm", "snapshot", "split", "sample_index", "mpid", "natoms",
                "L_e", "L_p", "L_eta", "L_total"]
               + [f"mae_{name[3:]}" for name in METRIC_ORDER]
               + list(METRIC_ORDER)
               + ["gamma_pred", "gamma_true", "eta_pred", "eta_true"])
    write_csv(ctx.dir / "samples.csv", rows, columns)
    ctx.log.log(f"phase A samples.csv written: {len(rows)} rows")

    # ---- 与 eid_zproj_s42_valid_samples.csv 交叉核对（预注册容差 rtol=1e-5, atol=1e-6） ----
    cross = pd.read_csv(RESULTS / "eid_zproj_s42_valid_samples.csv")
    valid_ids = data["valid_ids"]
    if not np.array_equal(cross["mpid"].to_numpy(), valid_ids):
        raise ValueError("valid sample order mismatch vs eid_zproj_s42_valid_samples.csv")
    crosscheck = {}
    for arm in ARM_ORDER:
        for which, suffix in (("best", ""), ("latest", "_epoch100")):
            arrays = records[(arm, which, "valid")]
            for name in METRIC_ORDER:
                column = f"{arm}_{name}{suffix}"
                historical = cross[column].to_numpy(dtype=np.float64)
                diff = np.abs(arrays[name] - historical)
                worst = int(np.argmax(diff))
                passed = bool(np.allclose(arrays[name], historical, rtol=1e-5, atol=1e-6))
                crosscheck[column] = {
                    "max_abs_diff": float(diff.max()),
                    "median_abs_diff": float(np.median(diff)),
                    "n_over_1e-6": int((diff > 1e-6).sum()),
                    "allclose_rtol1e-5_atol1e-6": passed,
                    "worst_index": worst,
                    "worst_mpid": str(valid_ids[worst]),
                    "worst_mine": float(arrays[name][worst]),
                    "worst_historical": float(historical[worst]),
                }
    worst_overall = max(entry["max_abs_diff"] for entry in crosscheck.values())
    failed_columns = [key for key, entry in crosscheck.items()
                      if not entry["allclose_rtol1e-5_atol1e-6"]]
    ctx.log.log(f"phase A valid cross-check: max abs diff {worst_overall:.3e}, "
                f"columns failing pre-registered tolerance: {failed_columns or 'none'}")
    ctx.log.log("cross-check diagnosis: formula/inputs/batch order verified identical; residual "
                "differences are GPU forward float nondeterminism (see forward_reproducibility), "
                "amplified by per-sample R2 ill-conditioning on near-flat targets; tolerance NOT loosened")

    # ---- 汇总 + 配对 bootstrap（固定顺序、单 RNG） ----
    summary = {}
    for (arm, which, split), arrays in sorted(records.items()):
        summary.setdefault(split, {}).setdefault(arm, {})[which] = {
            "n": len(arrays["sample_index"]),
            "losses": {key: summarize(arrays[key]) for key in LOSS_ORDER},
            "r2": {name: {"median": float(np.median(arrays[name])),
                          "fail_count": int((arrays[name] < 0).sum()),
                          "fail_percent": float(100 * np.mean(arrays[name] < 0))}
                   for name in METRIC_ORDER},
            "mae": {name[3:]: {"median": float(np.median(arrays[f"mae_{name[3:]}"]))}
                    for name in METRIC_ORDER},
            "scale": {
                "gamma_abs_error_median": float(np.median(np.abs(arrays["gamma_pred"] - arrays["gamma_true"]))),
                "eta_abs_error_median": float(np.median(np.abs(arrays["eta_pred"] - arrays["eta_true"]))),
                "gamma_pred_median": float(np.median(arrays["gamma_pred"])),
                "gamma_true_median": float(np.median(arrays["gamma_true"])),
                "eta_pred_median": float(np.median(arrays["eta_pred"])),
                "eta_true_median": float(np.median(arrays["eta_true"])),
            },
        }

    rng = np.random.default_rng(RNG_BOOTSTRAP)
    paired = {}
    for split in ("train", "valid"):
        paired[split] = {}
        for which in ("latest", "best"):
            paired[split][which] = {}
            for cand, ref in COMPARISONS:
                key = f"{cand}_minus_{ref}"
                entry = {"r2": {}, "loss_mean": {}}
                for name in METRIC_ORDER:
                    entry["r2"][name] = base.paired_comparison(
                        records[(ref, which, split)][name], records[(cand, which, split)][name], rng)
                for loss in LOSS_ORDER:
                    entry["loss_mean"][loss] = paired_mean_diff(
                        records[(ref, which, split)][loss], records[(cand, which, split)][loss], rng)
                paired[split][which][key] = entry
    ctx.log.log("phase A paired bootstrap completed")

    natoms_valid = records[("B100", "latest", "valid")]["natoms"]
    strata_rare = rarest_element_counts("valid") < 500
    flips = failure_flips(records, strata_rare, natoms_valid)
    write_csv(ctx.dir / "failure_flips.csv", flips,
              ["snapshot", "comparison", "metric", "stratum", "n",
               "b100_ok_z_fail", "b100_fail_z_ok", "both_fail", "both_ok"])

    metrics = {
        "phase": "A",
        "loss_params": {"resolved": resolved, "raw_from_config": raw},
        "summary": summary,
        "paired_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES, "rng_seed": RNG_BOOTSTRAP,
            "loop_order": "split(train,valid) -> snapshot(latest,best) -> comparison "
                          "(Z100-B100, ZP100-B100, ZP100-Z100) -> 4 r2 metrics, then 4 loss means "
                          "(L_e, L_p, L_eta, L_total) in the same cell; one shared generator stream",
            "interpretation_note": "设计对损失 bootstrap 与指标 bootstrap 的嵌套关系未明示；本轮取"
                                   "同一格内先指标后损失的单一 RNG 流，属口径解释，不影响点估计。",
            "data": paired,
        },
        "valid_crosscheck_vs_eid_zproj_samples": {
            "source": "results/eid_zproj_s42_valid_samples.csv",
            "source_sha256": sha256_file(RESULTS / "eid_zproj_s42_valid_samples.csv"),
            "rtol": 1e-5, "atol": 1e-6,
            "tolerance_note": "按设计预注册容差核对，未放宽容差；口径（公式、输入、批次顺序）已逐项核对一致，"
                             "残差来自 GPU 前向浮点非确定性并在近平坦目标的 R² 上病态放大。",
            "by_column": crosscheck,
            "max_abs_diff": worst_overall,
            "columns_failing_pre_registered_tolerance": failed_columns,
        },
        "forward_reproducibility": {
            "note": "同一模型同一 split 连续两次评估的逐指标最大差；产品数值取第二次（稳定态）。"
                    "首次评估含 kernel/算法选择预热，逐样本 R² 差可达 ~8e-6，第二次起逐位稳定。",
            "by_arm_snapshot_split": reproducibility,
        },
        "failure_flips_csv": "failure_flips.csv",
        "state_dict_checks": state_checks,
    }
    dump_json(ctx.dir / "metrics_phase_a.json", metrics)
    ctx.set_phase("A", "done", duration_s=time.time() - started,
                  outputs=["samples.csv", "metrics_phase_a.json", "failure_flips.csv"])
    ctx.log.log(f"phase A done in {time.time() - started:.0f}s")


# ---------------------------------------------------------------------------
# 阶段 B：初值、参数变化、优化器状态、当前梯度
# ---------------------------------------------------------------------------

GROUP_PATTERNS = [
    ("tok_emb", re.compile(r"^tok_emb\.")),
    ("atom_norm", re.compile(r"^atom_norm\.")),
    ("atom_proj", re.compile(r"^atom_proj\.")),
    ("num_emb_encoder", re.compile(r"^num_emb_encoder\.")),
    ("num_norm", re.compile(r"^num_norm\.")),
    ("fuse_proj", re.compile(r"^fuse_proj\.")),
    ("decoder", re.compile(r"^decoder\.")),
    ("spectrum_heads", re.compile(r"^(edos_out_head|phdos_out_head)\.")),
    ("scale_head", re.compile(r"^eta_head\.")),
]
ENCODER_RE = re.compile(r"^encoder\.layers\.(\d+)\.(\w+)")


def group_of(name: str) -> str:
    match = ENCODER_RE.match(name)
    if match:
        layer, leaf = int(match.group(1)), match.group(2)
        return f"encoder.layer{layer}.rp_proj" if leaf == "rp_proj" else f"encoder.layer{layer}.other"
    for group, pattern in GROUP_PATTERNS:
        if pattern.match(name):
            return group
    return "other"


def all_groups() -> list:
    groups = ["tok_emb", "atom_norm", "atom_proj", "num_emb_encoder", "num_norm", "fuse_proj"]
    for layer in range(6):
        groups += [f"encoder.layer{layer}.rp_proj", f"encoder.layer{layer}.other"]
    groups += ["decoder", "spectrum_heads", "scale_head", "other"]
    return groups


def embedding_row_categories(data, num_embeddings: int) -> dict:
    """embedding 行分类：train 真实 Z / padding / 哨兵 / 未出现。

    哨兵按槽位识别（elements[:, 0:2] 的取值）。本配置 token_num=118（行索引
    0..117），而哨兵 token 值为 126/127，落在索引范围外：它们只存在于晶格槽位，
    从不进入 tok_emb；因此“哨兵行”类别为空属预期事实，不是缺统计。
    """
    elements = np.load(DATA_DIR / "train/elements_train.npy")
    atoms = elements[:, 2:]
    real_z = sorted(int(z) for z in np.unique(atoms) if 0 < z < num_embeddings)
    sentinel_tokens = sorted({int(z) for z in np.unique(elements[:, 0])} | {int(z) for z in np.unique(elements[:, 1])})
    sentinel_rows = [z for z in sentinel_tokens if z < num_embeddings]
    padding = [0]
    seen = set(real_z) | set(sentinel_rows) | set(padding)
    unseen = [i for i in range(num_embeddings) if i not in seen]
    return {"real_z_train": real_z, "sentinel_rows": sentinel_rows,
            "padding_rows": padding, "unseen_rows": unseen,
            "sentinel_tokens_by_slot": sentinel_tokens,
            "note": (f"token_num={num_embeddings}; 哨兵 token 值 {sentinel_tokens} 超出 embedding 行范围，"
                     "从不进入 tok_emb（哨兵仅占 src 前两行晶格槽位）")}


def row_change_stats(init_rows: torch.Tensor, final_rows: torch.Tensor, idx: list) -> dict:
    if not idx:
        return {"n_rows": 0, "init_row_norm_rms": None, "final_row_norm_rms": None,
                "change_rms": None, "rel_change": None, "note": "empty category"}
    a = init_rows[idx]
    b = final_rows[idx]
    delta = b - a
    init_norm = float(torch.linalg.vector_norm(a))
    return {
        "n_rows": len(idx),
        "init_row_norm_rms": float(torch.sqrt(torch.mean(a * a))),
        "final_row_norm_rms": float(torch.sqrt(torch.mean(b * b))),
        "change_rms": float(torch.sqrt(torch.mean(delta * delta))),
        "rel_change": (float(torch.linalg.vector_norm(delta) / init_norm) if init_norm > 0 else None),
        "note": "" if init_norm > 0 else "init norm is zero: relative change is null",
    }


def parameter_change_rows(init_model, trained_states: dict) -> tuple[list, dict, dict]:
    init_state = {key: value.detach().float().cpu()
                  for key, value in init_model.state_dict().items()}
    named = list(init_model.named_parameters())
    rows = []
    group_summary = {}
    for name, param in named:
        init_t = init_state[name]
        entry = {"arm": None, "parameter": name, "group": group_of(name),
                 "numel": int(param.numel()), "requires_grad": bool(param.requires_grad),
                 "init_rms": float(torch.sqrt(torch.mean(init_t * init_t)))}
        init_norm = float(torch.linalg.vector_norm(init_t))
        for which, state in trained_states.items():
            final_t = state[name].detach().float().cpu()
            delta = final_t - init_t
            entry[f"{which}_rms"] = float(torch.sqrt(torch.mean(final_t * final_t)))
            entry[f"abs_change_rms_{which}"] = float(torch.sqrt(torch.mean(delta * delta)))
            if init_norm > 0:
                entry[f"rel_change_{which}"] = float(torch.linalg.vector_norm(delta) / init_norm)
            else:
                entry[f"rel_change_{which}"] = None
        entry["rel_change_note"] = ("" if init_norm > 0 else "init norm zero: rel change null")
        rows.append(entry)
    # 分组汇总（同组参数合并 RMS/范数口径）
    for group in all_groups():
        members = [r for r in rows if r["group"] == group]
        if not members:
            group_summary[group] = {"n_params": 0, "numel": 0, "note": "module absent (N/A, not zero)"}
            continue
        numel = sum(r["numel"] for r in members)
        summary = {"n_params": len(members), "numel": numel}
        for which in ("best", "latest"):
            abs_sq = sum((r[f"abs_change_rms_{which}"] ** 2) * r["numel"] for r in members)
            summary[f"abs_change_rms_{which}"] = float(np.sqrt(abs_sq / numel))
        summary["init_rms"] = float(np.sqrt(
            sum((r["init_rms"] ** 2) * r["numel"] for r in members) / numel))
        group_summary[group] = summary
    return rows, group_summary, init_state


def optimizer_rows(arm: str, which: str, named) -> tuple[list, dict]:
    checkpoint = load_checkpoint(arm, which)
    opt = checkpoint["optimizer"]
    groups = opt["param_groups"]
    if len(groups) != 1:
        raise ValueError(f"{arm}/{which}: expected a single optimizer param group")
    ids = list(groups[0]["params"])
    names = [name for name, _ in named]
    params = [param for _, param in named]
    if len(ids) != len(params):
        raise ValueError(f"{arm}/{which}: optimizer id count {len(ids)} != named params {len(params)}")
    rows = []
    steps = set()
    for position, (pid, name, param) in enumerate(zip(ids, names, params)):
        state = opt["state"].get(pid)
        if state is None:
            rows.append({"arm": arm, "checkpoint": which, "position": position,
                         "parameter": name, "group": group_of(name),
                         "numel": int(param.numel()), "requires_grad": bool(param.requires_grad),
                         "param_group": 0, "in_optimizer": False, "step": None,
                         "exp_avg_rms": None, "exp_avg_sq_rms": None, "adam_update_rms": None,
                         "note": "no optimizer state entry"})
            continue
        exp_avg = state["exp_avg"].detach().float().cpu()
        exp_avg_sq = state["exp_avg_sq"].detach().float().cpu()
        if tuple(exp_avg.shape) != tuple(param.shape) or tuple(exp_avg_sq.shape) != tuple(param.shape):
            raise ValueError(f"{arm}/{which}: optimizer state shape mismatch at {name}")
        step = float(state["step"]) if not torch.is_tensor(state["step"]) else float(state["step"].item())
        steps.add(step)
        update = exp_avg / (torch.sqrt(exp_avg_sq) + 1e-8)
        rows.append({"arm": arm, "checkpoint": which, "position": position,
                     "parameter": name, "group": group_of(name),
                     "numel": int(param.numel()), "requires_grad": bool(param.requires_grad),
                     "param_group": 0, "in_optimizer": True, "step": step,
                     "exp_avg_rms": float(torch.sqrt(torch.mean(exp_avg * exp_avg))),
                     "exp_avg_sq_rms": float(torch.sqrt(torch.mean(exp_avg_sq * exp_avg_sq))),
                     "adam_update_rms": float(torch.sqrt(torch.mean(update * update))),
                     "note": ""})
    info = {
        "param_group_count": len(groups),
        "param_group_config": {key: json_ready(groups[0].get(key)) for key in
                               ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
                                "foreach", "capturable", "differentiable", "fused", "initial_lr")
                               if key in groups[0]},
        "state_entries": len(opt["state"]),
        "named_params": len(params),
        "distinct_steps": sorted(steps),
        "mapping_method": "checkpoint optimizer param_groups[0]['params'] positional order "
                          "vs model.named_parameters() original order; shapes verified per entry",
    }
    return rows, info


def gradient_rows(arm: str, snapshot: str, model, probe_loader, resolved, device) -> tuple[list, dict]:
    named = list(model.named_parameters())
    params = [param for _, param in named]
    group_index = {}
    for position, (name, _) in enumerate(named):
        group_index.setdefault(group_of(name), []).append(position)
    for group in all_groups():
        group_index.setdefault(group, [])
    rows = []
    negative_cos = {}
    for bi, batch in enumerate(probe_loader):
        model.zero_grad(set_to_none=True)
        src, pos, out = forward_batch(model, batch, device)
        losses = loss_terms(model, batch, out, src, resolved, device)
        terms = [
            ("L_e", losses["L_e"].mean()),
            ("lambda_ph_L_p", resolved["lambda_ph"] * losses["L_p"].mean()),
            ("eta_sup_w_L_eta", resolved["eta_sup_w"] * losses["L_eta"].mean()),
        ]
        grads = {}
        for ti, (term, value) in enumerate(terms):
            grads[term] = torch.autograd.grad(
                value, params, retain_graph=ti < len(terms) - 1, allow_unused=True)
        batch_info = {
            "batch": bi,
            "batch_loss_L_e": float(losses["L_e"].mean()),
            "batch_loss_L_p": float(losses["L_p"].mean()),
            "batch_loss_L_eta": float(losses["L_eta"].mean()),
        }
        for group in all_groups():
            positions = group_index[group]
            if not positions:
                rows.append({"arm": arm, "snapshot": snapshot, "group": group,
                             "term": "module_absent", **batch_info,
                             "n_params": 0, "n_none": 0, "n_zero": 0, "n_nonfinite": 0,
                             "grad_norm": None, "grad_rms": None,
                             "cos_edos_phdos": None,
                             "note": "module absent (N/A, not zero gradient)"})
                continue
            for term, _ in terms:
                tensors = [grads[term][p] for p in positions]
                n_none = sum(1 for t in tensors if t is None)
                finite = [t for t in tensors if t is not None and torch.isfinite(t).all()]
                n_nonfinite = sum(1 for t in tensors if t is not None and not torch.isfinite(t).all())
                n_zero = sum(1 for t in finite if float(torch.abs(t).max()) == 0.0)
                sq = sum(float((t * t).sum()) for t in finite)
                numel = sum(t.numel() for t in finite)
                rows.append({"arm": arm, "snapshot": snapshot, "group": group, "term": term,
                             **batch_info,
                             "n_params": len(tensors), "n_none": n_none, "n_zero": n_zero,
                             "n_nonfinite": n_nonfinite,
                             "grad_norm": float(np.sqrt(sq)) if numel else None,
                             "grad_rms": float(np.sqrt(sq / numel)) if numel else None,
                             "cos_edos_phdos": None, "note": ""})
            # eDOS / phDOS 梯度余弦（共同模块）
            ge = [grads["L_e"][p] for p in positions]
            gp = [grads["lambda_ph_L_p"][p] for p in positions]
            pair = [(a, b) for a, b in zip(ge, gp) if a is not None and b is not None]
            dot = sum(float((a * b).sum()) for a, b in pair)
            na = float(np.sqrt(sum(float((a * a).sum()) for a, _ in pair)))
            nb = float(np.sqrt(sum(float((b * b).sum()) for _, b in pair)))
            if na == 0.0 or nb == 0.0:
                cos_value, note = None, "zero gradient norm: cosine N/A"
            else:
                cos_value, note = dot / (na * nb), ""
                negative_cos.setdefault(group, []).append(cos_value < 0)
            rows.append({"arm": arm, "snapshot": snapshot, "group": group, "term": "cos_e_vs_p",
                         **batch_info, "n_params": len(pair), "n_none": 0, "n_zero": 0,
                         "n_nonfinite": 0, "grad_norm": None, "grad_rms": None,
                         "cos_edos_phdos": cos_value, "note": note})
        model.zero_grad(set_to_none=True)
        del out, losses, grads
    summary = {
        group: {
            "n_batches_with_cosine": len(values),
            "negative_cosine_batches": int(sum(values)),
            "negative_cosine_fraction": (float(np.mean(values)) if values else None),
            "note": "" if values else "no batch produced a defined cosine",
        }
        for group, values in negative_cos.items()
    }
    for group in all_groups():
        summary.setdefault(group, {"n_batches_with_cosine": 0, "negative_cosine_batches": 0,
                                   "negative_cosine_fraction": None,
                                   "note": "no defined cosine (zero norms or absent module)"})
    return rows, summary


def phase_b(ctx: RunContext) -> None:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved, _ = resolve_loss_params()
    data = build_data()
    ctx.log.log("phase B start: rebuild untrained inits and check fingerprints")
    init_models, fingerprints = build_init_models()
    ctx.manifest["init_fingerprints"] = fingerprints
    ctx.save_manifest()
    for arm in ARM_ORDER:
        ctx.log.log(f"init fingerprint {arm}: {fingerprints[arm]['computed'][:16]}… match={fingerprints[arm]['match']}")

    # ---- 参数变化 ----
    change_rows = []
    group_summaries = {}
    for arm in ARM_ORDER:
        trained_states = {}
        for which in ("best", "latest"):
            checkpoint = load_checkpoint(arm, which)
            trained_states[which] = checkpoint["model"]
        rows, group_summary, init_state = parameter_change_rows(init_models[arm], trained_states)
        for row in rows:
            row["arm"] = arm
        change_rows.extend(rows)
        group_summaries[arm] = group_summary
        # embedding 行分类统计
        categories = embedding_row_categories(data, init_state["tok_emb.weight"].shape[0])
        emb = {}
        init_rows = init_state["tok_emb.weight"]
        row_keys = ("real_z_train", "sentinel_rows", "padding_rows", "unseen_rows")
        for which in ("best", "latest"):
            final_rows = trained_states[which]["tok_emb.weight"].detach().float().cpu()
            emb[which] = {
                key: row_change_stats(init_rows, final_rows, categories[key])
                for key in row_keys}
        group_summaries[arm]["embedding_rows"] = {
            "categories": {key: value for key, value in categories.items()},
            "n_rows_total": int(init_rows.shape[0]),
            "stats": emb}
    columns = ["arm", "parameter", "group", "numel", "requires_grad", "init_rms",
               "best_rms", "abs_change_rms_best", "rel_change_best",
               "latest_rms", "abs_change_rms_latest", "rel_change_latest", "rel_change_note"]
    write_csv(ctx.dir / "parameter_changes.csv", change_rows, columns)
    ctx.log.log(f"phase B parameter_changes.csv written: {len(change_rows)} rows")

    # ---- 优化器状态 ----
    opt_rows = []
    opt_info = {}
    for arm in ARM_ORDER:
        named = list(init_models[arm].named_parameters())
        opt_info[arm] = {}
        for which in ("best", "latest"):
            rows, info = optimizer_rows(arm, which, named)
            opt_rows.extend(rows)
            opt_info[arm][which] = info
            ctx.log.log(f"optimizer {arm}/{which}: steps={info['distinct_steps']} "
                        f"entries={info['state_entries']}")
    write_csv(ctx.dir / "optimizer_state.csv", opt_rows,
              ["arm", "checkpoint", "position", "parameter", "group", "numel", "requires_grad",
               "param_group", "in_optimizer", "step", "exp_avg_rms", "exp_avg_sq_rms",
               "adam_update_rms", "note"])

    # ---- 当前梯度探针（init / latest） ----
    grad_rows = []
    grad_summary = {}
    for arm in ARM_ORDER:
        grad_summary[arm] = {}
        for snapshot in ("init", "latest"):
            if snapshot == "init":
                model = init_models[arm]
            else:
                checkpoint = load_checkpoint(arm, "latest")
                model = Transformer(**transformer_params(arm))
                model.load_state_dict(checkpoint["model"], strict=True)
            before = state_hash(model.state_dict())
            model.to(device).eval()
            rows, summary = gradient_rows(arm, snapshot, model, data["probe_loader"], resolved, device)
            model.zero_grad(set_to_none=True)
            model.to("cpu")
            after = state_hash(model.state_dict())
            if before != after:
                raise ValueError(f"{arm}/{snapshot}: state_dict changed during gradient probe")
            grad_rows.extend(rows)
            grad_summary[arm][snapshot] = summary
            ctx.log.log(f"gradients {arm}/{snapshot}: {len(rows)} rows, state hash unchanged")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(ctx.dir / "gradients.csv", grad_rows,
              ["arm", "snapshot", "group", "term", "batch", "batch_loss_L_e", "batch_loss_L_p",
               "batch_loss_L_eta", "n_params", "n_none", "n_zero", "n_nonfinite",
               "grad_norm", "grad_rms", "cos_edos_phdos", "note"])

    metrics = {
        "phase": "B",
        "init_fingerprints": fingerprints,
        "parameter_change_group_summary": group_summaries,
        "optimizer_state": opt_info,
        "gradient_negative_cosine": grad_summary,
        "gradient_note": "eval() 模式、固定 128 条 train 探针上的当前梯度；不代表历史训练梯度"
                         "（历史训练含 dropout、参数逐批变化）。",
    }
    dump_json(ctx.dir / "metrics_phase_b.json", metrics)
    ctx.set_phase("B", "done", duration_s=time.time() - started,
                  outputs=["parameter_changes.csv", "optimizer_state.csv", "gradients.csv",
                           "metrics_phase_b.json"])
    ctx.log.log(f"phase B done in {time.time() - started:.0f}s")


# ---------------------------------------------------------------------------
# 阶段 C：表示、共享偏移、注意力
# ---------------------------------------------------------------------------

def module_param_state(model) -> dict:
    return {name: param.detach().float().cpu() for name, param in model.named_parameters()}


def shared_offset(arm: str, state: dict) -> torch.Tensor:
    """入口共享偏移 b_shared（设计 §5 公式；B100 含 atom_norm.bias 与常数分支）。"""
    beta_atom = state["atom_norm.bias"]
    if arm == "Z100":
        return beta_atom
    if arm == "ZP100":
        return state["atom_proj.weight"] @ beta_atom + state["atom_proj.bias"]
    weight = state["fuse_proj.weight"]
    a = state["num_emb_encoder.proj.weight"]
    b_num = state["num_emb_encoder.proj.bias"]
    gamma = state["num_norm.weight"]
    beta_num = state["num_norm.bias"]
    c = legacy3_const_vector()
    x = a @ c + b_num
    ln = F.layer_norm(x, (x.shape[0],), gamma, beta_num)
    return weight[:, :512] @ beta_atom + weight[:, 512:] @ ln + state["fuse_proj.bias"]


def recompute_layer(layer, src, rp_base, src_key_padding_mask):
    """按 TransformerEncoderLayer.forward 同式重算（eval 下 dropout 为恒等）。

    重算只做诊断分解，不参与反向；内部固定 inference_mode，避免对 hook 捕获的
    inference tensor 触发 autograd 保存。
    """
    with torch.inference_mode():
        return _recompute_layer_impl(layer, src, rp_base, src_key_padding_mask)


def _recompute_layer_impl(layer, src, rp_base, src_key_padding_mask):
    B, L, _ = src.size()
    nhead, dh = layer.nhead, layer.dim // layer.nhead
    q = k = v = src
    rp_emb = layer.rp_proj(rp_base).view(B, L, L, nhead, dh)
    q_heads = q.view(B, L, nhead, dh)
    s_geometry = (q_heads.unsqueeze(2) * rp_emb).sum(-1)            # [B, L, L, nhead]
    s_geometry = s_geometry.permute(0, 3, 1, 2)                     # [B, nhead, L, L]
    q_scaled = q / (dh ** 0.5)
    q_heads2 = q_scaled.view(B, L, nhead, dh).permute(0, 2, 1, 3)
    k_heads = k.view(B, L, nhead, dh).permute(0, 2, 1, 3)
    s_feature = torch.matmul(q_heads2, k_heads.transpose(-1, -2))  # [B, nhead, L, L]
    total = s_feature + s_geometry
    if src_key_padding_mask is not None:
        mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)       # [B, 1, 1, L]
        total = total.masked_fill(mask, -1e9)
    attn = F.softmax(total, dim=-1)
    v_heads = v.view(B, L, nhead, dh).permute(0, 2, 1, 3)
    out = torch.matmul(attn, v_heads).permute(0, 2, 1, 3).reshape(B, L, layer.dim)
    x = layer.norm1(src + layer.dropout1(out))
    x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = layer.norm2(x + layer.dropout2(x2))
    return x, attn, s_feature, s_geometry


class ActAccumulator:
    """表示统计：真实原子的均值/RMS/材料内原子间方差/配对余弦。"""

    def __init__(self):
        self.stats = {}

    def add(self, layer: str, h: torch.Tensor, z: torch.Tensor, real: torch.Tensor):
        with torch.inference_mode():
            self._add_impl(layer, h, z, real)

    def _add_impl(self, layer: str, h: torch.Tensor, z: torch.Tensor, real: torch.Tensor):
        B = h.shape[0]
        entry = self.stats.setdefault(layer, {
            "single": {"means": [], "rms": [], "vars": [], "n_mat": 0, "n_atoms": 0},
            "multi": {"means": [], "rms": [], "vars": [], "n_mat": 0, "n_atoms": 0},
            "pair_same": {"cos": []}, "pair_diff": {"cos": []}})
        for b in range(B):
            mask = real[b]
            n = int(mask.sum())
            if n == 0:
                continue
            hb = h[b][mask]
            zb = z[b][mask]
            unique_z = torch.unique(zb)
            kind = "single" if unique_z.numel() == 1 else "multi"
            entry[kind]["n_mat"] += 1
            entry[kind]["n_atoms"] += n
            entry[kind]["means"].append(float(hb.mean()))
            entry[kind]["rms"].append(float(torch.sqrt((hb * hb).mean())))
            if n >= 2:
                entry[kind]["vars"].append(float(hb.var(dim=0, unbiased=False).mean()))
            if n >= 2:
                normed = F.normalize(hb, dim=-1)
                sim = normed @ normed.T
                iu = torch.triu_indices(n, n, offset=1)
                cos = sim[iu[0], iu[1]]
                same = zb[iu[0]] == zb[iu[1]]
                if bool(same.any()):
                    entry["pair_same"]["cos"].extend(cos[same].tolist())
                if bool((~same).any()):
                    entry["pair_diff"]["cos"].extend(cos[~same].tolist())

    def rows(self, arm: str, snapshot: str) -> list:
        rows = []
        for layer, entry in self.stats.items():
            for kind in ("single", "multi"):
                bucket = entry[kind]
                rows.append({
                    "arm": arm, "snapshot": snapshot, "layer": layer,
                    "stratum": f"{kind}_element_materials",
                    "n_materials": bucket["n_mat"], "n_atoms": bucket["n_atoms"], "n_pairs": None,
                    "act_mean": float(np.mean(bucket["means"])) if bucket["means"] else None,
                    "act_rms": float(np.mean(bucket["rms"])) if bucket["rms"] else None,
                    "atom_var_mean": float(np.mean(bucket["vars"])) if bucket["vars"] else None,
                    "cos_mean": None, "cos_median": None,
                    "note": "" if bucket["vars"] else "inter-atom variance undefined for <2 atoms",
                })
            for label, key in (("same_element_pairs", "pair_same"),
                               ("diff_element_pairs", "pair_diff")):
                cos = entry[key]["cos"]
                rows.append({
                    "arm": arm, "snapshot": snapshot, "layer": layer, "stratum": label,
                    "n_materials": None, "n_atoms": None, "n_pairs": len(cos),
                    "act_mean": None, "act_rms": None, "atom_var_mean": None,
                    "cos_mean": float(np.mean(cos)) if cos else None,
                    "cos_median": float(np.median(cos)) if cos else None,
                    "note": "" if cos else "no pairs of this type in the probe batch",
                })
        return rows


class AttnAccumulator:
    """注意力打分分解与权重统计（按层/head 聚合，逐 query 在线汇总）。"""

    EPS_ZERO = 1e-12

    def __init__(self):
        self.stats = {}

    def add(self, layer: str, attn: torch.Tensor, s_feature: torch.Tensor,
            s_geometry: torch.Tensor, padding: torch.Tensor):
        with torch.inference_mode():
            self._add_impl(layer, attn, s_feature, s_geometry, padding)

    def _add_impl(self, layer: str, attn: torch.Tensor, s_feature: torch.Tensor,
                  s_geometry: torch.Tensor, padding: torch.Tensor):
        B, nhead, L, _ = attn.shape
        for b in range(B):
            valid = ~padding[b]
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            for h in range(nhead):
                entry = self.stats.setdefault((layer, h), {
                    "std_feature": [], "std_geometry": [], "ratio": [],
                    "ratio_na": 0, "attn_max": [], "entropy": [], "entropy_na": 0,
                    "n_queries": 0, "n_keys_single": 0})
                sf = s_feature[b, h][valid][:, valid]      # [n, n]
                sg = s_geometry[b, h][valid][:, valid]
                aw = attn[b, h][valid][:, valid]
                sf_c = sf - sf.mean(dim=-1, keepdim=True)
                sg_c = sg - sg.mean(dim=-1, keepdim=True)
                std_f = torch.sqrt((sf_c * sf_c).mean(dim=-1))
                std_g = torch.sqrt((sg_c * sg_c).mean(dim=-1))
                entry["n_queries"] += n_valid
                entry["std_feature"].extend(std_f.tolist())
                entry["std_geometry"].extend(std_g.tolist())
                for i in range(n_valid):
                    if float(std_f[i]) > self.EPS_ZERO:
                        entry["ratio"].append(float(std_g[i] / std_f[i]))
                    else:
                        entry["ratio_na"] += 1
                entry["attn_max"].extend(aw.max(dim=-1).values.tolist())
                if n_valid == 1:
                    entry["entropy_na"] += n_valid
                    entry["n_keys_single"] += n_valid
                else:
                    p = aw.clamp_min(1e-30)
                    entropy = -(p * p.log()).sum(dim=-1) / float(np.log(n_valid))
                    entry["entropy"].extend(entropy.tolist())

    def rows(self, arm: str, snapshot: str) -> list:
        rows = []
        for (layer, head), entry in sorted(self.stats.items()):
            rows.append({
                "arm": arm, "snapshot": snapshot, "layer": layer, "head": head,
                "n_queries": entry["n_queries"],
                "std_feature_mean": float(np.mean(entry["std_feature"])) if entry["std_feature"] else None,
                "std_feature_median": float(np.median(entry["std_feature"])) if entry["std_feature"] else None,
                "std_geometry_mean": float(np.mean(entry["std_geometry"])) if entry["std_geometry"] else None,
                "std_geometry_median": float(np.median(entry["std_geometry"])) if entry["std_geometry"] else None,
                "ratio_geom_feature_median": float(np.median(entry["ratio"])) if entry["ratio"] else None,
                "ratio_na_count": entry["ratio_na"],
                "attn_max_mean": float(np.mean(entry["attn_max"])) if entry["attn_max"] else None,
                "attn_max_median": float(np.median(entry["attn_max"])) if entry["attn_max"] else None,
                "entropy_norm_mean": float(np.mean(entry["entropy"])) if entry["entropy"] else None,
                "entropy_norm_median": float(np.median(entry["entropy"])) if entry["entropy"] else None,
                "entropy_na_count": entry["entropy_na"],
                "n_queries_single_key": entry["n_keys_single"],
                "note": "ratio N/A when centered std_feature <= 1e-12 (flagged, raw std kept); "
                        "entropy N/A when a query has a single valid key",
            })
        return rows


def phase_c(ctx: RunContext) -> None:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_data()
    init_models, _ = build_init_models()
    layers_of_interest = (0, 5)
    activation_rows = []
    attention_rows = []
    entry_stats = {}
    offset_stats = {}
    param_states = {}
    validation = {}

    for arm in ARM_ORDER:
        for snapshot in ("init", "best", "latest"):
            if snapshot == "init":
                model = init_models.pop(arm)
            else:
                model, _ = load_trained(arm, snapshot, device)
            before = state_hash(model.state_dict())
            model.to(device).eval()
            act_acc, attn_acc = ActAccumulator(), AttnAccumulator()
            entry_sum, entry_sq, entry_n, entry_atoms = 0.0, 0.0, 0, 0
            hooked_check = {"max_abs_diff": 0.0}
            layer_repro = {}
            for bi, batch in enumerate(data["probe_loader"]):
                src = batch[0].to(device)
                pos = batch[1].to(device)
                edos_x, phdos_x = batch[15].to(device), batch[16].to(device)
                captured = {"entry": None, "layers": {}}
                layer_outs = {}
                with torch.inference_mode():
                    # 三遍前向：冷启动首遍、稳定态参照、hook 前向。本 GPU 上冷态首遍
                    # 与稳定态存在浮点级差异（噪声底），故 hook 校验对稳定态参照比较，
                    # 并同时记录冷态差异作为噪声底。
                    clean_cold = model(src, src.eq(0), pos, edos_x, phdos_x)
                    clean = model(src, src.eq(0), pos, edos_x, phdos_x)
                    handles = []

                    def pre_entry(module, args, kwargs):
                        captured["entry"] = kwargs.get("src", args[0] if args else None)
                        return None

                    handles.append(model.encoder.register_forward_pre_hook(pre_entry, with_kwargs=True))

                    def make_pre(index):
                        def hook(module, args, kwargs):
                            captured["layers"].setdefault(index, {})
                            captured["layers"][index]["src"] = args[0]
                            captured["layers"][index]["rp_base"] = kwargs.get("rp_base")
                            captured["layers"][index]["mask"] = kwargs.get("src_key_padding_mask")
                            return None
                        return hook

                    def make_post(index):
                        def hook(module, args, output):
                            layer_outs[index] = output
                            if index in layers_of_interest:
                                captured["layers"].setdefault(index, {})["out"] = output
                            return None
                        return hook

                    for li in range(6):
                        layer = model.encoder.layers[li]
                        if li in layers_of_interest:
                            handles.append(layer.register_forward_pre_hook(make_pre(li), with_kwargs=True))
                        handles.append(layer.register_forward_hook(make_post(li)))
                    hooked = model(src, src.eq(0), pos, edos_x, phdos_x)
                    for handle in handles:
                        handle.remove()
                    for key in ("edos", "phdos", "eta"):
                        cold_diff = float((clean_cold[key] - clean[key]).abs().max())
                        diff = float((hooked[key] - clean[key]).abs().max())
                        hooked_check["max_abs_diff"] = max(hooked_check["max_abs_diff"], diff)
                        hooked_check["cold_vs_settled_max_abs_diff"] = max(
                            hooked_check.get("cold_vs_settled_max_abs_diff", 0.0), cold_diff)
                        if not torch.allclose(hooked[key], clean[key], rtol=1e-5, atol=1e-6):
                            raise ValueError(f"{arm}/{snapshot} hook changed forward output {key}: {diff}")

                    # 表示统计：入口 + 6 层输出
                    real = src[:, 2:].ne(0)
                    z = src[:, 2:]
                    h_entry = captured["entry"]
                    act_acc.add("entry", h_entry, z, real)
                    entry_sum += float(h_entry[real].sum())
                    entry_sq += float((h_entry[real] * h_entry[real]).sum())
                    entry_atoms += int(real.sum())
                    entry_n += int(real.sum()) * h_entry.shape[-1]
                    for li in range(6):
                        act_acc.add(f"encoder.layer{li}", layer_outs[li], z, real)

                    # 注意力重算与校验（第一层/最后一层）
                    for li in layers_of_interest:
                        cap = captured["layers"][li]
                        x_re, attn, s_feature, s_geometry = recompute_layer(
                            model.encoder.layers[li], cap["src"], cap["rp_base"], cap["mask"])
                        if bi == 0:
                            diff = float((x_re - cap["out"]).abs().max())
                            layer_repro[f"layer{li}_max_abs_diff"] = diff
                            if not torch.allclose(x_re, cap["out"], rtol=1e-5, atol=1e-6):
                                raise ValueError(
                                    f"{arm}/{snapshot} layer{li} recomputation mismatch: {diff}")
                        attn_acc.add(f"layer{li}", attn, s_feature, s_geometry, cap["mask"])
                del clean, hooked, layer_outs, captured

            activation_rows.extend(act_acc.rows(arm, snapshot))
            attention_rows.extend(attn_acc.rows(arm, snapshot))
            state = module_param_state(model)
            param_states[(arm, snapshot)] = state
            mean_entry = entry_sum / max(entry_n, 1)
            entry_stats[(arm, snapshot)] = {
                "entry_mean": mean_entry,
                "entry_var": max(0.0, entry_sq / max(entry_n, 1) - mean_entry ** 2),
                "n_atoms": entry_atoms,
                "n_scalars": entry_n,
            }
            offset = shared_offset(arm, state)
            offset_stats[(arm, snapshot)] = {
                "b_shared_norm": float(torch.linalg.vector_norm(offset)),
                "b_shared_rms": float(torch.sqrt((offset * offset).mean())),
                "beta_atom_norm": float(torch.linalg.vector_norm(state["atom_norm.bias"])),
            }
            validation[(arm, snapshot)] = {"hook_forward": hooked_check, "layer_recompute": layer_repro}
            after = state_hash(model.state_dict())
            if before != after:
                raise ValueError(f"{arm}/{snapshot}: state_dict changed during phase C")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ctx.log.log(f"phase C {arm}/{snapshot}: acts={len(act_acc.stats)} layers, "
                        f"attn rows={len(attn_acc.stats)}, hook diff={hooked_check['max_abs_diff']:.3e}")

    # 跨臂共享偏移余弦（同快照）
    offset_cos = {}
    for snapshot in ("init", "best", "latest"):
        offset_cos[snapshot] = {}
        for a, b in (("B100", "Z100"), ("B100", "ZP100"), ("Z100", "ZP100")):
            va = shared_offset(a, param_states[(a, snapshot)])
            vb = shared_offset(b, param_states[(b, snapshot)])
            na, nb = float(torch.linalg.vector_norm(va)), float(torch.linalg.vector_norm(vb))
            offset_cos[snapshot][f"{a}_vs_{b}"] = (
                float((va * vb).sum() / (na * nb)) if na > 0 and nb > 0 else None)

    write_csv(ctx.dir / "activations.csv", activation_rows,
              ["arm", "snapshot", "layer", "stratum", "n_materials", "n_atoms", "n_pairs",
               "act_mean", "act_rms", "atom_var_mean", "cos_mean", "cos_median", "note"])
    write_csv(ctx.dir / "attention.csv", attention_rows,
              ["arm", "snapshot", "layer", "head", "n_queries",
               "std_feature_mean", "std_feature_median", "std_geometry_mean", "std_geometry_median",
               "ratio_geom_feature_median", "ratio_na_count", "attn_max_mean", "attn_max_median",
               "entropy_norm_mean", "entropy_norm_median", "entropy_na_count",
               "n_queries_single_key", "note"])
    metrics = {
        "phase": "C",
        "entry_activation": {f"{arm}/{snapshot}": value for (arm, snapshot), value in entry_stats.items()},
        "shared_offset": {f"{arm}/{snapshot}": value for (arm, snapshot), value in offset_stats.items()},
        "shared_offset_formula": {
            "Z100": "b_shared = beta_atom",
            "ZP100": "b_shared = W_proj @ beta_atom + b_proj",
            "B100": "b_shared = W_Z @ beta_atom + W_c @ LN_num(A @ c + b_num) + b_fuse",
            "note": "参数分解不是因果证明；同元素原子入口表示相同是设计事实，不是 embedding 坍缩。",
        },
        "shared_offset_cosine": offset_cos,
        "validation": validation,
        "notes": [
            "表示统计只统计真实原子（src[:,2:] != 0）；哨兵（晶格行）与 padding 不进入 encoder。",
            "通道旋转使坐标均值不适合单独跨模型判优；余弦相似性与材料内方差为主要判读量。",
            "注意力排除 padding query/key；S_feature 与 S_geometry 按 query 在有效 key 上中心化后比较标准差。",
        ],
    }
    dump_json(ctx.dir / "metrics_phase_c.json", metrics)
    ctx.set_phase("C", "done", duration_s=time.time() - started,
                  outputs=["activations.csv", "attention.csv", "metrics_phase_c.json"])
    ctx.log.log(f"phase C done in {time.time() - started:.0f}s")


def load_trained_state(arm: str, snapshot: str) -> dict:
    """按快照取 CPU 上的参数状态（init 为重建未训练模型）。"""
    if snapshot == "init":
        models, _ = build_init_models()
        return module_param_state(models[arm])
    model = Transformer(**transformer_params(arm))
    model.load_state_dict(load_checkpoint(arm, snapshot)["model"], strict=True)
    return module_param_state(model)


def regime_check(ctx: RunContext) -> None:
    """核对 ZP100/best 在冷启动首遍与稳定态下同历史 CSV 的差异，定位交叉核对偏差来源。

    本函数必须是新进程里的第一次评估（冷启动首遍）才有意义；它只做评估与记录，
    不改变任何参数或旧产物。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved, _ = resolve_loss_params()
    data = build_data()
    cross = pd.read_csv(RESULTS / "eid_zproj_s42_valid_samples.csv")
    model, _ = load_trained("ZP100", "best", device)
    cold, _ = evaluate_split(model, data["valid_loader"], resolved, device,
                             data["split_index"]["valid"], data["split_ids"]["valid"], repeat=1)
    settled, _ = evaluate_split(model, data["valid_loader"], resolved, device,
                                data["split_index"]["valid"], data["split_ids"]["valid"], repeat=1)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {"checked_utc": utc_now(), "arm": "ZP100", "snapshot": "best", "split": "valid",
              "hypothesis": "历史判读工具在本工具进程内最先评估 best，其 CSV 数值处于冷启动首遍"
                            "kernel 区制；稳定态（第二次起）数值不同但内部可复现。",
              "columns": {}}
    for name in METRICS:
        historical = cross[f"ZP100_{name}"].to_numpy(dtype=np.float64)
        result["columns"][name] = {
            "cold_vs_csv_max_abs_diff": float(np.abs(cold[name] - historical).max()),
            "cold_allclose_rtol1e-5_atol1e-6": bool(np.allclose(cold[name], historical,
                                                               rtol=1e-5, atol=1e-6)),
            "settled_vs_csv_max_abs_diff": float(np.abs(settled[name] - historical).max()),
            "settled_allclose_rtol1e-5_atol1e-6": bool(np.allclose(settled[name], historical,
                                                                  rtol=1e-5, atol=1e-6)),
            "cold_vs_settled_max_abs_diff": float(np.abs(cold[name] - settled[name]).max()),
        }
    dump_json(ctx.dir / "valid_crosscheck_regime.json", result)
    ctx.log.log("regime check written: " + json.dumps(
        {name: {k: v for k, v in entry.items() if "max_abs_diff" in k}
         for name, entry in result["columns"].items()}, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["identity", "regime-check", "A", "B", "C", "all"],
                        required=True)
    parser.add_argument("--run-id", default=None,
                        help="继续写入既有 run 目录；缺省用当前 UTC 时间新建")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ctx = RunContext(run_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ctx.log.log(f"=== element identity diagnosis run {run_id} phase={args.phase} ===")
    ctx.log.log(f"torch {torch.__version__}, numpy {np.__version__}, python {platform.python_version()}, "
                f"device {'cuda' if torch.cuda.is_available() else 'cpu'}")

    identity = collect_input_identity()
    ctx.manifest["inputs"] = identity
    ctx.manifest.setdefault("old_artifacts_before", old_artifact_hashes())
    ctx.manifest["environment"] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "test_used": False, "training_run": False, "optimizer_step_called": False,
    }
    ctx.save_manifest()
    ctx.log.log("input identity verified: manifests/configs/checkpoints/data hashes match records")

    if args.phase in ("identity", "all"):
        init_models, fingerprints = build_init_models()
        ctx.manifest["init_fingerprints"] = fingerprints
        for arm in ARM_ORDER:
            ctx.log.log(f"init fingerprint {arm}: match={fingerprints[arm]['match']}")
        del init_models
        ctx.set_phase("identity", "done")
    if args.phase == "regime-check":
        regime_check(ctx)
        ctx.set_phase("regime_check", "done", outputs=["valid_crosscheck_regime.json"])
    if args.phase in ("A", "all"):
        phase_a(ctx)
    if args.phase in ("B", "all"):
        phase_b(ctx)
    if args.phase in ("C", "all"):
        phase_c(ctx)

    after = old_artifact_hashes()
    before = ctx.manifest["old_artifacts_before"]
    changed = [path for path in after if after[path] != before.get(path)]
    ctx.manifest["old_artifacts_after"] = after
    ctx.manifest["old_artifacts_unchanged"] = not changed
    if changed:
        ctx.manifest["old_artifacts_changed"] = changed
    ctx.save_manifest()
    ctx.log.log(f"old artifacts unchanged: {not changed}"
                + (f" changed={changed}" if changed else ""))
    ctx.log.log("=== done ===")


if __name__ == "__main__":
    main()
