"""Element-identity control experiment: pre-training checks and run manifests.

对应设计：``docs/design/design-element-identity-control.md`` 第 4 节。
三种用法（均从仓库根目录运行）：

* 默认：执行全部实施后、训练前检查，并写入两臂实验 manifest；
* ``--verify-run``：训练启动后核对每臂真实 ``config_used.yaml`` 与 manifest；
* ``--finalize``：训练完成后补记两臂 checkpoint 的 SHA-256 与 epoch 身份。

manifest 提供溯源，不能代替检查本身。任何检查失败都抛出异常，不写"通过"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"
DATA_DIR = ROOT / "data/train4ARPAT"
TABLE_CSV = ROOT / "utils/periodic_table_v2.csv"
SPLIT_YAML = ROOT / "index/split_v2.yaml"
DATA_MANIFEST = DATA_DIR / "manifest.json"
B7_CONFIG = OUTPUT / "ablation_m1_e9ctl/config_used.yaml"

# 文档第 2 节记录的指纹；实施时重新计算并核对，不因写有哈希就跳过。
EXPECTED_TABLE_SHA256 = "8cfba286b85f175c73431011a5c24bb8b110223ed60e81bae0357a02af9318fc"
EXPECTED_SPLIT_SHA256 = "8880b49652f72d5b5c5690afcc06ccaf643e3072c4789f58920256df42e43c41"
# 文档第 3 节预注册常数的记录值（约）；实际值由 F[1:119].mean(dim=0) 计算。
DOC_CONST = [0.49465635, 0.39238882, 0.40330893]
CONST_FORMULA = "c = F[1:119].mean(dim=0), F = PeriodicTable().atom_feature_map()"

ARMS = [
    {"arm": "A100", "tag": "_eidprop100_s42", "atom_feat": "legacy3"},
    {"arm": "B100", "tag": "_eidconst100_s42", "atom_feat": "legacy3_const"},
]

PLANNED_CLI = {
    "model": "M1", "epochs": 100, "batch_size": 32, "lr": 5e-5, "seed": 42,
    "norm": "sumnorm", "scale_mode": "eta", "skip_test_eval": True,
    "init_ckpt": "", "use_amp": False, "pair_aux_arm": "none", "pair_ratio": 0.0,
    "data_dir": "./data/train4ARPAT", "edos_num": 128, "phdos_num": 64,
    "decoder_layers": 6, "dropout": 0.05,
}

CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"preflight check failed: {name} | {detail}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def code_version() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    return {"git_commit": commit, "dirty": bool(status)}


def manifest_path(tag: str) -> Path:
    return RESULTS / f"{tag.lstrip('_')}_manifest.json"


def command_for(arm: dict) -> str:
    return (
        "python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 "
        "--batch_size 32 --lr 5e-5 --norm sumnorm --scale_mode eta "
        f"--atom_feat {arm['atom_feat']} --skip_test_eval --tag {arm['tag']}"
    )


def write_manifest(tag: str, payload: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = manifest_path(tag)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    print(f"manifest written: {path.relative_to(ROOT)}")


def load_manifest(tag: str) -> dict:
    path = manifest_path(tag)
    if not path.exists():
        raise FileNotFoundError(f"missing manifest {path}; run the preflight first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 检查 1：两个新模式的 feature_map 与预注册常数
# ---------------------------------------------------------------------------

def check_feature_maps() -> dict:
    from utils.atom_feature import AtomFeatureEncoder, PeriodicTable, legacy3_const_vector

    table = PeriodicTable().atom_feature_map()
    enc_a = AtomFeatureEncoder(3, 512, feat="legacy3")
    enc_b = AtomFeatureEncoder(3, 512, feat="legacy3_const")
    map_a, map_b = enc_a.feature_map, enc_b.feature_map

    record("feature_map shapes are [119,3]",
           tuple(map_a.shape) == (119, 3) and tuple(map_b.shape) == (119, 3),
           f"A={tuple(map_a.shape)} B={tuple(map_b.shape)}")
    record("A feature_map equals current legacy3 table bit-for-bit",
           bool(torch.equal(map_a, table)))
    c = legacy3_const_vector()
    record("c equals F[1:119].mean(dim=0) formula",
           bool(torch.equal(c, table[1:119].mean(dim=0))),
           f"c={c.tolist()}")
    record("c close to documented preregistered values",
           bool(np.allclose(c.numpy(), DOC_CONST, atol=1e-6)),
           f"documented={DOC_CONST}")
    record("B row 0 preserved (zeros)", bool(torch.equal(map_b[0], table[0])),
           f"B[0]={map_b[0].tolist()}")
    rows_ok = bool(torch.equal(map_b[1:], c.expand_as(map_b[1:])))
    record("B rows 1..118 identical to c", rows_ok)
    record("A rows keep element-to-element variation",
           bool((map_a[1:] != map_a[1:2]).any()))
    return {
        "table_sha256": sha256_file(TABLE_CSV),
        "feature_map_sha256": {"legacy3": sha256_tensor(map_a), "legacy3_const": sha256_tensor(map_b)},
        "const_vector_formula": CONST_FORMULA,
        "const_vector_float32": [float(x) for x in c.numpy()],
        "const_vector_documented": DOC_CONST,
    }


# ---------------------------------------------------------------------------
# 检查 2/3：初始权重、参数量、正反向、可复现、数据顺序
# ---------------------------------------------------------------------------

def build_transformer_params(atom_feat_mode: str) -> dict:
    """以冻结 B7 配方的 transformer 参数为参照，只替换性质模式。"""
    with B7_CONFIG.open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream)["config"]["model"]["params"]["sub_model"]["transformer"]
    params = dict(params)
    params["atom_feat_mode"] = atom_feat_mode
    expected = {"edos_num": 128, "phdos_num": 64, "num_decoder_layers": 6,
                "scale_mode": "eta", "dropout": 0.05, "head_type": "legacy"}
    for key, value in expected.items():
        if params.get(key) != value:
            raise AssertionError(f"B7 reference params drifted: {key}={params.get(key)!r}")
    for key in ("decoupled_decoder", "predict_scale", "use_periodic_manybody"):
        if bool(params.get(key)):
            raise AssertionError(f"B7 reference params drifted: {key}={params.get(key)!r}")
    return params


def build_transformer(atom_feat_mode: str, seed: int = 42) -> torch.nn.Module:
    from model.transformer import Transformer
    from run_ablation_experiments import setup_ablation_seed

    setup_ablation_seed(seed)
    model = Transformer(**build_transformer_params(atom_feat_mode))
    return model


def same_state(left: torch.nn.Module, right: torch.nn.Module) -> tuple[bool, str]:
    ld, rd = left.state_dict(), right.state_dict()
    if list(ld.keys()) != list(rd.keys()):
        return False, "state_dict keys differ"
    if "num_emb_encoder.feature_map" in ld:
        return False, "feature_map unexpectedly entered state_dict"
    for key in ld:
        if ld[key].shape != rd[key].shape:
            return False, f"shape mismatch at {key}"
        if not torch.equal(ld[key], rd[key]):
            return False, f"value mismatch at {key}"
    return True, "keys, shapes and values identical"


def check_initialization_and_forward() -> dict:
    model_a = build_transformer("legacy3")
    model_b = build_transformer("legacy3_const")
    ok, detail = same_state(model_a, model_b)
    record("same-seed initial state_dict identical across arms", ok, detail)

    def trainable_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    record("trainable parameter totals identical",
           trainable_count(model_a) == trainable_count(model_b),
           f"{trainable_count(model_a):,}")
    record("tok_emb not zero-initialized under new modes",
           float(model_a.tok_emb.weight.abs().sum()) > 0
           and float(model_b.tok_emb.weight.abs().sum()) > 0)
    record("mendeleev24 zero-init branch not triggered",
           model_a.atom_feat_mode != "mendeleev24" and model_b.atom_feat_mode != "mendeleev24",
           f"A mode={model_a.atom_feat_mode}, B mode={model_b.atom_feat_mode}")

    model_a2 = build_transformer("legacy3")
    model_b2 = build_transformer("legacy3_const")
    ok_a, det_a = same_state(model_a, model_a2)
    ok_b, det_b = same_state(model_b, model_b2)
    record("same-arm re-initialization reproducible", ok_a and ok_b, f"A: {det_a}; B: {det_b}")

    # 正反向数值有限 + M1 双谱输出与 H1 头形状。
    from torch.utils.data import DataLoader
    from datasets.dataset import Dos_Dataset

    dataset = Dos_Dataset(data_dir=str(DATA_DIR), split="valid", dos_minmax=True, dos_sumnorm=True)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    src = batch[0]
    pos = batch[1]
    edos_x, phdos_x = batch[15], batch[16]
    mask = src.eq(0)
    for name, model in (("A100", model_a), ("B100", model_b)):
        model.eval()
        out = model(src, mask, pos, edos_x, phdos_x)
        shapes = {key: tuple(out[key].shape) for key in ("edos", "phdos", "eta")}
        finite = all(bool(torch.isfinite(out[key]).all()) for key in ("edos", "phdos", "eta"))
        record(f"{name} forward shapes edos[4,128]/phdos[4,64]/eta[4,2] and finite",
               shapes == {"edos": (4, 128), "phdos": (4, 64), "eta": (4, 2)} and finite,
               f"shapes={shapes}")
        model.train()
        loss = out["edos"].sum() + out["phdos"].sum() + out["eta"].sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        record(f"{name} backward gradients finite",
               bool(torch.isfinite(loss)) and grads and all(bool(torch.isfinite(g).all()) for g in grads),
               f"{len(grads)} gradient tensors")
    return {"trainable_params": trainable_count(model_a)}


def check_sample_order() -> dict:
    """两臂使用同一数据管线；核对 train 样本 ID 顺序逐 epoch 一致。"""
    from utils.builder import ConfigBuilder

    orders = {}
    for arm in ARMS:
        with open(ROOT / "configs/default.yaml", encoding="utf-8") as stream:
            yaml_cfg = yaml.load(stream, Loader=yaml.FullLoader)
        for split in ("train", "valid", "test"):
            yaml_cfg["dataset"][split]["data_dir"] = str(DATA_DIR)
        yaml_cfg["model"]["params"]["sub_model"]["transformer"]["atom_feat_mode"] = arm["atom_feat"]
        builder = ConfigBuilder(**yaml_cfg)
        loader = builder.get_dataloader(split="train", dos_minmax=True, batch_size=32,
                                        dos_sumnorm=True, use_bucket_batch=False)
        sampler = loader.sampler
        per_epoch = []
        for epoch in (0, 1):
            sampler.set_epoch(epoch)
            per_epoch.append(list(sampler))
        orders[arm["arm"]] = per_epoch

    same = orders["A100"] == orders["B100"]
    record("train sample index order identical across arms (epochs 0-1)", same)
    ids = np.load(DATA_DIR / "train/train_index.npy")
    mapped = [ids[idx] for idx in orders["A100"][0]]
    record("sampler indices map to Q1 train IDs consistently",
           len(mapped) == len(ids) == 18706, f"train n={len(ids)}")
    return {"train_samples": int(len(ids)), "epochs_checked": [0, 1]}


# ---------------------------------------------------------------------------
# 检查 4：目标 tag 空闲
# ---------------------------------------------------------------------------

def check_tags_free() -> None:
    for arm in ARMS:
        tag = arm["tag"]
        save_dir = OUTPUT / f"ablation_m1{tag}"
        occupied = [name for name in ("checkpoint_latest.pth", "checkpoint_best.pth", "config_used.yaml")
                    if (save_dir / name).exists()]
        record(f"{arm['arm']} output tag free of checkpoints/config",
               not occupied, f"{save_dir.relative_to(ROOT)}: {occupied or 'empty'}")
        hist = RESULTS / f"history_m1{tag}.csv"
        record(f"{arm['arm']} history file absent", not hist.exists(), str(hist.name))
        summary = RESULTS / f"test_m1{tag}_summary.csv"
        record(f"{arm['arm']} test summary absent", not summary.exists(), str(summary.name))


# ---------------------------------------------------------------------------
# 检查 5 + 溯源：哈希与 manifest
# ---------------------------------------------------------------------------

def check_hashes() -> dict:
    table_sha = sha256_file(TABLE_CSV)
    split_sha = sha256_file(SPLIT_YAML)
    record("periodic_table_v2.csv SHA-256 matches design document",
           table_sha == EXPECTED_TABLE_SHA256, table_sha)
    record("split_v2.yaml SHA-256 matches design document",
           split_sha == EXPECTED_SPLIT_SHA256, split_sha)
    return {"table_sha256": table_sha, "split_sha256": split_sha,
            "data_manifest_sha256": sha256_file(DATA_MANIFEST)}


def build_manifest(arm: dict, fingerprint: dict, hashes: dict, init_info: dict,
                   order_info: dict) -> dict:
    return {
        "arm": arm["arm"],
        "tag": arm["tag"],
        "atom_feat_mode": arm["atom_feat"],
        "command": command_for(arm),
        "planned_cli": dict(PLANNED_CLI, atom_feat=arm["atom_feat"], tag=arm["tag"]),
        "planned_cli_sha256": hashlib.sha256(
            json.dumps(dict(PLANNED_CLI, atom_feat=arm["atom_feat"], tag=arm["tag"]),
                       sort_keys=True).encode()).hexdigest(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_version": code_version(),
        "hashes": hashes,
        "fingerprint": fingerprint,
        "initialization": init_info,
        "data_order": order_info,
        "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHECKS],
        "config_used_sha256": None,
        "checkpoints": {"checkpoint_best.pth": {"sha256": None, "epoch": None, "best_val_score": None},
                        "checkpoint_latest.pth": {"sha256": None, "epoch": None, "best_val_score": None}},
        "notes": "feature_map 不进入 state_dict；相同键和 strict=True 加载不能证明臂身份，"
                 "判读前必须核对本 manifest 的模式、表指纹与 checkpoint 哈希。",
    }


# ---------------------------------------------------------------------------
# --verify-run / --finalize
# ---------------------------------------------------------------------------

def verify_run(only: str | None = None) -> None:
    for arm in ARMS:
        if only and arm["arm"] != only:
            continue
        tag = arm["tag"]
        manifest = load_manifest(tag)
        config_path = OUTPUT / f"ablation_m1{tag}/config_used.yaml"
        if not config_path.exists():
            record(f"{arm['arm']} real run config exists", False, str(config_path))
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        cli = config["cli"]
        expected = {"model": "M1", "epochs": 100, "batch_size": 32, "lr": 5e-5, "seed": 42,
                    "norm": "sumnorm", "scale_mode": "eta", "atom_feat": arm["atom_feat"],
                    "skip_test_eval": True, "init_ckpt": "", "pair_aux_arm": "none"}
        bad = {k: (cli.get(k), v) for k, v in expected.items() if cli.get(k) != v}
        record(f"{arm['arm']} real config CLI matches frozen recipe", not bad, f"mismatch={bad}")
        record(f"{arm['arm']} real config use_amp disabled",
               not cli.get("use_amp"), f"use_amp={cli.get('use_amp')!r}")
        params = config["config"]["model"]["params"]
        tparams = params["sub_model"]["transformer"]
        record(f"{arm['arm']} real config effective atom_feat_mode",
               tparams.get("atom_feat_mode") == arm["atom_feat"],
               f"atom_feat_mode={tparams.get('atom_feat_mode')!r}")
        record(f"{arm['arm']} real config pair_aux disabled in model params",
               str(params.get("pair_aux_arm")) == "none", f"pair_aux_arm={params.get('pair_aux_arm')!r}")
        record(f"{arm['arm']} table hash unchanged since preflight",
               sha256_file(TABLE_CSV) == manifest["hashes"]["table_sha256"])
        record(f"{arm['arm']} split hash unchanged since preflight",
               sha256_file(SPLIT_YAML) == manifest["hashes"]["split_sha256"])
        record(f"{arm['arm']} data manifest hash unchanged since preflight",
               sha256_file(DATA_MANIFEST) == manifest["hashes"]["data_manifest_sha256"])
        manifest["config_used_sha256"] = sha256_file(config_path)
        manifest["verified_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_manifest(tag, manifest)


def finalize(only: str | None = None) -> None:
    for arm in ARMS:
        if only and arm["arm"] != only:
            continue
        tag = arm["tag"]
        manifest = load_manifest(tag)
        save_dir = OUTPUT / f"ablation_m1{tag}"
        for name in ("checkpoint_best.pth", "checkpoint_latest.pth"):
            path = save_dir / name
            if not path.exists():
                record(f"{arm['arm']} {name} exists for finalization", False, str(path))
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            identity = (ckpt.get("epoch"), ckpt.get("model_name"), ckpt.get("seed"))
            record(f"{arm['arm']} {name} identity (epoch, M1, 42)",
                   identity[1] == "M1" and identity[2] == 42 and identity[0] is not None,
                   f"identity={identity}")
            manifest["checkpoints"][name] = {
                "sha256": sha256_file(path),
                "epoch": int(ckpt.get("epoch")),
                "best_val_score": float(ckpt.get("best_val_score", float("nan"))),
            }
        config_path = save_dir / "config_used.yaml"
        if manifest.get("config_used_sha256") is None:
            manifest["config_used_sha256"] = sha256_file(config_path)
        elif sha256_file(config_path) != manifest["config_used_sha256"]:
            record(f"{arm['arm']} config_used.yaml unchanged since verify-run", False,
                   sha256_file(config_path))
        manifest["finalized_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_manifest(tag, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-run", action="store_true",
                        help="check each arm's real config_used.yaml and record its hash")
    parser.add_argument("--finalize", action="store_true",
                        help="record checkpoint SHA-256 and identity after training")
    parser.add_argument("--arm", choices=["A100", "B100"], default=None,
                        help="limit --verify-run/--finalize to one arm")
    args = parser.parse_args()

    if args.verify_run:
        verify_run(args.arm)
        return
    if args.finalize:
        finalize(args.arm)
        return

    print(f"code: {code_version()}")
    fingerprint = check_feature_maps()
    init_info = check_initialization_and_forward()
    order_info = check_sample_order()
    check_tags_free()
    hashes = check_hashes()
    for arm in ARMS:
        write_manifest(arm["tag"], build_manifest(arm, fingerprint, hashes, init_info, order_info))
    print(f"preflight: {sum(1 for _, ok, _ in CHECKS if ok)}/{len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
