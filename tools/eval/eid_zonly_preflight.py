"""Z100 (z_only) 单种子临时 baseline：实施后、训练前检查与实验 manifest。

对应设计：``docs/design/design-element-identity-zonly.md`` 第 4 节。
三种用法（均从仓库根目录运行）：

* 默认：执行全部实施后、训练前检查，并写入 Z100 实验 manifest
  ``results/eidzonly100_s42_manifest.json``（独立文件名，不触碰 A100/B100 manifest）；
* ``--verify-run``：训练启动后核对真实 ``config_used.yaml`` 与 manifest；
* ``--finalize``：训练完成后补记 checkpoint SHA-256 与 epoch 身份。

manifest 提供溯源，不能代替检查本身。任何检查失败都抛出异常，不写"通过"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import element_identity_preflight as base  # noqa: E402  复用哈希/版本/构造辅助

RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"
DATA_DIR = ROOT / "data/train4ARPAT"

ARM = {"arm": "Z100", "tag": "_eidzonly100_s42", "atom_feat": "z_only"}
REF_A = {"arm": "A100", "tag": "_eidprop100_s42", "atom_feat": "legacy3"}
REF_B = {"arm": "B100", "tag": "_eidconst100_s42", "atom_feat": "legacy3_const"}

# 参数量冻结预期：A100 可训练参数（A100/B100 前置检查实测）与删除模块的精确参数量。
EXPECTED_A100_PARAMS = 71_152_964
EXPECTED_DELTA_PARAMS = 527_872          # Linear(3,512)=2048 + LayerNorm(512)=1024 + Linear(1024,512)=524,800
EXPECTED_Z100_PARAMS = EXPECTED_A100_PARAMS - EXPECTED_DELTA_PARAMS  # 70,625,092

# z_only 编辑前（2026-09-30）用 /tmp/opencode/snapshot_init_hashes.py 记录的
# seed-42 初始 state_dict 哈希；改后重算须逐值一致，证明旧路径未被改变。
PRECHANGE_INIT_STATE_SHA256 = {
    "legacy3": "efca7088573a371fefe589a144ea66694273192498c9e7c15f2d111b717e06e0",
    "legacy3_const": "efca7088573a371fefe589a144ea66694273192498c9e7c15f2d111b717e06e0",
    "mendeleev24": "55a4776433a574b2bc634ff4b7e82e4d7d998831320ce09f2a202fe18452ce96",
}

PLANNED_CLI = dict(base.PLANNED_CLI, atom_feat="z_only", tag="_eidzonly100_s42")

TRAINING_CODE_FILES = [
    "model/transformer.py",
    "utils/atom_feature.py",
    "run_ablation_experiments.py",
    "utils/experiment_config.py",
    "utils/builder.py",
    "datasets/dataset.py",
    "model/model.py",
]

CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"preflight check failed: {name} | {detail}")


def manifest_path() -> Path:
    return RESULTS / "eidzonly100_s42_manifest.json"


def command_for() -> str:
    return (
        "python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 "
        "--batch_size 32 --lr 5e-5 --norm sumnorm --scale_mode eta "
        "--atom_feat z_only --skip_test_eval --tag _eidzonly100_s42"
    )


def state_hash(model: torch.nn.Module) -> str:
    """与改动前快照脚本完全一致的哈希口径（按 key 排序：key bytes + 张量字节）。"""
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def write_manifest(payload: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = manifest_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    print(f"manifest written: {path.relative_to(ROOT)}")


def load_manifest() -> dict:
    path = manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"missing manifest {path}; run the preflight first")
    return json.loads(path.read_text(encoding="utf-8"))


def build_z100(seed: int = 42) -> torch.nn.Module:
    from model.transformer import Transformer
    from run_ablation_experiments import setup_ablation_seed

    setup_ablation_seed(seed)
    return Transformer(**base.build_transformer_params("z_only"))


def build_old_mode(mode: str, seed: int = 42) -> torch.nn.Module:
    from model.transformer import Transformer
    from run_ablation_experiments import setup_ablation_seed

    setup_ablation_seed(seed)
    return Transformer(**base.build_transformer_params(mode))


def valid_batch(batch_size: int = 4):
    from torch.utils.data import DataLoader
    from datasets.dataset import Dos_Dataset

    dataset = Dos_Dataset(data_dir=str(DATA_DIR), split="valid", dos_minmax=True, dos_sumnorm=True)
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
    return dataset, batch


# ---------------------------------------------------------------------------
# 检查 1：z_only 模块结构、atom_src 公式、无性质表读取
# ---------------------------------------------------------------------------

def check_structure_and_formula() -> dict:
    model = build_z100()
    record("z_only does not instantiate num_emb_encoder/num_norm/fuse_proj",
           model.num_emb_encoder is None and model.num_norm is None and model.fuse_proj is None,
           f"types={type(model.num_emb_encoder).__name__},{type(model.num_norm).__name__},"
           f"{type(model.fuse_proj).__name__}")
    state = model.state_dict()
    forbidden = [k for k in state if ("num_emb_encoder" in k or "num_norm" in k or "fuse_proj" in k)]
    record("z_only state_dict free of numeric-branch/fusion keys", not forbidden, f"{forbidden[:5]}")
    record("z_only keeps tok_emb/atom_norm and geometry encoder",
           "tok_emb.weight" in state and "atom_norm.weight" in state
           and any(k.startswith("encoder.") for k in state)
           and any(k.startswith("decoder.") for k in state)
           and "eta_head.mlp.0.weight" in state)

    legacy = build_old_mode("legacy3")
    record("legacy3 path still instantiates numeric branch and fusion",
           isinstance(legacy.num_emb_encoder, torch.nn.Module)
           and isinstance(legacy.num_norm, torch.nn.Module)
           and isinstance(legacy.fuse_proj, torch.nn.Module))

    # forward 中 encoder 实际输入逐值等于 atom_norm(tok_emb(atom_idx))。
    _, batch = valid_batch()
    src, pos = batch[0], batch[1]
    mask = src.eq(0)
    captured = {}

    def _capture(_module, args, kwargs):
        captured["src"] = kwargs.get("src", args[0] if args else None)
        return None

    handle = model.encoder.register_forward_pre_hook(_capture, with_kwargs=True)
    model.eval()
    with torch.no_grad():
        out = model(src, mask, pos, batch[15], batch[16])
    handle.remove()
    expected = model.atom_norm(model.tok_emb(src[:, 2:]))
    record("encoder input equals atom_norm(tok_emb(atom_idx)) bit-for-bit",
           bool(torch.equal(captured["src"], expected)))
    shapes = {key: tuple(out[key].shape) for key in ("edos", "phdos", "eta")}
    record("z_only forward shapes edos[4,128]/phdos[4,64]/eta[4,2]",
           shapes == {"edos": (4, 128), "phdos": (4, 64), "eta": (4, 2)}, f"shapes={shapes}")

    # z_only 构造与前向不经过任何性质表读取路径。
    import utils.atom_feature as af

    def _boom(*_args, **_kwargs):
        raise AssertionError("property table read attempted under z_only")

    saved = (af.PeriodicTable.atom_feature_map, af.Mendeleev24.atom_feature_map)
    af.PeriodicTable.atom_feature_map = _boom
    af.Mendeleev24.atom_feature_map = _boom
    try:
        probe = build_z100()
        probe.eval()
        with torch.no_grad():
            probe(src, mask, pos, batch[15], batch[16])
        record("z_only construction and forward never read the property tables", True)
    finally:
        af.PeriodicTable.atom_feature_map, af.Mendeleev24.atom_feature_map = saved
    try:
        af.AtomFeatureEncoder(3, 512, feat="z_only")
    except ValueError as exc:
        record("AtomFeatureEncoder explicitly rejects z_only", True, str(exc))
    else:
        record("AtomFeatureEncoder explicitly rejects z_only", False, "no exception raised")
    return {"state_hash_z_only_init": state_hash(model)}


# ---------------------------------------------------------------------------
# 检查 2：参数量、RNG 消耗事实、同模式可复现、正反向数值
# ---------------------------------------------------------------------------

def check_params_and_rng() -> dict:
    from run_ablation_experiments import setup_ablation_seed

    model_z = build_z100()
    model_a = build_old_mode("legacy3")

    def trainable(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    a_params, z_params = trainable(model_a), trainable(model_z)
    record("A100-mode trainable params match the recorded 71,152,964",
           a_params == EXPECTED_A100_PARAMS, f"{a_params:,}")
    record("Z100 trainable params equal 71,152,964 - 527,872",
           z_params == EXPECTED_Z100_PARAMS, f"{z_params:,}")
    record("trainable parameter reduction is exactly 527,872",
           a_params - z_params == EXPECTED_DELTA_PARAMS, f"delta={a_params - z_params:,}")

    # RNG 消耗事实（信息性，但观察必须真实成立）：
    # 删除数值分支改变初始化随机数消耗；_reset_parameters 的 xavier 扫描按模块
    # 创建顺序对所有 dim>1 参数（含 tok_emb）赋值，抽签落点随之移动。因此同 seed
    # 下 Z100 与 A100 的共享参数初值也不逐值相同，包括 tok_emb 与 encoder。
    shared_keys = ("tok_emb.weight", "encoder.layers.0.linear1.weight")
    zs, as_ = model_z.state_dict(), model_a.state_dict()
    diffs = [k for k in shared_keys if not torch.equal(zs[k], as_[k])]
    record("RNG fact: shared params (incl. tok_emb) differ between z_only and legacy3 "
           "under the same seed", set(diffs) == set(shared_keys), f"differing={diffs}")

    model_z2 = build_z100()
    ok, detail = base.same_state(model_z, model_z2)
    record("same-mode (z_only) re-initialization reproducible", ok, detail)
    setup_ablation_seed(42)
    return {"a100_trainable_params": a_params, "z100_trainable_params": z_params,
            "delta_params": a_params - z_params,
            "rng_note": "删除数值分支改变初始化随机数消耗；_reset_parameters 的 xavier 扫描按模块创建顺序赋值，"
                        "抽签落点随之移动。同 seed 下 Z100 与 A100 不共享参数初值逐值相同（含 tok_emb 与 encoder）；"
                        "同模式重初始化可复现（上一项检查已实测）。"}


def check_forward_backward() -> dict:
    _, batch = valid_batch()
    src, pos = batch[0], batch[1]
    model = build_z100()
    model.train()
    out = model(src, src.eq(0), pos, batch[15], batch[16])
    finite_fwd = all(bool(torch.isfinite(out[key]).all()) for key in ("edos", "phdos", "eta"))
    loss = out["edos"].sum() + out["phdos"].sum() + out["eta"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite_bwd = bool(torch.isfinite(loss)) and grads and all(
        bool(torch.isfinite(g).all()) for g in grads)
    record("forward outputs finite", finite_fwd,
           f"min|edos|={out['edos'].abs().min().item():.3e} max|edos|={out['edos'].abs().max().item():.3e}")
    record("backward gradients finite and non-empty", bool(finite_bwd), f"{len(grads)} grad tensors")
    gmin = min(float(g.abs().min()) for g in grads)
    gmax = max(float(g.abs().max()) for g in grads)
    return {"grad_abs_min": gmin, "grad_abs_max": gmax, "n_grad_tensors": len(grads)}


# ---------------------------------------------------------------------------
# 检查 3：旧模式行为与 checkpoint 加载兼容性
# ---------------------------------------------------------------------------

def check_legacy_compat() -> dict:
    for mode in ("legacy3", "legacy3_const", "mendeleev24"):
        digest = state_hash(build_old_mode(mode))
        record(f"{mode} seed-42 initial state_dict hash equals pre-change snapshot",
               digest == PRECHANGE_INIT_STATE_SHA256[mode],
               f"now={digest[:16]}… pre={PRECHANGE_INIT_STATE_SHA256[mode][:16]}…")
    m24 = build_old_mode("mendeleev24")
    record("mendeleev24 tok_emb still zero-initialized",
           float(m24.tok_emb.weight.abs().sum()) == 0.0)

    loaded = {}
    for arm, mode in ((REF_A, "legacy3"), (REF_B, "legacy3_const")):
        save_dir = OUTPUT / f"ablation_m1{arm['tag']}"
        manifest = json.loads(
            (RESULTS / f"{arm['tag'].lstrip('_')}_manifest.json").read_text(encoding="utf-8"))
        ckpt_path = save_dir / "checkpoint_best.pth"
        if manifest["checkpoints"]["checkpoint_best.pth"]["sha256"] != base.sha256_file(ckpt_path):
            raise ValueError(f"{arm['arm']} checkpoint hash mismatch against its manifest")
        with (save_dir / "config_used.yaml").open(encoding="utf-8") as stream:
            params = yaml.safe_load(stream)["config"]["model"]["params"]["sub_model"]["transformer"]
        model = Transformer_from_params(params, mode)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"], strict=True)
        record(f"{arm['arm']} checkpoint strict-loads into {mode} model", True,
               f"epoch={ckpt.get('epoch')}")
        loaded[arm["arm"]] = {"checkpoint_sha256": base.sha256_file(ckpt_path),
                             "config_used_sha256": base.sha256_file(save_dir / "config_used.yaml"),
                             "epoch": int(ckpt.get("epoch"))}
    try:
        build_z100().load_state_dict(
            torch.load(OUTPUT / f"ablation_m1{REF_A['tag']}/checkpoint_best.pth",
                       map_location="cpu", weights_only=True)["model"], strict=True)
    except Exception as exc:
        record("z_only rejects legacy3 checkpoint weights (expected failure)", True,
               type(exc).__name__)
    else:
        record("z_only rejects legacy3 checkpoint weights (expected failure)", False,
               "strict load unexpectedly succeeded")
    return {"reference_arms": loaded}


def Transformer_from_params(params: dict, mode: str):
    from model.transformer import Transformer

    params = dict(params)
    params["atom_feat_mode"] = mode
    return Transformer(**params)


# ---------------------------------------------------------------------------
# 检查 4：数据/划分/网格/配方与 A100 一致；tag 空闲
# ---------------------------------------------------------------------------

def check_data_and_recipe() -> dict:
    hashes = {
        "table_sha256": base.sha256_file(ROOT / "utils/periodic_table_v2.csv"),
        "split_sha256": base.sha256_file(ROOT / "index/split_v2.yaml"),
        "data_manifest_sha256": base.sha256_file(DATA_DIR / "manifest.json"),
    }
    ref_manifest = json.loads(
        (RESULTS / f"{REF_A['tag'].lstrip('_')}_manifest.json").read_text(encoding="utf-8"))
    record("periodic table / split / data manifest hashes equal the A100 manifest record",
           all(hashes[k] == ref_manifest["hashes"][k] for k in hashes),
           json.dumps({k: v[:16] + "…" for k, v in hashes.items()}))
    record("table/split hashes equal the design-document constants",
           hashes["table_sha256"] == base.EXPECTED_TABLE_SHA256
           and hashes["split_sha256"] == base.EXPECTED_SPLIT_SHA256)

    n_train = int(np.load(DATA_DIR / "train/train_index.npy").shape[0])
    n_valid = int(np.load(DATA_DIR / "valid/valid_index.npy").shape[0])
    record("Q1 train/valid counts are 18,706 / 2,313",
           (n_train, n_valid) == (18706, 2313), f"train={n_train} valid={n_valid}")

    with (OUTPUT / f"ablation_m1{REF_A['tag']}/config_used.yaml").open(encoding="utf-8") as stream:
        ref_cfg = yaml.safe_load(stream)
    tparams = ref_cfg["config"]["model"]["params"]["sub_model"]["transformer"]
    record("A100 grid and dropout are 128/64 and 0.05",
           tparams.get("edos_num") == 128 and tparams.get("phdos_num") == 64
           and float(tparams.get("dropout")) == 0.05,
           f"edos={tparams.get('edos_num')} phdos={tparams.get('phdos_num')} "
           f"dropout={tparams.get('dropout')}")
    cli = ref_cfg["cli"]
    # dropout 在 cli 中允许为 None（模板值 0.05 生效），单独核对。
    bad = {k: (cli.get(k), PLANNED_CLI[k]) for k in PLANNED_CLI
           if k not in ("atom_feat", "tag", "dropout") and cli.get(k) != PLANNED_CLI[k]}
    record("planned CLI equals the A100 frozen recipe except atom_feat/tag/dropout", not bad,
           f"mismatch={bad}")
    record("planned dropout resolves to the 0.05 template value",
           cli.get("dropout") in (None, 0.05), f"A100 cli dropout={cli.get('dropout')!r}")
    record("planned run is FP32 with skip_test_eval, empty init_ckpt, pair_aux off",
           PLANNED_CLI["use_amp"] is False and PLANNED_CLI["skip_test_eval"] is True
           and PLANNED_CLI["init_ckpt"] == "" and PLANNED_CLI["pair_aux_arm"] == "none")
    return hashes


def check_sample_order() -> dict:
    from utils.builder import ConfigBuilder

    orders = {}
    for mode in ("z_only", "legacy3"):
        with open(ROOT / "configs/default.yaml", encoding="utf-8") as stream:
            yaml_cfg = yaml.load(stream, Loader=yaml.FullLoader)
        for split in ("train", "valid", "test"):
            yaml_cfg["dataset"][split]["data_dir"] = str(DATA_DIR)
        yaml_cfg["model"]["params"]["sub_model"]["transformer"]["atom_feat_mode"] = mode
        builder = ConfigBuilder(**yaml_cfg)
        loader = builder.get_dataloader(split="train", dos_minmax=True, batch_size=32,
                                        dos_sumnorm=True, use_bucket_batch=False)
        sampler = loader.sampler
        per_epoch = []
        for epoch in (0, 1):
            sampler.set_epoch(epoch)
            per_epoch.append(list(sampler))
        orders[mode] = per_epoch
    same = orders["z_only"] == orders["legacy3"]
    record("train sample index order identical to the legacy3 pipeline (epochs 0-1)", same)
    ids = np.load(DATA_DIR / "train/train_index.npy")
    record("sampler covers all 18,706 Q1 train samples",
           len(orders["z_only"][0]) == len(ids) == 18706, f"n={len(ids)}")
    return {"train_samples": int(len(ids)), "epochs_checked": [0, 1]}


def check_tags_free() -> None:
    tag = ARM["tag"]
    save_dir = OUTPUT / f"ablation_m1{tag}"
    occupied = [name for name in ("checkpoint_latest.pth", "checkpoint_best.pth", "config_used.yaml")
                if (save_dir / name).exists()]
    record("Z100 output tag free of checkpoints/config (no auto-resume)",
           not occupied, f"{save_dir.relative_to(ROOT)}: {occupied or 'empty'}")
    for path in (RESULTS / f"history_m1{tag}.csv",
                 RESULTS / f"test_m1{tag}_summary.csv",
                 RESULTS / "train_m1_eidzonly100_s42.log",
                 manifest_path()):
        record(f"Z100 artifact absent before launch: {path.name}", not path.exists())


# ---------------------------------------------------------------------------
# --verify-run / --finalize
# ---------------------------------------------------------------------------

def verify_run() -> None:
    manifest = load_manifest()
    config_path = OUTPUT / f"ablation_m1{ARM['tag']}/config_used.yaml"
    if not config_path.exists():
        record("Z100 real run config exists", False, str(config_path))
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    cli = config["cli"]
    expected = {"model": "M1", "epochs": 100, "batch_size": 32, "lr": 5e-5, "seed": 42,
                "norm": "sumnorm", "scale_mode": "eta", "atom_feat": "z_only",
                "skip_test_eval": True, "init_ckpt": "", "pair_aux_arm": "none"}
    bad = {k: (cli.get(k), v) for k, v in expected.items() if cli.get(k) != v}
    record("Z100 real config CLI matches the frozen recipe", not bad, f"mismatch={bad}")
    record("Z100 real config FP32 (use_amp disabled)", not cli.get("use_amp"),
           f"use_amp={cli.get('use_amp')!r}")
    record("Z100 real config dropout is 0.05",
           cli.get("dropout") in (None, 0.05), f"dropout={cli.get('dropout')!r}")
    params = config["config"]["model"]["params"]
    tparams = params["sub_model"]["transformer"]
    record("Z100 real config effective atom_feat_mode is z_only",
           tparams.get("atom_feat_mode") == "z_only",
           f"atom_feat_mode={tparams.get('atom_feat_mode')!r}")
    record("Z100 real config pair_aux disabled in model params",
           str(params.get("pair_aux_arm")) == "none", f"pair_aux_arm={params.get('pair_aux_arm')!r}")
    for key, name in (("table_sha256", "utils/periodic_table_v2.csv"),
                      ("split_sha256", "index/split_v2.yaml"),
                      ("data_manifest_sha256", "data/train4ARPAT/manifest.json")):
        record(f"Z100 {name} hash unchanged since preflight",
               base.sha256_file(ROOT / name) == manifest["hashes"][key])
    manifest["config_used_sha256"] = base.sha256_file(config_path)
    manifest["verified_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_manifest(manifest)


def finalize() -> None:
    manifest = load_manifest()
    save_dir = OUTPUT / f"ablation_m1{ARM['tag']}"
    for name in ("checkpoint_best.pth", "checkpoint_latest.pth"):
        path = save_dir / name
        if not path.exists():
            record(f"Z100 {name} exists for finalization", False, str(path))
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        identity = (ckpt.get("epoch"), ckpt.get("model_name"), ckpt.get("seed"))
        record(f"Z100 {name} identity (epoch, M1, 42)",
               identity[1] == "M1" and identity[2] == 42 and identity[0] is not None,
               f"identity={identity}")
        manifest["checkpoints"][name] = {
            "sha256": base.sha256_file(path),
            "epoch": int(ckpt.get("epoch")),
            "best_val_score": float(ckpt.get("best_val_score", float("nan"))),
        }
    config_path = save_dir / "config_used.yaml"
    if manifest.get("config_used_sha256") is None:
        manifest["config_used_sha256"] = base.sha256_file(config_path)
    elif base.sha256_file(config_path) != manifest["config_used_sha256"]:
        record("Z100 config_used.yaml unchanged since verify-run", False,
               base.sha256_file(config_path))
    manifest["finalized_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_manifest(manifest)


def build_manifest(fingerprint: dict, hashes: dict, init_info: dict, order_info: dict,
                   legacy_info: dict) -> dict:
    return {
        "arm": ARM["arm"],
        "tag": ARM["tag"],
        "atom_feat_mode": "z_only",
        "atom_src_formula": "atom_src = atom_norm(tok_emb(atom_idx))",
        "command": command_for(),
        "planned_cli": PLANNED_CLI,
        "planned_cli_sha256": hashlib.sha256(
            json.dumps(PLANNED_CLI, sort_keys=True).encode()).hexdigest(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_version": base.code_version(),
        "training_code_sha256": {name: base.sha256_file(ROOT / name)
                                 for name in TRAINING_CODE_FILES},
        "hashes": hashes,
        "fingerprint": fingerprint,
        "initialization": init_info,
        "data_order": order_info,
        "legacy_compat": legacy_info,
        "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHECKS],
        "config_used_sha256": None,
        "checkpoints": {"checkpoint_best.pth": {"sha256": None, "epoch": None, "best_val_score": None},
                        "checkpoint_latest.pth": {"sha256": None, "epoch": None, "best_val_score": None}},
        "notes": "z_only 不实例化数值分支；state_dict 不含 num_emb_encoder/num_norm/fuse_proj，"
                 "但删除模块改变随机数消耗，同 seed 与 A100 不共享参数初值逐值相同。"
                 "判读前必须核对本 manifest 的模式、指纹、配置哈希与 checkpoint 哈希。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-run", action="store_true",
                        help="check the real config_used.yaml and record its hash")
    parser.add_argument("--finalize", action="store_true",
                        help="record checkpoint SHA-256 and identity after training")
    args = parser.parse_args()

    if args.verify_run:
        verify_run()
        return
    if args.finalize:
        finalize()
        return

    if manifest_path().exists():
        raise FileExistsError(f"{manifest_path()} already exists; refusing to overwrite")

    print(f"code: {base.code_version()}")
    fingerprint = check_structure_and_formula()
    init_info = check_params_and_rng()
    init_info.update(check_forward_backward())
    legacy_info = check_legacy_compat()
    hashes = check_data_and_recipe()
    order_info = check_sample_order()
    check_tags_free()
    write_manifest(build_manifest(fingerprint, hashes, init_info, order_info, legacy_info))
    print(f"preflight: {sum(1 for _, ok, _ in CHECKS if ok)}/{len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
