"""ZP100 (z_only_proj) 对齐初始化候选：实施后、训练前检查与实验 manifest。

对应设计：``docs/design/design-element-identity-zonlyproj.md`` 第 4 节。
三种用法（均从仓库根目录运行）：

* 默认：执行全部实施后、训练前检查，并写入 ZP100 实验 manifest
  ``results/eidzproj100_s42_manifest.json``（独立文件名，不触碰 A100/B100/Z100 manifest）；
* ``--verify-run``：训练启动后核对真实 ``config_used.yaml``、对齐记录 ``zproj_align.json``
  与 manifest，并登记训练进程 PID；
* ``--finalize``：训练完成后补记 checkpoint SHA-256 与 epoch 身份。

manifest 提供溯源，不能代替检查本身。任何检查失败都抛出异常，不写"通过"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import element_identity_preflight as base  # noqa: E402  复用哈希/版本/构造辅助
import eid_zonly_preflight as zref  # noqa: E402  复用 Z100 参考构造与旧模式快照
from utils.zproj_alignment import (  # noqa: E402
    RNG_METHOD_NOTE, Z100_INIT_STATE_SHA256, apply_alignment, prepare_reference,
    restore_training_rng, rng_digest, rng_snapshot, shared_keys, state_hash, valid_batch)

RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"
DATA_DIR = ROOT / "data/train4ARPAT"

ARM = {"arm": "ZP100", "tag": "_eidzproj100_s42", "atom_feat": "z_only_proj"}
REF_Z = {"arm": "Z100", "tag": "_eidzonly100_s42", "atom_feat": "z_only"}
REF_A = {"arm": "A100", "tag": "_eidprop100_s42", "atom_feat": "legacy3"}
REF_B = {"arm": "B100", "tag": "_eidconst100_s42", "atom_feat": "legacy3_const"}

EXPECTED_Z100_PARAMS = 70_625_092
EXPECTED_ATOM_PROJ_PARAMS = 512 * 512 + 512      # Linear(512,512) 权重 + 偏置
EXPECTED_ZP100_PARAMS = EXPECTED_Z100_PARAMS + EXPECTED_ATOM_PROJ_PARAMS  # 70,887,748

PLANNED_CLI = dict(base.PLANNED_CLI, atom_feat="z_only_proj", tag="_eidzproj100_s42")

EXPECTED_RECIPE_CLI = {"model": "M1", "epochs": 100, "batch_size": 32, "lr": 5e-5, "seed": 42,
                       "norm": "sumnorm", "scale_mode": "eta", "atom_feat": "z_only_proj",
                       "skip_test_eval": True, "init_ckpt": "", "pair_aux_arm": "none"}

# 第 1 次启动（2026-10-01 20:56，PID 5575）在第 1 轮训练前因对齐记录写入函数的
# Path/str 缺陷崩溃：0 epoch、无 checkpoint、无 history。记录与证据保留如下，
# 不删除崩溃尝试留下的任何文件。
PRIOR_ATTEMPT = {
    "attempt": 1,
    "launched_utc": "2026-10-01T20:56:17Z",
    "pid": 5575,
    "crash": "utils/zproj_alignment.py::write_alignment_record 收到 str 路径，"
             "调用 Path.parent 抛 AttributeError；崩溃点在第 1 轮训练循环开始之前",
    "epochs_completed": 0,
    "fix": "write_alignment_record 现在接受 str 或 Path；未改任何参数或配方",
    "preserved_artifacts": [
        "results/train_m1_eidzproj100_s42_attempt1_crashed.log",
        "results/eidzproj100_s42_manifest_attempt1.json",
        "output/ablation_m1_eidzproj100_s42/config_used.yaml",
    ],
}

TRAINING_CODE_FILES = [
    "model/transformer.py",
    "utils/atom_feature.py",
    "utils/zproj_alignment.py",
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
    return RESULTS / "eidzproj100_s42_manifest.json"


def align_record_path() -> Path:
    return OUTPUT / "ablation_m1_eidzproj100_s42/zproj_align.json"


def command_for() -> str:
    return (
        "python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 "
        "--batch_size 32 --lr 5e-5 --norm sumnorm --scale_mode eta "
        "--atom_feat z_only_proj --skip_test_eval --tag _eidzproj100_s42"
    )


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def z100_manifest() -> dict:
    path = RESULTS / "eidzonly100_s42_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Z100 manifest {path}")
    return load_json(path)


def run_config(arm: dict) -> dict:
    """某臂实际运行记录的完整 ``config`` 段（runner 写入的 yaml_cfg）。"""
    with (OUTPUT / f"ablation_m1{arm['tag']}/config_used.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def candidate_yaml_cfg() -> dict:
    """Z100 运行配置 + 唯一差异 atom_feat_mode=z_only_proj（其余逐项沿用 Z100 记录）。"""
    cfg = run_config(REF_Z)["config"]
    params = cfg["model"]["params"]["sub_model"]["transformer"]
    if params.get("atom_feat_mode") != "z_only":
        raise ValueError("Z100 recorded config is not a z_only run")
    cfg["model"]["params"]["sub_model"]["transformer"]["atom_feat_mode"] = "z_only_proj"
    return cfg


def build_builder(yaml_cfg: dict):
    from utils.builder import ConfigBuilder

    return ConfigBuilder(**yaml_cfg)


def build_loaders(builder):
    """与 runner 相同的 loader 构造调用（顺序与随机数消耗一致）。"""
    train_loader = builder.get_dataloader(split="train", dos_minmax=True, batch_size=32,
                                          dos_sumnorm=True, use_bucket_batch=False)
    val_loader = builder.get_dataloader(split="valid", dos_minmax=True, batch_size=32,
                                        dos_sumnorm=True)
    return train_loader, val_loader


def trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 检查 1：Z100 未训练参考重建与初始参数指纹
# ---------------------------------------------------------------------------

def check_reference_rebuild() -> dict:
    manifest_z = z100_manifest()
    recorded = manifest_z["fingerprint"]["state_hash_z_only_init"]
    record("Z100 manifest initial fingerprint equals the frozen constant",
           recorded == Z100_INIT_STATE_SHA256,
           f"manifest={recorded[:16]}… constant={Z100_INIT_STATE_SHA256[:16]}…")

    model = zref.build_z100()
    digest = state_hash(model.state_dict())
    record("Z100 untrained model rebuilt from its recorded config and seed 42 "
           "matches the initial fingerprint", digest == recorded,
           f"now={digest[:16]}… record={recorded[:16]}…")

    # runner 构造路径（ConfigBuilder.get_model → basemodel → Transformer）同样复现该指纹，
    # 证明训练进程里的参考重建就是 Z100 的真实初始化路径。
    from run_ablation_experiments import setup_ablation_seed

    setup_ablation_seed(42)
    reference = prepare_reference(build_builder(candidate_yaml_cfg()))
    record("runner-path reference rebuild reproduces the Z100 init fingerprint",
           reference["ref_state_sha256"] == recorded,
           f"runner_path={reference['ref_state_sha256'][:16]}… record={recorded[:16]}…")
    record("reference model is the z_only architecture without atom_proj",
           "atom_proj.weight" not in reference["ref_state"]
           and "tok_emb.weight" in reference["ref_state"]
           and "atom_norm.weight" in reference["ref_state"],
           f"n_keys={len(reference['ref_state'])}")

    # Z100 对照身份未漂移：manifest 记录的 config 与 checkpoint 哈希仍与磁盘一致。
    save_z = OUTPUT / f"ablation_m1{REF_Z['tag']}"
    record("Z100 config_used.yaml hash still matches its manifest",
           manifest_z["config_used_sha256"] == base.sha256_file(save_z / "config_used.yaml"))
    for name in ("checkpoint_best.pth", "checkpoint_latest.pth"):
        record(f"Z100 {name} hash still matches its manifest",
               manifest_z["checkpoints"][name]["sha256"] == base.sha256_file(save_z / name),
               f"epoch={manifest_z['checkpoints'][name]['epoch']}")
    return {"reference": reference, "z100_manifest": manifest_z}


# ---------------------------------------------------------------------------
# 检查 2：唯一模型改动（z_only_proj 结构、公式、无性质表读取）
# ---------------------------------------------------------------------------

def check_architecture() -> dict:
    from model.transformer import Transformer

    params = base.build_transformer_params("z_only_proj")
    model = Transformer(**params)
    state = model.state_dict()
    z_state = zref.build_z100().state_dict()
    extra = sorted(set(state) - set(z_state))
    missing = sorted(set(z_state) - set(state))
    record("z_only_proj adds exactly atom_proj.{weight,bias} to the z_only keys",
           extra == ["atom_proj.bias", "atom_proj.weight"] and not missing,
           f"extra={extra} missing={missing}")
    record("z_only_proj keeps the numeric branch and fusion projection absent",
           model.num_emb_encoder is None and model.num_norm is None
           and model.fuse_proj is None and isinstance(model.atom_proj, torch.nn.Linear),
           f"atom_proj={model.atom_proj}")
    record("atom_proj is Linear(512 -> 512) with bias",
           tuple(model.atom_proj.weight.shape) == (512, 512)
           and tuple(model.atom_proj.bias.shape) == (512,),
           f"weight={tuple(model.atom_proj.weight.shape)} bias={tuple(model.atom_proj.bias.shape)}")
    record("z_only_proj trainable params equal 70,625,092 + 262,656",
           trainable(model) == EXPECTED_ZP100_PARAMS,
           f"{trainable(model):,} (delta {trainable(model) - EXPECTED_Z100_PARAMS:,})")

    # forward 公式：encoder 实际输入逐值等于 atom_proj(atom_norm(tok_emb(atom_idx)))。
    _, batch = zref.valid_batch()
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
    expected = model.atom_proj(model.atom_norm(model.tok_emb(src[:, 2:])))
    record("encoder input equals atom_proj(atom_norm(tok_emb(atom_idx))) bit-for-bit",
           bool(torch.equal(captured["src"], expected)))
    shapes = {key: tuple(out[key].shape) for key in ("edos", "phdos", "eta")}
    record("z_only_proj forward shapes edos[4,128]/phdos[4,64]/eta[4,2]",
           shapes == {"edos": (4, 128), "phdos": (4, 64), "eta": (4, 2)}, f"shapes={shapes}")

    # z_only 公式保持不变（本轮改动触及该分支，重新核对）。
    z_model = zref.build_z100()
    captured_z = {}
    handle = z_model.encoder.register_forward_pre_hook(
        lambda _m, args, kwargs: captured_z.update(
            {"src": kwargs.get("src", args[0] if args else None)}) or None,
        with_kwargs=True)
    z_model.eval()
    with torch.no_grad():
        z_model(src, mask, pos, batch[15], batch[16])
    handle.remove()
    record("z_only encoder input still equals atom_norm(tok_emb(atom_idx)) bit-for-bit",
           bool(torch.equal(captured_z["src"], z_model.atom_norm(z_model.tok_emb(src[:, 2:])))))

    # 纯原子序号表示：构造与前向不得读取任何元素性质表。
    import utils.atom_feature as af

    def _boom(*_args, **_kwargs):
        raise AssertionError("property table read attempted under z_only_proj")

    saved = (af.PeriodicTable.atom_feature_map, af.Mendeleev24.atom_feature_map)
    af.PeriodicTable.atom_feature_map = _boom
    af.Mendeleev24.atom_feature_map = _boom
    try:
        probe = Transformer(**params)
        probe.eval()
        with torch.no_grad():
            probe(src, mask, pos, batch[15], batch[16])
        record("z_only_proj construction and forward never read the property tables", True)
    finally:
        af.PeriodicTable.atom_feature_map, af.Mendeleev24.atom_feature_map = saved
    try:
        af.AtomFeatureEncoder(3, 512, feat="z_only_proj")
    except ValueError as exc:
        record("AtomFeatureEncoder explicitly rejects z_only_proj", True, str(exc))
    else:
        record("AtomFeatureEncoder explicitly rejects z_only_proj", False, "no exception raised")
    return {"shapes": shapes, "zp100_trainable_params": trainable(model)}


# ---------------------------------------------------------------------------
# 检查 3：旧模式逐值不变、checkpoint 加载边界
# ---------------------------------------------------------------------------

def check_legacy_compat() -> dict:
    from model.transformer import Transformer

    for mode in ("legacy3", "legacy3_const", "mendeleev24"):
        digest = state_hash(zref.build_old_mode(mode).state_dict())
        record(f"{mode} seed-42 initial state_dict hash equals pre-change snapshot",
               digest == zref.PRECHANGE_INIT_STATE_SHA256[mode],
               f"now={digest[:16]}… pre={zref.PRECHANGE_INIT_STATE_SHA256[mode][:16]}…")
    legacy = zref.build_old_mode("legacy3")
    record("legacy3 path still instantiates numeric branch and fusion",
           isinstance(legacy.num_emb_encoder, torch.nn.Module)
           and isinstance(legacy.num_norm, torch.nn.Module)
           and isinstance(legacy.fuse_proj, torch.nn.Module)
           and not hasattr(legacy, "atom_proj"))
    m24 = zref.build_old_mode("mendeleev24")
    record("mendeleev24 tok_emb still zero-initialized",
           float(m24.tok_emb.weight.abs().sum()) == 0.0)

    loaded = {}
    for arm, mode in ((REF_A, "legacy3"), (REF_B, "legacy3_const"), (REF_Z, "z_only")):
        save_dir = OUTPUT / f"ablation_m1{arm['tag']}"
        manifest = load_json(RESULTS / f"{arm['tag'].lstrip('_')}_manifest.json")
        ckpt_path = save_dir / "checkpoint_best.pth"
        if manifest["checkpoints"]["checkpoint_best.pth"]["sha256"] != base.sha256_file(ckpt_path):
            raise ValueError(f"{arm['arm']} checkpoint hash mismatch against its manifest")
        with (save_dir / "config_used.yaml").open(encoding="utf-8") as stream:
            tparams = yaml.safe_load(stream)["config"]["model"]["params"]["sub_model"]["transformer"]
        ref_model = Transformer(**dict(tparams, atom_feat_mode=mode))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ref_model.load_state_dict(ckpt["model"], strict=True)
        record(f"{arm['arm']} checkpoint strict-loads into its own {mode} model", True,
               f"epoch={ckpt.get('epoch')}")
        loaded[arm["arm"]] = {"checkpoint_sha256": base.sha256_file(ckpt_path),
                             "config_used_sha256": base.sha256_file(save_dir / "config_used.yaml"),
                             "epoch": int(ckpt.get("epoch"))}
        # z_only_proj 不得载入任何已有训练权重（键集不同 → strict 显式失败）。
        try:
            Transformer(**dict(tparams, atom_feat_mode="z_only_proj")).load_state_dict(
                ckpt["model"], strict=True)
        except Exception as exc:
            record(f"z_only_proj rejects {arm['arm']} trained weights (expected failure)",
                   True, type(exc).__name__)
        else:
            record(f"z_only_proj rejects {arm['arm']} trained weights (expected failure)",
                   False, "strict load unexpectedly succeeded")
    return {"reference_arms": loaded}


# ---------------------------------------------------------------------------
# 检查 4：对齐初始化全流程（与训练进程同一实现）+ 随机状态与数据顺序
# ---------------------------------------------------------------------------

def check_alignment_flow() -> dict:
    from run_ablation_experiments import setup_ablation_seed

    # 路径 A：完整重放训练进程的构造顺序（seed → loader → 参考 → 候选 → 对齐 → 恢复）。
    setup_ablation_seed(42)
    builder = build_builder(candidate_yaml_cfg())
    train_loader, val_loader = build_loaders(builder)
    reference = prepare_reference(builder)
    model = builder.get_model()
    info = apply_alignment(model.model["transformer"], reference, torch.device("cpu"),
                           batch=valid_batch(DATA_DIR), loader=train_loader)
    for check in info["checks"]:
        record(f"alignment: {check['name']}", check["pass"], check["detail"])
    restored = restore_training_rng(reference)
    record("training RNG state restored to the Z100 training-start state",
           restored == reference["ref_rng_sha256"] and restored == rng_digest(rng_snapshot()),
           f"restored={restored[:16]}… ref={reference['ref_rng_sha256'][:16]}…")

    # 路径 B：不经 loader、直接重建 z_only 参考模型，随机状态应逐位相同——
    # 实测证明 DataLoader/数据集构造不消耗随机数（训练开始时随机状态的对齐依据）。
    setup_ablation_seed(42)
    direct = prepare_reference(build_builder(candidate_yaml_cfg()))
    record("loader-free reference rebuild yields the same init and training-start RNG state",
           direct["ref_state_sha256"] == reference["ref_state_sha256"]
           and direct["ref_rng_sha256"] == reference["ref_rng_sha256"],
           f"state={direct['ref_state_sha256'][:16]}… rng={direct['ref_rng_sha256'][:16]}…")

    # 共享参数与投影初始化的独立复核（不依赖 apply_alignment 内部结论）。
    state = model.model["transformer"].state_dict()
    shared, proj = shared_keys(model.model["transformer"])
    shared_equal = all(torch.equal(state[key].detach().cpu(), reference["ref_state"][key])
                       for key in reference["ref_state"])
    record("independent re-check: every shared parameter equals the Z100 init value",
           shared_equal and set(proj) == {"atom_proj.weight", "atom_proj.bias"},
           f"n_shared={len(shared)}")
    record("independent re-check: atom_proj is identity weight / zero bias",
           bool(torch.equal(state["atom_proj.weight"], torch.eye(512)))
           and bool(torch.equal(state["atom_proj.bias"], torch.zeros(512))),
           f"weight_norm={float(state['atom_proj.weight'].norm()):.4f} "
           f"bias_norm={float(state['atom_proj.bias'].norm()):.4f}")

    # 初始前向等价：单位投影下 z_only_proj 与 z_only 参考输出数值一致。
    ref_cfg = candidate_yaml_cfg()
    ref_cfg["model"]["params"]["sub_model"]["transformer"]["atom_feat_mode"] = "z_only"
    ref_model = build_builder(ref_cfg).get_model().model["transformer"]
    ref_model.load_state_dict(reference["ref_state"], strict=True)
    batch = valid_batch(DATA_DIR)
    src, pos = batch[0], batch[1]
    ref_model.eval()
    model.model["transformer"].eval()
    with torch.no_grad():
        out_ref = ref_model(src, src.eq(0), pos, batch[15], batch[16])
        out_cand = model.model["transformer"](src, src.eq(0), pos, batch[15], batch[16])
    equivalence = {}
    for key in ("edos", "phdos", "eta"):
        diff = float((out_ref[key] - out_cand[key]).abs().max())
        equivalence[key] = {"bit_equal": bool(torch.equal(out_ref[key], out_cand[key])),
                            "max_abs_diff": diff}
        record(f"aligned forward equals the z_only reference on {key}",
               bool(torch.equal(out_ref[key], out_cand[key])) or diff <= 1e-6,
               f"bit_equal={equivalence[key]['bit_equal']} max_abs_diff={diff:.3e}")

    # 数据顺序：候选 loader 的 epoch 0/1 样本顺序与 z_only 管线逐值一致（顺序由
    # DistributedSampler(seed=0)+set_epoch 决定，与随机状态无关）。
    orders = {}
    for mode in ("z_only_proj", "z_only"):
        cfg = candidate_yaml_cfg()
        cfg["model"]["params"]["sub_model"]["transformer"]["atom_feat_mode"] = mode
        loader = build_builder(cfg).get_dataloader(split="train", dos_minmax=True, batch_size=32,
                                                   dos_sumnorm=True, use_bucket_batch=False)
        sampler = loader.sampler
        per_epoch = []
        for epoch in (0, 1):
            sampler.set_epoch(epoch)
            per_epoch.append(hashlib.sha256(
                np.asarray(list(sampler), dtype=np.int64).tobytes()).hexdigest())
        orders[mode] = per_epoch
    record("train sample index order identical to the z_only pipeline (epochs 0-1)",
           orders["z_only_proj"] == orders["z_only"],
           f"epoch0={orders['z_only_proj'][0][:16]}… epoch1={orders['z_only_proj'][1][:16]}…")
    record("in-run recorded sample order matches the preflight order",
           [item["sha256"] for item in info["sample_order"]] == orders["z_only_proj"],
           f"n={info['sample_order'][0]['n']}")
    ids = np.load(DATA_DIR / "train/train_index.npy")
    record("sampler covers all 18,706 Q1 train samples",
           info["sample_order"][0]["n"] == len(ids) == 18706, f"n={len(ids)}")
    return {"alignment": info, "reference": {
        "init_state_sha256": reference["ref_state_sha256"],
        "training_start_rng_sha256": reference["ref_rng_sha256"],
        "restored_rng_sha256": restored,
        "direct_rebuild_rng_sha256": direct["ref_rng_sha256"]},
        "forward_equivalence": equivalence,
        "sample_order_sha256": orders["z_only_proj"],
        "rng_method": RNG_METHOD_NOTE}


# ---------------------------------------------------------------------------
# 检查 5：数据、配方、网格、tag 空闲
# ---------------------------------------------------------------------------

def check_data_and_recipe() -> dict:
    hashes = {
        "table_sha256": base.sha256_file(ROOT / "utils/periodic_table_v2.csv"),
        "split_sha256": base.sha256_file(ROOT / "index/split_v2.yaml"),
        "data_manifest_sha256": base.sha256_file(DATA_DIR / "manifest.json"),
    }
    ref_manifest = load_json(RESULTS / f"{REF_A['tag'].lstrip('_')}_manifest.json")
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

    z_cfg = run_config(REF_Z)
    tparams = z_cfg["config"]["model"]["params"]["sub_model"]["transformer"]
    record("Z100 recorded grid and dropout are 128/64 and 0.05",
           tparams.get("edos_num") == 128 and tparams.get("phdos_num") == 64
           and float(tparams.get("dropout")) == 0.05,
           f"edos={tparams.get('edos_num')} phdos={tparams.get('phdos_num')} "
           f"dropout={tparams.get('dropout')}")
    cli = z_cfg["cli"]
    bad = {k: (cli.get(k), PLANNED_CLI[k]) for k in PLANNED_CLI
           if k not in ("atom_feat", "tag", "dropout") and cli.get(k) != PLANNED_CLI[k]}
    record("planned CLI equals the frozen recipe except atom_feat/tag/dropout", not bad,
           f"mismatch={bad}")
    record("planned dropout resolves to the 0.05 template value",
           cli.get("dropout") in (None, 0.05), f"Z100 cli dropout={cli.get('dropout')!r}")
    record("planned run is FP32 with skip_test_eval, empty init_ckpt, pair_aux off",
           PLANNED_CLI["use_amp"] is False and PLANNED_CLI["skip_test_eval"] is True
           and PLANNED_CLI["init_ckpt"] == "" and PLANNED_CLI["pair_aux_arm"] == "none")
    record("candidate config differs from Z100 only in atom_feat_mode",
           candidate_yaml_cfg()["model"]["params"]["sub_model"]["transformer"]["atom_feat_mode"]
           == "z_only_proj")
    return hashes


def check_tags_free() -> dict:
    """tag 现状核对：无 checkpoint 即无自动续跑风险；崩溃尝试的残留逐项如实记录。"""
    tag = ARM["tag"]
    save_dir = OUTPUT / f"ablation_m1{tag}"
    prior = dict(PRIOR_ATTEMPT, leftovers={})
    checkpoints = [name for name in ("checkpoint_latest.pth", "checkpoint_best.pth")
                   if (save_dir / name).exists()]
    record("ZP100 output tag carries no checkpoint (runner cannot auto-resume)",
           not checkpoints, f"{save_dir.relative_to(ROOT)}: {checkpoints or 'no checkpoint'}")
    record("ZP100 zproj_align.json absent before launch",
           not (save_dir / "zproj_align.json").exists())

    config_path = save_dir / "config_used.yaml"
    if config_path.exists():
        # 崩溃尝试在崩溃前写下的配置；内容须与冻结配方一致，且重启会写入同一内容。
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        bad = {k: (config["cli"].get(k), v) for k, v in EXPECTED_RECIPE_CLI.items()
               if config["cli"].get(k) != v}
        record("attempt-1 leftover config_used.yaml matches the frozen recipe", not bad,
               f"sha256={base.sha256_file(config_path)[:16]}… mismatch={bad}")
        prior["leftovers"]["config_used_yaml_sha256"] = base.sha256_file(config_path)
    else:
        record("save_dir has no leftover config_used.yaml", True, "absent")

    history = RESULTS / f"history_m1{tag}.csv"
    record("ZP100 history absent before launch (no prior results to overwrite)",
           not history.exists())
    summary = RESULTS / f"test_m1{tag}_summary.csv"
    record("ZP100 test summary absent before launch", not summary.exists())
    record("ZP100 manifest absent before launch", not manifest_path().exists())

    crashed_copy = RESULTS / "train_m1_eidzproj100_s42_attempt1_crashed.log"
    log_path = RESULTS / "train_m1_eidzproj100_s42.log"
    record("attempt-1 crashed log preserved at its own path", crashed_copy.exists(),
           f"{crashed_copy.name} sha256={base.sha256_file(crashed_copy)[:16]}…"
           if crashed_copy.exists() else "missing")
    if log_path.exists():
        record("launch log path holds only the attempt-1 crashed log (relaunch truncates it)",
               base.sha256_file(log_path) == base.sha256_file(crashed_copy),
               f"sha256={base.sha256_file(log_path)[:16]}…")
        prior["leftovers"]["launch_log_sha256_before_relaunch"] = base.sha256_file(log_path)
    else:
        record("launch log path free before launch", True, "absent")
    prior["leftovers"]["attempt1_manifest_sha256"] = base.sha256_file(
        RESULTS / "eidzproj100_s42_manifest_attempt1.json")
    return prior


# ---------------------------------------------------------------------------
# --verify-run / --finalize
# ---------------------------------------------------------------------------

def running_pids() -> list[int]:
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmdline = (Path("/proc") / entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "run_ablation_experiments" in cmdline and ARM["tag"] in cmdline:
            pids.append(int(entry))
    return sorted(pids)


def verify_run() -> None:
    manifest = load_manifest()
    save_dir = OUTPUT / f"ablation_m1{ARM['tag']}"
    config_path = save_dir / "config_used.yaml"
    if not config_path.exists():
        record("ZP100 real run config exists", False, str(config_path))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cli = config["cli"]
    expected = {"model": "M1", "epochs": 100, "batch_size": 32, "lr": 5e-5, "seed": 42,
                "norm": "sumnorm", "scale_mode": "eta", "atom_feat": "z_only_proj",
                "skip_test_eval": True, "init_ckpt": "", "pair_aux_arm": "none"}
    bad = {k: (cli.get(k), v) for k, v in expected.items() if cli.get(k) != v}
    record("ZP100 real config CLI matches the frozen recipe", not bad, f"mismatch={bad}")
    record("ZP100 real config FP32 (use_amp disabled)", not cli.get("use_amp"),
           f"use_amp={cli.get('use_amp')!r}")
    record("ZP100 real config dropout is 0.05",
           cli.get("dropout") in (None, 0.05), f"dropout={cli.get('dropout')!r}")
    params = config["config"]["model"]["params"]
    tparams = params["sub_model"]["transformer"]
    record("ZP100 real config effective atom_feat_mode is z_only_proj",
           tparams.get("atom_feat_mode") == "z_only_proj",
           f"atom_feat_mode={tparams.get('atom_feat_mode')!r}")
    record("ZP100 real config pair_aux disabled in model params",
           str(params.get("pair_aux_arm")) == "none", f"pair_aux_arm={params.get('pair_aux_arm')!r}")
    for key, name in (("table_sha256", "utils/periodic_table_v2.csv"),
                      ("split_sha256", "index/split_v2.yaml"),
                      ("data_manifest_sha256", "data/train4ARPAT/manifest.json")):
        record(f"ZP100 {name} hash unchanged since preflight",
               base.sha256_file(ROOT / name) == manifest["hashes"][key])

    align = load_json(align_record_path())
    record("in-run alignment record fingerprint matches the Z100 init",
           align["reference"]["fingerprint_match"] is True
           and align["reference"]["init_state_sha256"] == Z100_INIT_STATE_SHA256,
           f"init={align['reference']['init_state_sha256'][:16]}…")
    record("in-run alignment checks all passed",
           all(item["pass"] for item in align["checks"]),
           f"n_checks={len(align['checks'])}")
    record("in-run RNG restore equals the Z100 training-start state and was verified",
           align["rng"]["restored_training_start_sha256"]
           == align["rng"]["verified_before_loop_sha256"]
           == manifest["alignment"]["reference"]["training_start_rng_sha256"],
           f"restored={align['rng']['restored_training_start_sha256'][:16]}…")
    record("in-run sample order matches the preflight record",
           [item["sha256"] for item in align["sample_order"]]
           == manifest["alignment"]["sample_order_sha256"],
           f"epoch0={align['sample_order'][0]['sha256'][:16]}…")
    record("in-run shared-parameter copy count matches",
           align["candidate_init"]["n_shared_params"]
           == manifest["alignment"]["alignment"]["n_shared_params"],
           f"n={align['candidate_init']['n_shared_params']}")

    manifest["config_used_sha256"] = base.sha256_file(config_path)
    manifest["align_record_sha256"] = base.sha256_file(align_record_path())
    manifest["candidate_init_state_sha256"] = align["candidate_init"]["state_sha256"]
    manifest["process_pids"] = running_pids()
    manifest["log_path"] = "results/train_m1_eidzproj100_s42.log"
    manifest["verified_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_manifest(manifest)
    print(f"training process PIDs: {manifest['process_pids']}")


def finalize() -> None:
    manifest = load_manifest()
    save_dir = OUTPUT / f"ablation_m1{ARM['tag']}"
    for name in ("checkpoint_best.pth", "checkpoint_latest.pth"):
        path = save_dir / name
        if not path.exists():
            record(f"ZP100 {name} exists for finalization", False, str(path))
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        identity = (ckpt.get("epoch"), ckpt.get("model_name"), ckpt.get("seed"))
        record(f"ZP100 {name} identity (epoch, M1, 42)",
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
        record("ZP100 config_used.yaml unchanged since verify-run", False,
               base.sha256_file(config_path))
    if manifest.get("align_record_sha256") is not None:
        record("ZP100 zproj_align.json unchanged since verify-run",
               base.sha256_file(align_record_path()) == manifest["align_record_sha256"])
    manifest["finalized_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    history = pd.read_csv(RESULTS / f"history_m1{ARM['tag']}.csv")
    manifest["total_epoch_time_s"] = float(history["epoch_time_s"].sum())
    manifest["peak_vram_mb"] = float(history["peak_vram_mb"].max())
    write_manifest(manifest)


def build_manifest(fingerprint: dict, hashes: dict, alignment: dict, legacy_info: dict,
                   order_info: dict, arch_info: dict, prior_attempt: dict) -> dict:
    return {
        "arm": ARM["arm"],
        "tag": ARM["tag"],
        "atom_feat_mode": "z_only_proj",
        "atom_src_formula": "atom_src = atom_proj(atom_norm(tok_emb(atom_idx)))",
        "atom_proj_init": "weight = I(512), bias = 0",
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
        "architecture": arch_info,
        "alignment": alignment,
        "prior_attempt": prior_attempt,
        "legacy_compat": legacy_info,
        "data_order": order_info,
        "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHECKS],
        "config_used_sha256": None,
        "align_record_sha256": None,
        "candidate_init_state_sha256": None,
        "process_pids": [],
        "log_path": "results/train_m1_eidzproj100_s42.log",
        "checkpoints": {"checkpoint_best.pth": {"sha256": None, "epoch": None, "best_val_score": None},
                        "checkpoint_latest.pth": {"sha256": None, "epoch": None, "best_val_score": None}},
        "notes": "z_only_proj 在纯原子序号表示上新增 Linear(512→512) 投影；共享参数复制自 "
                 "Z100（z_only、seed 42）未训练参考模型，投影=单位矩阵/零偏置，训练随机状态恢复为 "
                 "Z100 训练开始时状态。判读前必须核对本 manifest 的模式、指纹、配置哈希、"
                 "对齐记录与 checkpoint 哈希。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-run", action="store_true",
                        help="check the real config_used.yaml / zproj_align.json and record hashes")
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
    ref_info = check_reference_rebuild()
    arch_info = check_architecture()
    legacy_info = check_legacy_compat()
    align_info = check_alignment_flow()
    hashes = check_data_and_recipe()
    prior_attempt = check_tags_free()
    fingerprint = {
        "z100_init_state_sha256": ref_info["reference"]["ref_state_sha256"],
        "z100_manifest_init_state_sha256": ref_info["z100_manifest"]["fingerprint"][
            "state_hash_z_only_init"],
        "candidate_init_state_sha256": align_info["alignment"]["candidate_state_sha256"],
    }
    write_manifest(build_manifest(fingerprint, hashes, align_info, legacy_info, {
        "train_samples": 18706, "epochs_checked": [0, 1],
        "sample_order_sha256": align_info["sample_order_sha256"],
        "note": "DistributedSampler(seed=0)+set_epoch 决定顺序，与随机状态无关；"
                "z_only_proj 与 z_only 管线逐值一致。"}, arch_info, prior_attempt))
    print(f"preflight: {sum(1 for _, ok, _ in CHECKS if ok)}/{len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
