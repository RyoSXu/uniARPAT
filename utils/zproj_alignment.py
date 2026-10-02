"""z_only_proj 对齐初始化：Z100 未训练参考、共享参数复制、单位投影与训练随机状态对齐。

本模块是该流程的唯一实现，由 ``run_ablation_experiments.py``（训练进程内执行）与
``tools/eval/eid_zproj_preflight.py``（训练前模拟与核对）共用，保证训练实际执行的
步骤与检查模拟的步骤一致。设计见 ``docs/design/design-element-identity-zonlyproj.md``。

这是历史 ZP100 实验协议：固定 512 维投影和 Z100 seed-42 初始指纹。ZP 的元素编码
本身不依赖参考模型；新 seed 或新 encoder 不应直接套用本协议的历史指纹。
当前模块职责与研究基线见 ``docs/design/design-element-initialization.md``。

步骤与随机数口径
----------------
1. :func:`prepare_reference`：把 ``ConfigBuilder`` 的 ``atom_feat_mode`` 暂时切到
   ``z_only``，以与 Z100 完全相同的构造路径（``get_model`` → ``basemodel`` →
   ``Transformer``）重建其未训练模型，随后把模式切回 ``z_only_proj``。
   Z100 的训练开始时随机状态 = ``setup_ablation_seed(seed)`` 后、模型构造消耗后的
   状态：DataLoader / ``Dos_Dataset`` 构造不消耗随机数（实测，preflight 另有核对），
   构造之后到训练循环之间也没有随机抽样（无 ``init_ckpt``、无 RNG reset、无 pair
   aux、不冻结），因此参考模型构造完的那一刻就是 Z100 的训练开始时随机状态。
2. :func:`apply_alignment`：把参考模型的**全部共享参数**复制进候选模型，
   ``atom_proj`` 置单位矩阵 / 零偏置（不消耗随机数），随后核对共享参数逐值一致、
   投影初始化正确、前向与梯度有限。这些检查会消耗随机数（dropout、DataLoader
   base seed 等）。
3. :func:`restore_training_rng`：检查结束后恢复第 1 步保存的随机状态，返回状态摘要
   哈希；runner 在进入训练循环前再次核对该哈希未变。数据顺序由
   ``DistributedSampler(seed=0) + set_epoch(epoch)`` 决定，只依赖 epoch，与模型初始化
   随机数无关；:func:`apply_alignment` 顺带记录 epoch 0/1 的样本顺序哈希供交叉核对。

对齐后，候选与 Z100 在训练开始时具有相同的参数初值（共享参数逐值相同、投影为恒等）、
相同的随机状态（dropout 流一致）与相同的数据顺序；唯一差别是新增的投影及其梯度。
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ATOM_PROJ_KEYS = ("atom_proj.weight", "atom_proj.bias")
Z100_INIT_STATE_SHA256 = ("4eda5029e186d2f3ef60adeb0616b0a170ee3ca8fad80bad6e3e46cd9735ca32")

RNG_METHOD_NOTE = (
    "参考模型以 z_only 模式按 runner 同一构造路径重建；构造结束时保存 python/numpy/"
    "torch CPU/torch CUDA 四路随机状态，候选模型构造与训练前检查之后整体恢复该状态，"
    "并在进入训练循环前复核状态摘要哈希未变。数据顺序由 DistributedSampler(seed=0)"
    "的 set_epoch 决定，与随机状态无关，另行核对 epoch 0/1 样本顺序。")


def state_hash(state: dict) -> str:
    """state_dict 指纹：按 key 排序，key bytes + 张量字节（与 Z100 manifest 同口径）。"""
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def rng_snapshot() -> dict:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def rng_restore(snapshot: dict) -> None:
    torch.set_rng_state(snapshot["torch_cpu"])
    if snapshot["torch_cuda"]:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])
    np.random.set_state(snapshot["numpy"])
    random.setstate(snapshot["python"])


def rng_digest(snapshot: dict) -> str:
    """四路随机状态的摘要哈希；numpy/python 状态按各自数组字节与标量字段混入。"""
    digest = hashlib.sha256()
    digest.update(snapshot["torch_cpu"].numpy().tobytes())
    for state in snapshot["torch_cuda"]:
        digest.update(state.numpy().tobytes())
    np_state = snapshot["numpy"]
    digest.update(str(np_state[0]).encode())
    digest.update(np.asarray(np_state[1]).tobytes())
    digest.update(repr(tuple(np_state[2:])).encode())
    py_state = snapshot["python"]
    digest.update(str(py_state[0]).encode())
    digest.update(repr(tuple(py_state[1])).encode())
    digest.update(repr(py_state[2:]).encode())
    return digest.hexdigest()


def identity_projection_(linear: torch.nn.Linear) -> None:
    """单位矩阵 / 零偏置初始化；只做赋值，不消耗随机数。"""
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.weight.shape[0], dtype=linear.weight.dtype,
                                      device=linear.weight.device))
        linear.bias.zero_()


def prepare_reference(builder, expected_mode: str = "z_only_proj") -> dict:
    """重建 Z100 未训练参考模型，返回其初始 state_dict 与训练开始时随机状态。

    调用时机：DataLoader 构造之后、候选模型构造之前（与 Z100 的 runner 顺序一致）。
    参考模型随后丢弃，只保留 CPU 上的 state_dict 与随机状态快照。
    """
    params = builder.model_params["params"]["sub_model"]["transformer"]
    if params.get("atom_feat_mode") != expected_mode:
        raise ValueError(f"prepare_reference expects atom_feat_mode={expected_mode!r}, "
                         f"got {params.get('atom_feat_mode')!r}")
    params["atom_feat_mode"] = "z_only"
    try:
        reference = builder.get_model()
    finally:
        params["atom_feat_mode"] = expected_mode
    ref_transformer = reference.model["transformer"]
    if not (ref_transformer.num_emb_encoder is None and ref_transformer.num_norm is None
            and ref_transformer.fuse_proj is None):
        raise ValueError("reference model is not the z_only architecture")
    if getattr(ref_transformer, "atom_proj", None) is not None:
        raise ValueError("reference model must not carry the z_only_proj projection")
    ref_state = {key: value.detach().cpu().clone()
                 for key, value in ref_transformer.state_dict().items()}
    rng = rng_snapshot()
    return {
        "ref_state": ref_state,
        "ref_state_sha256": state_hash(ref_state),
        "ref_rng": rng,
        "ref_rng_sha256": rng_digest(rng),
    }


def shared_keys(transformer: torch.nn.Module) -> tuple[list, list]:
    """返回 (共享键, 投影键)；共享键必须与参考 state_dict 完全一致。"""
    keys = list(transformer.state_dict().keys())
    proj = [key for key in keys if key in ATOM_PROJ_KEYS]
    shared = [key for key in keys if key not in ATOM_PROJ_KEYS]
    return shared, proj


def sample_order_hashes(loader) -> list[dict]:
    """训练 sampler 在 epoch 0/1 的样本顺序哈希（与随机状态无关，供交叉核对）。"""
    sampler = loader.sampler
    if not hasattr(sampler, "set_epoch"):
        raise ValueError("alignment expects an epoch-seeded sampler (DistributedSampler)")
    out = []
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        order = np.asarray(list(sampler), dtype=np.int64)
        out.append({"epoch": epoch, "n": int(order.size),
                    "sha256": hashlib.sha256(order.tobytes()).hexdigest()})
    sampler.set_epoch(0)
    return out


def valid_batch(data_dir, batch_size: int = 4):
    """训练前数值检查用的 Q1 valid 小批次（与训练 valid loader 同口径）。"""
    from torch.utils.data import DataLoader
    from datasets.dataset import Dos_Dataset

    dataset = Dos_Dataset(data_dir=str(data_dir), split="valid", dos_minmax=True,
                          dos_sumnorm=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return next(iter(loader))


def check_forward_backward(transformer, batch, device) -> dict:
    """训练前数值检查：双谱 + eta 前向有限、反向梯度有限且非空（不更新参数）。"""
    src, pos = batch[0].to(device), batch[1].to(device)
    mask = src.eq(0)
    was_training = transformer.training
    transformer.train()
    out = transformer(src, mask, pos, batch[15].to(device), batch[16].to(device))
    forward_finite = all(bool(torch.isfinite(out[key]).all()) for key in ("edos", "phdos", "eta"))
    loss = out["edos"].sum() + out["phdos"].sum() + out["eta"].sum()
    loss.backward()
    grads = [p.grad for p in transformer.parameters() if p.grad is not None]
    backward_finite = bool(torch.isfinite(loss)) and bool(grads) and all(
        bool(torch.isfinite(g).all()) for g in grads)
    transformer.zero_grad(set_to_none=True)
    transformer.train(was_training)
    return {
        "forward_finite": forward_finite,
        "backward_finite": bool(backward_finite),
        "loss_sum": float(loss.detach()),
        "n_grad_tensors": len(grads),
        "grad_abs_max": max(float(g.abs().max()) for g in grads) if grads else None,
        "shapes": {key: list(out[key].shape) for key in ("edos", "phdos", "eta")},
    }


def apply_alignment(transformer, reference: dict, device, batch, loader) -> dict:
    """复制共享参数、置单位投影，并完成训练前核对（会消耗随机数）。

    ``batch`` 是数值检查用的真实批次，``loader`` 是训练 dataloader（用于记录
    epoch 0/1 的样本顺序）；两者都是必填项，检查项不允许静默跳过。
    """
    ref_state = reference["ref_state"]
    shared, proj = shared_keys(transformer)
    if set(proj) != set(ATOM_PROJ_KEYS):
        raise ValueError(f"z_only_proj model must expose exactly {ATOM_PROJ_KEYS}, got {proj}")
    if set(shared) != set(ref_state):
        raise ValueError("shared parameter keys do not match the z_only reference: "
                         f"only_model={sorted(set(shared) - set(ref_state))[:5]}, "
                         f"only_ref={sorted(set(ref_state) - set(shared))[:5]}")
    if not hasattr(transformer, "atom_proj"):
        raise ValueError("z_only_proj model lacks atom_proj")
    linear = transformer.atom_proj
    if not (isinstance(linear, torch.nn.Linear) and linear.weight.shape == (512, 512)
            and linear.bias is not None):
        raise ValueError("atom_proj must be Linear(512 -> 512) with bias")

    # 全部共享参数复制自参考（未训练）模型；投影置单位矩阵/零偏置。一次 strict 加载
    # 同时完成复制与投影初始化，且不消耗随机数。
    new_state = dict(ref_state)
    identity_projection_(linear)
    new_state["atom_proj.weight"] = linear.weight.detach().cpu().clone()
    new_state["atom_proj.bias"] = linear.bias.detach().cpu().clone()
    transformer.load_state_dict(new_state, strict=True)

    state = transformer.state_dict()
    shared_equal = all(torch.equal(state[key].detach().cpu(), ref_state[key]) for key in ref_state)
    proj_device = state["atom_proj.weight"].device
    weight_ok = bool(torch.equal(state["atom_proj.weight"],
                                 torch.eye(512, dtype=state["atom_proj.weight"].dtype,
                                           device=proj_device)))
    bias_ok = bool(torch.equal(state["atom_proj.bias"],
                               torch.zeros_like(state["atom_proj.bias"])))

    numeric = check_forward_backward(transformer, batch, device)
    order = sample_order_hashes(loader)

    checks = [
        {"name": "reference init state fingerprint equals the Z100 record",
         "pass": reference["ref_state_sha256"] == Z100_INIT_STATE_SHA256,
         "detail": f"now={reference['ref_state_sha256'][:16]}… "
                   f"record={Z100_INIT_STATE_SHA256[:16]}…"},
        {"name": "all shared parameters copied bit-for-bit from the Z100 init model",
         "pass": shared_equal, "detail": f"n_shared={len(ref_state)}"},
        {"name": "atom_proj initialized to identity weight / zero bias",
         "pass": weight_ok and bias_ok, "detail": f"weight_eye={weight_ok} bias_zero={bias_ok}"},
        {"name": "forward outputs finite", "pass": bool(numeric.get("forward_finite")),
         "detail": f"shapes={numeric.get('shapes')}"},
        {"name": "backward gradients finite and non-empty",
         "pass": bool(numeric.get("backward_finite")),
         "detail": f"n_grad_tensors={numeric.get('n_grad_tensors')}"},
        {"name": "train sampler order recorded for epochs 0-1",
         "pass": len(order) == 2 and all(item["n"] == order[0]["n"] for item in order),
         "detail": f"n={order[0]['n'] if order else None}"},
    ]
    failed = [item["name"] for item in checks if not item["pass"]]
    if failed:
        raise AssertionError(f"z_only_proj alignment check failed: {failed}")
    return {
        "checks": checks,
        "candidate_state_sha256": state_hash(state),
        "n_shared_params": len(ref_state),
        "numeric": numeric,
        "sample_order": order,
        "summary": (f"shared params={len(ref_state)} copied from Z100 init "
                    f"({reference['ref_state_sha256'][:12]}…), atom_proj=identity/zero, "
                    f"fwd/bwd finite, order epochs 0-1 recorded"),
    }


def restore_training_rng(reference: dict) -> str:
    """检查结束后恢复 Z100 训练开始时的随机状态，返回恢复后的状态摘要哈希。"""
    rng_restore(reference["ref_rng"])
    digest = rng_digest(rng_snapshot())
    if digest != reference["ref_rng_sha256"]:
        raise AssertionError("restored RNG digest does not match the Z100 training-start state")
    return digest


def write_alignment_record(path, payload: dict) -> None:
    """把对齐记录写成 JSON；``path`` 可以是 str 或 Path（runner 传 os.path.join 结果）。"""
    path = Path(path)
    payload = dict(payload)
    payload["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
