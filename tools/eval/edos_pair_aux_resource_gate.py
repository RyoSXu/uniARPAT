#!/usr/bin/env python3
"""Stage-A calibration and V100 resource gate for the eDOS pair auxiliary path.

Only the Q1 train split is constructed. Calibration performs no optimizer step.
The short resource probe uses a fixed number of production train steps solely
for timing and peak-memory measurement; it does not save checkpoints or report
accuracy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.builder import ConfigBuilder
from utils.pair_aux_batches import PairPlanBatchSampler, build_pair_universe


B7_CHECKPOINT = ROOT / "output/ablation_m1_e9ctl/checkpoint_best.pth"
B7_CONFIG = ROOT / "output/ablation_m1_e9ctl/config_used.yaml"
B7_CHECKPOINT_SHA256 = "cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frozen_inputs(config_path: Path, checkpoint_path: Path, data_dir: Path):
    if file_sha256(checkpoint_path) != B7_CHECKPOINT_SHA256:
        raise ValueError("B7 checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    identity = (checkpoint.get("epoch"), checkpoint.get("model_name"), checkpoint.get("seed"))
    if identity != (33, "M1", 42):
        raise ValueError(f"unexpected B7 checkpoint identity: {identity}")
    with config_path.open() as stream:
        saved = yaml.safe_load(stream)
    config = copy.deepcopy(saved["config"])
    config["dataset"]["train"]["data_dir"] = str(data_dir)
    config["dataset"]["train"]["augment"] = False
    builder = ConfigBuilder(**config)
    dataset = builder.get_dataset(split="train", dos_minmax=True, dos_sumnorm=True)
    main_batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)))
    ids = np.load(data_dir / "train/train_index.npy")
    universe = build_pair_universe(dataset.elements, ids)
    sampler = PairPlanBatchSampler(universe, seed=42, pairs_per_batch=16)
    if len(universe) != 2591 or len(sampler) != 75:
        raise ValueError("Q1 pair universe no longer matches the frozen Stage-A census")
    pair_batch = next(iter(DataLoader(dataset, batch_sampler=sampler, num_workers=0)))
    return config, checkpoint, main_batch, pair_batch, sampler


def make_model(config, checkpoint, arm: str):
    model_config = copy.deepcopy(config)
    params = model_config["model"]["params"]
    params["pair_aux_arm"] = arm
    params["pair_ratio"] = 0.0 if arm == "none" else 0.10
    params["use_amp"] = False
    model = ConfigBuilder(**model_config).get_model()
    model.model["transformer"].load_state_dict(checkpoint["model"], strict=True)
    model.to(torch.device("cuda"))
    return model


def forward_edos_logits(model, batch, seed: int):
    set_seed(seed)
    transformer = model.model["transformer"]
    transformer.train()
    with torch.no_grad():
        data = model.data_preprocess(batch)
        output = transformer(data[0], data[2], data[1], data[-2], data[-1])
        logits = output["edos"].float()
        return logits.squeeze(1) if logits.dim() == 3 else logits


def parameters_match_checkpoint(model, checkpoint) -> bool:
    state = model.model["transformer"].state_dict()
    return all(
        torch.equal(value.detach().cpu(), checkpoint["model"][name])
        for name, value in state.items()
    )


def gradient_norm(model) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in model.model["transformer"].parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


def calibration_contract(config, checkpoint, main_batch, pair_batch, plan_hash):
    control = make_model(config, checkpoint, "control")
    candidate = make_model(config, checkpoint, "candidate")
    control._calibrate_edos_pair_weight(pair_batch)
    candidate._calibrate_edos_pair_weight(pair_batch)
    if control.optimizer["transformer"].state or candidate.optimizer["transformer"].state:
        raise RuntimeError("calibration unexpectedly initialized optimizer state")
    if not parameters_match_checkpoint(control, checkpoint) \
            or not parameters_match_checkpoint(candidate, checkpoint):
        raise RuntimeError("calibration changed B7 parameters or buffers")
    if control.pair_aux_calibration != candidate.pair_aux_calibration:
        raise RuntimeError("control and candidate calibration metadata differ")

    main_control = forward_edos_logits(control, main_batch, seed=143)
    main_candidate = forward_edos_logits(candidate, main_batch, seed=143)
    main_max_abs = float((main_control - main_candidate).abs().max().item())
    if main_max_abs != 0.0:
        raise RuntimeError(f"pre-gradient main outputs differ: {main_max_abs}")

    for model in (control, candidate):
        model.model["transformer"].zero_grad(set_to_none=True)
        set_seed(244)
        model.model["transformer"].train()
        pair_loss, *_ = model._pair_forward_loss(pair_batch)
        weight = model.pair_lambda if model.pair_aux_arm == "candidate" else 0.0
        (weight * pair_loss).backward()
    control_grad = gradient_norm(control)
    candidate_grad = gradient_norm(candidate)
    if control_grad != 0.0 or not np.isfinite(candidate_grad) or candidate_grad <= 0.0:
        raise RuntimeError(
            f"pair gradient contract failed: control={control_grad}, candidate={candidate_grad}")
    return {
        "epoch_plan_hash": plan_hash,
        "calibration": control.pair_aux_calibration,
        "optimizer_state_entries": 0,
        "parameters_unchanged": True,
        "main_output_max_abs": main_max_abs,
        "control_pair_gradient_norm": control_grad,
        "candidate_pair_gradient_norm": candidate_grad,
    }


def timed_arm(config, checkpoint, main_batch, pair_batch, arm, warmup, steps):
    model = make_model(config, checkpoint, arm)
    if arm != "none":
        model._calibrate_edos_pair_weight(pair_batch)
    model.model["transformer"].train()
    auxiliary_count = max(1, round(steps * 75 / 585)) if arm != "none" else 0
    auxiliary_steps = {
        (index + 1) * steps // auxiliary_count - 1
        for index in range(auxiliary_count)
    } if arm != "none" else set()

    for step in range(warmup):
        model.train_one_step(
            main_batch, step,
            pair_batch=pair_batch if arm != "none" and step == warmup - 1 else None,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for step in range(steps):
        start = time.perf_counter()
        model.train_one_step(
            main_batch, warmup + step,
            pair_batch=pair_batch if step in auxiliary_steps else None,
        )
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    return {
        "mean_step_ms": 1000.0 * float(np.mean(elapsed)),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        "timed_steps": steps,
        "auxiliary_steps": sorted(auxiliary_steps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=B7_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=B7_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/train4ARPAT")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/edos_pair_aux_stage_a_v100.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or "V100" not in torch.cuda.get_device_name():
        raise RuntimeError("this resource gate requires an NVIDIA V100")
    if args.warmup < 1 or args.steps < 8:
        raise ValueError("resource gate requires warmup>=1 and steps>=8")

    config, checkpoint, main_batch, pair_batch, sampler = load_frozen_inputs(
        args.config, args.checkpoint, args.data_dir,
    )
    calibration = calibration_contract(
        config, checkpoint, main_batch, pair_batch, sampler.plan_hash(0),
    )
    calibration["plan_hash"] = sampler.frozen_plan_hash()
    arms = {
        arm: timed_arm(
            config, checkpoint, main_batch, pair_batch, arm, args.warmup, args.steps,
        )
        for arm in ("none", "control", "candidate")
    }
    baseline = arms["none"]
    for arm in ("control", "candidate"):
        arms[arm]["time_ratio_vs_baseline"] = (
            arms[arm]["mean_step_ms"] / baseline["mean_step_ms"]
        )
        arms[arm]["vram_ratio_vs_baseline"] = (
            arms[arm]["peak_vram_mb"] / baseline["peak_vram_mb"]
        )
    max_time_ratio = max(arms[arm]["time_ratio_vs_baseline"] for arm in ("control", "candidate"))
    max_vram_ratio = max(arms[arm]["vram_ratio_vs_baseline"] for arm in ("control", "candidate"))
    if max_time_ratio > 1.25 or max_vram_ratio > 1.10:
        raise RuntimeError(
            f"resource gate failed: time_ratio={max_time_ratio:.4f}, "
            f"vram_ratio={max_vram_ratio:.4f}")
    report = {
        "stage": "A",
        "scope": "Q1 train-only calibration and resource gate; no accuracy result",
        "device": torch.cuda.get_device_name(),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "calibration": calibration,
        "arms": arms,
        "thresholds": {"time_ratio_max": 1.25, "vram_ratio_max": 1.10},
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
