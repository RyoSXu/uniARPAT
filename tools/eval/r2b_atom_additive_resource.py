#!/usr/bin/env python3
"""Measure R2a versus R2b on the same Q1 batch without an automatic threshold."""
import argparse
import csv
import json
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "../.."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.dataset import Dos_Dataset
from model.model import basemodel
from r2a_decoder_resource import production_params


def measure_arm(name, use_atom_additive_phdos, batch, warm_steps, device):
    logger = logging.getLogger(f"r2b-resource-{name}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    params = production_params(3)
    params["sub_model"]["transformer"]["use_atom_additive_phdos"] = bool(use_atom_additive_phdos)
    torch.manual_seed(42)
    model = basemodel(logger, **params)
    model.to(device)
    model.model["transformer"].train()
    n_params = sum(p.numel() for p in model.model["transformer"].parameters()
                   if p.requires_grad)

    # Two untimed steps settle allocator, optimizer and kernel state.
    for step in (0, 1):
        model.train_one_step(batch, step=step)
    torch.cuda.synchronize(device)

    times, peaks, losses = [], [], []
    for step in range(2, warm_steps + 2):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = model.train_one_step(batch, step=step)
        end.record()
        torch.cuda.synchronize(device)
        for key in ("loss", "loss_edos", "loss_phdos", "loss_eta"):
            if key not in result or not torch.isfinite(torch.tensor(result[key])):
                raise RuntimeError(f"{name} has non-finite {key}: {result.get(key)}")
        times.append(float(begin.elapsed_time(end)))
        peaks.append(float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
        losses.append(float(result["loss"]))
    return {
        "arm": name,
        "use_atom_additive_phdos": bool(use_atom_additive_phdos),
        "trainable_params": n_params,
        "warm_steps": int(warm_steps),
        "gpu_time_ms": times,
        "mean_gpu_time_ms": sum(times) / len(times),
        "peak_alloc_mb": max(peaks),
        "last_loss": losses[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./data/train4ARPAT")
    parser.add_argument("--warm_steps", type=int, default=5)
    parser.add_argument("--out_json", default="./results/r2b_atom_additive_resource_v100.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R2b resource measurement requires CUDA")
    if args.warm_steps < 1:
        raise ValueError("--warm_steps must be positive")
    device = torch.device("cuda:0")
    batch = next(iter(DataLoader(Dos_Dataset(
        data_dir=args.data_dir, split="train", dos_minmax=True, dos_sumnorm=True),
        batch_size=32, shuffle=False)))
    control = measure_arm("R2a", False, batch, args.warm_steps, device)
    torch.cuda.empty_cache()
    experiment = measure_arm("R2b", True, batch, args.warm_steps, device)
    out = {
        "device": torch.cuda.get_device_name(device),
        "batch_size": 32,
        "batch_atoms": int((batch[0][:, 2:] != 0).sum().item()),
        "same_q1_batch_reused_for_both_arms": True,
        "control": control,
        "experiment": experiment,
        "comparison": {
            "time_ratio": experiment["mean_gpu_time_ms"] / control["mean_gpu_time_ms"],
            "memory_ratio": experiment["peak_alloc_mb"] / control["peak_alloc_mb"],
        },
        "verdict": "MEASURED_NO_OOM",
    }
    out["comparison"]["time_diff_pct"] = (out["comparison"]["time_ratio"] - 1.0) * 100.0
    out["comparison"]["memory_diff_pct"] = (out["comparison"]["memory_ratio"] - 1.0) * 100.0
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as handle:
        json.dump(out, handle, indent=2)
    csv_path = args.out_json[:-5] + ".csv" if args.out_json.endswith(".json") else args.out_json + ".csv"
    fields = ("arm", "use_atom_additive_phdos", "trainable_params", "warm_steps",
              "mean_gpu_time_ms", "peak_alloc_mb", "last_loss")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({key: control[key] for key in fields})
        writer.writerow({key: experiment[key] for key in fields})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
