#!/usr/bin/env python3
"""Measure B7 6-layer versus R2a 3-layer decoder cost on the same Q1 batch.

The tool intentionally records cost rather than applying an autonomous resource
threshold.  It uses the production M1/Q1/SumNorm/H1 recipe and batch size 32.
"""
import argparse
import csv
import json
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets.dataset import Dos_Dataset
from model.model import basemodel


def production_params(decoder_layers: int):
    """Return the B7 production recipe with only decoder depth varied."""
    return dict(
        dos_minmax=True, dos_zscore=False, apply_log=False, scale_factor=1.0,
        loss_form="sumnorm_klw", use_mask=False, lambda_ph=1.0, grad_clip=0.0,
        w_w1=1.0, w_huber=1.0, huber_delta=0.02,
        tv_w=0.0, grad_w=0.0, peak_w=1.0, tail_w=1.0, tail_start=-1,
        scale_sup_w=1.0, eta_sup_w=1.0, delta_edos=0.09375,
        delta_phdos=19.6875, scalar_sup_w=1.0, c5_moe_balance_w=0.01,
        save_best="balanced_score", metrics_list=[],
        sub_model=dict(transformer=dict(
            token_num=118, d_model=512, nhead=8, edos_num=128, phdos_num=64,
            num_encoder_layers=6, num_decoder_layers=int(decoder_layers),
            dim_feedforward=2048, dropout=0.05, activation="gelu",
            normalize_before=False, decoupled_decoder=False,
            use_gated_cross_attn=False, head_type="legacy", predict_scale=False,
            atom_feat_mode="legacy3", energy_code="none", scale_mode="eta",
            scalar_mode="none", use_g1=False, use_g2=False)),
        optimizer=dict(transformer=dict(
            type="AdamW", params=dict(lr=5e-5, betas=[0.9, 0.99]))),
        lr_scheduler={},
    )


def measure_arm(name: str, decoder_layers: int, batch, warm_steps: int, device):
    logger = logging.getLogger(f"r2a-resource-{name}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    torch.manual_seed(42)
    model = basemodel(logger, **production_params(decoder_layers))
    model.to(device)
    model.model["transformer"].train()
    n_params = sum(p.numel() for p in model.model["transformer"].parameters()
                   if p.requires_grad)

    # The first optimization step allocates optimizer state and warms kernels.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train_one_step(batch, step=0)
    torch.cuda.synchronize(device)

    # A second, untimed step settles lazy CUDA/cuDNN allocations before the
    # comparable steady-state samples below.
    model.train_one_step(batch, step=1)
    torch.cuda.synchronize(device)

    times_ms = []
    peaks_mb = []
    losses = []
    for step in range(2, warm_steps + 2):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = model.train_one_step(batch, step=step)
        end.record()
        torch.cuda.synchronize(device)
        for key in ("loss", "loss_edos", "loss_phdos", "loss_eta"):
            if key not in result or not torch.isfinite(torch.tensor(result[key])):
                raise RuntimeError(f"{name} has non-finite {key}: {result.get(key)}")
        times_ms.append(float(begin.elapsed_time(end)))
        peaks_mb.append(float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
        losses.append(float(result["loss"]))

    return {
        "arm": name,
        "decoder_layers": int(decoder_layers),
        "trainable_params": n_params,
        "warm_steps": int(warm_steps),
        "gpu_time_ms": times_ms,
        "mean_gpu_time_ms": sum(times_ms) / len(times_ms),
        "peak_alloc_mb": max(peaks_mb),
        "last_loss": losses[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./data/train4ARPAT")
    parser.add_argument("--warm_steps", type=int, default=5)
    parser.add_argument("--out_json", default="./results/r2a_decoder_resource_v100.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R2a resource measurement requires CUDA")
    if args.warm_steps < 1:
        raise ValueError("--warm_steps must be positive")

    device = torch.device("cuda:0")
    dataset = Dos_Dataset(data_dir=args.data_dir, split="train", dos_minmax=True,
                          dos_sumnorm=True)
    batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))
    n_atoms = int((batch[0][:, 2:] != 0).sum().item())
    control = measure_arm("B7", 6, batch, args.warm_steps, device)
    torch.cuda.empty_cache()
    experiment = measure_arm("R2a", 3, batch, args.warm_steps, device)

    time_ratio = experiment["mean_gpu_time_ms"] / control["mean_gpu_time_ms"]
    mem_ratio = experiment["peak_alloc_mb"] / control["peak_alloc_mb"]
    out = {
        "device": torch.cuda.get_device_name(device),
        "batch_size": 32,
        "batch_atoms": n_atoms,
        "same_q1_batch_reused_for_both_arms": True,
        "control": control,
        "experiment": experiment,
        "comparison": {
            "time_ratio": time_ratio,
            "time_diff_pct": (time_ratio - 1.0) * 100.0,
            "memory_ratio": mem_ratio,
            "memory_diff_pct": (mem_ratio - 1.0) * 100.0,
        },
        "verdict": "MEASURED_NO_OOM",
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as handle:
        json.dump(out, handle, indent=2)
    csv_path = args.out_json[:-5] + ".csv" if args.out_json.endswith(".json") else args.out_json + ".csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "arm", "decoder_layers", "trainable_params", "warm_steps",
            "mean_gpu_time_ms", "peak_alloc_mb", "last_loss"))
        writer.writeheader()
        writer.writerow({key: control[key] for key in writer.fieldnames})
        writer.writerow({key: experiment[key] for key in writer.fieldnames})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
