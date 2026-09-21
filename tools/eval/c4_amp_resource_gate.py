"""C4 V100 gate: B7 FP32 versus CUDA FP16 AMP on one real Q1 batch.

No epoch is trained and no test split is read.  Both arms start from the B7
epoch-33 checkpoint; the timed steps only establish numerical and resource
eligibility for future training.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.builder import ConfigBuilder


def _make_model(config, checkpoint, use_amp):
    cfg = copy.deepcopy(config)
    cfg["model"]["params"]["use_amp"] = bool(use_amp)
    builder = ConfigBuilder(**cfg)
    model = builder.get_model()
    state = torch.load(checkpoint, map_location="cpu")["model"]
    model.model["transformer"].load_state_dict(state, strict=True)
    model.to(torch.device("cuda"))
    return model


def _forward_probabilities(model, batch):
    model.model["transformer"].eval()
    with torch.no_grad():
        inp, pos, mask, *_unused, edos_x, phdos_x = model.data_preprocess(batch)
        with model.amp_autocast():
            out = model.model["transformer"](inp, mask, pos, edos_x, phdos_x)
        out = model.fp32_outputs(out)
    return torch.softmax(out["edos"], -1), torch.softmax(out["phdos"], -1)


def _timed_steps(model, batch, warmup, steps):
    model.model["transformer"].train()
    for i in range(warmup):
        model.train_one_step(batch, step=i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = model.gscaler.state_dict().copy()
    losses = []
    start = time.perf_counter()
    for i in range(steps):
        result = model.train_one_step(batch, step=warmup + i)
        if not all(np.isfinite(value) for value in result.values()):
            raise RuntimeError("non-finite C4 train loss")
        losses.append(result["loss"])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    after = model.gscaler.state_dict().copy()
    return {"mean_step_ms": 1000.0 * elapsed / steps,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
            "loss_first": float(losses[0]), "loss_last": float(losses[-1]),
            "scaler_before": before, "scaler_after": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="output/ablation_m1_e9ctl/checkpoint_best.pth")
    parser.add_argument("--config", default="output/ablation_m1_e9ctl/config_used.yaml")
    parser.add_argument("--data-dir", default="./data/train4ARPAT")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("C4 resource gate requires CUDA")
    with open(args.config) as f:
        config = yaml.safe_load(f)["config"]
    config["dataset"]["train"]["data_dir"] = args.data_dir
    builder = ConfigBuilder(**config)
    dataset = builder.get_dataset(split="train", dos_minmax=True, dos_sumnorm=True)
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))

    fp32 = _make_model(config, args.checkpoint, False)
    amp = _make_model(config, args.checkpoint, True)
    e32, p32 = _forward_probabilities(fp32, batch)
    e16, p16 = _forward_probabilities(amp, batch)
    numeric = {"edos_mean_abs": float((e32 - e16).abs().mean()), "edos_max_abs": float((e32 - e16).abs().max()),
               "phdos_mean_abs": float((p32 - p16).abs().mean()), "phdos_max_abs": float((p32 - p16).abs().max())}
    if max(numeric["edos_mean_abs"], numeric["phdos_mean_abs"]) > 1e-4 or \
       max(numeric["edos_max_abs"], numeric["phdos_max_abs"]) > 2e-3:
        raise RuntimeError(f"C4 numeric gate failed: {numeric}")
    fp32_metrics = _timed_steps(fp32, batch, args.warmup, args.steps)
    amp_metrics = _timed_steps(amp, batch, args.warmup, args.steps)
    if not amp.gscaler.is_enabled() or not amp_metrics["scaler_after"]:
        raise RuntimeError("AMP GradScaler was not active")
    report = {"device": torch.cuda.get_device_name(), "checkpoint": args.checkpoint,
              "split": "train", "n_samples_read": args.batch_size, "warmup": args.warmup,
              "steps": args.steps, "numeric": numeric, "fp32": fp32_metrics, "amp": amp_metrics,
              "time_ratio_amp_over_fp32": amp_metrics["mean_step_ms"] / fp32_metrics["mean_step_ms"],
              "vram_ratio_amp_over_fp32": amp_metrics["peak_vram_mb"] / fp32_metrics["peak_vram_mb"]}
    os.makedirs(args.results_dir, exist_ok=True)
    json_path = os.path.join(args.results_dir, "c4_amp_resource_v100.json")
    csv_path = os.path.join(args.results_dir, "c4_amp_resource_v100.csv")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    pd.DataFrame([{"arm": name, "mean_step_ms": metrics["mean_step_ms"], "peak_vram_mb": metrics["peak_vram_mb"],
                   "loss_first": metrics["loss_first"], "loss_last": metrics["loss_last"]}
                  for name, metrics in (("fp32", fp32_metrics), ("amp", amp_metrics))]).to_csv(csv_path, index=False)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
