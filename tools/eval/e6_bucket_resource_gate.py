"""E6 V100 resource gate: fixed-width B7 batches versus bucket+trim batches.

This consumes only Q1 train batches, runs no epoch or test evaluation, and
reports cost rather than accuracy.  Batch membership deliberately differs.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.builder import ConfigBuilder


def _make_model(config, checkpoint):
    builder = ConfigBuilder(**copy.deepcopy(config))
    model = builder.get_model()
    state = torch.load(checkpoint, map_location="cpu")["model"]
    model.model["transformer"].load_state_dict(state, strict=True)
    model.to(torch.device("cuda"))
    return model


def _batches(config, batch_size, bucket, count):
    builder = ConfigBuilder(**copy.deepcopy(config))
    params = {**builder.dataloader_params, "num_workers": 0,
              "persistent_workers": False, "prefetch_factor": None}
    loader = builder.get_dataloader(split="train", dos_minmax=True, dos_sumnorm=True,
                                    batch_size=batch_size, use_bucket_batch=bucket,
                                    dataloader_params=params)
    sampler = loader.batch_sampler if bucket else loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    return [batch for _, batch in zip(range(count), loader)]


def _measure(model, batches, warmup, steps):
    model.model["transformer"].train()
    for i, batch in enumerate(batches[:warmup]):
        model.train_one_step(batch, i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses, slots = [], []
    for i, batch in enumerate(batches[warmup:warmup + steps]):
        result = model.train_one_step(batch, warmup + i)
        if not all(np.isfinite(value) for value in result.values()):
            raise RuntimeError("non-finite E6 loss")
        losses.append(result["loss"])
        slots.append(int(batch[0].shape[0] * (batch[0].shape[1] - 2)))
    torch.cuda.synchronize()
    return {"mean_step_ms": 1000 * (time.perf_counter() - start) / steps,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
            "mean_atom_slots": float(np.mean(slots)), "max_atom_slots": int(max(slots)),
            "loss_first": float(losses[0]), "loss_last": float(losses[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="output/ablation_m1_e9ctl/checkpoint_best.pth")
    parser.add_argument("--config", default="output/ablation_m1_e9ctl/config_used.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E6 resource gate requires CUDA")
    with open(args.config) as f:
        config = yaml.safe_load(f)["config"]
    count = args.warmup + args.steps
    fixed_batches = _batches(config, args.batch_size, False, count)
    bucket_batches = _batches(config, args.batch_size, True, count)
    fixed = _measure(_make_model(config, args.checkpoint), fixed_batches, args.warmup, args.steps)
    bucket = _measure(_make_model(config, args.checkpoint), bucket_batches, args.warmup, args.steps)
    report = {"device": torch.cuda.get_device_name(), "checkpoint": args.checkpoint,
              "split": "train", "batch_size": args.batch_size, "warmup": args.warmup,
              "steps": args.steps, "fixed": fixed, "bucket_trim": bucket,
              "time_ratio_bucket_over_fixed": bucket["mean_step_ms"] / fixed["mean_step_ms"],
              "vram_ratio_bucket_over_fixed": bucket["peak_vram_mb"] / fixed["peak_vram_mb"],
              "slot_ratio_bucket_over_fixed": bucket["mean_atom_slots"] / fixed["mean_atom_slots"]}
    Path(args.results_dir).mkdir(exist_ok=True)
    with open(Path(args.results_dir) / "e6_bucket_resource_v100.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    pd.DataFrame([{"arm": name, **data} for name, data in (("fixed", fixed), ("bucket_trim", bucket))]).to_csv(
        Path(args.results_dir) / "e6_bucket_resource_v100.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
