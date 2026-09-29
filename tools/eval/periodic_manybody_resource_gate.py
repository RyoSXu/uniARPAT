"""Measure the approved encoder candidate against B7 on real Q1 train batches."""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data._utils.collate import default_collate

from datasets.dataset import Dos_Dataset
from utils.builder import ConfigBuilder


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results/periodic_manybody_edge_audit_q1.csv"
OUTPUT = ROOT / "results/periodic_manybody_resource_gate_v100.json"
CHECKPOINT = ROOT / "output/ablation_m1_e9ctl/checkpoint_best.pth"


def make_batch(dataset, indices):
    return default_collate([dataset[int(i)] for i in indices])


def make_model(base_config, candidate):
    config = copy.deepcopy(base_config)
    config["model"]["params"]["sub_model"]["transformer"]["use_periodic_manybody"] = candidate
    model = ConfigBuilder(logger=logging.getLogger("manybody_gate"), **config).get_model()
    if not candidate:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.model["transformer"].load_state_dict(checkpoint["model"], strict=True)
    model.to(torch.device("cuda"))
    model.model["transformer"].train()
    return model


def measure(model, batch, warmup=3, repeat=8):
    torch.cuda.reset_peak_memory_stats()
    times = []
    losses = []
    for step in range(warmup + repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.train_one_step(batch, step=step)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        if not np.isfinite(result["loss"]):
            raise FloatingPointError("non-finite training loss")
        if step >= warmup:
            times.append(duration)
            losses.append(float(result["loss"]))
    return {
        "median_step_seconds": float(np.median(times)),
        "step_seconds": times,
        "losses": losses,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("resource gate requires CUDA")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with (ROOT / "output/ablation_m1_e9ctl/config_used.yaml").open() as stream:
        base_config = yaml.safe_load(stream)["config"]
    dataset = Dos_Dataset(data_dir=str(ROOT / "data/train4ARPAT"), split="train",
                          dos_minmax=True, dos_sumnorm=True)
    audit = pd.read_csv(AUDIT)
    audit = audit[audit.split == "train"].sort_values("idx")
    assert len(audit) == len(dataset)
    rng = np.random.default_rng(42)
    representative = rng.choice(len(dataset), size=32, replace=False).tolist()
    max_index = int(audit.loc[audit.n_edges.idxmax(), "idx"])
    stress = [max_index] + [i for i in representative if i != max_index][:31]
    if len(stress) < 32:
        stress.append(next(i for i in range(len(dataset)) if i not in stress))
    batches = {
        "representative": make_batch(dataset, representative),
        "one_max_graph": make_batch(dataset, stress),
    }
    report = {
        "gpu": torch.cuda.get_device_name(),
        "split": "Q1 train",
        "batch_size": 32,
        "representative_indices": representative,
        "stress_max_index": max_index,
        "stress_max_edges": int(audit.loc[audit.n_edges.idxmax(), "n_edges"]),
        "arms": {},
    }
    for name, candidate in (("b7", False), ("periodic_manybody", True)):
        print("measuring", name, flush=True)
        model = make_model(base_config, candidate)
        try:
            report["arms"][name] = {}
            for batch_name, batch in batches.items():
                print("batch", batch_name, flush=True)
                report["arms"][name][batch_name] = measure(model, batch)
                print(report["arms"][name][batch_name], flush=True)
        finally:
            del model
            torch.cuda.empty_cache()

    baseline = report["arms"]["b7"]
    candidate = report["arms"]["periodic_manybody"]
    report["max_step_ratio"] = max(
        candidate[key]["median_step_seconds"] / baseline[key]["median_step_seconds"]
        for key in batches)
    report["max_peak_allocated_gb"] = max(
        candidate[key]["peak_allocated_gb"] for key in batches)
    report["passed"] = (
        report["max_step_ratio"] <= 2.0
        and report["max_peak_allocated_gb"] <= 24.0)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("gate", "PASS" if report["passed"] else "STOP", report["max_step_ratio"],
          report["max_peak_allocated_gb"], flush=True)


if __name__ == "__main__":
    main()
