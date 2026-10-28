"""Real-batch preflight for the learned-spring phonon readout.

Builds the production assembly for the shape-auxiliary control and the phonon
candidate, runs two disposable updates each on the same train batches, and
checks that the candidate's training objective equals the evaluation metric.
No experiment checkpoint is written and test is never evaluated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.checks.check_long_budget import captured_config
from tools.eval.local_global_compare import training_command
from tools.train import configure_local_global_execution, evaluate_local_global_split, setup_ablation_seed
from utils.builder import ConfigBuilder
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import comparison_contract, initialize_comparison_model
from utils.local_structure_experiment import FROZEN
from utils.paths import file_sha256, input_path
from utils.spectral_balance import ENERGY_METRIC
from utils.spectral_balance_experiment import validate_balance_config

PROTOCOL = "results/runs/spectral_shape_aux_20261025/protocol.json"
TAG = "_sbal_phonon20261028_shape_a64_s42_e35"


def candidate_config(data_dir):
    return ExperimentConfig(**{**FROZEN, "local_route": "I3", "scale_mode": "per_atom",
                               "selection_metric": ENERGY_METRIC,
                               "spectral_balance_config": str(input_path(PROTOCOL)),
                               "spectral_shape_aux": True, "phonon_head": True,
                               "data_dir": str(input_path(data_dir)), "tag": TAG})


def build(cfg):
    setup_ablation_seed(cfg.seed)
    assembly = captured_config(cfg)
    builder = ConfigBuilder(**assembly)
    model = builder.get_model()
    params = assembly["model"]["params"]["sub_model"]["transformer"]
    init = initialize_comparison_model(model.model["transformer"], params, cfg.seed)
    model.to(torch.device("cuda"))
    configure_local_global_execution(model.model["transformer"])
    return builder, model, params, init


def updates(model, batches, pair_budget):
    model.model["transformer"].train()
    torch.cuda.reset_peak_memory_stats()
    timings, losses = [], []
    for index, batch in enumerate(batches):
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses.append(model.train_one_step(batch, index, pair_budget=pair_budget))
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        if not all(np.isfinite(value) for value in losses[-1].values()):
            raise ValueError("non-finite disposable training update")
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.model["transformer"].parameters()):
            raise ValueError("non-finite model gradient")
    return timings, losses, torch.cuda.max_memory_allocated() / 1024**2


def check(out, data_dir, batches=3):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"use a new preflight output directory: {out}")
    if not torch.cuda.is_available():
        raise ValueError("real-batch preflight requires CUDA")
    torch.set_num_threads(1)
    cfg = candidate_config(data_dir)
    validate_balance_config(cfg)
    control_cfg = replace(cfg, phonon_head=False, tag="_unit_phonon_control")

    builder, model, params, init = build(cfg)
    loader = builder.get_dataloader(split="train", batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
    iterator = iter(loader)
    chosen = [next(iterator) for _ in range(batches)]
    head = model.model["transformer"].phonon_head

    candidate_times, candidate_losses, candidate_peak = updates(model, chosen, cfg.pair_budget)
    model.model["transformer"].eval()
    for group in model.optimizer["transformer"].param_groups:
        group["lr"] = 0.
    before = model.train_one_step(chosen[0], 0, pair_budget=cfg.pair_budget)
    evaluated = evaluate_local_global_split(model, chosen[:1], pair_budget=cfg.pair_budget,
                                            return_spectra=True)
    auxiliary = .5 * (before["loss_shape_e"] + before["loss_shape_p"])
    gap = abs(before["loss"] - auxiliary - evaluated[ENERGY_METRIC])
    if gap > 1e-5 * max(1.0, abs(evaluated[ENERGY_METRIC])):
        raise ValueError("training/evaluation normalized blind objectives differ")
    shapes = evaluated["sample_spectra"]["pred_shape_phdos"]
    centers = np.linspace(-280 + 9.84375, 980 - 9.84375, 64)
    width_bins = head.min_width_bins + float(torch.nn.functional.softplus(head.width.detach()))
    physics = {"broadening_cm": width_bins * 19.6875,
               "mean_frequency_cm": float((shapes * centers).sum(-1).mean()),
               "weight_below_zero": float(shapes[:, centers < 0].sum(-1).mean())}
    del model
    torch.cuda.empty_cache()

    _, control, _, _ = build(control_cfg)
    control_times, _, control_peak = updates(control, chosen, cfg.pair_budget)

    result = {"schema": "phonon-head-preflight-v1", "passed": True,
              "configuration": asdict(cfg), "model_contract": comparison_contract(cfg, params),
              "initialization": init, "gpu": torch.cuda.get_device_name(),
              "batch_atoms": [int(batch[0].ne(0).sum()) for batch in chosen],
              "candidate_update_seconds": candidate_times, "candidate_peak_allocated_mib": candidate_peak,
              "control_update_seconds": control_times, "control_peak_allocated_mib": control_peak,
              "candidate_losses": candidate_losses, "loss_evaluation_difference": gap,
              "initial_physics_after_updates": physics,
              "source_sha256": {name: file_sha256(ROOT / name) for name in (
                  "model/phonon_head.py", "model/transformer.py", "utils/spectral_balance_experiment.py",
                  "tools/train.py", "tools/checks/check_phonon_head.py")},
              "command": training_command(cfg),
              "scope": "disposable updates on the first train batches; no checkpoint, no test",
              "full_training_started": False}
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default="data/crystal_q1_v1")
    args = parser.parse_args()
    result = check(input_path(args.out), args.data_dir)
    print(json.dumps({key: result[key] for key in (
        "passed", "gpu", "candidate_update_seconds", "control_update_seconds", "candidate_peak_allocated_mib",
        "control_peak_allocated_mib", "loss_evaluation_difference", "initial_physics_after_updates")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
