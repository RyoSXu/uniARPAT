"""Bounded real-data preflight of C0/C1 and the reusable B+ G6 reference.

Disposable models take a few optimizer steps to check the real loss, gradients,
resume contract and representative GPU cost. No formal training, test loader or
new checkpoint is created. A successful report records code/data fingerprints
and the exact configurations admitted by the sequential experiment entry point.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from model.transformer import Transformer
from tools.checks.check_global_depth import e3_case, optimizer_difference, take
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.builder import ConfigBuilder
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, restore_training_rng,
)
from utils.local_structure_experiment import (
    DEFAULT_REFERENCE, load_control_reference, parameter_counts, structure_control_configs,
)
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import rng_digest, rng_snapshot

CODE_FILES = (
    "model/local_structure_controls.py", "model/local_global_encoder.py", "model/transformer.py",
    "model/periodic_messages.py", "model/periodic_geometry_cache.py", "model/model.py",
    "utils/local_structure_experiment.py", "utils/local_global_experiment.py", "utils/builder.py",
    "utils/dataset.py", "utils/experiment_config.py", "utils/ablation_checkpoint.py",
    "utils/periodic_geometry.py", "utils/periodic_neighbors.py", "utils/crystal.py",
    "utils/crystal_cache.py", "utils/bucket_batch.py", "utils/spectral_inference.py",
    "tools/eval/lg_valid_export.py", "tools/eval/lg_paired_compare.py",
    "tools/eval/lg_structure_diagnosis.py", "tools/eval/lg_train_subset_diagnosis.py",
    "tools/train.py", "tools/training-defaults.yaml", "tools/eval/local_structure_compare.py",
    "tools/checks/check_local_structure_controls.py",
)


def code_fingerprints():
    return {name: file_sha256(ROOT / name) for name in CODE_FILES}


def run_directory(cfg):
    return repository_path(f"results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}")


def step_cost(model, batch, repetitions=3):
    """Warm complete optimizer-step cost, with AdamW state and the local cache."""
    model.model["transformer"].train()
    for _ in range(2):
        model.train_one_step(batch, step=0)
    elapsed = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repetitions):
        start = time.monotonic()
        values = model.train_one_step(batch, step=0)
        torch.cuda.synchronize()
        elapsed.append(time.monotonic() - start)
        if not np.isfinite(float(values["loss"])):
            raise ValueError("non-finite real-batch loss")
    return {"samples": len(batch[0]), "atoms": int(batch[0].ne(0).sum()),
            "padded_width": int(batch[0].shape[1]), "microbatches": model.last_microbatch_sizes,
            "wall_s": elapsed, "median_s": float(np.median(elapsed)),
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20}


def check(out: Path, reference_dir: Path, tag_prefix: str):
    if out.exists():
        raise FileExistsError(f"refusing existing preflight: {out}")
    if not sys.flags.utf8_mode or not torch.cuda.is_available():
        raise ValueError("this experiment requires PYTHONUTF8=1 and a CUDA GPU")
    torch.set_num_threads(1)
    setup_ablation_seed(42)
    start = time.monotonic()
    saved, reference_cfg = load_control_reference(reference_dir)
    controls = structure_control_configs(reference_cfg, tag_prefix)
    params = saved["config"]["model"]["params"]["sub_model"]["transformer"]
    reference = Transformer(**params)
    init = initialize_comparison_model(reference, params, 42)
    recorded = saved["runtime"]["local_global_init"]
    if init != recorded:
        raise ValueError("C2 initialization no longer reproduces its complete recorded identity")
    reference_count = parameter_counts(reference)
    del reference
    execution = saved["runtime"]["local_global_execution"]
    if (len(execution) != 1 or execution[0]["from_epoch"] != 1
            or execution[0]["route"] != "B+" or not execution[0]["shared_geometry"]
            or not execution[0]["geometry_cache"]["enabled"]):
        raise ValueError("C2 was not run with one fixed shared-geometry/caching policy")
    fingerprints = code_fingerprints()
    report = {"schema": "local-structure-preflight-v1", "passed": False,
              "scope": "disposable batch updates; no formal training; train/valid only",
              "reference_dir": str(reference_dir.resolve()), "tag_prefix": tag_prefix,
              "code_sha256": fingerprints, "data_contract": saved["cli"]["data_contract"],
              "reference": {"initialization": recorded, "parameter_counts": reference_count,
                            "execution": execution, "reuse": True,
                            "config_sha256": file_sha256(reference_dir / "config_used.yaml"),
                            "best_checkpoint_sha256": file_sha256(reference_dir / "checkpoint_best.pth"),
                            "rng_evidence": "historical runner resets seed=42 before the loop; "
                                            "a historical start-state digest was not saved"}, "controls": {}}
    builder = ConfigBuilder(**saved["config"])
    train_loader = builder.get_dataloader(split="train", dos_minmax=True, dos_sumnorm=True,
                                         batch_size=reference_cfg.batch_size)
    valid_loader = builder.get_dataloader(split="valid", dos_minmax=True, dos_sumnorm=True,
                                         batch_size=reference_cfg.batch_size)
    train_batches, _ = take(train_loader, 2)
    valid_batches, seen = take(valid_loader, 1)
    orders = {}
    for epoch in (0, 1):
        train_loader.sampler.set_epoch(epoch)
        order = np.asarray(list(train_loader.sampler), dtype=np.int64)
        orders[str(epoch)] = {"first_rows": order[:32].tolist(),
                              "sha256": hashlib.sha256(order.tobytes()).hexdigest()}
    report["data_order"] = {"sampler_seed": 0, "epoch_orders": orders,
                             "train_samples": len(train_loader.dataset),
                             "valid_samples": len(valid_loader.dataset),
                             "train_batches": len(train_loader), "valid_batches": len(valid_loader)}
    report["learning_rate_35"] = learning_rate_trace(35, 5e-5)
    valid_ids = [str(valid_loader.dataset.material_ids[i]) for i in list(valid_loader.sampler)[:seen]]
    for cfg in controls:
        artifacts = list(run_directory(cfg).glob("*"))
        history = repository_path(f"results/reports/history_{cfg.model_name.lower()}{cfg.tag}.csv")
        if artifacts or history.exists():
            raise ValueError(f"control tag already contains artifacts: {cfg.tag}")
        assembled = captured_config(cfg)
        common = deepcopy(assembled)
        common["model"]["params"]["sub_model"]["transformer"]["local_route"] = "B+"
        if common != saved["config"]:
            raise ValueError("actual runner configuration drifted outside the local route")
        build = ConfigBuilder(**assembled)
        model = build.get_model()
        transformer = model.model["transformer"]
        shaped = assembled["model"]["params"]["sub_model"]["transformer"]
        info = initialize_comparison_model(transformer, shaped, cfg.seed)
        if info["reference_state_sha256"] != recorded["state_sha256"]:
            raise ValueError("control does not refer to the reproduced C2 initial state")
        model.to(torch.device("cuda"))
        entry = {"configuration": asdict(cfg), "assembled_config_matches_reference": True,
                 "initialization": info, "contract": comparison_contract(cfg, shaped)}
        setup_ablation_seed(cfg.seed)
        entry["training_start_rng_sha256"] = rng_digest(rng_snapshot())
        losses = []
        transformer.train()
        for batch in train_batches:
            values = model.train_one_step(batch, step=0)
            losses.append(float(values["loss"]))
            missing = [name for name, value in transformer.named_parameters()
                       if value.requires_grad and value.grad is None]
            bad = [name for name, value in transformer.named_parameters()
                   if value.grad is not None and not torch.isfinite(value.grad).all()]
            if missing or bad or not np.isfinite(losses[-1]):
                raise ValueError(f"invalid gradients/loss: missing={missing}, nonfinite={bad}")
        entry["forward_backward"] = {"losses": losses, "all_gradients_exist_and_finite": True}
        evaluated = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True,
                                   sample_ids=valid_ids)
        frame = evaluated.pop("sample_df")
        if not np.isfinite(frame.select_dtypes("number").to_numpy()).all():
            raise ValueError("non-finite blind/oracle sample evaluation")
        entry["evaluation"] = {"samples": seen, "finite": True, "test_loader": False}
        transformer.eval()
        entry["e3_invariance"] = {}
        with torch.no_grad():
            for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
                transformer.to(dtype)
                worst = {}
                for operation in e3_operations():
                    for key, values in e3_case(model, transformer, valid_batches[0], operation, dtype).items():
                        worst[key] = max(worst.get(key, 0.), values["max_rel"])
                if any(value > tolerance for value in worst.values()):
                    raise ValueError(f"E(3) check exceeded {tolerance}: {worst}")
                entry["e3_invariance"][str(dtype)] = {"worst_relative": worst, "tolerance": tolerance}
        transformer.to(torch.float32)
        optimizer = model.optimizer["transformer"]
        checkpoint_rng_hash = rng_digest(rng_snapshot())
        payload = build_ablation_checkpoint(1, "M1", 42, False, transformer, optimizer, .5,
            model_contract=entry["contract"], training_rng=capture_training_rng())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=True, map_location="cpu")
        fresh = build.get_model()
        fresh.to(torch.device("cuda"))
        restore_ablation_checkpoint(payload, fresh.model["transformer"], fresh.optimizer["transformer"],
                                    False, model_contract=entry["contract"])
        for key, value in transformer.state_dict().items():
            if not torch.equal(value, fresh.model["transformer"].state_dict()[key]):
                raise ValueError("checkpoint weights did not round-trip: " + key)
        if optimizer_difference(optimizer.state_dict()["state"], fresh.optimizer["transformer"].state_dict()["state"]):
            raise ValueError("checkpoint optimizer did not round-trip")
        restore_training_rng(payload["training_rng"])
        if rng_digest(rng_snapshot()) != checkpoint_rng_hash:
            raise ValueError("checkpoint random streams did not round-trip")
        entry["checkpoint_roundtrip"] = {"weights": True, "optimizer": True, "rng": True,
                                          "weights_only_load": True}
        del fresh, payload, buffer
        gc.collect()
        torch.cuda.empty_cache()
        configure_local_global_execution(transformer)
        costs = [step_cost(model, batch) for batch in train_batches]
        cache = transformer._local_geometry_cache
        entry["costs"] = {"train": costs, "accounting": "warm full optimizer steps, including AdamW "
                           "state and lazily populated cache; excludes data loading and full-dataset cache",
                           "cache": cache.configuration() if cache is not None else {"enabled": False}}
        report["controls"][cfg.local_route] = entry
        print(json.dumps({"preflight_arm": cfg.local_route, "parameters": info["parameter_counts"],
                          "step_median_s": [cost["median_s"] for cost in costs]}), flush=True)
        del model, transformer, optimizer, build, evaluated, frame, assembled
        gc.collect()
        torch.cuda.empty_cache()
    entries = list(report["controls"].values())
    if (entries[0]["initialization"]["shared_control_state_sha256"]
            != entries[1]["initialization"]["shared_control_state_sha256"]
            or entries[0]["training_start_rng_sha256"] != entries[1]["training_start_rng_sha256"]):
        raise ValueError("controls do not share common initialization and training-start RNG")
    if fingerprints != code_fingerprints():
        raise ValueError("code changed during preflight")
    report.update(passed=True, seconds=time.monotonic() - start,
                  device=torch.cuda.get_device_name(), python_utf8_mode=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"preflight": "passed", "report": str(out), "seconds": report["seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check(args.out, args.reference, args.tag_prefix)


if __name__ == "__main__":
    main()
