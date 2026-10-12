"""Real-data G0 preflight: matched initial values, disposable updates and cost."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
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
from tools.checks.check_local_structure_controls import CODE_FILES, step_cost
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.builder import ConfigBuilder
from utils.global_only_experiment import LOCAL_PREFIX, global_only_config
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, restore_training_rng,
)
from utils.local_structure_experiment import DEFAULT_REFERENCE, load_control_reference
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash


def code_fingerprints():
    paths = (*CODE_FILES, "utils/global_only_experiment.py", "tools/checks/check_global_only.py",
             "tools/eval/global_only_compare.py")
    return {name: file_sha256(ROOT / name) for name in paths}


def check(out: Path, reference_dir: Path, tag_prefix: str):
    if out.exists():
        raise FileExistsError(f"refusing existing preflight: {out}")
    if not sys.flags.utf8_mode or not torch.cuda.is_available():
        raise ValueError("G0 preflight requires PYTHONUTF8=1 and a CUDA GPU")
    torch.set_num_threads(1)
    setup_ablation_seed(42)
    start = time.monotonic()
    saved, reference_cfg = load_control_reference(reference_dir)
    cfg = global_only_config(reference_cfg, tag_prefix)
    run_dir = repository_path(f"results/runs/ablation_m1{cfg.tag}")
    history = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
    if list(run_dir.glob("*")) or history.exists():
        raise ValueError("G0 preflight requires a fresh experiment tag")
    assembled = captured_config(cfg)
    restored = deepcopy(assembled)
    restored["model"]["params"]["sub_model"]["transformer"].update(
        local_route="B+", local_depth=2)
    if restored != saved["config"]:
        raise ValueError("G0 assembled configuration differs beyond removal of local updates")
    shaped = assembled["model"]["params"]["sub_model"]["transformer"]
    reference_params = saved["config"]["model"]["params"]["sub_model"]["transformer"]
    reference = Transformer(**reference_params)
    reference_info = initialize_comparison_model(reference, reference_params, 42)
    if reference_info != saved["runtime"]["local_global_init"]:
        raise ValueError("C2 untrained initial identity no longer reproduces")
    expected = {key: value.clone() for key, value in reference.state_dict().items()
                if not key.startswith(LOCAL_PREFIX)}
    del reference
    builder = ConfigBuilder(**assembled)
    model = builder.get_model()
    transformer = model.model["transformer"]
    before_rng = rng_digest(rng_snapshot())
    info = initialize_comparison_model(transformer, shaped, cfg.seed)
    if rng_digest(rng_snapshot()) != before_rng:
        raise ValueError("G0 initialization changed caller random streams")
    if (info["reference_state_sha256"] != reference_info["state_sha256"]
            or info["shared_state_sha256"] != state_hash(expected)
            or any(not torch.equal(transformer.state_dict()[key], value) for key, value in expected.items())):
        raise ValueError("G0 retained tensors are not identical to C2 initial values")
    del expected
    fingerprints = code_fingerprints()
    reference_files = {name: file_sha256(reference_dir / name)
                       for name in ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}
    report = {"schema": "global-only-preflight-v1", "passed": False,
              "scope": "disposable batch updates; no formal training; train/valid only",
              "reference_dir": str(reference_dir.resolve()), "reference_files": reference_files,
              "code_sha256": fingerprints, "configuration": asdict(cfg),
              "assembled_configuration": assembled, "initialization": info,
              "data_contract": saved["cli"]["data_contract"],
              "contract": comparison_contract(cfg, shaped),
              "learning_rate_35": learning_rate_trace(35, cfg.lr)}
    train_loader = builder.get_dataloader(split="train", dos_minmax=True, dos_sumnorm=True,
                                         batch_size=cfg.batch_size)
    valid_loader = builder.get_dataloader(split="valid", dos_minmax=True, dos_sumnorm=True,
                                         batch_size=cfg.batch_size)
    train_batches, _ = take(train_loader, 2)
    valid_batches, seen = take(valid_loader, 1)
    report["data_order"] = {"train_samples": len(train_loader.dataset),
                          "valid_samples": len(valid_loader.dataset),
                          "train_batches": len(train_loader), "valid_batches": len(valid_loader),
                          "sampler_seed": train_loader.sampler.seed}
    model.to(torch.device("cuda"))
    configure_local_global_execution(transformer)
    setup_ablation_seed(cfg.seed)
    report["training_start_rng_sha256"] = rng_digest(rng_snapshot())
    losses = []
    transformer.train()
    for batch in train_batches:
        values = model.train_one_step(batch, step=0)
        losses.append(float(values["loss"]))
        if not np.isfinite(losses[-1]) or any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in transformer.parameters() if parameter.requires_grad
        ):
            raise ValueError("G0 real-batch loss/gradients are missing or non-finite")
    report["forward_backward"] = {"losses": losses, "all_gradients_exist_and_finite": True}
    valid_ids = [str(valid_loader.dataset.material_ids[i])
                 for i in list(valid_loader.sampler)[:seen]]
    evaluated = evaluate_split(model, valid_batches, sumnorm=True,
                               return_sample_level=True, sample_ids=valid_ids)
    if not np.isfinite(evaluated["sample_df"].select_dtypes("number").to_numpy()).all():
        raise ValueError("non-finite G0 blind/oracle evaluation")
    report["evaluation"] = {"samples": seen, "finite": True, "test_loader": False}
    transformer.eval()
    report["e3_invariance"] = {}
    with torch.no_grad():
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
            transformer.to(dtype)
            worst = {}
            for operation in e3_operations():
                for key, values in e3_case(model, transformer, valid_batches[0], operation, dtype).items():
                    worst[key] = max(worst.get(key, 0.), values["max_rel"])
            if any(value > tolerance for value in worst.values()):
                raise ValueError("G0 E(3) check exceeded tolerance")
            report["e3_invariance"][str(dtype)] = worst
    transformer.to(torch.float32)
    optimizer = model.optimizer["transformer"]
    checkpoint_rng = rng_digest(rng_snapshot())
    buffer = io.BytesIO()
    torch.save(build_ablation_checkpoint(1, "M1", 42, False, transformer, optimizer, .5,
        model_contract=report["contract"], training_rng=capture_training_rng()), buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True, map_location="cpu")
    fresh = builder.get_model()
    fresh.to(torch.device("cuda"))
    restore_ablation_checkpoint(payload, fresh.model["transformer"], fresh.optimizer["transformer"],
                                False, model_contract=report["contract"])
    if any(not torch.equal(value, fresh.model["transformer"].state_dict()[key])
           for key, value in transformer.state_dict().items()):
        raise ValueError("G0 checkpoint weights did not round-trip")
    if optimizer_difference(optimizer.state_dict()["state"],
                            fresh.optimizer["transformer"].state_dict()["state"]):
        raise ValueError("G0 checkpoint optimizer did not round-trip")
    restore_training_rng(payload["training_rng"])
    if rng_digest(rng_snapshot()) != checkpoint_rng:
        raise ValueError("G0 checkpoint random streams did not round-trip")
    report["checkpoint_roundtrip"] = {"weights": True, "optimizer": True, "rng": True}
    del fresh, payload, buffer
    gc.collect()
    torch.cuda.empty_cache()
    report["costs"] = [step_cost(model, batch) for batch in train_batches]
    report["cost_scope"] = "warm full optimizer steps on two disposable train batches"
    if fingerprints != code_fingerprints() or any(
        file_sha256(reference_dir / name) != digest for name, digest in reference_files.items()
    ):
        raise ValueError("source or reference changed during G0 preflight")
    report.update(passed=True, wall_s=time.monotonic() - start)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "out": str(out), "wall_s": report["wall_s"],
                      "step_median_s": [row["median_s"] for row in report["costs"]]}), flush=True)
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
