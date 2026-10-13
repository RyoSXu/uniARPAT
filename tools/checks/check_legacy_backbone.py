"""Bounded O0/O1 GPU preflight; all updated models are disposable."""
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.checks.check_global_depth import e3_case, optimizer_difference, take
from tools.checks.check_local_structure_controls import CODE_FILES, step_cost
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval.legacy_backbone_compare import PRIOR_REPORT, reference_identity, run_directory
from tools.eval.lg_structure_diagnosis import dump
from tools.eval.lg_valid_export import load_run
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.builder import ConfigBuilder
from utils.legacy_backbone_experiment import legacy_backbone_configs
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, restore_training_rng,
)
from utils.local_structure_experiment import DEFAULT_REFERENCE
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import rng_digest, rng_snapshot


def code_fingerprints():
    paths = (*CODE_FILES, "utils/lg_baseline_experiment.py", "utils/global_only_experiment.py",
             "utils/legacy_backbone_experiment.py", "tools/checks/check_legacy_backbone.py",
             "tools/eval/legacy_backbone_compare.py")
    return {name: file_sha256(ROOT / name) for name in paths}


def check(out, reference_dir, tag_prefix):
    if out.exists():
        raise FileExistsError(f"refusing existing preflight: {out}")
    if not sys.flags.utf8_mode or not torch.cuda.is_available():
        raise ValueError("O0/O1 preflight requires PYTHONUTF8=1 and a CUDA GPU")
    torch.set_num_threads(1)
    setup_ablation_seed(42)
    start = time.monotonic()
    saved, reference, identities = reference_identity(reference_dir)
    controls = legacy_backbone_configs(reference, tag_prefix)
    fingerprints = code_fingerprints()
    baseline_record, _, _ = load_run(Path(identities["original_baseline"]["run_dir"]))
    c1_record, _, _ = load_run(Path(identities["C1"]["run_dir"]))
    report = {"schema": "legacy-backbone-preflight-v1", "passed": False,
              "scope": "disposable real train-batch updates; train/valid only; no formal checkpoints",
              "references": identities, "prior_report_sha256": file_sha256(PRIOR_REPORT),
              "code_sha256": fingerprints, "data_contract": saved["cli"]["data_contract"],
              "learning_rate_35": learning_rate_trace(35, reference.lr), "controls": {}}
    for cfg in controls:
        history = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
        if list(run_directory(cfg).glob("*")) or history.exists():
            raise ValueError("O0/O1 preflight requires fresh tags")
        setup_ablation_seed(42)
        assembled = captured_config(cfg)
        restored = deepcopy(assembled)
        restored["model"]["params"]["sub_model"]["transformer"].update(local_route="B+", local_depth=2)
        if restored != saved["config"]:
            raise ValueError("O0/O1 assembly changed the frozen C2 recipe")
        params = assembled["model"]["params"]["sub_model"]["transformer"]
        builder = ConfigBuilder(**assembled)
        model = builder.get_model()
        transformer = model.model["transformer"]
        before_rng = rng_digest(rng_snapshot())
        info = initialize_comparison_model(transformer, params, 42)
        if rng_digest(rng_snapshot()) != before_rng:
            raise ValueError("O0/O1 initializer changed caller random streams")
        if info["baseline_initialization"] != baseline_record["runtime"]["lg_baseline_init"]:
            raise ValueError("original baseline untrained identity no longer reproduces")
        if cfg.local_route == "O1" and info["c1_initialization"] != c1_record["runtime"]["local_global_init"]:
            raise ValueError("unchanged C1 untrained identity no longer reproduces")
        if "O0" in report["controls"] and info["shared_old_state_sha256"] != \
                report["controls"]["O0"]["initialization"]["shared_old_state_sha256"]:
            raise ValueError("O0/O1 retained tensors do not share identical initial values")
        entry = {"configuration": asdict(cfg), "assembled_configuration": assembled,
                 "initialization": info, "contract": comparison_contract(cfg, params)}
        train_loader = builder.get_dataloader(split="train", dos_minmax=True, dos_sumnorm=True,
                                             batch_size=cfg.batch_size)
        valid_loader = builder.get_dataloader(split="valid", dos_minmax=True, dos_sumnorm=True,
                                             batch_size=cfg.batch_size)
        train_batches, _ = take(train_loader, 2)
        valid_batches, seen = take(valid_loader, 1)
        entry["data_order"] = {"train_samples": len(train_loader.dataset), "valid_samples": len(valid_loader.dataset),
                              "sampler_seed": train_loader.sampler.seed,
                              "first_train_ids": [str(train_loader.dataset.material_ids[i])
                                                  for i in list(train_loader.sampler)[:cfg.batch_size]]}
        model.to(torch.device("cuda"))
        configure_local_global_execution(transformer)
        setup_ablation_seed(42)
        entry["training_start_rng_sha256"] = rng_digest(rng_snapshot())
        losses = []
        with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring used")):
            for batch in train_batches:
                values = model.train_one_step(batch, step=0)
                losses.append(float(values["loss"]))
                if not np.isfinite(losses[-1]) or any(
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                    for parameter in transformer.parameters() if parameter.requires_grad
                ):
                    raise ValueError("O0/O1 real loss/gradients missing or non-finite")
            ids = [str(valid_loader.dataset.material_ids[i]) for i in list(valid_loader.sampler)[:seen]]
            evaluated = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True, sample_ids=ids)
            if not np.isfinite(evaluated["sample_df"].select_dtypes("number").to_numpy()).all():
                raise ValueError("non-finite blind/oracle evaluation")
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
                        raise ValueError("O0/O1 E(3) check exceeded tolerance")
                    entry["e3_invariance"][str(dtype)] = worst
        transformer.to(torch.float32)
        entry["forward_backward"] = {"losses": losses, "all_gradients_exist_and_finite": True,
                                     "old_geometry_path_not_called": True}
        entry["evaluation"] = {"samples": seen, "finite": True, "test_loader": False}
        optimizer = model.optimizer["transformer"]
        checkpoint_rng = rng_digest(rng_snapshot())
        buffer = io.BytesIO()
        torch.save(build_ablation_checkpoint(1, "M1", 42, False, transformer, optimizer, .5,
            model_contract=entry["contract"], training_rng=capture_training_rng()), buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=True, map_location="cpu")
        fresh = builder.get_model()
        fresh.to(torch.device("cuda"))
        restore_ablation_checkpoint(payload, fresh.model["transformer"], fresh.optimizer["transformer"],
                                    False, model_contract=entry["contract"])
        if any(not torch.equal(value, fresh.model["transformer"].state_dict()[key])
               for key, value in transformer.state_dict().items()):
            raise ValueError("O0/O1 checkpoint weights did not round-trip")
        if optimizer_difference(optimizer.state_dict()["state"], fresh.optimizer["transformer"].state_dict()["state"]):
            raise ValueError("O0/O1 checkpoint optimizer did not round-trip")
        restore_training_rng(payload["training_rng"])
        if rng_digest(rng_snapshot()) != checkpoint_rng:
            raise ValueError("O0/O1 checkpoint RNG did not round-trip")
        entry["checkpoint_roundtrip"] = {"weights": True, "optimizer": True, "rng": True}
        del fresh, payload, buffer
        gc.collect()
        torch.cuda.empty_cache()
        entry["costs"] = [step_cost(model, batch) for batch in train_batches]
        entry["cost_scope"] = "warm full optimizer steps on two disposable train batches"
        report["controls"][cfg.local_route] = entry
        print(json.dumps({"checked": cfg.local_route,
                          "median_step_s": [cost["median_s"] for cost in entry["costs"]]}), flush=True)
        del transformer, model, builder, train_loader, valid_loader, train_batches, valid_batches, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    if fingerprints != code_fingerprints() or file_sha256(PRIOR_REPORT) != report["prior_report_sha256"]:
        raise ValueError("source or prior report changed during preflight")
    for identity in identities.values():
        if any(file_sha256(Path(identity["run_dir"]) / name) != digest for name, digest in identity["files"].items()):
            raise ValueError("existing reference artifact changed during preflight")
    report.update(passed=True, wall_s=time.monotonic()-start)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, report)
    print(json.dumps({"passed": True, "out": str(out), "wall_s": report["wall_s"]}), flush=True)
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
