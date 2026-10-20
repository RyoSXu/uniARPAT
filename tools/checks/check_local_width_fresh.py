"""Preflight both fresh capacity arms using only Git sources and crystal data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
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

from tools.checks.check_global_depth import e3_case, take
from tools.checks.check_global_norm import _train_batch_updates
from tools.checks.check_local_readout import checkpoint_roundtrip, permutation_case
from tools.checks.check_local_structure_controls import step_cost
from tools.checks.check_local_width import code_fingerprints as width_fingerprints
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval import local_width_fresh_compare as runner
from tools.eval.lg_structure_diagnosis import dump
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.bucket_batch import crystal_collate
from utils.builder import ConfigBuilder
from utils.i1_a_experiment import validate_i3_model
from utils.local_global_experiment import (
    _is_local_parameter, capture_training_rng, comparison_contract, initialize_comparison_model,
)
from utils.local_width_experiment import validate_width_model, width_parameter_counts
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash


def code_fingerprints():
    return {**width_fingerprints(), **{name: file_sha256(ROOT / name) for name in
        ("tools/checks/check_local_width_fresh.py", "tools/eval/local_width_fresh_compare.py", "utils/paths.py")}}


def check(out, tag_prefix):
    if out.exists():
        raise FileExistsError(f"refusing existing fresh capacity preflight: {out}")
    if not sys.flags.utf8_mode:
        raise ValueError("fresh capacity preflight requires PYTHONUTF8=1")
    runtime = runner.runtime_identity()
    torch.set_num_threads(1)
    setup_ablation_seed(42)
    started = time.monotonic()
    controls = runner.comparison_configs(tag_prefix)
    for cfg in controls:
        if (any(runner.queue.run_directory(cfg).glob("*"))
                or repository_path(f"results/reports/history_m1{cfg.tag}.csv").exists()):
            raise ValueError("fresh capacity preflight requires two unused training tags")
    fingerprints = code_fingerprints()
    contract = runner.data_identity(controls[0])
    assembled = {cfg.local_route: captured_config(cfg) for cfg in controls}
    if runner.restore_narrow_assembly(assembled["W1"]) != assembled["I3"]:
        raise ValueError("fresh capacity assembly changed more than local widths")
    builder = ConfigBuilder(**assembled["I3"])
    train_loader = builder.get_dataloader(split="train", dos_minmax=True, dos_sumnorm=True, batch_size=32)
    valid_loader = builder.get_dataloader(split="valid", dos_minmax=True, dos_sumnorm=True, batch_size=32)
    train_batches, _ = take(train_loader, 2)
    valid_batches, seen = take(valid_loader, 1)
    order = list(train_loader.sampler)
    data_order = {"train_samples": len(train_loader.dataset), "valid_samples": len(valid_loader.dataset),
        "sampler_seed": train_loader.sampler.seed, "first_train_ids": [
            str(train_loader.dataset.material_ids[i]) for i in order[:32]]}
    largest = int(train_loader.dataset.atom_counts.argmax())
    tail_indices = [i for i in order if i != largest][:31] + [largest]
    tail_batch = crystal_collate([train_loader.dataset[i] for i in tail_indices])
    symmetry_batch = crystal_collate([valid_loader.dataset[i] for i in list(valid_loader.sampler)[:4]])
    report = {"schema": runner.PREFLIGHT_SCHEMA, "passed": False, "runtime": runtime,
        "scope": "disposable real-batch updates; no historical artifacts, formal checkpoints or test evaluation",
        "code_sha256": fingerprints, "data_contract": contract,
        "learning_rate_35": learning_rate_trace(35, controls[0].lr), "controls": {}}
    params_by_route = {route: configuration["model"]["params"]["sub_model"]["transformer"]
                       for route, configuration in assembled.items()}
    for cfg in controls:
        route = cfg.local_route
        builder = ConfigBuilder(**assembled[route])
        params = params_by_route[route]
        model = builder.get_model()
        transformer = model.model["transformer"]
        before_rng = rng_digest(rng_snapshot())
        info = initialize_comparison_model(transformer, params, 42)
        if before_rng != rng_digest(rng_snapshot()):
            raise ValueError("fresh initializer changed caller random streams")
        (validate_i3_model if route == "I3" else validate_width_model)(transformer, params, require_identity=True)
        common_hash = state_hash({key: value for key, value in transformer.state_dict().items()
                                  if not _is_local_parameter(key)})
        entry = {"configuration": asdict(cfg), "assembled_configuration": assembled[route],
            "initialization": info, "matched_nonlocal_state_sha256": common_hash,
            "parameter_counts": width_parameter_counts(transformer), "contract": comparison_contract(cfg, params),
            "data_order": data_order, "stress_batch": {
                "ids": [str(train_loader.dataset.material_ids[i]) for i in tail_indices],
                "largest_structure_atoms": int(train_loader.dataset.atom_counts[largest])}}
        model.to(torch.device("cuda"))
        configure_local_global_execution(transformer)
        setup_ablation_seed(42)
        entry["training_start_rng_sha256"] = rng_digest(rng_snapshot())
        with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring used")):
            entry["forward_backward"] = _train_batch_updates(model, train_batches)
            ids = [str(valid_loader.dataset.material_ids[i]) for i in list(valid_loader.sampler)[:seen]]
            evaluated = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True, sample_ids=ids)
            if not np.isfinite(evaluated["sample_df"].select_dtypes("number").to_numpy()).all():
                raise ValueError("non-finite fresh capacity blind/oracle evaluation")
            entry["evaluation_finite"] = True
            transformer.eval()
            entry["symmetry"] = {}
            with torch.no_grad():
                for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
                    transformer.to(dtype)
                    worst = {}
                    for operation in e3_operations():
                        for key, value in e3_case(model, transformer, symmetry_batch, operation, dtype).items():
                            worst[key] = max(worst.get(key, 0.), value["max_rel"])
                    permutation = permutation_case(model, transformer, symmetry_batch, dtype)
                    if any(value > tolerance for value in worst.values()) or any(
                            value["max_rel"] > tolerance for value in permutation.values()):
                        raise ValueError("fresh capacity E(3)/permutation exceeded tolerance")
                    entry["symmetry"][str(dtype)] = {"e3_max_rel": worst, "permutation": permutation}
            transformer.to(torch.float32)
            entry["checkpoint_roundtrip"] = checkpoint_roundtrip(model, builder, entry, contract)
            other = next(candidate for candidate in controls if candidate.local_route != route)
            payload = build_ablation_checkpoint(2, "M1", 42, False, transformer, model.optimizer["transformer"], .5,
                model_contract=entry["contract"], data_contract=contract, training_rng=capture_training_rng())
            try:
                restore_ablation_checkpoint(payload, transformer, model.optimizer["transformer"], False,
                    model_contract=comparison_contract(other, params_by_route[other.local_route]), data_contract=contract)
            except ValueError as error:
                if "different model contract" not in str(error):
                    raise
            else:
                raise ValueError("fresh widths must reject each other's checkpoints")
            entry["other_width_checkpoint_rejected"] = True
            del payload
        entry["costs"] = [{"kind": "ordinary", **step_cost(model, batch)} for batch in train_batches]
        entry["costs"].append({"kind": "largest_structure", **step_cost(model, tail_batch)})
        entry["geometry_cache"] = transformer._local_geometry_cache.configuration()
        report["controls"][route] = entry
        print(json.dumps({"arm": runner.ROUTE_LABELS[route], "parameters": info["parameter_count"],
                          "costs": entry["costs"]}), flush=True)
        del transformer, model, builder
        gc.collect()
        torch.cuda.empty_cache()
    report.update(passed=True, elapsed_s=time.monotonic() - started)
    runner.validate_preflight(report, controls)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, report)
    print(json.dumps({"passed": True, "elapsed_s": report["elapsed_s"], "report": str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.tag_prefix)


if __name__ == "__main__":
    main()
