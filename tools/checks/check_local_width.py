"""Bounded real-batch width checks and matched A64/W1 optimizer-step costs."""
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
from tools.checks.check_local_module import code_fingerprints as module_fingerprints
from tools.checks.check_local_readout import checkpoint_roundtrip, permutation_case
from tools.checks.check_local_structure_controls import step_cost
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval.legacy_backbone_compare import run_directory
from tools.eval.lg_structure_diagnosis import dump
from tools.eval.local_width_compare import (
    DEFAULT_EVIDENCE, PREFLIGHT_SCHEMA, reference_identity, restore_narrow_assembly,
)
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.bucket_batch import crystal_collate
from utils.builder import ConfigBuilder
from utils.local_global_experiment import (
    _is_local_parameter, capture_training_rng, comparison_contract, initialize_comparison_model,
)
from utils.local_width_experiment import local_width_config, validate_width_model
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash


def code_fingerprints():
    return {**module_fingerprints(), **{name: file_sha256(ROOT / name) for name in
        ("utils/local_width_experiment.py", "tools/checks/check_local_width.py",
         "tools/eval/local_width_compare.py")}}


def check(out, evidence_dir, tag_prefix):
    if out.exists():
        raise FileExistsError(f"refusing existing width preflight: {out}")
    if not sys.flags.utf8_mode or not torch.cuda.is_available():
        raise ValueError("width preflight requires PYTHONUTF8=1 and an idle CUDA GPU")
    torch.set_num_threads(1)
    setup_ablation_seed(42)
    started = time.monotonic()
    saved, reference, identities = reference_identity(evidence_dir)
    cfg = local_width_config(reference, tag_prefix)
    if (list(run_directory(cfg).glob("*"))
            or repository_path(f"results/reports/history_m1{cfg.tag}.csv").exists()):
        raise ValueError("width preflight requires a fresh candidate tag")
    fingerprints = code_fingerprints()
    assembled = captured_config(cfg)
    if restore_narrow_assembly(assembled) != saved["config"]:
        raise ValueError("width assembly changed the frozen A64 recipe beyond local widths")
    reference_builder = ConfigBuilder(**saved["config"])
    train_loader = reference_builder.get_dataloader(split="train", dos_minmax=True, dos_sumnorm=True,
                                                     batch_size=cfg.batch_size)
    valid_loader = reference_builder.get_dataloader(split="valid", dos_minmax=True, dos_sumnorm=True,
                                                     batch_size=cfg.batch_size)
    train_batches, _ = take(train_loader, 2)
    valid_batches, seen = take(valid_loader, 1)
    order = list(train_loader.sampler)
    data_order = {"train_samples": len(train_loader.dataset), "valid_samples": len(valid_loader.dataset),
                  "sampler_seed": train_loader.sampler.seed, "first_train_ids": [
                      str(train_loader.dataset.material_ids[i]) for i in order[:cfg.batch_size]]}
    if data_order != identities["A64"]["data_order"]:
        raise ValueError("width data views or sampler changed")
    largest = int(train_loader.dataset.atom_counts.argmax())
    tail_indices = [i for i in order if i != largest][:cfg.batch_size - 1] + [largest]
    tail_batch = crystal_collate([train_loader.dataset[i] for i in tail_indices])
    symmetry_batch = crystal_collate([valid_loader.dataset[i] for i in list(valid_loader.sampler)[:4]])
    report = {"schema": PREFLIGHT_SCHEMA, "passed": False,
        "scope": "disposable real-batch updates and cost probes; no formal training, checkpoints or test loader",
        "references": identities, "code_sha256": fingerprints, "data_contract": saved["cli"]["data_contract"],
        "gpu": torch.cuda.get_device_name(), "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "learning_rate_35": learning_rate_trace(35, cfg.lr), "controls": {}}
    for label, configuration in (("A64", saved["config"]), ("W1", assembled)):
        builder = ConfigBuilder(**configuration)
        params = configuration["model"]["params"]["sub_model"]["transformer"]
        model = builder.get_model()
        transformer = model.model["transformer"]
        before_rng = rng_digest(rng_snapshot())
        info = initialize_comparison_model(transformer, params, 42)
        if before_rng != rng_digest(rng_snapshot()):
            raise ValueError("width initializer changed caller random streams")
        common_hash = state_hash({key: value for key, value in transformer.state_dict().items()
                                  if not _is_local_parameter(key)})
        if (info if label == "A64" else info["narrow_initialization"]) != saved["runtime"]["local_global_init"]:
            raise ValueError("historical full-A untrained initialization no longer reproduces")
        entry = {"configuration": asdict(reference if label == "A64" else cfg),
                 "assembled_configuration": configuration, "initialization": info,
                 "matched_nonlocal_state_sha256": common_hash,
                 "contract": comparison_contract(reference if label == "A64" else cfg, params),
                 "data_order": data_order, "stress_batch": {
                     "ids": [str(train_loader.dataset.material_ids[i]) for i in tail_indices],
                     "largest_structure_atoms": int(train_loader.dataset.atom_counts[largest])}}
        if label == "W1":
            validate_width_model(transformer, params, require_identity=True)
            if common_hash != report["narrow_probe"]["matched_nonlocal_state_sha256"]:
                raise ValueError("A64/W1 nonlocal initial tensors differ")
        model.to(torch.device("cuda"))
        configure_local_global_execution(transformer)
        setup_ablation_seed(42)
        entry["training_start_rng_sha256"] = rng_digest(rng_snapshot())
        if entry["training_start_rng_sha256"] != saved["runtime"]["structure_training_rng"][0]["state_sha256"]:
            raise ValueError("width training-start RNG differs from A64")
        with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring used")):
            cold_start = time.monotonic()
            entry["forward_backward"] = _train_batch_updates(model, train_batches)
            torch.cuda.synchronize()
            entry["cold_train_batches_elapsed_s"] = time.monotonic() - cold_start
            if label == "W1":
                ids = [str(valid_loader.dataset.material_ids[i]) for i in list(valid_loader.sampler)[:seen]]
                evaluated = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True, sample_ids=ids)
                if not np.isfinite(evaluated["sample_df"].select_dtypes("number").to_numpy()).all():
                    raise ValueError("non-finite width blind/oracle evaluation")
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
                            raise ValueError("width E(3)/permutation check exceeded tolerance")
                        entry["symmetry"][str(dtype)] = {"e3_max_rel": worst, "permutation": permutation}
                transformer.to(torch.float32)
                entry["checkpoint_roundtrip"] = checkpoint_roundtrip(model, builder, entry, report["data_contract"])
                optimizer = model.optimizer["transformer"]
                payload = build_ablation_checkpoint(2, "M1", 42, False, transformer, optimizer, .5,
                    model_contract=entry["contract"], training_rng=capture_training_rng(),
                    data_contract=report["data_contract"])
                try:
                    restore_ablation_checkpoint(payload, transformer, optimizer, False,
                        model_contract=report["narrow_probe"]["contract"], data_contract=report["data_contract"])
                except ValueError as error:
                    if "different model contract" not in str(error):
                        raise
                else:
                    raise ValueError("width and A64 checkpoint contracts must be isolated")
                entry["narrow_checkpoint_rejected"] = True
                del payload
        entry["costs"] = [{"kind": "ordinary", **step_cost(model, batch)} for batch in train_batches]
        entry["costs"].append({"kind": "largest_structure", **step_cost(model, tail_batch)})
        entry["geometry_cache"] = transformer._local_geometry_cache.configuration()
        entry["cache_statistics_after_probe"] = transformer._local_geometry_cache.statistics()
        if label == "A64":
            report["narrow_probe"] = entry
        else:
            report["controls"]["W1"] = entry
        print(json.dumps({"preflight_arm": label, "parameters": info["parameter_count"],
                          "costs": entry["costs"]}), flush=True)
        del transformer, model, builder
        gc.collect()
        torch.cuda.empty_cache()
    if identities != reference_identity(evidence_dir)[2] or fingerprints != code_fingerprints():
        raise ValueError("width source or historical references changed during preflight")
    report.update(passed=True, elapsed_s=time.monotonic() - started)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, report)
    print(json.dumps({"passed": True, "elapsed_s": report["elapsed_s"], "report": str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.references, args.tag_prefix)


if __name__ == "__main__":
    main()
