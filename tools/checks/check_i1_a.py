"""Bounded corrected-backbone A checks and real optimizer-step GPU costs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
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
from tools.checks.check_global_norm import _train_batch_updates
from tools.checks.check_i1_bplus import code_fingerprints as bplus_fingerprints
from tools.checks.check_local_structure_controls import step_cost
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval.i1_a_compare import reference_identity
from tools.eval.legacy_backbone_compare import run_directory
from tools.eval.lg_structure_diagnosis import dump
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.bucket_batch import crystal_collate
from utils.builder import ConfigBuilder
from utils.i1_a_experiment import DEFAULT_I2_REFERENCE, I2_EVIDENCE, i1_a_config, validate_i3_model
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, restore_training_rng,
)
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import rng_digest, rng_snapshot


def code_fingerprints():
    return {**bplus_fingerprints(), **{name: file_sha256(ROOT / name) for name in
        ("utils/i1_a_experiment.py", "tools/checks/check_i1_a.py", "tools/eval/i1_a_compare.py")}}


def check(out, reference_dir, tag_prefix, *, identity_loader=None, configuration_factory=None,
          fingerprint_provider=None, report_schema="i1-a-preflight-v1", permutation_probe=None):
    """Reuse the full A checks with an explicitly supplied comparison protocol.

    Defaults retain the historical Oracle-selected experiment. New protocols
    supply their own reference authentication, configuration and source gate.
    """
    if out.exists():
        raise FileExistsError(f"refusing existing I3 preflight: {out}")
    if not sys.flags.utf8_mode or not torch.cuda.is_available():
        raise ValueError("I3 real-data preflight requires PYTHONUTF8=1 and a CUDA GPU")
    torch.set_num_threads(1)
    setup_ablation_seed(42)
    started = time.monotonic()
    load_identity = identity_loader or reference_identity
    make_configuration = configuration_factory or i1_a_config
    fingerprints_for_run = fingerprint_provider or code_fingerprints
    saved, reference, identities = load_identity(reference_dir)
    cfg = make_configuration(reference, tag_prefix)
    if list(run_directory(cfg).glob("*")) or repository_path(f"results/reports/history_m1{cfg.tag}.csv").exists():
        raise ValueError("I3 preflight requires a fresh candidate tag")
    fingerprints = fingerprints_for_run()
    assembled = captured_config(cfg)
    restored = deepcopy(assembled)
    restored["model"]["params"]["sub_model"]["transformer"].update(local_route="I2")
    if restored != saved["config"]:
        raise ValueError("I3 assembly changes the frozen I2 joint recipe beyond local content")
    params = assembled["model"]["params"]["sub_model"]["transformer"]
    builder = ConfigBuilder(**assembled)
    model = builder.get_model()
    transformer = model.model["transformer"]
    before_rng = rng_digest(rng_snapshot())
    info = initialize_comparison_model(transformer, params, 42)
    if rng_digest(rng_snapshot()) != before_rng:
        raise ValueError("I3 initializer changed caller CPU/CUDA random streams")
    if info["i2_initialization"] != saved["runtime"]["local_global_init"]:
        raise ValueError("the complete historical untrained I2 state no longer reproduces")
    if info["shared_control_state_sha256"] != info["i2_initialization"]["shared_control_state_sha256"]:
        raise ValueError("I3 and I2 common initial tensors differ")
    validate_i3_model(transformer, params, require_identity=True)
    entry = {"configuration": asdict(cfg), "assembled_configuration": assembled,
             "initialization": info, "contract": comparison_contract(cfg, params)}
    train_loader = builder.get_dataloader(split="train", dos_minmax=True, dos_sumnorm=True,
                                         batch_size=cfg.batch_size)
    valid_loader = builder.get_dataloader(split="valid", dos_minmax=True, dos_sumnorm=True,
                                         batch_size=cfg.batch_size)
    train_batches, _ = take(train_loader, 2)
    valid_batches, seen = take(valid_loader, 1)
    train_order = list(train_loader.sampler)
    entry["data_order"] = {"train_samples": len(train_loader.dataset), "valid_samples": len(valid_loader.dataset),
                          "sampler_seed": train_loader.sampler.seed,
                          "first_train_ids": [str(train_loader.dataset.material_ids[i])
                                              for i in train_order[:cfg.batch_size]]}
    historical = json.loads((ROOT / I2_EVIDENCE / "preflight_final.json").read_text())["controls"]["I2"]
    if entry["data_order"] != historical["data_order"]:
        raise ValueError("I3 data views or sampler differ from the common protocol")
    largest = int(train_loader.dataset.atom_counts.argmax())
    tail_indices = [i for i in train_order if i != largest][:cfg.batch_size - 1] + [largest]
    tail_batch = crystal_collate([train_loader.dataset[i] for i in tail_indices])
    symmetry_batch = crystal_collate([valid_loader.dataset[i] for i in list(valid_loader.sampler)[:4]])
    entry["stress_batch"] = {"ids": [str(train_loader.dataset.material_ids[i]) for i in tail_indices],
                             "largest_structure_atoms": int(train_loader.dataset.atom_counts[largest])}
    model.to(torch.device("cuda"))
    configure_local_global_execution(transformer)
    if transformer._local_geometry_cache.route != "A":
        raise ValueError("I3 must cache full A geometry")
    setup_ablation_seed(42)
    entry["training_start_rng_sha256"] = rng_digest(rng_snapshot())
    if entry["training_start_rng_sha256"] != saved["runtime"]["structure_training_rng"][0]["state_sha256"]:
        raise ValueError("I3 and historical I2 training-start RNG streams differ")
    with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring used")):
        train_started = time.monotonic()
        entry["forward_backward"] = _train_batch_updates(model, train_batches)
        torch.cuda.synchronize()
        entry["cold_train_batches_elapsed_s"] = time.monotonic() - train_started
        ids = [str(valid_loader.dataset.material_ids[i]) for i in list(valid_loader.sampler)[:seen]]
        evaluated = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True, sample_ids=ids)
        if not np.isfinite(evaluated["sample_df"].select_dtypes("number").to_numpy()).all():
            raise ValueError("non-finite I3 blind/oracle evaluation")
        entry["e3_invariance"] = {"samples": 4, "scope": "real valid structures, both outputs and spectra"}
        transformer.eval()
        with torch.no_grad():
            for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
                transformer.to(dtype)
                worst = {}
                for operation in e3_operations():
                    for key, value in e3_case(model, transformer, symmetry_batch, operation, dtype).items():
                        worst[key] = max(worst.get(key, 0.), value["max_rel"])
                if any(value > tolerance for value in worst.values()):
                    raise ValueError(f"I3 {dtype} E(3) check exceeded tolerance")
                entry["e3_invariance"][str(dtype)] = worst
                if permutation_probe is not None:
                    permutation = permutation_probe(model, transformer, symmetry_batch, dtype)
                    if any(value["max_rel"] > tolerance for value in permutation.values()):
                        raise ValueError(f"I3 {dtype} permutation check exceeded tolerance")
                    entry.setdefault("permutation_invariance", {})[str(dtype)] = permutation
    transformer.to(torch.float32)
    optimizer = model.optimizer["transformer"]
    payload = build_ablation_checkpoint(2, "M1", 42, False, transformer, optimizer, .5,
        model_contract=entry["contract"], training_rng=capture_training_rng(),
        data_contract=saved["cli"]["data_contract"])
    expected_rng = rng_digest(rng_snapshot())
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    fresh = builder.get_model()
    fresh.to(torch.device("cuda"))
    restore_ablation_checkpoint(checkpoint, fresh.model["transformer"], fresh.optimizer["transformer"], False,
        model_contract=entry["contract"], data_contract=saved["cli"]["data_contract"])
    if any(not torch.equal(value, fresh.model["transformer"].state_dict()[key])
           for key, value in transformer.state_dict().items()) or optimizer_difference(
               optimizer.state_dict()["state"], fresh.optimizer["transformer"].state_dict()["state"]):
        raise ValueError("I3 weights or optimizer restore differs")
    restore_training_rng(checkpoint["training_rng"])
    if rng_digest(rng_snapshot()) != expected_rng:
        raise ValueError("I3 RNG restore differs")
    try:
        restore_ablation_checkpoint({**checkpoint, "model_contract": saved["model_contract"]},
            fresh.model["transformer"], fresh.optimizer["transformer"], False,
            model_contract=entry["contract"], data_contract=saved["cli"]["data_contract"])
    except ValueError as error:
        if "different model contract" not in str(error):
            raise
    else:
        raise ValueError("I3 accepted an I2 checkpoint contract")
    alternate_metric = ("balanced_oracle_mae" if cfg.selection_metric == "balanced_blind_r2"
                        else "balanced_blind_r2")
    other_contract = comparison_contract(replace(cfg, selection_metric=alternate_metric), params)
    try:
        restore_ablation_checkpoint({**checkpoint, "model_contract": other_contract},
            fresh.model["transformer"], fresh.optimizer["transformer"], False,
            model_contract=entry["contract"], data_contract=saved["cli"]["data_contract"])
    except ValueError as error:
        if "different model contract" not in str(error):
            raise
    else:
        raise ValueError("I3 accepted a different checkpoint selection protocol")
    entry.update(checkpoint_roundtrip=True, i2_checkpoint_rejected=True,
                 selection_checkpoint_rejected=True)
    del fresh, checkpoint, payload, buffer
    gc.collect()
    torch.cuda.empty_cache()
    entry["costs"] = [{"kind": "ordinary", **step_cost(model, batch)} for batch in train_batches]
    entry["costs"].append({"kind": "largest_structure", **step_cost(model, tail_batch)})
    entry["geometry_cache"] = transformer._local_geometry_cache.configuration()
    if identities != load_identity(reference_dir)[2] or fingerprints != fingerprints_for_run():
        raise ValueError("I2/I1 references or I3 source changed during preflight")
    report = {"schema": report_schema, "passed": True,
              "scope": "disposable real train-batch updates; train/valid only; no formal checkpoints",
              "references": identities, "code_sha256": fingerprints, "data_contract": saved["cli"]["data_contract"],
              "learning_rate_35": learning_rate_trace(35, cfg.lr), "controls": {"I3": entry},
              "elapsed_s": time.monotonic() - started}
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, report)
    print(json.dumps({"passed": True, "route": "I3", "parameters": info["parameter_count"],
                      "costs": entry["costs"], "elapsed_s": report["elapsed_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_I2_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.reference, args.tag_prefix)


if __name__ == "__main__":
    main()
