"""Bounded I2 real-data checks; only disposable models receive updates."""
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

from tools.checks.check_attention_projection import code_fingerprints as projection_fingerprints
from tools.checks.check_global_depth import e3_case, optimizer_difference, take
from tools.checks.check_global_norm import _train_batch_updates
from tools.checks.check_local_structure_controls import step_cost
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval.i1_bplus_compare import reference_identity
from tools.eval.legacy_backbone_compare import run_directory
from tools.eval.lg_structure_diagnosis import dump
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.builder import ConfigBuilder
from utils.i1_bplus_experiment import DEFAULT_I1_REFERENCE, i1_bplus_config, validate_i2_model
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, restore_training_rng,
)
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import rng_digest, rng_snapshot


def code_fingerprints():
    return {**projection_fingerprints(), **{name: file_sha256(ROOT / name) for name in
        ("utils/i1_bplus_experiment.py", "tools/checks/check_i1_bplus.py", "tools/eval/i1_bplus_compare.py",
         "tools/eval/local_global_compare.py", "utils/zproj_alignment.py")}}


def check(out, reference_dir, tag_prefix):
    if out.exists():
        raise FileExistsError(f"refusing existing I2 preflight: {out}")
    if not sys.flags.utf8_mode or not torch.cuda.is_available():
        raise ValueError("I2 real-data preflight requires PYTHONUTF8=1 and a CUDA GPU")
    torch.set_num_threads(1)
    setup_ablation_seed(42)
    started = time.monotonic()
    saved, reference, identities = reference_identity(reference_dir)
    cfg = i1_bplus_config(reference, tag_prefix)
    if list(run_directory(cfg).glob("*")) or repository_path(f"results/reports/history_m1{cfg.tag}.csv").exists():
        raise ValueError("I2 preflight requires a fresh candidate tag")
    fingerprints = code_fingerprints()
    assembled = captured_config(cfg)
    restored = deepcopy(assembled)
    restored["model"]["params"]["sub_model"]["transformer"].update(local_route="I1")
    if restored != saved["config"]:
        raise ValueError("I2 assembly changes the frozen I1 recipe beyond local content")
    params = assembled["model"]["params"]["sub_model"]["transformer"]
    builder = ConfigBuilder(**assembled)
    model = builder.get_model()
    transformer = model.model["transformer"]
    before_rng = rng_digest(rng_snapshot())
    info = initialize_comparison_model(transformer, params, 42)
    if rng_digest(rng_snapshot()) != before_rng:
        raise ValueError("I2 initializer changed caller CPU/CUDA random streams")
    if info["i1_initialization"] != saved["runtime"]["local_global_init"]:
        raise ValueError("the complete historical untrained I1 state no longer reproduces")
    validate_i2_model(transformer, params, require_identity=True)
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
    historical = json.loads(Path(identities["I1"]["preflight_path"]).read_text())["controls"]["I1"]
    if entry["data_order"] != historical["data_order"]:
        raise ValueError("I2 data views or sampler differ from the common protocol")
    model.to(torch.device("cuda"))
    configure_local_global_execution(transformer)
    if transformer._local_geometry_cache.route != "B+":
        raise ValueError("I2 must cache full B+ geometry")
    setup_ablation_seed(42)
    entry["training_start_rng_sha256"] = rng_digest(rng_snapshot())
    if entry["training_start_rng_sha256"] != saved["runtime"]["structure_training_rng"][0]["state_sha256"]:
        raise ValueError("I2 and historical I1 training-start RNG streams differ")
    with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring used")):
        train_started = time.monotonic()
        entry["forward_backward"] = _train_batch_updates(model, train_batches)
        torch.cuda.synchronize()
        entry["cold_train_batches_elapsed_s"] = time.monotonic() - train_started
        ids = [str(valid_loader.dataset.material_ids[i]) for i in list(valid_loader.sampler)[:seen]]
        evaluated = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True, sample_ids=ids)
        if not np.isfinite(evaluated["sample_df"].select_dtypes("number").to_numpy()).all():
            raise ValueError("non-finite I2 blind/oracle evaluation")
        entry["e3_invariance"] = {}
        transformer.eval()
        with torch.no_grad():
            for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
                transformer.to(dtype)
                worst = {}
                for operation in e3_operations():
                    for key, value in e3_case(model, transformer, valid_batches[0], operation, dtype).items():
                        worst[key] = max(worst.get(key, 0.), value["max_rel"])
                if any(value > tolerance for value in worst.values()):
                    raise ValueError(f"I2 {dtype} E(3) check exceeded tolerance")
                entry["e3_invariance"][str(dtype)] = worst
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
        raise ValueError("I2 weights or optimizer restore differs")
    restore_training_rng(checkpoint["training_rng"])
    if rng_digest(rng_snapshot()) != expected_rng:
        raise ValueError("I2 RNG restore differs")
    try:
        restore_ablation_checkpoint({**checkpoint, "model_contract": saved["model_contract"]},
            fresh.model["transformer"], fresh.optimizer["transformer"], False,
            model_contract=entry["contract"], data_contract=saved["cli"]["data_contract"])
    except ValueError as error:
        if "different model contract" not in str(error):
            raise
    else:
        raise ValueError("I2 accepted an I1 checkpoint contract")
    entry.update(checkpoint_roundtrip=True, i1_checkpoint_rejected=True)
    del fresh, checkpoint, payload, buffer
    gc.collect()
    torch.cuda.empty_cache()
    entry["costs"] = [step_cost(model, batch) for batch in train_batches]
    entry["geometry_cache"] = transformer._local_geometry_cache.configuration()
    if identities != reference_identity(reference_dir)[2] or fingerprints != code_fingerprints():
        raise ValueError("I1 reference or I2 source changed during preflight")
    report = {"schema": "i1-bplus-preflight-v1", "passed": True,
              "scope": "disposable real train-batch updates; train/valid only; no formal checkpoints",
              "references": identities, "code_sha256": fingerprints, "data_contract": saved["cli"]["data_contract"],
              "learning_rate_35": learning_rate_trace(35, cfg.lr), "controls": {"I2": entry},
              "elapsed_s": time.monotonic() - started}
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, report)
    print(json.dumps({"passed": True, "route": "I2", "parameters": info["parameter_count"],
                      "costs": entry["costs"], "elapsed_s": report["elapsed_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_I1_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.reference, args.tag_prefix)


if __name__ == "__main__":
    main()
