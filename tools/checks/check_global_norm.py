"""Disposable real-data N1 checks; formal training remains a separate command."""
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
from tools.checks.check_legacy_backbone import code_fingerprints as legacy_code_fingerprints
from tools.checks.check_local_structure_controls import step_cost
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval.global_norm_compare import evidence_fingerprints, reference_identity, run_directory
from tools.eval.lg_structure_diagnosis import dump
from tools.eval.lg_valid_export import load_run
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.builder import ConfigBuilder
from utils.global_norm_experiment import global_norm_config
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, restore_training_rng,
)
from utils.local_structure_experiment import DEFAULT_REFERENCE
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import rng_digest, rng_snapshot


def code_fingerprints():
    return {**legacy_code_fingerprints(), **{name: file_sha256(ROOT / name) for name in
        ("utils/global_norm_experiment.py", "tools/checks/check_global_norm.py", "tools/eval/global_norm_compare.py")}}


def _train_batch_updates(model, batches):
    """The builder starts in eval; disposable gradient checks must train."""
    transformer = model.model["transformer"]
    transformer.train()
    if not transformer.training or any(not layer.attn.training for layer in transformer.encoder.global_layers):
        raise ValueError("real train-batch probes require training-mode attention")
    losses = []
    for batch in batches:
        values = model.train_one_step(batch, step=0)
        losses.append(float(values["loss"]))
        if not np.isfinite(losses[-1]) or any(
            p.grad is None or not torch.isfinite(p.grad).all()
            for p in transformer.parameters() if p.requires_grad
        ):
            raise ValueError("real train-batch loss/gradients missing or non-finite")
    return {"batches": len(losses), "losses": losses, "all_parameter_gradients_finite": True,
            "training_mode": transformer.training,
            "attention_training_modes": [layer.attn.training for layer in transformer.encoder.global_layers]}


def check(out, reference_dir, tag_prefix, *, route="N1"):
    if route == "N1":
        schema = "global-norm-preflight-v1"
        identity_reader, evidence_reader, fingerprint_reader = reference_identity, evidence_fingerprints, code_fingerprints
        config_factory = global_norm_config
        reference_label, initializer_key = "C1", "c1_initialization"
    elif route == "D1":
        from tools.checks.check_attention_dropout import code_fingerprints as dropout_fingerprints
        from tools.eval.attention_dropout_compare import (
            evidence_fingerprints as dropout_evidence, reference_identity as dropout_references,
        )
        from utils.attention_dropout_experiment import attention_dropout_config

        schema = "attention-dropout-preflight-v1"
        identity_reader, evidence_reader, fingerprint_reader = dropout_references, dropout_evidence, dropout_fingerprints
        config_factory = attention_dropout_config
        reference_label, initializer_key = "N1", "n1_initialization"
    elif route in ("I1", "I0"):
        from tools.checks.check_attention_projection import code_fingerprints as projection_fingerprints
        from tools.eval.attention_projection_compare import (
            evidence_fingerprints as projection_evidence, reference_identity as projection_references,
        )
        from utils.attention_projection_experiment import attention_projection_configs

        schema = "attention-projection-arm-preflight-v1"
        identity_reader, evidence_reader = projection_references, projection_evidence
        fingerprint_reader = projection_fingerprints
        def config_factory(reference, prefix):
            return next(cfg for cfg in attention_projection_configs(reference, prefix) if cfg.local_route == route)
        reference_label, initializer_key = "D1", "d1_initialization"
    else:
        raise ValueError("the shared real-data preflight supports only N1/D1/I1/I0")
    if out.exists():
        raise FileExistsError(f"refusing existing {route} preflight: {out}")
    if not sys.flags.utf8_mode or not torch.cuda.is_available():
        raise ValueError(f"{route} preflight requires PYTHONUTF8=1 and a CUDA GPU")
    torch.set_num_threads(1)
    setup_ablation_seed(42)
    start = time.monotonic()
    saved, reference, identities = identity_reader(reference_dir)
    cfg = config_factory(reference, tag_prefix)
    history = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
    if list(run_directory(cfg).glob("*")) or history.exists():
        raise ValueError(f"{route} preflight requires a fresh tag")
    fingerprints, evidence = fingerprint_reader(), evidence_reader()
    reference_record, _, _ = load_run(Path(identities[reference_label]["run_dir"]))
    assembled = captured_config(cfg)
    restored = deepcopy(assembled)
    restored["model"]["params"]["sub_model"]["transformer"].update(local_route="B+")
    if restored != saved["config"]:
        raise ValueError(f"{route} assembly changed the frozen recipe")
    if route in ("D1", "I1", "I0"):
        n1_assembled = deepcopy(assembled)
        n1_assembled["model"]["params"]["sub_model"]["transformer"].update(local_route=reference_label)
        if n1_assembled != reference_record["config"]:
            raise ValueError(f"{route} assembly changed the {reference_label} recipe beyond the route")
    params = assembled["model"]["params"]["sub_model"]["transformer"]
    builder = ConfigBuilder(**assembled)
    model = builder.get_model()
    transformer = model.model["transformer"]
    before_rng = rng_digest(rng_snapshot())
    info = initialize_comparison_model(transformer, params, 42)
    if rng_digest(rng_snapshot()) != before_rng:
        raise ValueError(f"{route} initializer changed caller random streams")
    if info[initializer_key] != reference_record["runtime"]["local_global_init"]:
        raise ValueError(f"complete untrained {reference_label} initialization no longer reproduces")
    attention_dropout = 0.0 if route in ("D1", "I1", "I0") else params["dropout"]
    if ((route in ("N1", "D1") and info["state_sha256"] != info[initializer_key]["state_sha256"])
            or any(layer.norm_first or layer.attn.dropout != attention_dropout
                   or layer.dropout_attn.p != params["dropout"]
                   or layer.dropout_ffn.p != params["dropout"] or layer.ffn[2].p != params["dropout"]
                   for layer in transformer.encoder.global_layers)):
        raise ValueError(f"{route} changed more than its agreed single factor")
    if route in ("I1", "I0"):
        from utils.attention_projection_experiment import validate_projection_model
        validate_projection_model(transformer, params, require_identity=True)
    entry = {"configuration": asdict(cfg), "assembled_configuration": assembled,
             "initialization": info, "contract": comparison_contract(cfg, params),
             "global_attention_probability_dropout": attention_dropout,
             "residual_and_ffn_dropout": params["dropout"]}
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
    with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring used")):
        entry["forward_backward"] = _train_batch_updates(model, train_batches)
        if route == "I0":
            validate_projection_model(transformer, params, require_identity=True)
            if any(p.grad is not None or p in model.optimizer["transformer"].state
                   for layer in transformer.encoder.global_layers for p in layer.attn.parameters()):
                raise ValueError("fixed identity projections received gradients or optimizer state")
        ids = [str(valid_loader.dataset.material_ids[i]) for i in list(valid_loader.sampler)[:seen]]
        evaluated = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True, sample_ids=ids)
        if not np.isfinite(evaluated["sample_df"].select_dtypes("number").to_numpy()).all():
            raise ValueError(f"non-finite {route} blind/oracle evaluation")
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
                    raise ValueError(f"{route} E(3) check exceeded tolerance")
                entry["e3_invariance"][str(dtype)] = worst
    transformer.to(torch.float32)
    optimizer = model.optimizer["transformer"]
    checkpoint = build_ablation_checkpoint(2, "M1", 42, False, transformer, optimizer, .5,
        model_contract=entry["contract"], training_rng=capture_training_rng(),
        data_contract=saved["cli"]["data_contract"])
    expected_rng = rng_digest(rng_snapshot())
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=True)
    fresh = builder.get_model()
    fresh.to(torch.device("cuda"))
    restore_ablation_checkpoint(payload, fresh.model["transformer"], fresh.optimizer["transformer"], False,
        model_contract=entry["contract"], data_contract=saved["cli"]["data_contract"])
    if any(not torch.equal(value, fresh.model["transformer"].state_dict()[key])
           for key, value in transformer.state_dict().items()) or optimizer_difference(
               optimizer.state_dict()["state"], fresh.optimizer["transformer"].state_dict()["state"]):
        raise ValueError(f"{route} weights/optimizer restore differs")
    restore_training_rng(payload["training_rng"])
    if rng_digest(rng_snapshot()) != expected_rng:
        raise ValueError(f"{route} RNG restore differs")
    entry["checkpoint_roundtrip"] = True
    del fresh, payload, checkpoint, buffer
    gc.collect()
    torch.cuda.empty_cache()
    entry["costs"] = [step_cost(model, batch) for batch in train_batches]
    if route == "I0":
        validate_projection_model(transformer, params, require_identity=True)
        entry["fixed_projection_after_updates"] = True
    for label, identity in identities.items():
        if any(file_sha256(Path(identity["run_dir"]) / name) != digest for name, digest in identity["files"].items()):
            raise ValueError(f"{label} reference changed during {route} preflight")
    if fingerprints != fingerprint_reader() or evidence != evidence_reader():
        raise ValueError(f"source or evidence changed during {route} preflight")
    report = {"schema": schema, "passed": True,
              "scope": "disposable train-batch updates; train/valid only; no formal checkpoints",
              "references": identities, "reference_evidence_sha256": evidence,
              "code_sha256": fingerprints, "data_contract": saved["cli"]["data_contract"],
              "learning_rate_35": learning_rate_trace(35, reference.lr), "controls": {route: entry},
              "elapsed_s": time.monotonic()-start}
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, report)
    print(json.dumps({"passed": True, "route": route, "initial_tensors": info["n_tensors"],
                      "parameters": info["parameter_count"], "elapsed_s": report["elapsed_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.reference, args.tag_prefix)


if __name__ == "__main__":
    main()
