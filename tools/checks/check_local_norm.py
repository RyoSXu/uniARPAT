"""Bounded preflight for complete A internal pre-norm and frozen A64 reuse."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from model.transformer import Transformer
from tools.checks.check_global_depth import e3_case, take
from tools.checks.check_global_norm import _train_batch_updates
from tools.checks.check_local_readout import checkpoint_roundtrip, permutation_case
from tools.checks.check_local_structure_controls import step_cost
from tools.checks.check_local_width_fresh import code_fingerprints as fresh_fingerprints
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval import local_norm_compare as runner
from tools.eval.lg_structure_diagnosis import dump
from tools.eval.lg_valid_export import select_best
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.bucket_batch import crystal_collate
from utils.builder import ConfigBuilder
from utils.local_global_experiment import capture_training_rng, comparison_contract, initialize_comparison_model
from utils.local_norm_experiment import local_norm_config, validate_local_norm_model
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import rng_digest, rng_snapshot


def code_fingerprints():
    return {**fresh_fingerprints(), **{name: file_sha256(ROOT / name) for name in
        ("utils/local_norm_experiment.py", "tools/eval/local_norm_compare.py", "tools/checks/check_local_norm.py")}}


def historical_module(name, reference):
    source = runner.historical_source(reference["reference_commit"], name, reference["historical_source_sha256"][name])
    module_name = "_local_norm_historical_" + Path(name).stem
    module = ModuleType(module_name)
    module.__file__ = str(ROOT / name)
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def post_norm_compatibility(params, reference):
    """Execute authenticated historical modules, comparing the full DOS model and gradients."""
    from tests.model.test_local_global_encoder import fixture

    messages = historical_module("model/periodic_messages.py", reference)
    encoder = historical_module("model/local_global_encoder.py", reference)
    encoder.PeriodicLocalMessage = messages.PeriodicLocalMessage
    encoder.prepare_shared_message_geometry = messages.prepare_shared_message_geometry
    historical = historical_module("model/transformer.py", reference)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        current = Transformer(**params)
        info = initialize_comparison_model(current, params, 42)
        with patch("model.local_global_encoder.LocalGlobalEncoder", encoder.LocalGlobalEncoder):
            previous = historical.Transformer(**params)
        previous.load_state_dict(current.state_dict(), strict=True)
    result = {"passed": False, "reference_commit": reference["reference_commit"],
              "initialization": info, "dtypes": {}}
    for dtype in (torch.float32, torch.float64):
        z, geometry = fixture(dtype)
        ex, px = torch.linspace(-6, 6, params["edos_num"], dtype=dtype), torch.linspace(0, 1200, params["phdos_num"], dtype=dtype)
        outputs, gradients = [], []
        for model in (previous, current):
            model.to(dtype).eval()
            model.zero_grad(set_to_none=True)
            prediction = model(z, z.eq(0), geometry, ex, px)
            sum(value.square().mean() for value in prediction.values()).backward()
            outputs.append({key: value.detach().clone() for key, value in prediction.items()})
            gradients.append({key: value.grad.detach().clone() for key, value in model.named_parameters()
                              if value.grad is not None})
        if gradients[0].keys() != dict(current.named_parameters()).keys() or gradients[0].keys() != gradients[1].keys():
            raise ValueError("historical post-norm gradient coverage changed")
        for left, right in ((outputs[0], outputs[1]), (gradients[0], gradients[1])):
            for key in left:
                torch.testing.assert_close(left[key], right[key], atol=0, rtol=0, msg=lambda text: key + ": " + text)
        result["dtypes"][str(dtype)] = {"outputs_bitwise_equal": True, "parameter_gradients_bitwise_equal": True,
                                        "gradient_tensors": len(gradients[0])}
        del outputs, gradients, prediction
    del current, previous
    gc.collect()
    result["passed"] = True
    return result


def replay_reference(saved, reference, batches, ids):
    """Replay a bounded real valid batch from the old selected checkpoint without updating it."""
    run = Path(reference["run_dir"])
    picked = select_best(run, saved["model_contract"]["selection_metric"], saved)
    model = ConfigBuilder(**saved["config"]).get_model()
    transformer = model.model["transformer"]
    transformer.load_state_dict(picked["checkpoint"]["model"], strict=True)
    del picked
    model.to(torch.device("cuda"))
    configure_local_global_execution(transformer)
    evaluated = evaluate_split(model, batches, sumnorm=True, return_sample_level=True, return_spectra=True, sample_ids=ids)
    with np.load(reference["exports"]["A64_valid_spectra.npz"]["source"]) as frozen:
        lookup = {str(value): i for i, value in enumerate(frozen["material_ids"])}
        order = [lookup[value] for value in ids]
        errors = {}
        for key, actual in evaluated["sample_spectra"].items():
            expected = frozen[key][order]
            error = float(np.max(np.abs(actual - expected)))
            relative = error / max(float(np.max(np.abs(expected))), 1e-12)
            if relative > 1e-4 or key.startswith("tgt_") and not np.array_equal(actual, expected):
                raise ValueError("post-norm reference prediction replay differs: " + key)
            errors[key] = {"max_abs": error, "max_rel": relative}
    del transformer, model
    gc.collect()
    torch.cuda.empty_cache()
    return {"passed": True, "materials": len(ids), "material_ids": ids, "spectra": errors}


def check(out, evidence_dir, tag_prefix, reference_commit=runner.REFERENCE_COMMIT):
    if out.exists():
        raise FileExistsError("existing local norm preflight must be preserved")
    if not sys.flags.utf8_mode:
        raise ValueError("local norm preflight requires PYTHONUTF8=1")
    torch.set_num_threads(1)
    started = time.monotonic()
    saved, reference_cfg, reference = runner.reference_identity(evidence_dir, reference_commit)
    cfg = local_norm_config(reference_cfg, tag_prefix)
    if (any(runner.queue.run_directory(cfg).glob("*"))
            or repository_path(f"results/reports/history_m1{cfg.tag}.csv").exists()):
        raise ValueError("preflight requires an unused LN1 training tag")
    fingerprints = code_fingerprints()
    assembled = captured_config(cfg)
    if runner.restore_post_assembly(assembled) != saved["config"]:
        raise ValueError("LN1 changes more than the internal normalization route")
    params = assembled["model"]["params"]["sub_model"]["transformer"]
    compatibility = post_norm_compatibility(saved["model_contract"]["transformer"], reference)
    if compatibility["initialization"] != saved["runtime"]["local_global_init"]:
        raise ValueError("current canonical post-norm initialization differs from historical A64")
    print(json.dumps({"reference_reuse": "authenticated", "post_norm_compatibility": "passed"}), flush=True)
    builder = ConfigBuilder(**assembled)
    train_loader = builder.get_dataloader(split="train", dos_minmax=True, dos_sumnorm=True, batch_size=32)
    valid_loader = builder.get_dataloader(split="valid", dos_minmax=True, dos_sumnorm=True, batch_size=32)
    train_batches, _ = take(train_loader, 2)
    valid_batches, seen = take(valid_loader, 1)
    order = list(train_loader.sampler)
    data_order = {"train_samples": len(train_loader.dataset), "valid_samples": len(valid_loader.dataset),
        "sampler_seed": train_loader.sampler.seed,
        "first_train_ids": [str(train_loader.dataset.material_ids[i]) for i in order[:32]]}
    if data_order != reference["data_order"]:
        raise ValueError("LN1 sampling or complete train/valid views differ from A64")
    largest = int(train_loader.dataset.atom_counts.argmax())
    tail_indices = [i for i in order if i != largest][:31] + [largest]
    tail_batch = crystal_collate([train_loader.dataset[i] for i in tail_indices])
    symmetry_batch = crystal_collate([valid_loader.dataset[i] for i in list(valid_loader.sampler)[:4]])
    ids = [str(valid_loader.dataset.material_ids[i]) for i in list(valid_loader.sampler)[:seen]]
    replay = replay_reference(saved, reference, valid_batches, ids)
    print(json.dumps({"reference_prediction_replay": "passed", "materials": seen}), flush=True)
    model = builder.get_model()
    transformer = model.model["transformer"]
    info = initialize_comparison_model(transformer, params, 42)
    validate_local_norm_model(transformer, params, require_identity=True)
    entry = {"configuration": asdict(cfg), "assembled_configuration": assembled, "initialization": info,
             "contract": comparison_contract(cfg, params), "data_order": data_order,
             "stress_batch": {"ids": [str(train_loader.dataset.material_ids[i]) for i in tail_indices],
                              "largest_structure_atoms": int(train_loader.dataset.atom_counts[largest])}}
    model.to(torch.device("cuda"))
    configure_local_global_execution(transformer)
    setup_ablation_seed(42)
    entry["training_start_rng_sha256"] = rng_digest(rng_snapshot())
    with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring used")):
        entry["forward_backward"] = _train_batch_updates(model, train_batches)
        evaluated = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True, sample_ids=ids)
        if not np.isfinite(evaluated["sample_df"].select_dtypes("number").to_numpy()).all():
            raise ValueError("non-finite pre-norm blind/oracle evaluation")
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
                    raise ValueError("internal pre-norm E(3)/permutation exceeded tolerance")
                entry["symmetry"][str(dtype)] = {"e3_max_rel": worst, "permutation": permutation}
        transformer.to(torch.float32)
        entry["checkpoint_roundtrip"] = checkpoint_roundtrip(model, builder, entry, saved["cli"]["data_contract"])
        payload = build_ablation_checkpoint(2, "M1", 42, False, transformer, model.optimizer["transformer"], .5,
            model_contract=entry["contract"], data_contract=saved["cli"]["data_contract"], training_rng=capture_training_rng())
        try:
            restore_ablation_checkpoint(payload, transformer, model.optimizer["transformer"], False,
                model_contract=saved["model_contract"], data_contract=saved["cli"]["data_contract"])
        except ValueError as error:
            if "different model contract" not in str(error):
                raise
        else:
            raise ValueError("pre/post norm must reject each other's same-shape checkpoint")
        entry["post_norm_checkpoint_rejected"] = True
        del payload
    entry["costs"] = [{"kind": "ordinary", **step_cost(model, batch)} for batch in train_batches]
    entry["costs"].append({"kind": "largest_structure", **step_cost(model, tail_batch)})
    entry["geometry_cache"] = transformer._local_geometry_cache.configuration()
    report = {"schema": runner.PREFLIGHT_SCHEMA, "passed": True, "tag_prefix": tag_prefix,
        "scope": "disposable real-batch steps, historical post-norm replay; no formal training or test evaluation",
        "code_sha256": fingerprints, "runtime": runner.fresh.runtime_identity(),
        "data_contract": saved["cli"]["data_contract"], "references": {"A64": reference},
        "post_norm_compatibility": compatibility, "reference_prediction_replay": replay,
        "learning_rate_35": learning_rate_trace(35, cfg.lr), "controls": {"LN1": entry},
        "elapsed_s": time.monotonic() - started}
    del transformer, model, builder
    gc.collect()
    torch.cuda.empty_cache()
    runner.validate_preflight(report, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, report)
    print(json.dumps({"passed": True, "elapsed_s": report["elapsed_s"], "costs": entry["costs"], "report": str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=runner.DEFAULT_EVIDENCE)
    parser.add_argument("--reference-commit", default=runner.REFERENCE_COMMIT)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.references, args.tag_prefix, args.reference_commit)


if __name__ == "__main__":
    main()
