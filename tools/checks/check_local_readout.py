"""Bounded real-data I1/I2/I4 checks using disposable models only."""
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

from tools.checks.check_global_depth import difference, e3_case, optimizer_difference, take
from tools.checks.check_global_norm import _train_batch_updates
from tools.checks.check_i1_bplus import code_fingerprints as bplus_fingerprints
from tools.checks.check_local_structure_controls import step_cost
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval.legacy_backbone_compare import run_directory
from tools.eval.lg_structure_diagnosis import dump
from tools.eval.local_readout_compare import PREFLIGHT_SCHEMA, recipe_identity
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split, setup_ablation_seed
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.builder import ConfigBuilder
from utils.crystal import CrystalGeometry, atom_padding_mask
from utils.i1_bplus_experiment import DEFAULT_I1_REFERENCE
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, restore_training_rng,
)
from utils.local_readout_experiment import GATE_KEY, local_readout_configs, validate_i4_model
from utils.paths import file_sha256, repository_path
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import rng_digest, rng_snapshot


def code_fingerprints():
    return {**bplus_fingerprints(), **{name: file_sha256(ROOT / name) for name in
        ("utils/local_readout_experiment.py", "tools/eval/local_readout_compare.py",
         "tools/checks/check_local_readout.py", "model/losses.py", "model/heads.py", "utils/metrics.py")}}


def gate_zero_equivalence(candidate, reference, batch):
    """Full training objective, backward, atom-input gradients and dropout RNG.

    No optimizer step is taken. The actual training entry's loss is reused so
    H1 eta/gamma supervision is included rather than reconstructed separately.
    """
    start_rng = capture_training_rng()
    captures = []
    try:
        # Repeating the unchanged I2 backward measures the GPU's own numerical
        # variation rather than assuming that a second execution is bitwise.
        for model in (reference, reference, candidate):
            transformer = model.model["transformer"]
            transformer.train()
            restore_training_rng(start_rng)
            capture = {}

            def input_hook(_module, inputs):
                capture["input"] = inputs[0]
                inputs[0].retain_grad()

            def output_hook(_module, _inputs, outputs):
                capture["outputs"] = {key: value.detach().clone() for key, value in outputs.items()}

            handles = [transformer.encoder.register_forward_pre_hook(input_hook),
                       transformer.register_forward_hook(output_hook)]
            try:
                capture["loss"] = model.train_one_step(batch, 0, _accumulation=(True, False, 1.))
            finally:
                for handle in handles:
                    handle.remove()
            capture["rng_sha256"] = rng_digest(rng_snapshot())
            capture["gradients"] = {name: parameter.grad.detach().clone()
                                    for name, parameter in transformer.named_parameters()}
            captures.append(capture)
        first, left, right = captures
        if len({item["rng_sha256"] for item in captures}) != 1:
            raise ValueError("zero-gate I4 changed the I2 training random-number stream")
        dtype = next(reference.model["transformer"].parameters()).dtype
        atol, rtol = (2e-5, 1e-4) if dtype == torch.float32 else (1e-10, 1e-9)
        outputs = {}
        for key in left["outputs"]:
            torch.testing.assert_close(right["outputs"][key], left["outputs"][key], atol=atol, rtol=rtol)
            outputs[key] = difference(left["outputs"][key], right["outputs"][key])
        torch.testing.assert_close(right["input"].grad, left["input"].grad, atol=atol, rtol=rtol)
        input_gradient = difference(left["input"].grad, right["input"].grad)

        def gradient_error(a, b):
            squares, baseline_squares, maximum = [], [], 0.
            for name, gradient in a["gradients"].items():
                other = b["gradients"][name]
                torch.testing.assert_close(other, gradient, atol=atol, rtol=rtol)
                delta = other - gradient
                maximum = max(maximum, float(delta.abs().max()))
                squares.append(delta.double().square().sum())
                baseline_squares.append(gradient.double().square().sum())
            norm = torch.stack(baseline_squares).sum().sqrt().clamp_min(1e-30)
            relative = float(torch.stack(squares).sum().sqrt() / norm)
            return {"max_abs": maximum, "l2_relative": relative}

        repeat = gradient_error(first, left)
        comparison = gradient_error(left, right)
        norm_limit = 2e-4 if dtype == torch.float32 else 1e-9
        if max(repeat["l2_relative"], comparison["l2_relative"]) > norm_limit:
            raise ValueError("zero-gate common gradient relative error exceeded its precision bound")
        if dtype == torch.float32 and comparison["l2_relative"] > 3 * repeat["l2_relative"] + 1e-5:
            raise ValueError("zero-gate FP32 difference exceeds the measured I2 repeat variation")
        gate_gradient = float(candidate.model["transformer"].encoder.local_readout_gate.grad)
        if not np.isfinite(gate_gradient) or gate_gradient == 0:
            raise ValueError("the zero-initialized local readout gate has no finite learning signal")
        if any(model.optimizer["transformer"].state for model in (reference, candidate)):
            raise ValueError("equivalence checks must not update either optimizer")
        return {"passed": True, "outputs": outputs, "atom_input_gradient": input_gradient,
                "common_gradient_error": comparison, "reference_repeat_gradient_error": repeat,
                "common_gradient_tolerance": {"atol": atol, "rtol": rtol, "l2_relative": norm_limit,
                    "fp32_repeat_comparison": "candidate <= 3*I2_repeat + 1e-5"},
                "gate_gradient": gate_gradient, "rng_sha256": left["rng_sha256"],
                "optimizer_updates": 0, "includes_H1_supervision": True}
    finally:
        for model in (reference, candidate):
            model.optimizer["transformer"].zero_grad(set_to_none=True)
        restore_training_rng(start_rng)


def permutation_case(model, transformer, batch, dtype):
    values = model.data_preprocess(batch)
    inp, mask, nval = values[0], values[2], values[15]
    pos = CrystalGeometry(values[1].cell.to(dtype), values[1].frac_coords.to(dtype))
    ex, px = values[16].to(dtype), values[17].to(dtype)
    reference = transformer(inp, mask, pos, ex, px)
    atoms = (~atom_padding_mask(pos, mask)).sum(-1)
    reference.update(reconstruct_h1_spectra(reference, atoms, nval))
    order = torch.arange(inp.shape[1] - 1, -1, -1, device=inp.device)
    shuffled, shuffled_mask = inp[:, order], mask[:, order]
    coords = pos.frac_coords[:, order].clone().masked_fill(shuffled_mask[..., None], float("nan"))
    actual = transformer(shuffled, shuffled_mask, CrystalGeometry(pos.cell, coords), ex, px)
    actual.update(reconstruct_h1_spectra(actual, atoms, nval))
    return {key: difference(reference[key], actual[key]) for key in reference}


def checkpoint_roundtrip(model, builder, entry, data_contract):
    transformer, optimizer = model.model["transformer"], model.optimizer["transformer"]
    payload = build_ablation_checkpoint(2, "M1", 42, False, transformer, optimizer, .5,
        model_contract=entry["contract"], data_contract=data_contract, training_rng=capture_training_rng())
    expected_rng = rng_digest(rng_snapshot())
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    fresh = builder.get_model()
    fresh.to(torch.device("cuda"))
    restore_ablation_checkpoint(checkpoint, fresh.model["transformer"], fresh.optimizer["transformer"], False,
        model_contract=entry["contract"], data_contract=data_contract)
    if (any(not torch.equal(value, fresh.model["transformer"].state_dict()[key])
            for key, value in transformer.state_dict().items())
            or optimizer_difference(optimizer.state_dict()["state"],
                                    fresh.optimizer["transformer"].state_dict()["state"])):
        raise ValueError("local readout checkpoint weights or optimizer restore differs")
    restore_training_rng(checkpoint["training_rng"])
    if rng_digest(rng_snapshot()) != expected_rng:
        raise ValueError("local readout checkpoint RNG restore differs")
    # A distinct architecture or checkpoint selection protocol must fail before
    # state loading, even if many tensor names and shapes happen to coincide.
    for contract in ({**entry["contract"], "selection_metric": "balanced_oracle_mae"},
                     {**entry["contract"], "schema": "different-interface"}):
        try:
            restore_ablation_checkpoint({**checkpoint, "model_contract": contract},
                fresh.model["transformer"], fresh.optimizer["transformer"], False,
                model_contract=entry["contract"], data_contract=data_contract)
        except ValueError as error:
            if "different model contract" not in str(error):
                raise
        else:
            raise ValueError("local readout accepted a different checkpoint contract")
    del fresh, checkpoint, payload, stream
    gc.collect()
    torch.cuda.empty_cache()
    return {"weights_optimizer_rng_equal": True, "foreign_contracts_rejected": True}


def check(out, reference_dir, tag_prefix):
    if out.exists():
        raise FileExistsError(f"refusing existing local readout preflight: {out}")
    if not sys.flags.utf8_mode or not torch.cuda.is_available():
        raise ValueError("local readout preflight requires PYTHONUTF8=1 and a CUDA GPU")
    torch.set_num_threads(1)
    setup_ablation_seed(42)
    started = time.monotonic()
    saved, reference, identity = recipe_identity(reference_dir)
    controls = local_readout_configs(reference, tag_prefix)
    fingerprints = code_fingerprints()
    historical = json.loads(Path(identity["I1"]["preflight_path"]).read_text())["controls"]["I1"]
    report = {"schema": PREFLIGHT_SCHEMA, "passed": False,
              "scope": "disposable real train-batch updates and cost probes; train/valid only; no formal checkpoints",
              "recipe_reference": identity, "code_sha256": fingerprints,
              "data_contract": saved["cli"]["data_contract"], "controls": {},
              "learning_rate_35": learning_rate_trace(35, reference.lr)}
    for cfg in controls:
        history = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
        if list(run_directory(cfg).glob("*")) or history.exists():
            raise ValueError("local readout preflight requires three fresh tags")
        assembled = captured_config(cfg)
        restored = deepcopy(assembled)
        restored["model"]["params"]["sub_model"]["transformer"]["local_route"] = "I1"
        if restored != saved["config"]:
            raise ValueError("local readout assembly changed the frozen recipe beyond the route")
        params = assembled["model"]["params"]["sub_model"]["transformer"]
        builder = ConfigBuilder(**assembled)
        model = builder.get_model()
        transformer = model.model["transformer"]
        before_rng = rng_digest(rng_snapshot())
        info = initialize_comparison_model(transformer, params, 42)
        if before_rng != rng_digest(rng_snapshot()):
            raise ValueError("local readout initialization changed caller random-number streams")
        entry = {"configuration": asdict(cfg), "assembled_configuration": assembled,
                 "initialization": info, "contract": comparison_contract(cfg, params)}
        train_loader = builder.get_dataloader(split="train", dos_minmax=True, dos_sumnorm=True,
                                             batch_size=cfg.batch_size)
        valid_loader = builder.get_dataloader(split="valid", dos_minmax=True, dos_sumnorm=True,
                                             batch_size=cfg.batch_size)
        train_batches, _ = take(train_loader, 2)
        valid_batches, seen = take(valid_loader, 1)
        entry["data_order"] = {"train_samples": len(train_loader.dataset), "valid_samples": len(valid_loader.dataset),
            "sampler_seed": train_loader.sampler.seed, "first_train_ids": [
                str(train_loader.dataset.material_ids[i]) for i in list(train_loader.sampler)[:cfg.batch_size]]}
        if entry["data_order"] != historical["data_order"]:
            raise ValueError("local readout data views or sampler order changed")
        model.to(torch.device("cuda"))
        configure_local_global_execution(transformer)
        setup_ablation_seed(42)
        entry["training_start_rng_sha256"] = rng_digest(rng_snapshot())
        with patch("utils.rp_encoding.RPEncoding.forward", side_effect=AssertionError("old geometry scoring used")):
            if cfg.local_route == "I4":
                validate_i4_model(transformer, params, require_zero=True, require_identity=True)
                common_config = deepcopy(assembled)
                common_params = common_config["model"]["params"]["sub_model"]["transformer"]
                common_params["local_route"] = "I2"
                common = ConfigBuilder(**common_config).get_model()
                initialize_comparison_model(common.model["transformer"], common_params, 42)
                common.to(torch.device("cuda"))
                configure_local_global_execution(common.model["transformer"])
                setup_ablation_seed(42)
                precision = {}
                for dtype in (torch.float64, torch.float32):
                    transformer.to(dtype)
                    common.model["transformer"].to(dtype)
                    setup_ablation_seed(42)
                    precision[str(dtype)] = gate_zero_equivalence(model, common, train_batches[0])
                    print(json.dumps({"zero_gate_precision": str(dtype), **precision[str(dtype)]}), flush=True)
                report["gate_zero_equivalence"] = {"passed": True, "precision": precision}
                del common
                gc.collect()
                torch.cuda.empty_cache()
                setup_ablation_seed(42)
            entry["forward_backward"] = _train_batch_updates(model, train_batches)
            ids = [str(valid_loader.dataset.material_ids[i]) for i in list(valid_loader.sampler)[:seen]]
            evaluated = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True, sample_ids=ids)
            if not np.isfinite(evaluated["sample_df"].select_dtypes("number").to_numpy()).all():
                raise ValueError("non-finite local readout blind/oracle evaluation")
            transformer.eval()
            entry["symmetry"] = {}
            gate = getattr(transformer.encoder, "local_readout_gate", None)
            learned_gate = None if gate is None else float(gate.detach())
            if gate is not None:
                with torch.no_grad():
                    gate.fill_(.25)
                entry["nonzero_symmetry_gate"] = .25
            with torch.no_grad():
                for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
                    transformer.to(dtype)
                    worst = {}
                    for operation in e3_operations():
                        for key, value in e3_case(model, transformer, valid_batches[0], operation, dtype).items():
                            worst[key] = max(worst.get(key, 0.), value["max_rel"])
                    permutation = permutation_case(model, transformer, valid_batches[0], dtype)
                    if any(value > tolerance for value in worst.values()) or any(
                            value["max_rel"] > tolerance for value in permutation.values()):
                        raise ValueError("local readout E(3)/permutation check exceeded tolerance")
                    entry["symmetry"][str(dtype)] = {"e3_max_rel": worst, "permutation": permutation}
            transformer.to(torch.float32)
            if learned_gate is not None:
                with torch.no_grad():
                    transformer.encoder.local_readout_gate.fill_(learned_gate)
                entry["gate_after_two_batches"] = learned_gate
        entry["checkpoint_roundtrip"] = checkpoint_roundtrip(model, builder, entry, report["data_contract"])
        entry["costs"] = [step_cost(model, batch) for batch in train_batches]
        entry["geometry_cache"] = transformer._local_geometry_cache.configuration()
        report["controls"][cfg.local_route] = entry
        print(json.dumps({"preflight_arm": cfg.local_route, "parameters": info.get("parameter_count"),
                          "costs": entry["costs"]}), flush=True)
        del model, transformer, builder, train_loader, valid_loader, train_batches, valid_batches
        gc.collect()
        torch.cuda.empty_cache()
    infos = {route: report["controls"][route]["initialization"] for route in report["controls"]}
    if (infos["I1"] != saved["runtime"]["local_global_init"]
            or infos["I2"]["i1_initialization"] != infos["I1"]
            or infos["I4"]["bplus_initialization"] != infos["I2"]
            or infos["I4"]["common_state_sha256"] != infos["I2"]["state_sha256"]
            or any(entry["training_start_rng_sha256"]
                   != saved["runtime"]["structure_training_rng"][0]["state_sha256"]
                   for entry in report["controls"].values())):
        raise ValueError("the fresh controls do not match their complete untrained reference or start RNG")
    if identity != recipe_identity(reference_dir)[2] or fingerprints != code_fingerprints():
        raise ValueError("source or historical recipe changed during local readout preflight")
    report.update(passed=True, elapsed_s=time.monotonic() - started)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, report)
    print(json.dumps({"passed": True, "routes": list(report["controls"]),
                      "gate_zero_equivalence": report["gate_zero_equivalence"],
                      "elapsed_s": report["elapsed_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_I1_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.reference, args.tag_prefix)


if __name__ == "__main__":
    main()
