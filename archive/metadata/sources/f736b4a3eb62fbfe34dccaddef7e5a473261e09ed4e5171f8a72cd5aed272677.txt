"""固定16对train材料，只更新G2消息和读出，复用冻结／全encoder对照。"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_ablation_experiments import setup_ablation_seed
from tools.eval.g2_encoder_adaptation_probe import AdaptiveEdosProbe, encoder_batch, gradient_equivalence
from tools.eval.g2_frozen_readout_probe import SOURCE, contrast_mse, local_pairs, metric_tables
from tools.eval.g2_small_fit_probe import (
    EVAL_EVERY, N_PAIRS, PLAN, STEPS, configure_encoder_scope, fit_arm, pair_scores,
    select_train_pairs, state_change,
)
from tools.eval.g2_structure_path_audit import file_hash
from utils.ablation_checkpoint import atomic_torch_save
from utils.builder import ConfigBuilder

DESIGN = REPO_ROOT / "docs/design/design-g2-value-fit-probe.md"
PRIOR = REPO_ROOT / "results/g2_small_fit_q1"
PRIOR_RUN = REPO_ROOT / "output/g2_small_fit_q1"
ARM = "g2_only"
RELOAD_TV_TOLERANCE = 1e-5


def reference_path(suffix, prefix=PRIOR):
    return prefix.with_name(prefix.name + suffix)


def load_reference(design=DESIGN, runner_path=None, comparison_prefix=PRIOR):
    """保留旧执行来源；只允许公共小样本helper演进，不改写历史哈希。"""
    prior = json.loads(PRIOR.with_suffix(".json").read_text())
    if (prior["status"], prior["split"], prior["n_pairs"], prior["steps_per_arm"],
            prior["evaluation_interval"], prior["valid_evaluated"], prior["test_evaluated"]) != (
            "complete", "train", N_PAIRS, STEPS, EVAL_EVERY, False, False):
        raise ValueError("incompatible small-fit reference protocol")
    protected = []
    for name, expected in prior["input_sha256"].items():
        path = REPO_ROOT / name
        historical_path = PRIOR_RUN / "script_executed.py" if name == "tools/eval/g2_small_fit_probe.py" else path
        if file_hash(historical_path) != expected:
            raise ValueError(f"historical source changed: {historical_path}")
        protected.extend((path, historical_path))
    cache_path = PRIOR_RUN / "train_cache.pt"
    if file_hash(cache_path) != prior["cache_sha256"]:
        raise ValueError("fixed train cache changed")
    selection = pd.read_csv(reference_path("_selection.csv"))
    pd.testing.assert_frame_equal(selection, select_train_pairs(pd.read_csv(PLAN)), check_dtype=False)
    protected.extend((design, Path(__file__), runner_path or Path(__file__), cache_path, PRIOR.with_suffix(".json")))
    protected.extend(reference_path(suffix) for suffix in (
        "_selection.csv", "_history.csv", "_pair_history.csv", "_samples.csv", "_final_pairs.csv",
        "_summary.csv", "_checkpoint_verification.json"))
    if comparison_prefix != PRIOR:
        comparison = json.loads(comparison_prefix.with_suffix(".json").read_text())
        for key in ("status", "split", "n_pairs", "n_materials", "source_epoch", "initialization_seed",
                    "selection_seed", "steps_per_arm", "evaluation_interval", "valid_evaluated",
                    "test_evaluated", "dropout", "optimizer", "cache_sha256"):
            if comparison[key] != prior[key]:
                raise ValueError(f"comparison protocol differs: {key}")
        pd.testing.assert_frame_equal(selection, pd.read_csv(reference_path("_selection.csv", comparison_prefix)))
        old_run = REPO_ROOT / "output" / comparison_prefix.name
        old_runner = comparison.get("runner", "tools/eval/g2_value_fit_probe.py")
        old_shared = old_run / "value_fit_helper_executed.py"
        training_snapshots = {old_runner: old_run / "script_executed.py",
                              "tools/eval/g2_value_fit_probe.py": old_shared if old_shared.exists() else old_run / "script_executed.py",
                              "tools/eval/g2_small_fit_probe.py": old_run / "small_fit_helper_executed.py"}
        for phase, snapshots in (
                ("input_sha256", training_snapshots),
                ("finalization_input_sha256", {"tools/eval/g2_value_fit_probe.py": old_run / "finalizer_executed.py",
                                               "tools/eval/g2_small_fit_probe.py": old_run / "small_fit_helper_finalized.py"})):
            for name, expected in comparison.get(phase, {}).items():
                path = snapshots.get(name, REPO_ROOT / name)
                if file_hash(path) != expected:
                    raise ValueError(f"comparison source changed: {path}")
                protected.append(path)
        protected.append(comparison_prefix.with_suffix(".json"))
        protected.extend(reference_path(suffix, comparison_prefix) for suffix in (
            "_selection.csv", "_history.csv", "_pair_history.csv", "_samples.csv", "_final_pairs.csv",
            "_summary.csv", "_checkpoint_verification.json"))
    hashes = {str(path.relative_to(REPO_ROOT)): file_hash(path) for path in protected}
    return prior, selection, hashes


def gradient_boundary_check(probe, batch, target, pairs, arm=ARM):
    """实际反向检查：可训练层需收到穿过冻结模块的梯度。"""
    if arm not in ("g2_only", "non_g2", "last_layer"):
        raise ValueError("boundary check requires a complementary encoder scope")
    configure_encoder_scope(probe, arm)
    probe.eval()
    a, b = torch.as_tensor(pairs, device=target.device).T
    prediction = probe(batch)
    loss = contrast_mse(prediction[a], prediction[b], target[a], target[b]).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite preflight objective")
    loss.backward()
    layers = []
    last_layer = len(probe.encoder.layers) - 1
    try:
        for name, parameter in probe.named_parameters():
            is_g2 = name.startswith("encoder.g2_msgs.")
            if arm == "last_layer":
                encoder_allowed = name.startswith(f"encoder.layers.{last_layer}.")
            else:
                encoder_allowed = name.startswith("encoder.") and is_g2 == (arm == "g2_only")
            allowed = name.startswith("readout.") or encoder_allowed
            if parameter.requires_grad != allowed:
                raise RuntimeError(f"unexpected trainable scope: {name}")
            if not allowed and parameter.grad is not None:
                raise RuntimeError(f"frozen parameter received gradient: {name}")
            if allowed and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"missing or nonfinite gradient: {name}")
        modules = probe.encoder.g2_msgs if arm == "g2_only" else probe.encoder.layers
        active_modules = [(last_layer, modules[-1])] if arm == "last_layer" else list(enumerate(modules))
        for i, module in active_modules:
            norm = sum(float(p.grad.square().sum()) for p in module.parameters()) ** .5
            if not norm > 0.:
                raise RuntimeError(f"{arm} layer {i} received no gradient")
            layers.append(dict(layer=i, gradient_l2=norm,
                               trainable_values=sum(p.numel() for p in module.parameters())))
    finally:
        probe.zero_grad(set_to_none=True)
    return dict(loss=float(loss.detach()), layers=layers, arm=arm,
                encoder_total_parameters=sum(p.numel() for p in probe.encoder.parameters()),
                encoder_trainable_parameters=sum(p.numel() for p in probe.encoder.parameters() if p.requires_grad),
                readout_trainable_parameters=sum(p.numel() for p in probe.readout.parameters() if p.requires_grad),
                frozen_gradients_absent=True)


def verify_checkpoints(prototype, run_dir, cache, batch, pairs, first_fit, final_prediction, arm=ARM):
    """从磁盘严格重载见证与最终权重，重新执行encoder→读出。"""
    probe = copy.deepcopy(prototype).to(batch["src"].device).eval()
    configure_encoder_scope(probe, arm)
    frozen_names = {name for name, parameter in probe.named_parameters() if not parameter.requires_grad}
    frozen_names.update(name for name, _ in probe.named_buffers())
    frozen_before = {name: value.detach().cpu() for name, value in prototype.state_dict().items() if name in frozen_names}
    paths = [("final", STEPS)]
    if first_fit is not None:
        paths.insert(0, ("first_fit", first_fit))
    rows = []
    for label, expected_step in paths:
        path = run_dir / f"{arm}_{label}.pth"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["arm"] != arm or saved["step"] != expected_step:
            raise ValueError("checkpoint identity mismatch")
        probe.load_state_dict(saved["probe"], strict=True)
        frozen_change = state_change(frozen_before, probe)
        if frozen_change["changed_values"]:
            raise RuntimeError("reloaded checkpoint changed frozen state")
        with torch.no_grad():
            repeats = [probe(batch).cpu() for _ in range(5)]
        prediction = repeats[0]
        scores = pair_scores(prediction, cache["target"], pairs)
        repeat_tv = max(float(.5 * (p - prediction).abs().sum(-1).max()) for p in repeats)
        for p in repeats[1:]:
            if not np.array_equal(pair_scores(p, cache["target"], pairs).passed, scores.passed):
                raise RuntimeError("pair gate changes across repeated forward passes")
        if repeat_tv > RELOAD_TV_TOLERANCE:
            raise RuntimeError("repeated predictions exceed numerical tolerance")
        if label == "first_fit" and not scores.passed.all():
            raise RuntimeError("reloaded witness failed the all-pair gate")
        maximum_tv = float(.5 * (prediction - final_prediction).abs().sum(-1).max()) if label == "final" else None
        if label == "final" and maximum_tv > RELOAD_TV_TOLERANCE:
            raise RuntimeError("reloaded final checkpoint differs from reported predictions")
        rows.append(dict(checkpoint=str(path.relative_to(REPO_ROOT)), sha256=file_hash(path), step=expected_step,
                         n_pairs_passed=int(scores.passed.sum()), all_pairs_passed=bool(scores.passed.all()),
                         max_mse_ratio=float(scores.mse_ratio.max()), max_tv_ratio=float(scores.tv_ratio.max()),
                         max_prediction_tv_vs_final=maximum_tv, max_repeat_prediction_tv=repeat_tv,
                         frozen_state_changed_values=frozen_change["changed_values"]))
    return dict(status="passed", arm=arm, checkpoints=rows, repeat_forwards=5,
                numerical_tv_tolerance=RELOAD_TV_TOLERANCE)


def run_probe(run_dir, output_prefix, device, finalize_existing=False, *, arm=ARM, design=DESIGN,
              runner_path=None, comparison_prefix=PRIOR):
    os.chdir(REPO_ROOT)
    if arm not in ("g2_only", "non_g2", "last_layer"):
        raise ValueError("unknown complementary encoder arm")
    runner_path = Path(runner_path or __file__).resolve()
    run_dir, output_prefix = Path(run_dir).resolve(), Path(output_prefix).resolve()
    if (run_dir.exists() and not finalize_existing) or list(output_prefix.parent.glob(output_prefix.name + "*")):
        raise FileExistsError("run or result prefix already exists; refusing to overwrite")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    prior, selection, hashes = load_reference(design, runner_path, comparison_prefix)
    comparison = json.loads(comparison_prefix.with_suffix(".json").read_text())
    if finalize_existing:
        metadata = json.loads((run_dir / "config.json").read_text())
        if metadata["arm"] != arm:
            raise ValueError("cannot finalize a different arm")
        shared_snapshot = run_dir / "value_fit_helper_executed.py"
        snapshots = {str(runner_path.relative_to(REPO_ROOT)): run_dir / "script_executed.py",
                     "tools/eval/g2_value_fit_probe.py": shared_snapshot if shared_snapshot.exists() else run_dir / "script_executed.py",
                     "tools/eval/g2_small_fit_probe.py": run_dir / "small_fit_helper_executed.py"}
        for name, expected in metadata["input_sha256"].items():
            path = snapshots.get(name, REPO_ROOT / name)
            if file_hash(path) != expected:
                raise RuntimeError(f"executed input changed: {path}")
        history = pd.read_csv(run_dir / f"{arm}_history.csv")
        if history.step.tolist() != list(range(0, STEPS + 1, EVAL_EVERY)) or not (history.arm == arm).all():
            raise ValueError("cannot finalize an incomplete training history")
        fit_rows = history.loc[history.all_pairs_passed]
        first_fit = int(fit_rows.step.iloc[0]) if len(fit_rows) else None
        for path in (run_dir / "config.json", run_dir / f"{arm}_history.csv", run_dir / f"{arm}_final.pth", *snapshots.values()):
            hashes[str(path.relative_to(REPO_ROOT))] = file_hash(path)
        finalizer_path = run_dir / "finalizer_executed.py"
        if finalizer_path.exists() and file_hash(finalizer_path) != file_hash(Path(__file__)):
            raise FileExistsError("a different finalizer snapshot already exists")
        if not finalizer_path.exists():
            shutil.copyfile(__file__, finalizer_path)
        metadata.update(finalized_from_existing_checkpoint=True, finalization_input_sha256=hashes.copy(),
                        finalizer_snapshot=str(finalizer_path.relative_to(REPO_ROOT)))
    saved_cache = torch.load(PRIOR_RUN / "train_cache.pt", map_location="cpu", weights_only=True)
    cache, inputs, blind_factor = saved_cache["cache"], saved_cache["inputs"], saved_cache["blind_factor"]
    cache["records"] = pd.DataFrame(cache["records"])
    if len(inputs) != 2 * N_PAIRS or len(cache["target"]) != len(inputs):
        raise ValueError("unexpected fixed cache size")
    pairs = local_pairs(selection, cache)
    config = yaml.safe_load((SOURCE / "config_used.yaml").read_text())
    setup_ablation_seed(42)
    model = ConfigBuilder(**copy.deepcopy(config["config"])).get_model()
    source = model.model["transformer"]
    checkpoint = torch.load(SOURCE / "checkpoint_latest.pth", map_location="cpu", weights_only=True)
    source.load_state_dict(checkpoint["model"], strict=True)
    prototype = AdaptiveEdosProbe(source)
    if len(prototype.encoder.g2_msgs) != 6:
        raise ValueError("expected six source G2 message modules")
    del model, source, checkpoint, saved_cache
    probe = copy.deepcopy(prototype).to(device)
    batch = encoder_batch(inputs, range(len(inputs)), device)
    with torch.no_grad():
        initial, gamma = probe(batch, with_gamma=True)
        maximum_tv = float(.5 * (initial.cpu() - cache["original"]).abs().sum(-1).max())
        expected_scale = torch.tensor(cache["records"].blind_scale.to_numpy(), dtype=torch.float32)
        scale_error = float(((gamma.cpu() * blind_factor - expected_scale).abs() / expected_scale.abs().clamp_min(1e-12)).max())
    if maximum_tv > 1e-5 or scale_error > 1e-5:
        raise ValueError("initial cached encoder chain or blind scale failed equivalence")
    if not prior["positive_logit_control"]["passed"]:
        raise ValueError("reference objective positive control failed")
    if not finalize_existing:
        equivalence = dict(max_prediction_tv=maximum_tv, max_relative_blind_scale_error=scale_error,
                           **gradient_equivalence(probe, {"inputs": inputs}, cache, selection, device))
        boundary = gradient_boundary_check(probe, batch, cache["target"].to(device), pairs, arm=arm)
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(runner_path, run_dir / "script_executed.py")
        shutil.copyfile(__file__, run_dir / "value_fit_helper_executed.py")
        shutil.copyfile(REPO_ROOT / "tools/eval/g2_small_fit_probe.py", run_dir / "small_fit_helper_executed.py")
        shutil.copyfile(design, run_dir / "design_executed.md")
        metadata = {key: prior[key] for key in (
            "source_epoch", "initialization_seed", "selection_seed", "steps_per_arm", "evaluation_interval",
            "n_pairs", "n_materials", "split", "valid_evaluated", "test_evaluated", "dropout", "optimizer")}
        metadata.update(status="running", arm=arm, design=str(design.relative_to(REPO_ROOT)), input_sha256=hashes.copy(),
                        runner=str(runner_path.relative_to(REPO_ROOT)),
                        reference=str(PRIOR.with_suffix(".json").relative_to(REPO_ROOT)),
                        comparison_reference=str(comparison_prefix.with_suffix(".json").relative_to(REPO_ROOT)),
                        reference_artifact_deviations=comparison.get("artifact_deviations", comparison.get("reference_artifact_deviations", {})),
                        reference_script_snapshot=str((PRIOR_RUN / "script_executed.py").relative_to(REPO_ROOT)),
                        cache_sha256=prior["cache_sha256"], device=str(device),
                        comparison_factor="encoder parameter updates restricted to " + {
                            "g2_only": "G2 messages", "non_g2": "non-G2 encoder", "last_layer": "last ordinary encoder layer"}[arm],
                        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                        equivalence=equivalence, gradient_boundary=boundary,
                        positive_logit_control_reused=prior["positive_logit_control"])
        (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        print(f"preflight passed: equivalence={equivalence}; gradient boundary={boundary}", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    setup_ablation_seed(42)
    start = time.monotonic()
    if finalize_existing:
        saved = torch.load(run_dir / f"{arm}_final.pth", map_location="cpu", weights_only=True)
        if saved["arm"] != arm or saved["step"] != STEPS:
            raise ValueError("cannot finalize a checkpoint with a different arm or step")
        probe.load_state_dict(saved["probe"], strict=True)
        del saved
    else:
        history, trajectory, first_fit = fit_arm(probe, arm, cache, batch, pairs, device, run_dir)
    with torch.no_grad():
        prediction, gamma = probe(batch, with_gamma=True)
    prediction, scale = prediction.cpu(), gamma.cpu() * blind_factor
    scores = pair_scores(prediction, cache["target"], pairs)
    if finalize_existing:
        recorded = history.iloc[-1]
        actual = [scores.contrast_mse.mean(), scores.contrast_error_tv.mean(), scores.mse_ratio.max(), scores.tv_ratio.max()]
        np.testing.assert_allclose(actual, recorded[["mean_mse", "mean_tv", "max_mse_ratio", "max_tv_ratio"]].to_numpy(dtype=float),
                                   rtol=1e-4, atol=1e-7)
        if int(scores.passed.sum()) != int(recorded.n_pairs_passed):
            raise RuntimeError("reloaded final pass count differs from training history")
        pair_history_path = run_dir / f"{arm}_pair_history.csv"
        if pair_history_path.exists():
            trajectory = pd.read_csv(pair_history_path)
            if sorted(trajectory.step.unique()) != history.step.tolist() or len(trajectory) != len(history) * N_PAIRS:
                raise ValueError("incomplete saved pair history")
        else:
            trajectory = pd.concat([pair_scores(initial.cpu(), cache["target"], pairs).assign(arm=arm, step=0),
                                    scores.assign(arm=arm, step=STEPS)], ignore_index=True)
            trajectory["measurement_source"] = "recomputed_endpoint"
            metadata["artifact_deviations"] = dict(missing_pair_history_steps=history.step.tolist()[1:-1],
                                                    cause="export failed before per-pair trajectory was persisted",
                                                    aggregate_history_complete=True, training_repeated=False)
        metadata["final_prediction_source"] = "reloaded_final_checkpoint"
    atomic_torch_save(dict(predictions={arm: prediction}, blind_scale={arm: scale}), str(run_dir / "final_predictions.pt"))
    changes = {name: state_change(getattr(prototype, name).state_dict(), getattr(probe, name))
               for name in ("encoder", "readout", "eta_head")}
    changes["g2_messages"] = state_change(prototype.encoder.g2_msgs.state_dict(), probe.encoder.g2_msgs)
    non_g2_before = {name: value for name, value in prototype.encoder.state_dict().items() if not name.startswith("g2_msgs.")}
    changes["encoder_except_g2"] = state_change(non_g2_before, probe.encoder)
    frozen_key, active_key = ("encoder_except_g2", "g2_messages") if arm == "g2_only" else ("g2_messages", "encoder_except_g2")
    if arm == "last_layer":
        last_index = len(prototype.encoder.layers) - 1
        last_prefix = f"layers.{last_index}."
        fixed_before = {name: value for name, value in prototype.encoder.state_dict().items() if not name.startswith(last_prefix)}
        changes["encoder_except_last_layer"] = state_change(fixed_before, probe.encoder)
        changes["encoder_last_layer"] = state_change(prototype.encoder.layers[-1].state_dict(), probe.encoder.layers[-1])
        frozen_key, active_key = "encoder_except_last_layer", "encoder_last_layer"
    if changes[frozen_key]["changed_values"] or changes["eta_head"]["changed_values"]:
        raise RuntimeError("frozen parameter boundary violated")
    if not changes[active_key]["changed_values"] or not changes["readout"]["changed_values"]:
        raise RuntimeError("expected parameters did not update")
    verification = verify_checkpoints(prototype, run_dir, cache, batch, pairs, first_fit, prediction, arm=arm)
    arm_cache = {**cache, "records": cache["records"].assign(blind_scale=scale.numpy())}
    samples, final_pairs = metric_tables({arm: {"train": prediction}}, {"train": arm_cache}, selection)
    summary = dict(arm=arm, n_pairs=len(scores), informative_pairs=int(scores.informative.sum()),
                   n_materials=len(samples), n_pairs_passed=int(scores.passed.sum()),
                   mean_mse=float(scores.contrast_mse.mean()), mean_tv=float(scores.contrast_error_tv.mean()),
                   aggregate_mse_ratio=float(scores.contrast_mse.mean() / max(scores.zero_mse.mean(), 1e-12)),
                   aggregate_tv_ratio=float(scores.contrast_error_tv.mean() / max(scores.target_tv.mean(), 1e-12)),
                   max_mse_ratio=float(scores.mse_ratio.max()), max_tv_ratio=float(scores.tv_ratio.max()),
                   first_fit_step=first_fit if first_fit is not None else -1,
                   final_all_pairs_passed=bool(scores.passed.all()))
    for metric in ("r2_edos_oracle", "r2_edos_blind"):
        summary[metric + "_median"] = float(samples[metric].median())
        summary[metric + "_fail_pct"] = float((samples[metric] < 0).mean() * 100)
    tables = dict(history=history,
                  pair_history=trajectory.merge(selection.reset_index(names="pair_index"), on="pair_index", validate="many_to_one"),
                  samples=samples, final_pairs=final_pairs, summary=pd.DataFrame([summary]))
    for name, frame in tables.items():
        reference = pd.read_csv(reference_path("_" + name + ".csv", comparison_prefix))
        if (reference.arm == arm).any():
            raise ValueError("comparison already contains the requested arm")
        if name == "pair_history":
            if "measurement_source" not in reference:
                reference["measurement_source"] = "recorded_during_training"
            if "measurement_source" not in frame:
                frame = frame.assign(measurement_source="recorded_during_training")
        tables[name] = pd.concat([reference, frame], ignore_index=True)
    tables["selection"] = selection
    for name, expected in hashes.items():
        if file_hash(REPO_ROOT / name) != expected:
            raise RuntimeError(f"protected input changed: {name}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
            raise FloatingPointError(f"nonfinite {name} results")
        with output_prefix.with_name(output_prefix.name + "_" + name + ".csv").open("x") as stream:
            frame.to_csv(stream, index=False)
    verdict_prefix = "g2_message" if arm == "g2_only" else arm
    metadata.update(status="complete", verdict=verdict_prefix + ("_updates_sufficient_for_small_fit" if first_fit is not None else "_small_fit_not_demonstrated"),
                    first_fit_step={**comparison["first_fit_step"], arm: first_fit}, final_all_pairs_passed=bool(scores.passed.all()),
                    parameter_changes=changes, checkpoint_verification=verification,
                    elapsed_seconds=time.monotonic() - start,
                    timing_scope="finalization_only" if finalize_existing else "training_and_finalization",
                    training_elapsed_seconds=float(history.iloc[-1].elapsed_seconds),
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                    peak_memory_scope="finalization_only" if finalize_existing else "training_and_finalization",
                    limitations="One fixed train selection and initialization; restricted weight updates do not freeze attention activations; no generalization claim.")
    for suffix, value in ((".json", metadata), ("_checkpoint_verification.json", verification)):
        with output_prefix.with_name(output_prefix.name + suffix).open("x") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    (run_dir / "completed.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(tables["summary"].to_string(index=False), flush=True)
    print(f"verdict: {metadata['verdict']}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "output/g2_value_fit_q1")
    parser.add_argument("--output-prefix", type=Path, default=REPO_ROOT / "results/g2_value_fit_q1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--finalize-existing", action="store_true", help="仅汇总已完成2000步的现有权重，不执行训练")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_probe(args.run_dir, args.output_prefix, torch.device(args.device), finalize_existing=args.finalize_existing)


if __name__ == "__main__":
    main()
