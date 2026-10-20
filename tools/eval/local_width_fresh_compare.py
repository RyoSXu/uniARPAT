"""Rebuild the A64/W1 capacity comparison from code and the frozen data only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools.checks.check_long_budget import learning_rate_trace
from tools.eval import global_norm_compare as norm
from tools.eval import i1_bplus_compare as bplus
from tools.eval import legacy_backbone_compare as queue
from tools.eval.lg_paired_compare import load
from tools.eval.lg_structure_diagnosis import dump, paired_group_interval
from tools.eval.lg_valid_export import export, load_run
from tools.eval.local_global_compare import summarize, training_command
from tools.eval.local_width_compare import restore_narrow_assembly
from utils.builder import ConfigBuilder
from utils.checkpoint_selection import BLIND_R2_METRIC
from utils.crystal_cache import crystal_data_contract
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import comparison_contract, validate_local_global_config
from utils.local_structure_experiment import FROZEN
from utils.paths import file_sha256, input_path, repository_path

PREFLIGHT_SCHEMA = "local-width-fresh-preflight-v1"
LABELS = ("W1", "A64")
ROUTE_LABELS = {"I3": "A64", "W1": "W1"}
BOOTSTRAP_SEED = 20261016
SCOPE = ("fresh same-platform A scalar128/MLP256 versus scalar64/MLP128 capacity package; "
         "fixed formulas/residuals/staged L2/G6, joint eDOS/phDOS, seed42, 35 epochs, crystal_q1_v1; "
         "equal-task valid blind median R2 selection; intervals conditional on selected checkpoints; "
         "no historical trained reference and no I1 baseline-adoption conclusion")


def comparison_configs(tag_prefix):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("isolated alphanumeric tag prefix required")
    controls = [ExperimentConfig(**{**FROZEN, "selection_metric": BLIND_R2_METRIC, "local_route": route,
        "tag": f"_lawfresh_{tag_prefix}_{label.lower()}_k2_g6_staged_s42_e35"})
        for route, label in ROUTE_LABELS.items()]
    for cfg in controls:
        validate_local_global_config(cfg)
    return controls


def data_identity(cfg):
    contract = crystal_data_contract(input_path(cfg.data_dir), normalization=cfg.norm, pair_budget=cfg.pair_budget)
    if contract is None:
        raise ValueError("fresh capacity comparison requires the complete crystal_q1_v1 manifest and payloads")
    return contract


def runtime_identity():
    if not torch.cuda.is_available():
        raise ValueError("fresh capacity GPU preflight and training require CUDA")
    device = torch.cuda.current_device()
    return {"python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "numpy", "e3nn", "timm", "pandas", "scipy")},
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "device_memory_bytes": torch.cuda.get_device_properties(device).total_memory}


def validate_preflight(report, controls):
    """Validate a new report without opening any historical run or checkpoint."""
    from tools.checks.check_local_width_fresh import code_fingerprints

    expected_routes = set(ROUTE_LABELS)
    if ([cfg.local_route for cfg in controls] != list(ROUTE_LABELS)
            or report.get("schema") != PREFLIGHT_SCHEMA or report.get("passed") is not True
            or report.get("code_sha256") != code_fingerprints()
            or report.get("runtime") != runtime_identity()
            or report.get("data_contract") != data_identity(controls[0])
            or set(report.get("controls", {})) != expected_routes
            or report.get("learning_rate_35") != learning_rate_trace(35, controls[0].lr)):
        raise ValueError("fresh capacity source, runtime, data or preflight protocol changed")
    for cfg in controls:
        validate_local_global_config(cfg)
        entry = report["controls"][cfg.local_route]
        params = entry["assembled_configuration"]["model"]["params"]["sub_model"]["transformer"]
        if (entry.get("configuration") != asdict(cfg)
                or entry.get("contract") != comparison_contract(cfg, params)
                or entry.get("forward_backward", {}).get("all_parameter_gradients_finite") is not True
                or entry.get("evaluation_finite") is not True
                or set(entry.get("symmetry", {})) != {"torch.float32", "torch.float64"}
                or entry.get("checkpoint_roundtrip") != {
                    "weights_optimizer_rng_equal": True, "foreign_contracts_rejected": True}
                or entry.get("other_width_checkpoint_rejected") is not True
                or len(entry.get("costs", [])) != 3):
            raise ValueError("fresh capacity configuration, contract or real-batch checks changed")
    narrow, wide = (report["controls"][route] for route in ROUTE_LABELS)
    if (restore_narrow_assembly(wide["assembled_configuration"]) != narrow["assembled_configuration"]
            or wide["initialization"].get("narrow_initialization") != narrow["initialization"]
            or wide["initialization"].get("matched_nonlocal_state_sha256") != narrow["matched_nonlocal_state_sha256"]
            or wide["matched_nonlocal_state_sha256"] != narrow["matched_nonlocal_state_sha256"]
            or wide["data_order"] != narrow["data_order"]
            or wide["data_order"].get("train_samples") != 18704 or wide["data_order"].get("valid_samples") != 2313
            or wide["training_start_rng_sha256"] != narrow["training_start_rng_sha256"]):
        raise ValueError("fresh capacity arms differ in common initialization, recipe, data order or start RNG")
    return report


def verify_preflight(path, controls):
    return validate_preflight(json.loads(path.read_text()), controls)


def verify_completed(cfg, preflight):
    queue.verify_completed(cfg, preflight)
    saved, _, _ = load_run(queue.run_directory(cfg))
    if saved["config"] != preflight["controls"][cfg.local_route]["assembled_configuration"]:
        raise ValueError("completed fresh run differs from its assembled preflight recipe")


def artifacts(cfg):
    run = queue.run_directory(cfg)
    history = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
    return {"run_dir": str(run), "files": {name: file_sha256(run / name) for name in
        ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")},
        "history_path": str(history), "history_sha256": file_sha256(history)}


def comparison_report(exports, draws=2000):
    report = norm.comparison_report(exports, draws, labels=LABELS, scope=SCOPE, seed=BOOTSTRAP_SEED)
    report["decisions"] = {
        "width_upgrade_conditions_met": report["joint_blind"]["W1_minus_A64"]["eligible_for_adoption"],
        "I1_baseline_adoption": "not evaluated; no same-platform I1 in this comparison",
        "automatic_baseline_replacement": False, "automatic_followup_training": False}
    return report


def structural_report(pairs):
    result = {"diagnostic_only": True, "scope": "normalized signed differences, composition groups equally weighted",
        "bootstrap": {"draws": 2000, "unit": "composition group", "seed": 42,
                      "training_seed_uncertainty": False}, "summaries": {}, "contrasts": {"W1_minus_A64": {}}}
    for split in ("train", "valid"):
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                filtered = {}
                for label in LABELS:
                    key = f"{label}_{split}_{grouping}_{task}"
                    result["summaries"][key], filtered[label] = bplus.group_summary(pairs[label, split, grouping, task], task)
                result["contrasts"]["W1_minus_A64"][f"{split}_{grouping}_{task}"] = paired_group_interval(
                    filtered["A64"], filtered["W1"])
    return result


def analyze(controls, preflight, out):
    validate_preflight(preflight, controls)
    for cfg in controls:
        verify_completed(cfg, preflight)
    if out.exists():
        raise FileExistsError(f"refusing existing fresh capacity analysis: {out}")
    out.mkdir(parents=True)
    before = {ROUTE_LABELS[cfg.local_route]: artifacts(cfg) for cfg in controls}
    pairs, exports = {}, {}
    for cfg in controls:
        label = ROUTE_LABELS[cfg.local_route]
        run = queue.run_directory(cfg)
        export(run, out / f"{label}_valid_samples.csv", out / f"{label}_summary.json",
               spectra_out=out / f"{label}_valid_spectra.npz")
        exports[label] = load(out / f"{label}_summary.json", out / f"{label}_valid_samples.csv")
        saved, _, _ = load_run(run)
        dataset = ConfigBuilder(**saved["config"]).get_dataset(split="valid", dos_minmax=True, dos_sumnorm=True)
        if len(dataset) != 2313:
            raise ValueError("fresh capacity diagnosis requires complete valid data")
        valid_pairs = bplus.spectrum_pairs(out / f"{label}_valid_spectra.npz", out / f"{label}_valid_samples.csv", dataset)
        train_pairs = bplus.export_train(run, out, label)
        for split, tables in (("train", train_pairs), ("valid", valid_pairs)):
            for (grouping, task), table in tables.items():
                table.to_csv(out / f"{label}_{split}_{grouping}_{task}_pairs.csv", index=False)
                pairs[label, split, grouping, task] = table
        del dataset, valid_pairs, train_pairs
        gc.collect()
        torch.cuda.empty_cache()
    for split in ("train", "valid"):
        bplus.assert_paired_targets(out / f"A64_{split}_spectra.npz", out / f"W1_{split}_spectra.npz")
    report = comparison_report(exports)
    report.update(schema="local-width-fresh-analysis-v1", analysis_complete=True,
        structural_diagnosis=structural_report(pairs), run_artifacts=before,
        code_sha256=preflight["code_sha256"], runtime=preflight["runtime"], data_contract=preflight["data_contract"],
        initializations={ROUTE_LABELS[route]: entry["initialization"] for route, entry in preflight["controls"].items()},
        historical_artifacts_required=False)
    validate_preflight(preflight, controls)
    if before != {ROUTE_LABELS[cfg.local_route]: artifacts(cfg) for cfg in controls}:
        raise ValueError("fresh checkpoints, configs or histories changed during frozen analysis")
    dump(out / "report.json", report)
    return report


def execute(controls, preflight, out):
    validate_preflight(preflight, controls)
    if (out / "analysis").exists():
        raise FileExistsError("existing fresh analysis must be preserved; use --summarize")
    for cfg, row in zip(controls, summarize(controls)):
        if row["status"] == "partial" or row["status"] == "pending" and any(queue.run_directory(cfg).glob("*")):
            raise ValueError("partial fresh run requires a recovery decision; refusing automatic resume")
        if row["status"] == "complete":
            verify_completed(cfg, preflight)
    return queue.execute(controls, preflight, out, analyze_fn=analyze, queue_schema="local-width-fresh-queue-v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    controls = comparison_configs(args.tag_prefix)
    if args.summarize:
        print(json.dumps(summarize(controls), ensure_ascii=False, indent=2))
    elif args.run or args.analyze:
        if args.preflight is None or args.out is None:
            parser.error("--run/--analyze require --preflight and --out")
        preflight = verify_preflight(args.preflight, controls)
        if args.run:
            execute(controls, preflight, args.out)
        else:
            analyze(controls, preflight, args.out / "analysis")
    else:
        for cfg in controls:
            print(shlex.join(training_command(cfg)))


if __name__ == "__main__":
    main()
