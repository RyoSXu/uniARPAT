"""Train the full-A 128/256 candidate; reuse verified A64 and I1 results."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.checks.check_long_budget import learning_rate_trace
from tools.eval import global_norm_compare as norm
from tools.eval import i1_bplus_compare as bplus
from tools.eval import local_module_compare as narrow
from tools.eval.legacy_backbone_compare import run_directory, verify_completed
from tools.eval.lg_paired_compare import load
from tools.eval.lg_structure_diagnosis import dump, paired_group_interval
from tools.eval.lg_valid_export import export, load_run
from tools.eval.local_global_compare import summarize, training_command
from utils.builder import ConfigBuilder
from utils.checkpoint_selection import BLIND_R2_METRIC
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import comparison_contract, validate_local_global_config
from utils.local_width_experiment import local_width_config, narrow_reference_parameters
from utils.paths import file_sha256, repository_path

DEFAULT_EVIDENCE = ROOT / "results/runs/local_module_selection_20261016"
PREFLIGHT_SCHEMA = "local-width-preflight-v1"
LABELS = ("W1", "A64", "I1")
BOOTSTRAP_SEED = 20261016
EXPORT_SUFFIXES = ("summary.json", "valid_samples.csv", "valid_spectra.npz",
                   "train_summary.json", "train_samples.csv", "train_spectra.npz")
SCOPE = ("full A scalar128/MLP256 versus scalar64/MLP128 capacity package; fixed formulas, residuals, "
         "two internal blocks per call, staged L2/G6, joint eDOS/phDOS, seed42, 35 epochs, crystal_q1_v1; "
         "equal-task valid blind median R2 selection; I1 remains the performance reference; intervals "
         "conditional on selected checkpoints, not training seeds")


def reference_identity(evidence_dir):
    """Bind completed A64 evidence to its independent audit, retaining old sources."""
    evidence_dir = evidence_dir.resolve()
    paths = {name: evidence_dir / name for name in
             ("preflight_final.json", "analysis/report.json", "analysis/verification_codex.json", "queue.json")}
    pre, report, verified, queue = (json.loads(path.read_text()) for path in paths.values())
    if (pre.get("schema") != narrow.PREFLIGHT_SCHEMA or pre.get("passed") is not True
            or report.get("schema") != "local-module-analysis-v1" or report.get("analysis_complete") is not True
            or verified.get("passed") is not True or verified.get("protocol_and_completed_checkpoints_passed") is not True
            or queue.get("schema") != "local-module-queue-v1" or queue.get("status") != "complete"
            or queue.get("arms") != {"I3": "complete"}
            or verified.get("preflight_sha256") != file_sha256(paths["preflight_final.json"])
            or verified.get("input_artifacts_sha256", {}).get("report.json") != file_sha256(paths["analysis/report.json"])
            or report.get("code_sha256") != pre["code_sha256"]
            or report.get("data_contract") != pre["data_contract"]):
        raise ValueError("A64 evidence differs from its independent verification")
    _, _, common = narrow.reference_identity(Path(pre["references"]["I2"]["evidence_dir"]))
    if pre["references"] != common or report["run_artifacts"]["I1"] != common["I1"]:
        raise ValueError("A64 and I1 reference provenance differs")
    artifact = report["run_artifacts"]["I3"]
    run = Path(artifact["run_dir"])
    saved, params, contract = load_run(run)
    cfg = ExperimentConfig(**{**contract["training"], "tag": run.name.removeprefix("ablation_m1")})
    validate_local_global_config(cfg)
    entry = pre["controls"]["I3"]
    if (cfg.local_route != "I3" or cfg.selection_metric != BLIND_R2_METRIC
            or contract != comparison_contract(cfg, params) or contract != entry["contract"]
            or entry["configuration"] != asdict(cfg) or saved["config"] != entry["assembled_configuration"]
            or saved["runtime"]["local_global_init"] != entry["initialization"]
            or saved["cli"]["data_contract"] != pre["data_contract"]
            or entry["data_order"] != common["I1"]["data_order"]):
        raise ValueError("A64 recipe, initialization, data or blind selection changed")
    verify_completed(cfg, pre)
    files = {name: file_sha256(run / name) for name in
             ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}
    history = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
    if (any(artifact[name] != digest for name, digest in files.items())
            or str(history) != artifact["history_path"] or file_sha256(history) != artifact["history_sha256"]):
        raise ValueError("A64 checkpoint or history changed after verification")
    identities = {"A64": {"run_dir": str(run), "files": files, "history_path": str(history),
        "history_sha256": artifact["history_sha256"], "data_order": entry["data_order"],
        "evidence_dir": str(evidence_dir), "evidence_sha256": {
            name: file_sha256(path) for name, path in paths.items()}}, "I1": deepcopy(common["I1"])}
    for label, source_label in (("A64", "I3"), ("I1", "I1")):
        exports = {}
        for suffix in EXPORT_SUFFIXES:
            source = evidence_dir / "analysis" / f"{source_label}_{suffix}"
            digest = file_sha256(source)
            if verified["input_artifacts_sha256"].get(source.name) != digest:
                raise ValueError("frozen width-reference exports differ: " + source.name)
            if label == "I1" and digest != common["I2"]["export_files_sha256"][source.name]:
                raise ValueError("copied I1 export no longer matches the original")
            exports[f"{label}_{suffix}"] = {"source": str(source), "sha256": digest}
        identities[label]["exports"] = exports
    return saved, cfg, identities


def restore_narrow_assembly(assembled):
    restored = deepcopy(assembled)
    params = restored["model"]["params"]["sub_model"]["transformer"]
    restored["model"]["params"]["sub_model"]["transformer"] = narrow_reference_parameters(params)
    return restored


def verify_preflight(path, evidence_dir, cfg):
    from tools.checks.check_local_width import code_fingerprints

    report = json.loads(path.read_text())
    saved, _, identities = reference_identity(evidence_dir)
    entry = report.get("controls", {}).get("W1", {})
    params = entry.get("assembled_configuration", {}).get("model", {}).get("params", {}).get(
        "sub_model", {}).get("transformer", {})
    narrow_entry = report.get("narrow_probe", {})
    if (report.get("schema") != PREFLIGHT_SCHEMA or report.get("passed") is not True
            or report.get("code_sha256") != code_fingerprints() or report.get("references") != identities
            or report.get("data_contract") != saved["cli"]["data_contract"]
            or set(report.get("controls", {})) != {"W1"} or entry.get("configuration") != asdict(cfg)
            or not params or restore_narrow_assembly(entry["assembled_configuration"]) != saved["config"]
            or entry.get("contract") != comparison_contract(cfg, params)
            or entry.get("initialization", {}).get("narrow_initialization") != saved["runtime"]["local_global_init"]
            or narrow_entry.get("initialization") != saved["runtime"]["local_global_init"]
            or narrow_entry.get("matched_nonlocal_state_sha256")
                != entry.get("initialization", {}).get("matched_nonlocal_state_sha256")
            or entry.get("training_start_rng_sha256") != saved["runtime"]["structure_training_rng"][0]["state_sha256"]
            or entry.get("data_order") != identities["A64"]["data_order"]
            or report.get("learning_rate_35") != learning_rate_trace(35, cfg.lr)
            or entry.get("checkpoint_roundtrip") != {
                "weights_optimizer_rng_equal": True, "foreign_contracts_rejected": True}
            or entry.get("narrow_checkpoint_rejected") is not True
            or set(entry.get("symmetry", {})) != {"torch.float32", "torch.float64"}
            or any(len(item.get("costs", [])) != 3 for item in (entry, narrow_entry))
            or entry.get("geometry_cache", {}).get("workspace_guard") != "local-pairs-width-v1"):
        raise ValueError("width preflight source, references, recipe, initialization or checks changed")
    return report


def comparison_report(exports, draws=2000):
    report = norm.comparison_report(exports, draws, labels=LABELS, scope=SCOPE, seed=BOOTSTRAP_SEED)
    report["decisions"] = {
        "width_upgrade_conditions_met": report["joint_blind"]["W1_minus_A64"]["eligible_for_adoption"],
        "baseline_upgrade_conditions_met": report["joint_blind"]["W1_minus_I1"]["eligible_for_adoption"],
        "automatic_baseline_replacement": False, "automatic_followup_training": False}
    return report


def structural_report(pairs):
    result = {"diagnostic_only": True, "scope": "composition-group weighted normalized signed spectral differences",
              "bootstrap": {"draws": 2000, "unit": "composition group", "seed": 42,
                            "training_seed_uncertainty": False}, "summaries": {}, "contrasts": {}}
    for split in ("train", "valid"):
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                filtered = {}
                for label in LABELS:
                    summary, filtered[label] = bplus.group_summary(pairs[label, split, grouping, task], task)
                    result["summaries"][f"{label}_{split}_{grouping}_{task}"] = summary
                for reference in LABELS[1:]:
                    result["contrasts"].setdefault(f"W1_minus_{reference}", {})[
                        f"{split}_{grouping}_{task}"] = paired_group_interval(filtered[reference], filtered["W1"])
    return result


def analyze(cfg, preflight, out):
    from tools.checks.check_local_width import code_fingerprints

    if preflight["code_sha256"] != code_fingerprints():
        raise ValueError("width source changed during training")
    _, _, identities = reference_identity(Path(preflight["references"]["A64"]["evidence_dir"]))
    if identities != preflight["references"]:
        raise ValueError("width references changed during training")
    verify_completed(cfg, preflight)
    if out.exists():
        raise FileExistsError(f"refusing existing width analysis: {out}")
    out.mkdir(parents=True)
    for label in LABELS[1:]:
        for name, entry in identities[label]["exports"].items():
            shutil.copyfile(entry["source"], out / name)
            if file_sha256(out / name) != entry["sha256"]:
                raise ValueError("copied frozen width-reference export differs: " + name)
    run = run_directory(cfg)
    before = {name: file_sha256(run / name) for name in
              ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}
    export(run, out / "W1_valid_samples.csv", out / "W1_summary.json", spectra_out=out / "W1_valid_spectra.npz")
    train_pairs = bplus.export_train(run, out, "W1")
    saved, _, _ = load_run(run)
    pairs = {}
    for split, size in (("train", 18704), ("valid", 2313)):
        dataset = ConfigBuilder(**saved["config"]).get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
        if len(dataset) != size:
            raise ValueError("width diagnosis requires complete train/valid views")
        for label in LABELS:
            bplus.assert_paired_targets(out / f"W1_{split}_spectra.npz", out / f"{label}_{split}_spectra.npz")
            tables = (train_pairs if split == "train" and label == "W1" else bplus.spectrum_pairs(
                out / f"{label}_{split}_spectra.npz", out / f"{label}_{split}_samples.csv", dataset))
            for (grouping, task), table in tables.items():
                table.to_csv(out / f"{label}_{split}_{grouping}_{task}_pairs.csv", index=False)
                pairs[label, split, grouping, task] = table
        del dataset
        gc.collect()
    exports = {label: load(out / f"{label}_summary.json", out / f"{label}_valid_samples.csv") for label in LABELS}
    report = comparison_report(exports)
    history = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
    report.update(schema="local-width-analysis-v1", analysis_complete=True,
        structural_diagnosis=structural_report(pairs), code_sha256=preflight["code_sha256"],
        data_contract=preflight["data_contract"], initialization=preflight["controls"]["W1"]["initialization"],
        export_reuse={label: identities[label]["exports"] for label in LABELS[1:]},
        run_artifacts={"W1": {"run_dir": str(run), **before, "history_path": str(history),
                               "history_sha256": file_sha256(history)}, **identities})
    if (any(file_sha256(run / name) != digest for name, digest in before.items())
            or reference_identity(Path(identities["A64"]["evidence_dir"]))[2] != identities
            or code_fingerprints() != preflight["code_sha256"]):
        raise ValueError("candidate, width references or source changed during analysis")
    dump(out / "report.json", report)
    return report


def execute(cfg, preflight, out):
    from tools.checks.check_local_width import code_fingerprints

    if preflight["code_sha256"] != code_fingerprints():
        raise ValueError("width source changed after preflight")
    status = summarize([cfg])[0]["status"]
    if status == "partial" or status == "pending" and any(run_directory(cfg).glob("*")):
        raise ValueError("partial width run requires a recovery decision; refusing automatic resume")
    if (out / "analysis").exists():
        raise FileExistsError("existing width analysis must be preserved; use --summarize")
    return norm.execute(cfg, preflight, out, analyze_fn=analyze, queue_schema="local-width-queue-v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    _, reference, _ = reference_identity(args.references)
    cfg = local_width_config(reference, args.tag_prefix)
    if args.summarize:
        print(json.dumps(summarize([cfg]), ensure_ascii=False, indent=2))
    elif args.run or args.analyze:
        if args.preflight is None or args.out is None:
            parser.error("--run/--analyze require --preflight and --out")
        preflight = verify_preflight(args.preflight, args.references, cfg)
        if args.run:
            execute(cfg, preflight, args.out)
        else:
            analyze(cfg, preflight, args.out / "analysis")
    else:
        print(shlex.join(training_command(cfg)))


if __name__ == "__main__":
    main()
