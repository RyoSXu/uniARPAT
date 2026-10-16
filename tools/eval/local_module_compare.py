"""Train full A with joint-blind selection; reuse authenticated I1/B+ results."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
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
from tools.eval import i1_a_compare as historical_a
from tools.eval import i1_bplus_compare as bplus
from tools.eval.legacy_backbone_compare import run_directory, verify_completed
from tools.eval.lg_paired_compare import compare, load
from tools.eval.lg_structure_diagnosis import dump
from tools.eval.lg_valid_export import export, load_run
from tools.eval.local_global_compare import summarize, training_command
from utils.builder import ConfigBuilder
from utils.checkpoint_selection import BLIND_R2_METRIC
from utils.experiment_config import ExperimentConfig
from utils.i1_a_experiment import i1_a_config
from utils.local_global_experiment import comparison_contract, validate_local_global_config
from utils.paths import file_sha256, input_path, repository_path

DEFAULT_EVIDENCE = ROOT / "results/runs/local_readout_interface_20261015"
PREFLIGHT_SCHEMA = "local-module-preflight-v1"
LABELS = ("I3", "I2", "I1")
BOOTSTRAP_SEED = 20261016
SCOPE = ("full A versus full B+ at scalar width64, two internal blocks per call and staged L2/G6; "
         "joint eDOS/phDOS, seed42, 35 epochs, crystal_q1_v1; equal-task valid blind median R2 "
         "checkpoint selection; I1 is the performance reference; local formula and capacity "
         "change together; intervals conditional on selected checkpoints, not training seeds")


def candidate_config(reference, tag_prefix):
    if reference.local_route != "I2" or reference.selection_metric != BLIND_R2_METRIC:
        raise ValueError("local module selection requires the completed joint-blind I2 recipe")
    cfg = i1_a_config(replace(reference, selection_metric="balanced_oracle_mae"), tag_prefix)
    cfg = replace(cfg, selection_metric=BLIND_R2_METRIC,
                  tag=f"_lms_{tag_prefix}_i3_k2_g6_staged_s42_e35")
    validate_local_global_config(cfg)
    return cfg


def reference_identity(evidence_dir):
    """Authenticate historical evidence without requiring its code to be current."""
    evidence_dir = evidence_dir.resolve()
    paths = {name: evidence_dir / name for name in
             ("preflight_final.json", "analysis/report.json", "analysis/verification_codex.json", "queue.json")}
    pre, report, verified, queue = (json.loads(paths[name].read_text()) for name in paths)
    if (pre.get("schema") != "local-readout-preflight-v1" or pre.get("passed") is not True
            or report.get("schema") != "local-readout-analysis-v1" or report.get("analysis_complete") is not True
            or verified.get("passed") is not True or verified.get("protocol_and_completed_checkpoints_passed") is not True
            or queue.get("schema") != "local-readout-queue-v1" or queue.get("status") != "complete"
            or queue.get("arms") != {"I1": "complete", "I2": "complete", "I4": "complete"}
            or verified.get("preflight_sha256") != file_sha256(paths["preflight_final.json"])
            or verified.get("input_artifacts_sha256", {}).get("report.json") != file_sha256(paths["analysis/report.json"])
            or report.get("code_sha256") != pre["code_sha256"]
            or report.get("data_contract") != pre["data_contract"]):
        raise ValueError("completed joint-blind reference evidence differs from its verification")
    identities = {}
    for label in ("I1", "I2"):
        artifact = report["run_artifacts"][label]
        run = Path(artifact["run_dir"])
        saved, params, contract = load_run(run)
        cfg = ExperimentConfig(**{**contract["training"], "tag": run.name.removeprefix("ablation_m1")})
        validate_local_global_config(cfg)
        entry = pre["controls"][label]
        if (cfg.local_route != label or cfg.selection_metric != BLIND_R2_METRIC
                or contract != comparison_contract(cfg, params) or contract != entry["contract"]
                or entry["configuration"] != asdict(cfg) or saved["config"] != entry["assembled_configuration"]
                or saved["runtime"]["local_global_init"] != entry["initialization"]
                or saved["cli"]["data_contract"] != pre["data_contract"]):
            raise ValueError(f"{label} recipe, initialization or blind selection changed")
        verify_completed(cfg, pre)
        files = {name: file_sha256(run / name) for name in
                 ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}
        history = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
        if (any(files[name] != artifact[name] for name in files)
                or str(history) != artifact["history_path"] or file_sha256(history) != artifact["history_sha256"]):
            raise ValueError(f"{label} artifacts changed after independent verification")
        identities[label] = {"run_dir": str(run), "files": files, "history_path": str(history),
                             "history_sha256": artifact["history_sha256"], "data_order": entry["data_order"]}
        del saved
        gc.collect()
    i1, i2 = pre["controls"]["I1"], pre["controls"]["I2"]
    if (i2["initialization"]["i1_initialization"] != i1["initialization"]
            or i1["data_order"] != i2["data_order"]
            or i1["training_start_rng_sha256"] != i2["training_start_rng_sha256"]):
        raise ValueError("I1/I2 common initialization, start RNG or sampler differ")
    saved, _, contract = load_run(Path(identities["I2"]["run_dir"]))
    cfg = ExperimentConfig(**{**contract["training"],
                             "tag": Path(identities["I2"]["run_dir"]).name.removeprefix("ablation_m1")})
    data_dir = Path(input_path(cfg.data_dir))
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (file_sha256(manifest_path) != pre["data_contract"]["manifest_sha256"]
            or manifest["schema"] != pre["data_contract"]["schema"]):
        raise ValueError("current data manifest differs from the completed references")
    for name, record in manifest["files"].items():
        if Path(name).parts[0] in ("train", "valid") and file_sha256(data_dir / name) != record["sha256"]:
            raise ValueError("train/valid payload differs: " + name)
    exports = {name: file_sha256(evidence_dir / "analysis" / name) for name in historical_a.EXPORT_FILES}
    if any(verified["input_artifacts_sha256"].get(name) != digest for name, digest in exports.items()):
        raise ValueError("frozen reference exports differ from independent verification")
    identities["I2"].update(evidence_dir=str(evidence_dir), export_dir=str(evidence_dir / "analysis"),
                            export_files_sha256=exports,
                            evidence_sha256={name: file_sha256(path) for name, path in paths.items()})
    return saved, cfg, identities


def verify_preflight(path, evidence_dir, cfg):
    from tools.checks.check_local_module import code_fingerprints

    report = json.loads(path.read_text())
    saved, _, identities = reference_identity(evidence_dir)
    entry = report.get("controls", {}).get("I3", {})
    assembled = deepcopy(entry.get("assembled_configuration", {}))
    params = assembled.get("model", {}).get("params", {}).get("sub_model", {}).get("transformer", {})
    if params:
        params["local_route"] = "I2"
    if (report.get("schema") != PREFLIGHT_SCHEMA or report.get("passed") is not True
            or report.get("code_sha256") != code_fingerprints() or report.get("references") != identities
            or report.get("data_contract") != saved["cli"]["data_contract"]
            or set(report.get("controls", {})) != {"I3"} or entry.get("configuration") != asdict(cfg)
            or assembled != saved["config"]
            or entry.get("initialization", {}).get("i2_initialization") != saved["runtime"]["local_global_init"]
            or entry.get("training_start_rng_sha256") != saved["runtime"]["structure_training_rng"][0]["state_sha256"]
            or entry.get("data_order") != identities["I2"]["data_order"]
            or report.get("learning_rate_35") != learning_rate_trace(35, cfg.lr)
            or any(entry.get(key) is not True for key in
                   ("checkpoint_roundtrip", "i2_checkpoint_rejected", "selection_checkpoint_rejected"))
            or set(entry.get("permutation_invariance", {})) != {"torch.float32", "torch.float64"}):
        raise ValueError("A preflight source, references, recipe, RNG or checks changed")
    original_params = entry["assembled_configuration"]["model"]["params"]["sub_model"]["transformer"]
    if entry.get("contract") != comparison_contract(cfg, original_params):
        raise ValueError("A preflight checkpoint contract differs")
    return report


def comparison_report(exports, draws=2000):
    report = norm.comparison_report(exports, draws, labels=LABELS, scope=SCOPE, seed=BOOTSTRAP_SEED)
    # The reverse direction has its own task protections; negating a score CI
    # alone cannot establish that B+ meets those protections against A.
    reverse = compare(*exports["I3"], [("I2", *exports["I2"])], draws, BOOTSTRAP_SEED, scope=SCOPE)
    report["contrasts"]["I2_minus_I3"] = reverse["comparisons"]["I2"]
    report["joint_blind"]["I2_minus_I3"] = reverse["joint_blind"]["I2"]
    checks = report["joint_blind"]
    a_wins, bp_wins = (checks[name]["eligible_for_adoption"] for name in ("I3_minus_I2", "I2_minus_I3"))
    report["decisions"] = {"route_reference": "A" if a_wins else "B+" if bp_wins else "undetermined",
        "route_conditions": {"A_vs_Bplus": a_wins, "Bplus_vs_A": bp_wins},
        "A_vs_I1_numerical_adoption_conditions_met": checks["I3_minus_I1"]["eligible_for_adoption"],
        "automatic_baseline_replacement": False, "automatic_followup_training": False}
    return report


def analyze(cfg, preflight, out):
    from tools.checks.check_local_module import code_fingerprints

    if preflight["code_sha256"] != code_fingerprints():
        raise ValueError("A source changed during training")
    saved, _, identities = reference_identity(Path(preflight["references"]["I2"]["evidence_dir"]))
    if identities != preflight["references"] or saved["cli"]["data_contract"] != preflight["data_contract"]:
        raise ValueError("completed references changed during A training")
    verify_completed(cfg, preflight)
    if out.exists():
        raise FileExistsError(f"refusing existing local module analysis: {out}")
    out.mkdir(parents=True)
    source = Path(identities["I2"]["export_dir"])
    for name, digest in identities["I2"]["export_files_sha256"].items():
        shutil.copyfile(source / name, out / name)
        if file_sha256(out / name) != digest:
            raise ValueError("copied frozen reference export differs: " + name)
    candidate_run = run_directory(cfg)
    artifact_files = ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")
    before = {name: file_sha256(candidate_run / name) for name in artifact_files}
    export(candidate_run, out / "I3_valid_samples.csv", out / "I3_summary.json",
           spectra_out=out / "I3_valid_spectra.npz")
    train_pairs = bplus.export_train(candidate_run, out, "I3")
    candidate_saved, _, _ = load_run(candidate_run)
    pairs = {}
    for split, size in (("train", 18704), ("valid", 2313)):
        dataset = ConfigBuilder(**candidate_saved["config"]).get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
        if len(dataset) != size:
            raise ValueError("analysis requires complete train/valid views")
        for label in LABELS:
            bplus.assert_paired_targets(out / f"I3_{split}_spectra.npz", out / f"{label}_{split}_spectra.npz")
            tables = (train_pairs if split == "train" and label == "I3" else bplus.spectrum_pairs(
                out / f"{label}_{split}_spectra.npz", out / f"{label}_{split}_samples.csv", dataset))
            for (grouping, task), table in tables.items():
                table.to_csv(out / f"{label}_{split}_{grouping}_{task}_pairs.csv", index=False)
                pairs[label, split, grouping, task] = table
        del dataset
        gc.collect()
    exports = {label: load(out / f"{label}_summary.json", out / f"{label}_valid_samples.csv") for label in LABELS}
    report = comparison_report(exports)
    structure = historical_a.structural_report(pairs)
    structure.pop("primary")
    structure.pop("primary_contrast")
    structure["diagnostic_only"] = True
    report.update(schema="local-module-analysis-v1", analysis_complete=True,
        structural_diagnosis=structure, code_sha256=preflight["code_sha256"], data_contract=preflight["data_contract"],
        parameter_counts=preflight["controls"]["I3"]["initialization"]["parameter_counts"],
        reference_parameter_counts=preflight["controls"]["I3"]["initialization"]["reference_parameter_counts"],
        export_reuse={"labels": ["I2", "I1"], "source": str(source),
                      "files_sha256": identities["I2"]["export_files_sha256"]},
        run_artifacts={"I3": {"run_dir": str(candidate_run), **before},
                       **{label: identities[label] for label in LABELS[1:]}})
    history = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
    report["run_artifacts"]["I3"].update(history_path=str(history), history_sha256=file_sha256(history))
    if (any(file_sha256(candidate_run / name) != digest for name, digest in before.items())
            or reference_identity(Path(identities["I2"]["evidence_dir"]))[2] != identities
            or code_fingerprints() != preflight["code_sha256"]):
        raise ValueError("candidate, references or source changed during A analysis")
    dump(out / "report.json", report)
    return report


def execute(cfg, preflight, out):
    from tools.checks.check_local_module import code_fingerprints

    if preflight["code_sha256"] != code_fingerprints():
        raise ValueError("A source changed after preflight")
    status = summarize([cfg])[0]["status"]
    if status == "partial" or status == "pending" and any(run_directory(cfg).glob("*")):
        raise ValueError("partial A run requires an explicit recovery decision; refusing automatic resume")
    if (out / "analysis").exists():
        raise FileExistsError("existing analysis must be preserved; use --summarize")
    return norm.execute(cfg, preflight, out, analyze_fn=analyze, queue_schema="local-module-queue-v1")


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
    cfg = candidate_config(reference, args.tag_prefix)
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
