"""Train one internal pre-norm A candidate and reuse authenticated post-norm A64."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
import gc
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.checks.check_long_budget import learning_rate_trace
from tools.eval import global_norm_compare as norm
from tools.eval import i1_bplus_compare as bplus
from tools.eval import legacy_backbone_compare as queue
from tools.eval import local_width_fresh_compare as fresh
from tools.eval.lg_paired_compare import load
from tools.eval.lg_structure_diagnosis import dump, paired_group_interval
from tools.eval.lg_valid_export import export, load_run
from tools.eval.local_global_compare import summarize, training_command
from utils.builder import ConfigBuilder
from utils.checkpoint_selection import BLIND_R2_METRIC
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import comparison_contract, validate_local_global_config
from utils.local_norm_experiment import local_norm_config, post_norm_parameters, validate_local_norm_model
from utils.paths import file_sha256, repository_path

DEFAULT_EVIDENCE = ROOT / "results/runs/local_a_width_fresh_20261019"
REFERENCE_COMMIT = "1c29ba0d50ec2479f837624b6836ad98e2f453f0"
PREFLIGHT_SCHEMA = "local-internal-norm-preflight-v1"
LABELS = ("LN1", "A64")
EXPORT_SUFFIXES = ("summary.json", "valid_samples.csv", "valid_spectra.npz",
                   "train_summary.json", "train_samples.csv", "train_spectra.npz")
# Every other historical source must still be byte-identical. These five
# extensions are additionally checked against Git-loaded historical modules.
COMPATIBILITY_FILES = ("model/periodic_messages.py", "model/local_global_encoder.py", "model/transformer.py",
                       "utils/local_global_experiment.py", "tools/train.py")
SCOPE = ("complete A64 internal pre-norm versus post-norm; same complete initial tensors, external residuals, "
         "staged L2/G6 and joint losses; seed42, 35 epochs, crystal_q1_v1 train/valid only; "
         "equal-task blind median R2 selection; intervals conditional on selected checkpoints")


@dataclass(frozen=True)
class NormProtocol:
    """Only the declared local candidate varies; evidence and statistics are shared."""

    route: str = "LN1"
    stem: str = "local-internal-norm"
    config_factory: Callable = local_norm_config
    reference_parameters: Callable = post_norm_parameters
    validate_model: Callable = validate_local_norm_model
    reference_init_key: str = "post_norm_initialization"
    rejected_checkpoint_key: str = "post_norm_checkpoint_rejected"
    compatibility_key: str = "post_norm_compatibility"
    execution_fields: tuple = (("internal_normalization", ("pre-norm",)*4),)
    decision_key: str = "pre_norm_conditions_met"
    scope: str = SCOPE
    source_files: tuple = ()


INTERNAL_NORM = NormProtocol()


def historical_source(commit, name, expected_sha256):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("the historical reference must be pinned to a full Git commit")
    content = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT, check=True,
                             capture_output=True).stdout
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("historical Git source differs from the original preflight: " + name)
    return content.decode("utf-8")


def reference_identity(evidence_dir, commit=REFERENCE_COMMIT):
    """Authenticate the fresh A64 result without reusing old source fingerprints as current ones."""
    evidence_dir = evidence_dir.resolve()
    paths = {name: evidence_dir / name for name in
        ("preflight_final.json", "analysis/report.json", "analysis/verification_codex.json", "queue.json")}
    pre, report, audit, state = (json.loads(path.read_text()) for path in paths.values())
    if (pre.get("schema") != fresh.PREFLIGHT_SCHEMA or pre.get("passed") is not True
            or report.get("schema") != "local-width-fresh-analysis-v1" or report.get("analysis_complete") is not True
            or audit.get("passed") is not True or audit.get("protocol_and_completed_checkpoints_passed") is not True
            or state.get("schema") != "local-width-fresh-queue-v1" or state.get("status") != "complete"
            or state.get("arms") != {"I3": "complete", "W1": "complete"}
            or audit.get("preflight_sha256") != file_sha256(paths["preflight_final.json"])
            or audit.get("input_artifacts_sha256", {}).get("report.json") != file_sha256(paths["analysis/report.json"])
            or report.get("code_sha256") != pre["code_sha256"]
            or report.get("data_contract") != pre["data_contract"] or pre["runtime"] != fresh.runtime_identity()):
        raise ValueError("fresh A64 evidence, runtime or independent verification differs")
    changes = {}
    for name, digest in pre["code_sha256"].items():
        historical_source(commit, name, digest)
        current = file_sha256(ROOT / name)
        if current != digest:
            if name not in COMPATIBILITY_FILES:
                raise ValueError("unapproved historical source drift: " + name)
            changes[name] = {"historical": digest, "current": current}
    artifact = report["run_artifacts"]["A64"]
    run = Path(artifact["run_dir"])
    saved, params, contract = load_run(run)
    cfg = ExperimentConfig(**{**contract["training"], "tag": run.name.removeprefix("ablation_m1")})
    validate_local_global_config(cfg)
    entry = pre["controls"]["I3"]
    if (cfg.local_route != "I3" or cfg.selection_metric != BLIND_R2_METRIC
            or contract != comparison_contract(cfg, params) or entry["contract"] != contract
            or entry["configuration"] != asdict(cfg) or saved["config"] != entry["assembled_configuration"]
            or saved["runtime"]["local_global_init"] != entry["initialization"]
            or saved["cli"]["data_contract"] != pre["data_contract"]
            or fresh.data_identity(cfg) != pre["data_contract"]):
        raise ValueError("fresh A64 configuration, data or initialization differs")
    queue.verify_completed(cfg, pre)
    if fresh.artifacts(cfg) != artifact:
        raise ValueError("A64 checkpoint, configuration or history changed after verification")
    exports = {}
    for name, digest in audit["input_artifacts_sha256"].items():
        if name.startswith("A64_"):
            source = evidence_dir / "analysis" / name
            if file_sha256(source) != digest:
                raise ValueError("authenticated A64 export changed: " + name)
            exports[name] = {"source": str(source), "sha256": digest}
    if any(f"A64_{suffix}" not in exports for suffix in EXPORT_SUFFIXES):
        raise ValueError("complete train/valid A64 exports required")
    identity = {**artifact, "evidence_dir": str(evidence_dir), "reference_commit": commit,
        "evidence_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "exports": exports, "data_order": entry["data_order"],
        "historical_source_sha256": pre["code_sha256"], "source_extensions": changes}
    return saved, cfg, identity


def restore_post_assembly(assembled):
    return restore_reference_assembly(assembled)


def restore_reference_assembly(assembled, *, protocol=INTERNAL_NORM):
    result = deepcopy(assembled)
    params = result["model"]["params"]["sub_model"]["transformer"]
    result["model"]["params"]["sub_model"]["transformer"] = protocol.reference_parameters(params)
    return result


def validate_preflight(report, cfg, *, protocol=INTERNAL_NORM):
    from tools.checks.check_local_norm import code_fingerprints

    ref = report["references"]["A64"]
    saved, reference_cfg, identity = reference_identity(Path(ref["evidence_dir"]), ref["reference_commit"])
    entry = report.get("controls", {}).get(protocol.route, {})
    params = entry.get("assembled_configuration", {}).get("model", {}).get("params", {}).get(
        "sub_model", {}).get("transformer", {})
    if (report.get("schema") != f"{protocol.stem}-preflight-v1" or report.get("passed") is not True
            or report.get("code_sha256") != code_fingerprints(protocol=protocol) or ref != identity
            or report.get("runtime") != fresh.runtime_identity() or report.get("data_contract") != saved["cli"]["data_contract"]
            or set(report.get("controls", {})) != {protocol.route} or entry.get("configuration") != asdict(cfg)
            or not params or restore_reference_assembly(entry["assembled_configuration"], protocol=protocol) != saved["config"]
            or entry.get("contract") != comparison_contract(cfg, params)
            or entry.get("initialization", {}).get(protocol.reference_init_key) != saved["runtime"]["local_global_init"]
            or entry["initialization"].get("state_sha256") != saved["runtime"]["local_global_init"]["state_sha256"]
            or entry.get("data_order") != ref["data_order"]
            or entry.get("training_start_rng_sha256") != saved["runtime"]["structure_training_rng"][0]["state_sha256"]
            or entry.get("forward_backward", {}).get("all_parameter_gradients_finite") is not True
            or entry.get("evaluation_finite") is not True
            or set(entry.get("symmetry", {})) != {"torch.float32", "torch.float64"}
            or entry.get("checkpoint_roundtrip") != {
                "weights_optimizer_rng_equal": True, "foreign_contracts_rejected": True}
            or entry.get(protocol.rejected_checkpoint_key) is not True
            or len(entry.get("costs", [])) != 3
            or report.get(protocol.compatibility_key, {}).get("passed") is not True
            or report.get("reference_prediction_replay", {}).get("passed") is not True
            or report.get("learning_rate_35") != learning_rate_trace(35, cfg.lr)):
        raise ValueError("local comparison source, recipe, initialization, compatibility or real-batch preflight differs")
    # Reconstruct the only allowed change from the authenticated A64 recipe.
    if cfg != protocol.config_factory(reference_cfg, report["tag_prefix"]):
        raise ValueError("candidate queue differs from the authenticated A64 recipe")
    validate_local_global_config(cfg)
    return report


def verify_preflight(path, cfg, *, protocol=INTERNAL_NORM):
    return validate_preflight(json.loads(path.read_text()), cfg, protocol=protocol)


def comparison_report(exports, draws=2000, *, protocol=INTERNAL_NORM):
    report = norm.comparison_report(exports, draws, labels=(protocol.route, "A64"), scope=protocol.scope, seed=20261016)
    report["decisions"] = {
        protocol.decision_key: report["joint_blind"][f"{protocol.route}_minus_A64"]["eligible_for_adoption"],
        "automatic_baseline_replacement": False, "automatic_followup_training": False}
    return report


def structural_report(pairs, *, protocol=INTERNAL_NORM):
    contrast = f"{protocol.route}_minus_A64"
    report = {"diagnostic_only": True, "bootstrap": {"draws": 2000, "unit": "composition group", "seed": 42,
              "training_seed_uncertainty": False}, "summaries": {}, "contrasts": {contrast: {}}}
    for split in ("train", "valid"):
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                filtered = {}
                for label in (protocol.route, "A64"):
                    key = f"{label}_{split}_{grouping}_{task}"
                    report["summaries"][key], filtered[label] = bplus.group_summary(pairs[label, split, grouping, task], task)
                report["contrasts"][contrast][f"{split}_{grouping}_{task}"] = paired_group_interval(
                    filtered["A64"], filtered[protocol.route])
    return report


def verify_completed(cfg, preflight, *, protocol=INTERNAL_NORM):
    queue.verify_completed(cfg, preflight)
    saved, _, contract = load_run(queue.run_directory(cfg))
    entry = preflight["controls"][protocol.route]
    execution = saved.get("runtime", {}).get("local_global_execution", [])
    if (saved["config"] != entry["assembled_configuration"] or contract != entry["contract"]
            or not execution or any(record.get(key) != list(expected)
                for record in execution for key, expected in protocol.execution_fields)):
        raise ValueError(f"completed {protocol.route} assembly, contract or actual normalization differs from preflight")


def analyze(controls, preflight, out, *, protocol=INTERNAL_NORM):
    cfg, = controls
    route = protocol.route
    validate_preflight(preflight, cfg, protocol=protocol)
    verify_completed(cfg, preflight, protocol=protocol)
    if out.exists():
        raise FileExistsError("existing local norm analysis must be preserved")
    out.mkdir(parents=True)
    reference = preflight["references"]["A64"]
    for suffix in EXPORT_SUFFIXES:
        name = f"A64_{suffix}"
        source = reference["exports"][name]
        if file_sha256(Path(source["source"])) != source["sha256"]:
            raise ValueError("reference export changed before copying")
        shutil.copyfile(source["source"], out / name)
    run, before = queue.run_directory(cfg), fresh.artifacts(cfg)
    export(run, out / f"{route}_valid_samples.csv", out / f"{route}_summary.json", spectra_out=out / f"{route}_valid_spectra.npz")
    train_pairs = bplus.export_train(run, out, route)
    saved, _, _ = load_run(run)
    pairs = {}
    for split, size in (("train", 18704), ("valid", 2313)):
        bplus.assert_paired_targets(out / f"{route}_{split}_spectra.npz", out / f"A64_{split}_spectra.npz")
        dataset = ConfigBuilder(**saved["config"]).get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
        if len(dataset) != size:
            raise ValueError("local norm diagnosis requires complete train/valid views")
        for label in (route, "A64"):
            tables = train_pairs if (label, split) == (route, "train") else bplus.spectrum_pairs(
                out / f"{label}_{split}_spectra.npz", out / f"{label}_{split}_samples.csv", dataset)
            for (grouping, task), table in tables.items():
                table.to_csv(out / f"{label}_{split}_{grouping}_{task}_pairs.csv", index=False)
                pairs[label, split, grouping, task] = table
        del dataset
        gc.collect()
    exports = {label: load(out / f"{label}_summary.json", out / f"{label}_valid_samples.csv") for label in (route, "A64")}
    report = comparison_report(exports, protocol=protocol)
    report.update(schema=f"{protocol.stem}-analysis-v1", analysis_complete=True,
        structural_diagnosis=structural_report(pairs, protocol=protocol), code_sha256=preflight["code_sha256"],
        data_contract=preflight["data_contract"], initialization=preflight["controls"][route]["initialization"],
        run_artifacts={route: before, "A64": reference})
    validate_preflight(preflight, cfg, protocol=protocol)
    if fresh.artifacts(cfg) != before:
        raise ValueError("candidate checkpoint, configuration or history changed during analysis")
    dump(out / "report.json", report)
    return report


def execute(cfg, preflight, out, *, protocol=INTERNAL_NORM):
    validate_preflight(preflight, cfg, protocol=protocol)
    status = summarize([cfg])[0]["status"]
    if status == "partial" or status == "pending" and any(queue.run_directory(cfg).glob("*")):
        raise ValueError("partial local norm run requires a recovery decision; refusing automatic resume")
    if status == "complete":
        verify_completed(cfg, preflight, protocol=protocol)
    if (out / "analysis").exists():
        raise FileExistsError("existing local norm analysis must be preserved; use --summarize")
    return queue.execute([cfg], preflight, out, analyze_fn=partial(analyze, protocol=protocol),
                         queue_schema=f"{protocol.stem}-queue-v1")


def main(*, protocol=INTERNAL_NORM):
    parser = argparse.ArgumentParser(description=protocol.scope)
    parser.add_argument("--references", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--reference-commit", default=REFERENCE_COMMIT)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    _, reference, _ = reference_identity(args.references, args.reference_commit)
    cfg = protocol.config_factory(reference, args.tag_prefix)
    if args.summarize:
        print(json.dumps(summarize([cfg]), ensure_ascii=False, indent=2))
    elif args.run or args.analyze:
        if args.preflight is None or args.out is None:
            parser.error("--run/--analyze require --preflight and --out")
        preflight = verify_preflight(args.preflight, cfg, protocol=protocol)
        if (Path(preflight["references"]["A64"]["evidence_dir"]) != args.references.resolve()
                or preflight["references"]["A64"]["reference_commit"] != args.reference_commit):
            raise ValueError("CLI reference differs from the authenticated preflight")
        if args.run:
            execute(cfg, preflight, args.out, protocol=protocol)
        else:
            analyze([cfg], preflight, args.out / "analysis", protocol=protocol)
    else:
        print(shlex.join(training_command(cfg)))


if __name__ == "__main__":
    main()
