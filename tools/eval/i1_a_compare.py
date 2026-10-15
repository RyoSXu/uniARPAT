"""Train I3/full A only; reuse verified I2/I1 exports for joint comparison."""
from __future__ import annotations

import argparse
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

import torch

from tools.eval import global_norm_compare as norm
from tools.eval import i1_bplus_compare as bplus
from tools.eval.legacy_backbone_compare import run_directory, verify_completed
from tools.eval.lg_paired_compare import load
from tools.eval.lg_structure_diagnosis import dump, paired_group_interval
from tools.eval.lg_valid_export import export, load_run
from tools.eval.local_global_compare import summarize, training_command
from utils.builder import ConfigBuilder
from utils.experiment_config import ExperimentConfig
from utils.i1_a_experiment import DEFAULT_I2_REFERENCE, I2_EVIDENCE, i1_a_config
from utils.i1_bplus_experiment import DEFAULT_I1_REFERENCE
from utils.paths import file_sha256, repository_path

LABELS = ("I3", "I2", "I1")
SCOPE = ("full A versus full B+ on the corrected joint I1 backbone, with radial I1 as reference; "
         "seed42, 35 epochs, crystal_q1_v1, staged L2/G6; common initial tensors matched; "
         "local packages and capacity change; selected-checkpoint intervals, not training-seed uncertainty")
EXPORT_FILES = tuple(f"{label}_{suffix}" for label in ("I2", "I1") for suffix in
                     ("summary.json", "valid_samples.csv", "valid_spectra.npz",
                      "train_summary.json", "train_samples.csv", "train_spectra.npz"))


def reference_identity(reference_dir):
    """Authenticate completed I2 provenance rather than rerun its old code gate.

    Historical preflight source hashes stay immutable. The new I3 preflight
    locks current source and reconstructs the complete untrained I2 identity.
    Existing I2 verification/initialization gates are not changed or bypassed.
    """
    i1_saved, _, identities = bplus.reference_identity(ROOT / DEFAULT_I1_REFERENCE)
    saved, params, contract = load_run(reference_dir)
    cfg = ExperimentConfig(**{**contract["training"],
                             "tag": reference_dir.name.removeprefix("ablation_m1")})
    if cfg.local_route != "I2":
        raise ValueError("complete A comparison requires the completed I2 reference")
    evidence_dir = ROOT / I2_EVIDENCE
    preflight_path = evidence_dir / "preflight_final.json"
    report_path = evidence_dir / "analysis/report.json"
    verification_path = evidence_dir / "analysis/verification_codex.json"
    old_preflight = json.loads(preflight_path.read_text())
    old_report = json.loads(report_path.read_text())
    verification = json.loads(verification_path.read_text())
    queue = json.loads((evidence_dir / "queue.json").read_text())
    if (old_preflight.get("schema") != "i1-bplus-preflight-v1" or old_preflight.get("passed") is not True
            or old_report.get("schema") != "i1-bplus-analysis-v1" or old_report.get("analysis_complete") is not True
            or queue.get("schema") != "i1-bplus-queue-v1" or queue.get("status") != "complete"
            or queue.get("arms") != {"I2": "complete"}
            or (evidence_dir / "monitor/exit_code").read_text().strip() != "0"
            or verification.get("passed") is not True
            or old_report["preflight"] != old_preflight["code_sha256"]
            or old_preflight["references"] != identities
            or old_preflight["data_contract"] != i1_saved["cli"]["data_contract"]
            or saved["cli"]["data_contract"] != i1_saved["cli"]["data_contract"]):
        raise ValueError("I2 completion, historical verification or common data differ")
    verify_completed(cfg, old_preflight)
    files = {name: file_sha256(reference_dir / name) for name in
             ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}
    recorded = old_report["run_artifacts"]["I2"]
    if (recorded["run_dir"] != str(reference_dir.resolve())
            or recorded["configuration_sha256"] != files["config_used.yaml"]
            or recorded["best_checkpoint_sha256"] != files["checkpoint_best.pth"]):
        raise ValueError("I2 differs from its completed, verified result")
    for label in ("I2", "I1"):
        expected = old_report["run_artifacts"][label]
        identity = {"run_dir": str(reference_dir.resolve()), "files": files} if label == "I2" else identities[label]
        if (expected["run_dir"] != identity["run_dir"]
                or expected["configuration_sha256"] != identity["files"]["config_used.yaml"]
                or expected["best_checkpoint_sha256"] != identity["files"]["checkpoint_best.pth"]):
            raise ValueError("frozen exports no longer describe their reference checkpoints")
    export_hashes = {name: file_sha256(evidence_dir / "analysis" / name) for name in EXPORT_FILES}
    if (verification["input_artifacts_sha256"]["report.json"] != file_sha256(report_path)
            or any(verification["input_artifacts_sha256"].get(name) != digest
                   for name, digest in export_hashes.items())):
        raise ValueError("I2/I1 frozen exports differ from the independently verified artifacts")
    history_path = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
    evidence_paths = (preflight_path, report_path, verification_path, evidence_dir / "queue.json",
                      evidence_dir / "monitor/exit_code")
    identities["I2"] = {"run_dir": str(reference_dir.resolve()), "files": files,
        "history_path": str(history_path), "history_sha256": file_sha256(history_path),
        "evidence_sha256": {str(path): file_sha256(path) for path in evidence_paths},
        "export_dir": str((evidence_dir / "analysis").resolve()), "export_files_sha256": export_hashes}
    return saved, cfg, identities


def verify_preflight(path, reference_dir, cfg):
    from tools.checks.check_i1_a import code_fingerprints

    report = json.loads(path.read_text())
    saved, _, identities = reference_identity(reference_dir)
    if (report.get("schema") != "i1-a-preflight-v1" or report.get("passed") is not True
            or report.get("code_sha256") != code_fingerprints() or report.get("references") != identities
            or report.get("data_contract") != saved["cli"]["data_contract"]
            or set(report.get("controls", {})) != {"I3"}
            or report["controls"]["I3"]["configuration"] != asdict(cfg)
            or report["controls"]["I3"]["initialization"]["i2_initialization"]
                != saved["runtime"]["local_global_init"]
            or report["controls"]["I3"]["training_start_rng_sha256"]
                != saved["runtime"]["structure_training_rng"][0]["state_sha256"]):
        raise ValueError("I3 source, references, data or initialization changed after preflight")
    return report


def structural_report(pairs):
    result = {"scope": "normalized signed spectral differences; composition groups equally weighted",
              "primary": "valid_absolute_edos", "primary_contrast": "I3_minus_I2",
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
                    result["contrasts"].setdefault(f"I3_minus_{reference}", {})[
                        f"{split}_{grouping}_{task}"] = paired_group_interval(filtered[reference], filtered["I3"])
    return result


def analyze(cfg, preflight, out):
    from tools.checks.check_i1_a import code_fingerprints

    if preflight["code_sha256"] != code_fingerprints():
        raise ValueError("I3 source changed during training")
    saved, _, identities = reference_identity(Path(preflight["references"]["I2"]["run_dir"]))
    if identities != preflight["references"] or saved["cli"]["data_contract"] != preflight["data_contract"]:
        raise ValueError("completed references changed during I3 training")
    verify_completed(cfg, preflight)
    if out.exists():
        raise FileExistsError(f"refusing existing I3 analysis: {out}")
    out.mkdir(parents=True)
    reference_exports = Path(identities["I2"]["export_dir"])
    for name, digest in identities["I2"]["export_files_sha256"].items():
        shutil.copyfile(reference_exports / name, out / name)
        if file_sha256(out / name) != digest:
            raise ValueError("copied frozen reference export differs: " + name)
    candidate_run = run_directory(cfg)
    export(candidate_run, out / "I3_valid_samples.csv", out / "I3_summary.json",
           spectra_out=out / "I3_valid_spectra.npz")
    train_pairs = bplus.export_train(candidate_run, out, "I3")
    pairs = {}
    for split in ("train", "valid"):
        candidate_saved, _, _ = load_run(candidate_run)
        dataset = ConfigBuilder(**candidate_saved["config"]).get_dataset(
            split=split, dos_minmax=True, dos_sumnorm=True)
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
    report = norm.comparison_report(exports, labels=LABELS, scope=SCOPE)
    report.update(schema="i1-a-analysis-v1", analysis_complete=True,
        structural_generalization=structural_report(pairs),
        preflight=preflight["code_sha256"],
        parameter_counts=preflight["controls"]["I3"]["initialization"]["parameter_counts"],
        reference_parameter_counts=preflight["controls"]["I3"]["initialization"]["reference_parameter_counts"],
        export_reuse={"labels": ["I2", "I1"], "source": str(reference_exports),
                      "files_sha256": identities["I2"]["export_files_sha256"]})
    runs = {"I3": candidate_run, **{label: Path(identities[label]["run_dir"]) for label in LABELS[1:]}}
    report["run_artifacts"] = {label: {"run_dir": str(run.resolve()),
        "configuration_sha256": file_sha256(run / "config_used.yaml"),
        "best_checkpoint_sha256": file_sha256(run / "checkpoint_best.pth")} for label, run in runs.items()}
    if (reference_identity(Path(identities["I2"]["run_dir"]))[2] != identities
            or preflight["code_sha256"] != code_fingerprints()):
        raise ValueError("references or source changed during I3 analysis")
    dump(out / "report.json", report)
    return report


def execute(cfg, preflight, out):
    return norm.execute(cfg, preflight, out, analyze_fn=analyze, queue_schema="i1-a-queue-v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_I2_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    _, reference, _ = reference_identity(args.reference)
    cfg = i1_a_config(reference, args.tag_prefix)
    if args.summarize:
        print(json.dumps(summarize([cfg]), ensure_ascii=False, indent=2))
    elif args.run or args.analyze:
        if args.preflight is None or args.out is None:
            parser.error("--run/--analyze require --preflight and --out")
        preflight = verify_preflight(args.preflight, args.reference, cfg)
        if args.run:
            execute(cfg, preflight, args.out)
        else:
            analyze(cfg, preflight, args.out / "analysis")
    else:
        print(shlex.join(training_command(cfg)))


if __name__ == "__main__":
    main()
