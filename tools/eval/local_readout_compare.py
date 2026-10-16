"""Plan or run fresh I1/I2/I4 joint-blind local-to-readout comparisons."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from tools.eval import legacy_backbone_compare as queue
from tools.eval.global_norm_compare import comparison_report
from tools.eval.i1_bplus_compare import (
    assert_paired_targets, export_train, group_summary, reference_identity as recipe_identity,
    spectrum_pairs,
)
from tools.eval.lg_paired_compare import load
from tools.eval.lg_structure_diagnosis import dump, paired_group_interval
from tools.eval.lg_valid_export import export, load_run, select_best
from tools.eval.local_global_compare import summarize, training_command
from utils.builder import ConfigBuilder
from utils.i1_bplus_experiment import DEFAULT_I1_REFERENCE
from utils.local_global_experiment import comparison_contract
from utils.local_readout_experiment import GATE_KEY, READOUT_ROUTES, local_readout_configs
from utils.paths import file_sha256, repository_path

LABELS = ("I4", "I2", "I1")
PREFLIGHT_SCHEMA = "local-readout-preflight-v1"
SCOPE = ("fresh seed42, 35-epoch joint eDOS/phDOS training on crystal_q1_v1; "
         "all runs selected by equal-task valid blind median R2; "
         "I4 vs I2 adds only a zero-initialized local-state residual gate; "
         "I1 is the development reference; not a causal localization of information loss; "
         "intervals conditional on selected checkpoints, not training-seed uncertainty")


def verify_preflight(path, reference_dir, controls):
    from tools.checks.check_local_readout import code_fingerprints

    report = json.loads(path.read_text())
    saved, _, identity = recipe_identity(reference_dir)
    if (report.get("schema") != PREFLIGHT_SCHEMA or report.get("passed") is not True
            or report.get("code_sha256") != code_fingerprints()
            or report.get("recipe_reference") != identity
            or report.get("data_contract") != saved["cli"]["data_contract"]
            or set(report.get("controls", {})) != set(READOUT_ROUTES)
            or [cfg.local_route for cfg in controls] != list(READOUT_ROUTES)
            or report.get("gate_zero_equivalence", {}).get("passed") is not True):
        raise ValueError("local readout preflight source, data, recipe or equivalence changed")
    for cfg in controls:
        entry = report["controls"][cfg.local_route]
        params = entry["assembled_configuration"]["model"]["params"]["sub_model"]["transformer"]
        if (entry["configuration"] != asdict(cfg) or entry["contract"] != comparison_contract(cfg, params)
                or entry["initialization"].get("seed") != cfg.seed
                or entry["training_start_rng_sha256"]
                    != saved["runtime"]["structure_training_rng"][0]["state_sha256"]):
            raise ValueError("local readout preflight configuration, contract or start RNG changed")
    infos = {label: report["controls"][label]["initialization"] for label in READOUT_ROUTES}
    if (infos["I1"] != saved["runtime"]["local_global_init"]
            or infos["I2"]["i1_initialization"] != infos["I1"]
            or infos["I4"]["bplus_initialization"] != infos["I2"]
            or infos["I4"]["common_state_sha256"] != infos["I2"]["state_sha256"]):
        raise ValueError("local readout common untrained initialization changed")
    return report


def gate_diagnostics(cfg):
    run = queue.run_directory(cfg)
    saved, _, contract = load_run(run)
    picked = select_best(run, cfg.selection_metric, saved)
    history = picked["history"]
    if "local_readout_gate" not in history or not np.isfinite(history.local_readout_gate).all():
        raise ValueError("I4 must record a finite readout gate in every epoch")
    selected_gate = float(picked["checkpoint"]["model"][GATE_KEY])
    if not np.isclose(selected_gate, float(picked["row"].local_readout_gate), rtol=1e-7, atol=1e-10):
        raise ValueError("I4 selected checkpoint gate differs from its history")
    if comparison_contract(cfg, saved["config"]["model"]["params"]["sub_model"]["transformer"]) != contract:
        raise ValueError("I4 gate diagnostics require the current candidate contract")
    return {"diagnostic_only": True, "best_epoch": int(picked["row"].epoch),
            "selected_gate": selected_gate,
            "trajectory": history[["epoch", "local_readout_gate"]].to_dict("records"),
            "scope": "gate magnitude does not prove structural use or causal information loss"}


def analyze(controls, preflight, out):
    from tools.checks.check_local_readout import code_fingerprints

    if out.exists():
        raise FileExistsError(f"refusing existing local readout analysis: {out}")
    if code_fingerprints() != preflight["code_sha256"]:
        raise ValueError("local readout source changed during training")
    for cfg in controls:
        queue.verify_completed(cfg, preflight)
    out.mkdir(parents=True)
    runs = {cfg.local_route: queue.run_directory(cfg) for cfg in controls}
    exports, pairs, artifacts = {}, {}, {}
    for label, run in runs.items():
        artifacts[label] = {"run_dir": str(run), **{
            name: file_sha256(run / name) for name in
            ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}}
        samples, summary, spectra = (out / f"{label}_{suffix}" for suffix in
                                    ("valid_samples.csv", "summary.json", "valid_spectra.npz"))
        export(run, samples, summary, spectra_out=spectra)
        exports[label] = load(summary, samples)
        saved, _, _ = load_run(run)
        dataset = ConfigBuilder(**saved["config"]).get_dataset(split="valid", dos_minmax=True, dos_sumnorm=True)
        if len(dataset) != 2313:
            raise ValueError("local readout analysis requires the complete valid view")
        valid_pairs = spectrum_pairs(spectra, samples, dataset)
        train_pairs = export_train(run, out, label)
        for split, tables in (("valid", valid_pairs), ("train", train_pairs)):
            for (grouping, task), table in tables.items():
                table.to_csv(out / f"{label}_{split}_{grouping}_{task}_pairs.csv", index=False)
                pairs[label, split, grouping, task] = table
        del dataset, valid_pairs, train_pairs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = comparison_report(exports, labels=LABELS, scope=SCOPE)
    structure = {"diagnostic_only": True, "scope": "normalized signed differences, groups equally weighted",
                 "summaries": {}, "contrasts": {}}
    for split in ("train", "valid"):
        for label in ("I2", "I4"):
            assert_paired_targets(out / f"I1_{split}_spectra.npz", out / f"{label}_{split}_spectra.npz")
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                filtered = {}
                for label in READOUT_ROUTES:
                    key = f"{label}_{split}_{grouping}_{task}"
                    structure["summaries"][key], filtered[label] = group_summary(
                        pairs[label, split, grouping, task], task)
                for reference in ("I2", "I1"):
                    structure["contrasts"].setdefault(f"I4_minus_{reference}", {})[
                        f"{split}_{grouping}_{task}"] = paired_group_interval(filtered[reference], filtered["I4"])
    checks = report["joint_blind"]
    report.update(schema="local-readout-analysis-v1", analysis_complete=True,
                  structural_diagnosis=structure, gate=gate_diagnostics(controls[-1]),
                  numerical_adoption_conditions={
                      "vs_I1": checks["I4_minus_I1"]["eligible_for_adoption"],
                      "vs_I2": checks["I4_minus_I2"]["eligible_for_adoption"],
                      "both_required": all(entry["eligible_for_adoption"] for entry in checks.values()),
                      "automatic_baseline_replacement": False},
                  run_artifacts=artifacts, code_sha256=preflight["code_sha256"],
                  data_contract=preflight["data_contract"])
    for cfg in controls:
        run = runs[cfg.local_route]
        if any(file_sha256(run / name) != value for name, value in artifacts[cfg.local_route].items()
               if name != "run_dir"):
            raise ValueError("a trained run changed during frozen analysis")
        path = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
        artifacts[cfg.local_route].update(history_path=str(path), history_sha256=file_sha256(path))
    if code_fingerprints() != preflight["code_sha256"]:
        raise ValueError("local readout source changed during analysis")
    dump(out / "report.json", report)
    return report


def execute(controls, preflight, out):
    from tools.checks.check_local_readout import code_fingerprints

    if code_fingerprints() != preflight["code_sha256"]:
        raise ValueError("local readout source changed after preflight")
    return queue.execute(controls, preflight, out, analyze_fn=analyze, queue_schema="local-readout-queue-v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_I1_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    _, reference, _ = recipe_identity(args.reference)
    controls = local_readout_configs(reference, args.tag_prefix)
    if args.summarize:
        print(json.dumps(summarize(controls), ensure_ascii=False, indent=2))
    elif args.run or args.analyze:
        if args.preflight is None or args.out is None:
            parser.error("--run/--analyze require --preflight and --out")
        preflight = verify_preflight(args.preflight, args.reference, controls)
        if args.run:
            execute(controls, preflight, args.out)
        else:
            analyze(controls, preflight, args.out / "analysis")
    else:
        for cfg in controls:
            print(shlex.join(training_command(cfg)))


if __name__ == "__main__":
    main()
