"""One N1 run: isolate global normalization placement with unchanged C1 initial values."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from tools.eval.legacy_backbone_compare import (
    PRIOR_REPORT, reference_identity as legacy_reference_identity, run_directory, verify_completed,
)
from tools.eval.lg_paired_compare import compare, load, run_line
from tools.eval.lg_structure_diagnosis import dump, pair_errors, paired_group_interval, summarize_pairs
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position
from tools.eval.lg_valid_export import export, load_run
from tools.eval.local_global_compare import summarize, training_command
from utils.builder import ConfigBuilder
from utils.global_norm_experiment import global_norm_config
from utils.legacy_backbone_experiment import legacy_backbone_configs
from utils.local_structure_experiment import DEFAULT_REFERENCE, load_control_reference
from utils.paths import file_sha256

LEGACY_ROOT = ROOT / "results/runs/legacy_backbone_attribution_20261012"
LABELS = ("N1", "C1", "O1", "original_baseline")
SCOPE = ("seed 42, 35 epochs, crystal_q1_v1; N1 minus reference; N1/C1 differ only in global "
         "normalization placement, with identical initial tensors; O1/baseline are package references; "
         "intervals conditional on selected checkpoints, not training-seed uncertainty")


def reference_identity(reference_dir):
    saved, cfg, identities = legacy_reference_identity(reference_dir)
    old_preflight = json.loads((LEGACY_ROOT / "preflight_final.json").read_text())
    if old_preflight.get("passed") is not True or old_preflight.get("schema") != "legacy-backbone-preflight-v1":
        raise ValueError("completed O1 reference requires its historical successful preflight")
    old_cfg = legacy_backbone_configs(cfg, "attrib20261012")[1]
    verify_completed(old_cfg, old_preflight)
    recorded = json.loads((LEGACY_ROOT / "analysis/report.json").read_text())["run_artifacts"]["O1"]
    run = run_directory(old_cfg)
    files = {name: file_sha256(run / name) for name in
             ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}
    if (str(run) != recorded["run_dir"] or files["config_used.yaml"] != recorded["configuration_sha256"]
            or files["checkpoint_best.pth"] != recorded["best_checkpoint_sha256"]
            or old_preflight["data_contract"] != saved["cli"]["data_contract"]):
        raise ValueError("O1 no longer matches its verified result or the shared data")
    identities["O1"] = {"run_dir": str(run), "files": files}
    return saved, cfg, identities


def evidence_fingerprints():
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in
            (PRIOR_REPORT, LEGACY_ROOT / "preflight_final.json", LEGACY_ROOT / "analysis/report.json")}


def verify_preflight(path, reference_dir, cfg):
    from tools.checks.check_global_norm import code_fingerprints

    report = json.loads(path.read_text())
    if report.get("schema") != "global-norm-preflight-v1" or report.get("passed") is not True:
        raise ValueError("successful N1 real-data preflight required")
    _, _, references = reference_identity(reference_dir)
    if (report["controls"]["N1"]["configuration"] != asdict(cfg)
            or report["code_sha256"] != code_fingerprints() or report["references"] != references
            or report["reference_evidence_sha256"] != evidence_fingerprints()):
        raise ValueError("N1 source, reference or queue configuration changed after preflight")
    return report


def comparison_report(exports, draws=2000, *, labels=LABELS, scope=SCOPE, seed=20261013):
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("at least two integer bootstrap draws required")
    if len(labels) < 2 or len(set(labels)) != len(labels) or set(exports) != set(labels):
        raise ValueError("one candidate and distinct reference labels must match exports")
    candidate = labels[0]
    frames = [exports[label][1] for label in labels]
    if (not len(frames[0]) or any(not frame.index.is_unique or not np.isfinite(frame.to_numpy()).all()
                                or not frame.index.equals(frames[0].index) for frame in frames)):
        raise ValueError("finite unique material-ID partitions must pair for all four runs")
    result = {"scope": scope, "runs": {label: run_line(exports[label][0]) for label in labels},
              "bootstrap": {"draws": draws, "seed": seed, "unit": "material", "paired": True},
              "contrasts": {}}
    for reference in labels[1:]:
        summary, frame = exports[reference]
        compared = compare(summary, frame, [(candidate, *exports[candidate])], draws, seed, scope=scope)
        result["contrasts"][f"{candidate}_minus_{reference}"] = compared["comparisons"][candidate]
        if "joint_blind" in compared:
            result["primary_metric"] = compared["primary_metric"]
            result.setdefault("joint_blind", {})[f"{candidate}_minus_{reference}"] = compared["joint_blind"][candidate]
        result["pairing"] = compared["pairing"]
    return result


def composition_contrasts(pairs, *, labels=LABELS):
    result = {}
    candidate = labels[0]
    for reference in labels[1:]:
        result[f"{candidate}_minus_{reference}"] = {
            f"valid_{grouping}_{task}": paired_group_interval(
                pairs[reference, grouping, task], pairs[candidate, grouping, task])
            for grouping in ("absolute", "reduced") for task in ("edos", "phdos")}
    return result


def analyze(cfg, preflight, out, *, labels=LABELS, scope=SCOPE, additional_runs=None,
            comparison_fn=None, composition_fn=None):
    if labels[0] != cfg.local_route:
        raise ValueError("analysis candidate must match the configured route")
    additional_runs = {} if additional_runs is None else additional_runs
    if set(additional_runs) - set(labels[1:]) or set(additional_runs) & set(preflight["references"]):
        raise ValueError("additional completed runs cannot override a frozen reference or the candidate")
    verify_completed(cfg, preflight)
    if out.exists():
        raise FileExistsError(f"refusing existing {cfg.local_route} analysis: {out}")
    for label, entry in preflight["references"].items():
        if any(file_sha256(Path(entry["run_dir"]) / name) != digest for name, digest in entry["files"].items()):
            raise ValueError(f"{label} reference changed during N1 training")
    runs = {labels[0]: run_directory(cfg), **{label: Path(additional_runs[label] if label in additional_runs
                                                       else preflight["references"][label]["run_dir"])
                                             for label in labels[1:]}}
    out.mkdir(parents=True)
    exports, pairs, summaries = {}, {}, {}
    for label, run in runs.items():
        samples, summary, spectrum_path = (out / f"{label}_{suffix}" for suffix in
                                           ("valid_samples.csv", "summary.json", "valid_spectra.npz"))
        export(run, samples, summary, spectra_out=spectrum_path)
        exports[label] = load(summary, samples)
        frame = pd.read_csv(samples)
        saved, _, _ = load_run(run)
        dataset = ConfigBuilder(**saved["config"]).get_dataset(split="valid", dos_minmax=True, dos_sumnorm=True)
        lookup = {str(value): i for i, value in enumerate(dataset.material_ids)}
        rows = [lookup[value] for value in frame.material_id]
        keys = composition_keys_by_position(dataset, rows)
        with np.load(spectrum_path) as spectra:
            if spectra["material_ids"].tolist() != frame.material_id.tolist():
                raise ValueError("spectrum and material order differ")
            for grouping in ("absolute", "reduced"):
                for task in ("edos", "phdos"):
                    table = pair_errors(spectra[f"pred_shape_{task}"], spectra[f"tgt_shape_{task}"],
                                        keys[grouping], frame.material_id.tolist())
                    table.to_csv(out / f"{label}_valid_{grouping}_{task}_pairs.csv", index=False)
                    pairs[label, grouping, task] = table
                    summaries[f"{label}_{grouping}_{task}"] = summarize_pairs(table)
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = (comparison_report(exports, labels=labels, scope=scope) if comparison_fn is None
              else comparison_fn(exports))
    contrasts = (composition_contrasts(pairs, labels=labels) if composition_fn is None
                 else composition_fn(pairs))
    report["same_composition_pairs"] = {"summaries": summaries, "contrasts": contrasts}
    report["run_artifacts"] = {label: {"run_dir": str(run),
        "configuration_sha256": file_sha256(run / "config_used.yaml"),
        "best_checkpoint_sha256": file_sha256(run / "checkpoint_best.pth")} for label, run in runs.items()}
    report["preflight"] = preflight["code_sha256"]
    dump(out / "report.json", report)
    return report


def execute(cfg, preflight, out, *, analyze_fn=None, queue_schema="global-norm-queue-v1"):
    route = cfg.local_route
    analyze_fn = analyze if analyze_fn is None else analyze_fn
    out.mkdir(parents=True, exist_ok=True)
    state = {"schema": queue_schema, "pid": os.getpid(), "status": "running",
             "started_at_unix": time.time(), "arms": {route: "pending"}}
    try:
        if summarize([cfg])[0]["status"] != "complete":
            with (out / f"{route.lower()}_train.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(training_command(cfg), cwd=ROOT,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=log, stderr=subprocess.STDOUT)
                state.update(active_pid=process.pid)
                state["arms"][route] = "running"
                dump(out / "queue.json", state)
                print(json.dumps({"started": route, "pid": process.pid}), flush=True)
                if process.wait():
                    raise RuntimeError(f"{route} failed; see {route.lower()}_train.log")
        verify_completed(cfg, preflight)
        state["arms"][route] = "complete"
        state.update(status="analyzing", active_pid=None)
        dump(out / "queue.json", state)
        analyze_fn(cfg, preflight, out / "analysis")
        state.update(status="complete", finished_at_unix=time.time())
        dump(out / "queue.json", state)
        print(json.dumps({"queue": "complete", "report": str(out / "analysis/report.json")}), flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error), stopped_at_unix=time.time())
        dump(out / "queue.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    _, reference = load_control_reference(args.reference)
    cfg = global_norm_config(reference, args.tag_prefix)
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
