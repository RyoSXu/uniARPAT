"""Train only O0/O1, then compare old/new globals and local-update interaction."""
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

from tools.eval.lg_paired_compare import load, run_line
from tools.eval.lg_structure_diagnosis import dump, pair_errors, paired_group_interval, summarize_pairs
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position
from tools.eval.lg_valid_export import export, load_run, select_best
from tools.eval.local_global_compare import summarize, training_command
from utils.builder import ConfigBuilder
from utils.legacy_backbone_experiment import legacy_backbone_configs
from utils.local_global_experiment import comparison_contract
from utils.local_structure_experiment import DEFAULT_REFERENCE, load_control_reference
from utils.paths import file_sha256, repository_path

PRIOR_REPORT = ROOT / "results/runs/global_only_attribution_20261012/analysis/report.json"
LABELS = ("O0", "O1", "G0", "C1", "original_baseline")
CONTRASTS = {
    "baseline_minus_O0": {"original_baseline": 1, "O0": -1},
    "O1_minus_O0": {"O1": 1, "O0": -1},
    "C1_minus_G0": {"C1": 1, "G0": -1},
    "O0_minus_G0": {"O0": 1, "G0": -1},
    "O1_minus_C1": {"O1": 1, "C1": -1},
    "interaction_old_minus_new": {"O1": 1, "O0": -1, "C1": -1, "G0": 1},
}
SCOPE = ("seed 42, 35 epochs, crystal_q1_v1; old globals omit geometric scoring; "
         "old/new backbone contrast includes encoder parameterization and initial values; "
         "interaction=(O1-O0)-(C1-G0); intervals conditional on selected checkpoints")


def run_directory(cfg):
    return repository_path(f"results/runs/ablation_m1{cfg.tag}")


def reference_identity(reference_dir):
    saved, cfg = load_control_reference(reference_dir)
    prior = json.loads(PRIOR_REPORT.read_text())
    artifacts = {}
    for label in ("G0", "C1", "original_baseline"):
        entry = prior["run_artifacts"][label]
        run = Path(entry["run_dir"])
        record, _, contract = load_run(run)
        if record["cli"]["data_contract"] != saved["cli"]["data_contract"]:
            raise ValueError(f"{label} data differ from the C2 protocol")
        expected = {"seed": 42, "epochs": 35, "batch_size": 32, "lr": 5e-5,
                    "selection_metric": "balanced_oracle_mae"}
        if any(contract["training"][key] != value for key, value in expected.items()):
            raise ValueError(f"{label} comparison recipe differs")
        picked = select_best(run, cfg.selection_metric, record)
        if picked["history"].epoch.tolist() != list(range(1, 36)):
            raise ValueError(f"{label} must have exactly 35 completed epochs")
        latest = torch.load(run / "checkpoint_latest.pth", map_location="cpu", weights_only=True)
        if (latest.get("epoch") != 35 or latest.get("model_contract") != contract
                or latest.get("data_contract") != saved["cli"]["data_contract"]):
            raise ValueError(f"{label} latest identity differs")
        files = {name: file_sha256(run / name) for name in
                 ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}
        if (files["config_used.yaml"] != entry["configuration_sha256"]
                or files["checkpoint_best.pth"] != entry["best_checkpoint_sha256"]):
            raise ValueError(f"{label} no longer matches the verified G0 report")
        artifacts[label] = {"run_dir": str(run), "files": files}
        del picked, latest
        gc.collect()
    artifacts["C2"] = {"run_dir": str(reference_dir.resolve()), "files": {
        name: file_sha256(reference_dir / name) for name in
        ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}}
    return saved, cfg, artifacts


def verify_preflight(path, reference_dir, controls):
    from tools.checks.check_legacy_backbone import code_fingerprints

    report = json.loads(path.read_text())
    if report.get("schema") != "legacy-backbone-preflight-v1" or report.get("passed") is not True:
        raise ValueError("successful O0/O1 real-data preflight required")
    _, _, references = reference_identity(reference_dir)
    if (report["code_sha256"] != code_fingerprints() or report["references"] != references
            or report["prior_report_sha256"] != file_sha256(PRIOR_REPORT)
            or any(report["controls"][cfg.local_route]["configuration"] != asdict(cfg) for cfg in controls)):
        raise ValueError("source, reference or queue configuration changed after preflight")
    return report


def verify_completed(cfg, preflight):
    run = run_directory(cfg)
    saved, params, contract = load_run(run)
    entry = preflight["controls"][cfg.local_route]
    if (contract != comparison_contract(cfg, params)
            or saved["runtime"]["local_global_init"] != entry["initialization"]
            or saved["cli"]["data_contract"] != preflight["data_contract"]):
        raise ValueError("completed O0/O1 identity differs from preflight")
    picked = select_best(run, cfg.selection_metric, saved)
    if picked["history"].epoch.tolist() != list(range(1, 36)):
        raise ValueError("O0/O1 require exactly 35 completed epochs")
    latest = torch.load(run / "checkpoint_latest.pth", map_location="cpu", weights_only=True)
    if (latest.get("epoch") != 35 or latest.get("model_contract") != contract
            or latest.get("data_contract") != preflight["data_contract"]):
        raise ValueError("O0/O1 latest checkpoint does not complete the experiment")
    rng = saved["runtime"]["structure_training_rng"][0]
    if (rng["from_epoch"] != 1 or rng["method"] != "seed-reset-before-loop"
            or rng["state_sha256"] != entry["training_start_rng_sha256"]):
        raise ValueError("O0/O1 training-start RNG differs from preflight")


def factorial_report(exports, draws=2000):
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("at least two integer bootstrap draws required")
    frames = {label: exports[label][1] for label in LABELS}
    index = frames["O0"].index
    if not index.is_unique or not len(index) or any(not index.equals(frame.index) for frame in frames.values()):
        raise ValueError("five unique, identically ordered material-ID partitions required")
    if any(not np.isfinite(frame.to_numpy()).all() for frame in frames.values()):
        raise ValueError("finite paired sample metrics required")
    report = {"scope": SCOPE, "runs": {label: run_line(exports[label][0]) for label in LABELS},
              "pairing": {"materials": len(index), "ids_identical": True},
              "bootstrap": {"draws": draws, "seed": 20261012, "unit": "material", "paired": True},
              "contrast_coefficients": CONTRASTS, "contrasts": {name: {} for name in CONTRASTS}}
    rng = np.random.default_rng(20261012)
    for mode in ("blind", "oracle"):
        for task in ("edos", "phdos"):
            for statistic in ("r2_median", "mae_median", "fail_rate_pct"):
                column = f"{mode}_{'mae' if statistic == 'mae_median' else 'r2'}_{task}"
                arrays = {label: frame[column].to_numpy() for label, frame in frames.items()}
                reducer = (lambda values, axis: (values < 0).mean(axis=axis) * 100
                           ) if statistic == "fail_rate_pct" else np.median
                points = {label: float(reducer(values, axis=0)) for label, values in arrays.items()}
                boot = {label: np.empty(draws) for label in LABELS}
                for start in range(0, draws, 128):
                    stop = min(start + 128, draws)
                    indices = rng.integers(0, len(index), size=(stop-start, len(index)))
                    for label in LABELS:
                        boot[label][start:stop] = reducer(arrays[label][indices], axis=1)
                key = f"{mode}_{task}_{statistic}"
                for name, coefficients in CONTRASTS.items():
                    differences = sum(coefficient * boot[label] for label, coefficient in coefficients.items())
                    interval = np.percentile(differences, [2.5, 97.5]).tolist()
                    report["contrasts"][name][key] = {
                        "difference": sum(coefficient * points[label] for label, coefficient in coefficients.items()),
                        "components": {label: points[label] for label in coefficients}, "ci95": interval,
                        "ci95_contains_zero": bool(interval[0] <= 0 <= interval[1])}
    return report


def composition_contrasts(pairs):
    """Use composition groups, including for the old-minus-new interaction."""
    result = {}
    for name, coefficients in CONTRASTS.items():
        result[name] = {}
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                template = pairs["O0", grouping, task]
                for label in coefficients:
                    if not template[["left_id", "right_id", "composition"]].equals(
                            pairs[label, grouping, task][["left_id", "right_id", "composition"]]):
                        raise ValueError("composition-pair identities differ")
                negative, positive = template.copy(), template.copy()
                negative["difference_error"] = sum(-coefficient * pairs[label, grouping, task].difference_error
                    for label, coefficient in coefficients.items() if coefficient < 0)
                positive["difference_error"] = sum(coefficient * pairs[label, grouping, task].difference_error
                    for label, coefficient in coefficients.items() if coefficient > 0)
                result[name][f"valid_{grouping}_{task}"] = paired_group_interval(negative, positive)
    return result


def analyze(controls, preflight, out):
    if out.exists():
        raise FileExistsError(f"refusing existing analysis: {out}")
    for cfg in controls:
        verify_completed(cfg, preflight)
    runs = {cfg.local_route: run_directory(cfg) for cfg in controls}
    runs.update({label: Path(preflight["references"][label]["run_dir"])
                 for label in ("G0", "C1", "original_baseline")})
    out.mkdir(parents=True)
    exports, pairs, pair_summaries = {}, {}, {}
    for label in LABELS:
        run = runs[label]
        samples, summary, spectra_path = (out / f"{label}_{suffix}" for suffix in
                                          ("valid_samples.csv", "summary.json", "valid_spectra.npz"))
        export(run, samples, summary, spectra_out=spectra_path)
        exports[label] = load(summary, samples)
        frame = pd.read_csv(samples)
        record, _, _ = load_run(run)
        dataset = ConfigBuilder(**record["config"]).get_dataset(split="valid", dos_minmax=True, dos_sumnorm=True)
        lookup = {str(value): i for i, value in enumerate(dataset.material_ids)}
        rows = [lookup[value] for value in frame.material_id]
        keys = composition_keys_by_position(dataset, rows)
        with np.load(spectra_path) as spectra:
            if spectra["material_ids"].tolist() != frame.material_id.tolist():
                raise ValueError("spectrum and material-ID order differ")
            for grouping in ("absolute", "reduced"):
                for task in ("edos", "phdos"):
                    table = pair_errors(spectra[f"pred_shape_{task}"], spectra[f"tgt_shape_{task}"],
                                        keys[grouping], frame.material_id.tolist())
                    table.to_csv(out / f"{label}_valid_{grouping}_{task}_pairs.csv", index=False)
                    pairs[label, grouping, task] = table
                    pair_summaries[f"{label}_{grouping}_{task}"] = summarize_pairs(table)
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = factorial_report(exports)
    report["same_composition_pairs"] = {"summaries": pair_summaries, "contrasts": composition_contrasts(pairs)}
    report["run_artifacts"] = {label: {"run_dir": str(run), "configuration_sha256": file_sha256(run / "config_used.yaml"),
        "best_checkpoint_sha256": file_sha256(run / "checkpoint_best.pth")} for label, run in runs.items()}
    report["preflight"] = preflight["code_sha256"]
    dump(out / "report.json", report)
    return report


def execute(controls, preflight, out, *, analyze_fn=None, queue_schema="legacy-backbone-queue-v1"):
    analyze_fn = analyze if analyze_fn is None else analyze_fn
    out.mkdir(parents=True, exist_ok=True)
    state = {"schema": queue_schema, "pid": os.getpid(), "status": "running",
             "started_at_unix": time.time(), "arms": {cfg.local_route: "pending" for cfg in controls}}
    try:
        for cfg in controls:
            if summarize([cfg])[0]["status"] != "complete":
                log_path = out / f"{cfg.local_route.lower()}_train.log"
                with log_path.open("a", encoding="utf-8") as log:
                    process = subprocess.Popen(training_command(cfg), cwd=ROOT,
                        env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                             "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                        stdout=log, stderr=subprocess.STDOUT)
                    state.update(active_pid=process.pid)
                    state["arms"][cfg.local_route] = "running"
                    dump(out / "queue.json", state)
                    print(json.dumps({"started": cfg.local_route, "pid": process.pid, "log": str(log_path)}), flush=True)
                    if process.wait():
                        raise RuntimeError(f"{cfg.local_route} failed; see {log_path}")
            verify_completed(cfg, preflight)
            state["arms"][cfg.local_route] = "complete"
            dump(out / "queue.json", state)
        state.update(status="analyzing", active_pid=None)
        dump(out / "queue.json", state)
        analyze_fn(controls, preflight, out / "analysis")
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
    controls = legacy_backbone_configs(reference, args.tag_prefix)
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
