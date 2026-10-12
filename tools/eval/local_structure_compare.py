"""Frozen C0/C1 attribution queue, reusing the verified 35-epoch B+ G6 result.

Default prints the two commands. --run requires a matching successful real-data
preflight, runs fresh/resumed arms sequentially and exports the paired analysis.
--summarize reads histories; --analyze evaluates completed arms without training.
The only training budget is seed 42, 35 epochs, batch 32, L2/G6 staged. Exceptions
stop the queue and preserve artifacts. No test selection or evaluation is made.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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

from tools.eval.local_global_compare import summarize, training_command
from tools.eval.lg_paired_compare import compare
from tools.eval.lg_structure_diagnosis import (
    dump, loader, pair_errors, paired_group_interval, summarize_pairs,
)
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position, spectral_summary
from tools.eval.lg_valid_export import load_run, select_best
from tools.train import evaluate_split
from utils.builder import ConfigBuilder
from utils.local_global_experiment import comparison_contract
from utils.local_structure_experiment import (
    DEFAULT_REFERENCE, load_control_reference, structure_control_configs,
)
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import state_hash

PRIOR_DIAGNOSIS = ROOT / "results/runs/lg_structure_diagnosis_20261009"


def run_directory(cfg):
    return repository_path(f"results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}")


def verify_preflight(path, reference_dir, controls):
    from tools.checks.check_local_structure_controls import code_fingerprints

    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "local-structure-preflight-v1" or report.get("passed") is not True:
        raise ValueError("successful C0/C1 real-data preflight required")
    if (Path(report["reference_dir"]).resolve() != reference_dir.resolve()
            or report["code_sha256"] != code_fingerprints()):
        raise ValueError("source code or reference changed after preflight")
    if (file_sha256(reference_dir / "config_used.yaml") != report["reference"]["config_sha256"]
            or file_sha256(reference_dir / "checkpoint_best.pth")
            != report["reference"]["best_checkpoint_sha256"]):
        raise ValueError("C2 reference changed after preflight")
    for cfg in controls:
        if report["controls"][cfg.local_route]["configuration"] != asdict(cfg):
            raise ValueError("queue configuration differs from preflight")
    return report


def verify_completed_control(cfg, preflight):
    saved, params, contract = load_run(run_directory(cfg))
    expected_init = preflight["controls"][cfg.local_route]["initialization"]
    if (contract != comparison_contract(cfg, params)
            or saved["runtime"]["local_global_init"] != expected_init
            or saved["cli"]["data_contract"] != preflight["data_contract"]):
        raise ValueError("trained control identity differs from preflight")
    picked = select_best(run_directory(cfg), cfg.selection_metric, saved)
    if int(picked["history"].epoch.max()) != cfg.epochs:
        raise ValueError("analysis requires exactly 35 completed epochs")
    rng = saved["runtime"]["structure_training_rng"][0]
    if (rng["from_epoch"] != 1 or rng["method"] != "seed-reset-before-loop"
            or rng["state_sha256"] != preflight["controls"][cfg.local_route]["training_start_rng_sha256"]):
        raise ValueError("actual control training-start random streams differ from preflight")
    return saved, picked


def analyze(reference_dir, controls, preflight, out):
    """Same best-epoch rule; fixed train subset and full valid; signed pair errors."""
    if out.exists():
        raise FileExistsError(f"refusing existing analysis: {out}")
    prior = json.loads((PRIOR_DIAGNOSIS / "report.json").read_text())
    if prior["groups"]["bp35"]["checkpoint_sha256"] != preflight["reference"]["best_checkpoint_sha256"]:
        raise ValueError("prior diagnostic C2 is a different checkpoint")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [(cfg.local_route, run_directory(cfg), cfg) for cfg in controls] + [("C2", reference_dir, None)]
    report = {"scope": "C0/C1/C2 local content on the same ZP L2/G6 staged backbone; "
                       "seed 42, 35 epochs; parameter matching is not equal expressivity; "
                       "C1/C2 differs in more than angles",
              "selection_metric": "balanced_oracle_mae", "train_subset_source": str(PRIOR_DIAGNOSIS),
              "preflight": deepcopy(preflight["reference"]), "models": {}, "paired_metrics": {},
              "same_composition_pair_differences": {}}
    summaries, frames, tables = {}, {}, {}
    for label, run, cfg in runs:
        saved, _, contract = load_run(run)
        if cfg is None:
            picked = select_best(run, contract["selection_metric"], saved)
        else:
            saved, picked = verify_completed_control(cfg, preflight)
        build = ConfigBuilder(**saved["config"])
        model = build.get_model()
        transformer = model.model["transformer"]
        transformer.load_state_dict(picked["checkpoint"]["model"], strict=True)
        del picked["checkpoint"]
        model.to(device)
        transformer.eval()
        before = state_hash(transformer.state_dict())
        entry = {"run_dir": str(run), "best_epoch": int(picked["row"].epoch),
                 "checkpoint_sha256": file_sha256(run / "checkpoint_best.pth"), "splits": {}}
        for split in ("train", "valid"):
            dataset = build.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
            rows = prior["splits"]["train"]["rows"] if split == "train" else list(range(len(dataset)))
            ids = [str(dataset.material_ids[int(i)]) for i in rows]
            if ids != prior["splits"][split]["material_ids"]:
                raise ValueError("material identities differ from the fixed prior diagnosis")
            measured = evaluate_split(model, loader(dataset, rows), sumnorm=True,
                return_sample_level=True, return_spectra=True, sample_ids=ids)
            frame, spectra = measured["sample_df"], measured["sample_spectra"]
            if not np.isfinite(frame.select_dtypes("number").to_numpy()).all():
                raise ValueError("non-finite evaluated sample metrics")
            frame.to_csv(out / f"{label}_{split}_samples.csv", index=False)
            np.savez(out / f"{label}_{split}_spectra.npz", material_ids=np.asarray(ids), **spectra)
            keys = composition_keys_by_position(dataset, rows)
            metrics = spectral_summary(frame)
            entry["splits"][split] = {"samples": len(ids), "metrics": metrics, "pairs": {}}
            for grouping in ("absolute", "reduced"):
                for task in ("edos", "phdos"):
                    pairs = pair_errors(spectra[f"pred_shape_{task}"], spectra[f"tgt_shape_{task}"],
                                        keys[grouping], ids)
                    pairs.to_csv(out / f"{label}_{split}_{grouping}_{task}_pairs.csv", index=False)
                    tables[label, split, grouping, task] = pairs
                    entry["splits"][split]["pairs"][f"{grouping}_{task}"] = summarize_pairs(pairs)
                    if label == "C0" and grouping == "absolute" and len(pairs):
                        if float(pairs.predicted_tv.max()) > 1e-4:
                            raise ValueError("C0 unexpectedly distinguishes identical absolute compositions")
            if split == "valid":
                frames[label] = frame.set_index("material_id").sort_index()
                history, row = picked["history"], picked["row"]
                summaries[label] = {"tag": picked["tag"], "best_epoch": int(row.epoch), "epochs": 35,
                    "best_score": float(row.balanced_score), "contract_schema": contract["schema"],
                    "init_protocol": contract["init_protocol"],
                    "epoch_time_s_median": float(history.epoch_time_s.median()),
                    "epoch_time_s_sum": float(history.epoch_time_s.sum()),
                    "peak_vram_mb_median": float(history.peak_vram_mb.median()),
                    "peak_vram_mb_max": float(history.peak_vram_mb.max())}
            print(json.dumps({"analyzed": label, "split": split, "samples": len(ids)}), flush=True)
            del dataset, measured, frame, spectra
        if state_hash(transformer.state_dict()) != before:
            raise ValueError("read-only analysis changed model state")
        if label == "C2":
            previous = pd.read_csv(PRIOR_DIAGNOSIS / "bp35_valid_samples.csv").set_index("material_id").sort_index()
            if list(previous.index) != list(frames[label].index):
                raise ValueError("C2 valid partition changed")
            for column in previous:
                if column.startswith(("blind_", "oracle_")):
                    np.testing.assert_allclose(frames[label][column], previous[column], rtol=2e-4, atol=2e-5)
                    if "_r2_" in column:
                        np.testing.assert_array_equal(frames[label][column] < 0, previous[column] < 0)
            entry["previous_valid_metrics_reproduced"] = True
        report["models"][label] = entry
        del model, transformer, build
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for base, candidates in (("C0", ("C1", "C2")), ("C1", ("C2",))):
        report["paired_metrics"]["versus_" + base] = compare(summaries[base], frames[base],
            [(other, summaries[other], frames[other]) for other in candidates], 2000, 20261009,
            scope=report["scope"])
        for other in candidates:
            contrast = {}
            for split in ("train", "valid"):
                for grouping in ("absolute", "reduced"):
                    for task in ("edos", "phdos"):
                        contrast[f"{split}_{grouping}_{task}"] = paired_group_interval(
                            tables[base, split, grouping, task], tables[other, split, grouping, task])
            report["same_composition_pair_differences"][f"{other}_minus_{base}"] = contrast
    dump(out / "report.json", report)
    return report


def execute(reference_dir, controls, preflight, out):
    out.mkdir(parents=True, exist_ok=True)
    state = {"schema": "local-structure-queue-v1", "pid": os.getpid(), "status": "running",
             "started_at_unix": time.time(), "arms": {cfg.local_route: "pending" for cfg in controls}}
    try:
        for cfg in controls:
            row = summarize([cfg])[0]
            if row["status"] == "complete":
                verify_completed_control(cfg, preflight)
                state["arms"][cfg.local_route] = "complete"
                continue
            log_path = out / f"{cfg.local_route.lower()}_train.log"
            state["arms"][cfg.local_route] = "running"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(training_command(cfg), cwd=ROOT,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    stdout=log, stderr=subprocess.STDOUT)
                state["active_pid"] = process.pid
                dump(out / "queue.json", state)
                print(json.dumps({"started": cfg.local_route, "pid": process.pid, "log": str(log_path)}), flush=True)
                code = process.wait()
                if code:
                    raise RuntimeError(f"{cfg.local_route} exited with {code}; see {log_path}")
            verify_completed_control(cfg, preflight)
            state["arms"][cfg.local_route] = "complete"
            print(json.dumps({"completed": cfg.local_route}), flush=True)
        state.update(status="analyzing", active_pid=None)
        dump(out / "queue.json", state)
        analyze(reference_dir, controls, preflight, out / "analysis")
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
    controls = structure_control_configs(reference, args.tag_prefix)
    if args.summarize:
        print(json.dumps(summarize(controls), ensure_ascii=False, indent=2))
    elif args.run or args.analyze:
        if args.preflight is None or args.out is None:
            parser.error("--run/--analyze requires --preflight and --out")
        preflight = verify_preflight(args.preflight, args.reference, controls)
        if args.run:
            execute(args.reference, controls, preflight, args.out)
        else:
            analyze(args.reference, controls, preflight, args.out)
    else:
        for cfg in controls:
            print(shlex.join(training_command(cfg)))


if __name__ == "__main__":
    main()
