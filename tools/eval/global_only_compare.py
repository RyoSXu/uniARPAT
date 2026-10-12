"""Plan, train or analyze one frozen G0 control against existing 35-epoch runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.lg_paired_compare import compare, load
from tools.eval.lg_valid_export import export, load_run, select_best
from tools.eval.local_global_compare import summarize, training_command
from tools.train import train_and_eval
from utils.global_only_experiment import global_only_config
from utils.local_global_experiment import comparison_contract
from utils.local_structure_experiment import DEFAULT_REFERENCE, load_control_reference
from utils.paths import file_sha256, repository_path

CONTROL_ANALYSIS = ROOT / "results/runs/local_structure_attribution_20261009/analysis/report.json"
BASELINE = ROOT / "results/runs/ablation_m1_lg_baseline_q1v1_e35_s42"


def run_directory(cfg):
    return repository_path(f"results/runs/ablation_m1{cfg.tag}")


def verify_preflight(path, reference_dir, cfg):
    from tools.checks.check_global_only import code_fingerprints

    report = json.loads(path.read_text())
    if report.get("schema") != "global-only-preflight-v1" or report.get("passed") is not True:
        raise ValueError("successful G0 real-data preflight required")
    if (report["configuration"] != asdict(cfg)
            or Path(report["reference_dir"]).resolve() != reference_dir.resolve()
            or report["code_sha256"] != code_fingerprints()
            or any(file_sha256(reference_dir / name) != digest
                   for name, digest in report["reference_files"].items())):
        raise ValueError("G0 configuration, source or reference changed after preflight")
    return report


def verify_completed(cfg, preflight):
    saved, params, contract = load_run(run_directory(cfg))
    if (contract != comparison_contract(cfg, params)
            or saved["runtime"]["local_global_init"] != preflight["initialization"]
            or saved["cli"]["data_contract"] != preflight["data_contract"]):
        raise ValueError("trained G0 identity differs from the preflight")
    picked = select_best(run_directory(cfg), cfg.selection_metric, saved)
    if picked["history"].epoch.tolist() != list(range(1, 36)):
        raise ValueError("G0 requires exactly 35 completed epochs")
    import torch
    latest = torch.load(run_directory(cfg) / "checkpoint_latest.pth", weights_only=True, map_location="cpu")
    if (latest.get("epoch") != 35 or latest.get("model_contract") != contract
            or latest.get("data_contract") != preflight["data_contract"]):
        raise ValueError("G0 latest checkpoint is not the completed experiment")
    rng = saved["runtime"]["structure_training_rng"][0]
    if (rng["from_epoch"] != 1 or rng["method"] != "seed-reset-before-loop"
            or rng["state_sha256"] != preflight["training_start_rng_sha256"]):
        raise ValueError("G0 training-start RNG differs from preflight")


def comparison_report(exports, draws=2000):
    """Keep G0 distinct from the original baseline in both labels and differences."""
    g0_summary, g0_frame = exports["G0"]
    arms = [(label, *exports[label]) for label in ("C0", "C1", "C2", "original_baseline")]
    report = compare(g0_summary, g0_frame, arms, draws, 20261012,
        scope="arm minus G0; seed 42, 35 epochs, crystal_q1_v1; local arms have identical retained "
              "initial values; original baseline matches outside the encoder and is an encoder package")
    report["runs"]["G0"] = report["runs"].pop("baseline")
    report["comparison_reference"] = "G0"
    return report


def analyze(reference_dir, cfg, preflight, out):
    verify_completed(cfg, preflight)
    if out.exists():
        raise FileExistsError(f"refusing existing G0 analysis: {out}")
    prior = json.loads(CONTROL_ANALYSIS.read_text())
    runs = {"G0": run_directory(cfg), "C0": Path(prior["models"]["C0"]["run_dir"]),
            "C1": Path(prior["models"]["C1"]["run_dir"]), "C2": reference_dir, "original_baseline": BASELINE}
    # Validate every reused run before creating output; evaluation never retrains it.
    for label, run in runs.items():
        saved, _, contract = load_run(run)
        if saved["cli"]["data_contract"] != preflight["data_contract"]:
            raise ValueError(f"{label} uses a different data contract")
        training = contract["training"]
        if any(training[key] != value for key, value in
               {"seed": 42, "epochs": 35, "batch_size": 32, "lr": 5e-5,
                "selection_metric": "balanced_oracle_mae"}.items()):
            raise ValueError(f"{label} has a different comparison recipe")
        picked = select_best(run, cfg.selection_metric, saved)
        if picked["history"].epoch.tolist() != list(range(1, 36)):
            raise ValueError(f"{label} has not completed the exact 35-epoch budget")
        if label in ("C0", "C1", "C2") and file_sha256(run / "checkpoint_best.pth") \
                != prior["models"][label]["checkpoint_sha256"]:
            raise ValueError(f"{label} differs from the completed attribution report")
    out.mkdir(parents=True)
    exports = {}
    for label, run in runs.items():
        samples, summary = out / f"{label}_valid_samples.csv", out / f"{label}_summary.json"
        export(run, samples, summary)
        exports[label] = load(summary, samples)
    report = comparison_report(exports)
    report["run_artifacts"] = {label: {"run_dir": str(run),
        "configuration_sha256": file_sha256(run / "config_used.yaml"),
        "best_checkpoint_sha256": file_sha256(run / "checkpoint_best.pth")}
        for label, run in runs.items()}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"analysis": "complete", "report": str(out / "report.json")}), flush=True)


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
    if args.analyze and args.out is None:
        parser.error("--analyze requires --out")
    _, reference = load_control_reference(args.reference)
    cfg = global_only_config(reference, args.tag_prefix)
    if args.summarize:
        print(json.dumps(summarize([cfg]), ensure_ascii=False, indent=2))
    elif args.run or args.analyze:
        if args.preflight is None:
            parser.error("--run/--analyze requires --preflight")
        preflight = verify_preflight(args.preflight, args.reference, cfg)
        if args.run:
            if summarize([cfg])[0]["status"] != "complete":
                train_and_eval(cfg)
            verify_completed(cfg, preflight)
        if args.analyze or args.out is not None:
            if args.out is None:
                parser.error("--analyze requires --out")
            analyze(args.reference, cfg, preflight, args.out)
    else:
        print(shlex.join(training_command(cfg)))


if __name__ == "__main__":
    main()
