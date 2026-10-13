"""Train I1/I0 only; separate projection initialization from trainability."""
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

from tools.eval import attention_dropout_compare as dropout
from tools.eval import global_norm_compare as norm
from tools.eval import legacy_backbone_compare as legacy
from tools.eval.lg_valid_export import load_run
from tools.eval.local_global_compare import summarize, training_command
from utils.attention_dropout_experiment import attention_dropout_config
from utils.attention_projection_experiment import attention_projection_configs
from utils.local_structure_experiment import DEFAULT_REFERENCE, load_control_reference
from utils.paths import file_sha256

DROPOUT_ROOT = ROOT / "results/runs/attention_dropout_attribution_20261013"
LABELS = ("I0", "I1", "D1", "O1", "original_baseline")
SCOPE = ("seed 42, 35 epochs, crystal_q1_v1; I1-D1 changes only QKVO initialization to "
         "identities/zero biases; I0-I1 changes only QKVO trainability with all initial tensors "
         "identical, including a change in effective trainable capacity; all retain C1 locals, "
         "post-norm and zero probability dropout; O1/baseline are package references; "
         "intervals conditional on selected checkpoints, not training-seed uncertainty")
DIRECT_CONTRASTS = {"I1_minus_D1": "projection initialization at fixed trainable architecture",
                    "I0_minus_I1": "projection trainability/capacity at fixed identity initialization"}


def reference_identity(reference_dir):
    saved, cfg, identities = dropout.reference_identity(reference_dir)
    preflight = json.loads((DROPOUT_ROOT / "preflight_final.json").read_text())
    queue = json.loads((DROPOUT_ROOT / "queue.json").read_text())
    if (preflight.get("schema") != "attention-dropout-preflight-v1" or preflight.get("passed") is not True
            or queue.get("status") != "complete" or queue.get("arms") != {"D1": "complete"}
            or (DROPOUT_ROOT / "run.exit").read_text().strip() != "0"):
        raise ValueError("projection controls require the completed, historically verified D1")
    d1_cfg = attention_dropout_config(cfg, "drop20261013")
    legacy.verify_completed(d1_cfg, preflight)
    run = legacy.run_directory(d1_cfg)
    recorded = json.loads((DROPOUT_ROOT / "analysis/report.json").read_text())["run_artifacts"]["D1"]
    files = {name: file_sha256(run / name) for name in
             ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}
    record, _, _ = load_run(run)
    if (str(run) != recorded["run_dir"] or files["config_used.yaml"] != recorded["configuration_sha256"]
            or files["checkpoint_best.pth"] != recorded["best_checkpoint_sha256"]
            or record["cli"]["data_contract"] != saved["cli"]["data_contract"]
            or preflight["data_contract"] != saved["cli"]["data_contract"]):
        raise ValueError("D1 differs from its completed result or the common data")
    identities["D1"] = {"run_dir": str(run), "files": files}
    return saved, cfg, identities


def evidence_fingerprints():
    return {**dropout.evidence_fingerprints(), **{
        str(path.relative_to(ROOT)): file_sha256(path) for path in
        (DROPOUT_ROOT / "preflight_final.json", DROPOUT_ROOT / "analysis/report.json",
         DROPOUT_ROOT / "queue.json", DROPOUT_ROOT / "run.exit")}}


def verify_preflight(path, reference_dir, controls):
    from tools.checks.check_attention_projection import code_fingerprints

    report = json.loads(path.read_text())
    if report.get("schema") != "attention-projection-preflight-v1" or report.get("passed") is not True:
        raise ValueError("successful real-data I1/I0 preflight required")
    if [cfg.local_route for cfg in controls] != ["I1", "I0"]:
        raise ValueError("projection queue requires exactly I1 then I0")
    _, _, references = reference_identity(reference_dir)
    if (report["code_sha256"] != code_fingerprints() or report["references"] != references
            or report["reference_evidence_sha256"] != evidence_fingerprints()
            or set(report["controls"]) != {"I1", "I0"}
            or any(report["controls"][cfg.local_route]["configuration"] != asdict(cfg) for cfg in controls)):
        raise ValueError("projection source, references or configurations changed after preflight")
    left, right = (report["controls"][route]["initialization"] for route in ("I1", "I0"))
    if (left["state_sha256"] != right["state_sha256"]
            or left["d1_initialization"] != right["d1_initialization"]
            or left["frozen_parameter_names"] or len(right["frozen_parameter_names"]) != 24
            or any(report["controls"][route]["training_start_rng_sha256"]
                   != report["controls"]["I1"]["training_start_rng_sha256"] for route in ("I1", "I0"))):
        raise ValueError("I1/I0 must pair complete initial tensors and differ only in QKVO trainability")
    return report


def comparison_report(exports, draws=2000):
    left = norm.comparison_report({label: exports[label] for label in LABELS}, draws,
                                  labels=LABELS, scope=SCOPE)
    right_labels = ("I1", "D1", "O1", "original_baseline")
    right = norm.comparison_report({label: exports[label] for label in right_labels}, draws,
                                   labels=right_labels, scope=SCOPE)
    left["contrasts"].update(right["contrasts"])
    left["direct_contrasts"] = DIRECT_CONTRASTS
    return left


def composition_contrasts(pairs):
    return {**norm.composition_contrasts(pairs, labels=LABELS),
            **norm.composition_contrasts(pairs, labels=("I1", "D1", "O1", "original_baseline"))}


def analyze(controls, preflight, out):
    if [cfg.local_route for cfg in controls] != ["I1", "I0"]:
        raise ValueError("projection analysis requires I1/I0")
    for cfg in controls:
        legacy.verify_completed(cfg, preflight)
    return norm.analyze(controls[1], preflight, out, labels=LABELS, scope=SCOPE,
                        additional_runs={"I1": legacy.run_directory(controls[0])},
                        comparison_fn=comparison_report, composition_fn=composition_contrasts)


def execute(controls, preflight, out):
    return legacy.execute(controls, preflight, out, analyze_fn=analyze,
                          queue_schema="attention-projection-queue-v1")


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
    controls = attention_projection_configs(reference, args.tag_prefix)
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
