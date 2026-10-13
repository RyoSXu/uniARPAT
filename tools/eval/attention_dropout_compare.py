"""Train only D1, then isolate N1's global attention-probability dropout."""
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

from tools.eval import global_norm_compare as norm
from tools.eval.lg_valid_export import load_run
from tools.eval.local_global_compare import summarize, training_command
from utils.attention_dropout_experiment import attention_dropout_config
from utils.global_norm_experiment import global_norm_config
from utils.local_structure_experiment import DEFAULT_REFERENCE, load_control_reference
from utils.paths import file_sha256

NORM_ROOT = ROOT / "results/runs/global_norm_attribution_20261013"
LABELS = ("D1", "N1", "O1", "original_baseline")
SCOPE = ("seed 42, 35 epochs, crystal_q1_v1; D1 minus reference; D1/N1 differ only in global "
         "attention-probability dropout (0 vs 0.05), with identical untrained tensors; "
         "post-norm, learned QKVO, residual/FFN/decoder dropout and C1 locals remain fixed; "
         "O1/baseline are package references; intervals conditional on selected checkpoints, "
         "not training-seed uncertainty")


def reference_identity(reference_dir):
    saved, cfg, identities = norm.reference_identity(reference_dir)
    preflight = json.loads((NORM_ROOT / "preflight_final.json").read_text())
    queue = json.loads((NORM_ROOT / "queue.json").read_text())
    if (preflight.get("schema") != "global-norm-preflight-v1" or preflight.get("passed") is not True
            or queue.get("status") != "complete" or queue.get("arms") != {"N1": "complete"}
            or (NORM_ROOT / "run.exit").read_text().strip() != "0"):
        raise ValueError("D1 requires the completed, historically verified N1 reference")
    n1_cfg = global_norm_config(cfg, "norm20261013")
    norm.verify_completed(n1_cfg, preflight)
    run = norm.run_directory(n1_cfg)
    recorded = json.loads((NORM_ROOT / "analysis/report.json").read_text())["run_artifacts"]["N1"]
    files = {name: file_sha256(run / name) for name in
             ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")}
    record, _, _ = load_run(run)
    if (str(run) != recorded["run_dir"] or files["config_used.yaml"] != recorded["configuration_sha256"]
            or files["checkpoint_best.pth"] != recorded["best_checkpoint_sha256"]
            or record["cli"]["data_contract"] != saved["cli"]["data_contract"]
            or preflight["data_contract"] != saved["cli"]["data_contract"]):
        raise ValueError("N1 no longer matches its completed result or the shared data")
    identities["N1"] = {"run_dir": str(run), "files": files}
    return saved, cfg, identities


def evidence_fingerprints():
    # Historical evidence is bound as it stands, not re-signed after new code.
    return {**norm.evidence_fingerprints(), **{
        str(path.relative_to(ROOT)): file_sha256(path) for path in
        (NORM_ROOT / "preflight_final.json", NORM_ROOT / "analysis/report.json",
         NORM_ROOT / "queue.json", NORM_ROOT / "run.exit")}}


def verify_preflight(path, reference_dir, cfg):
    from tools.checks.check_attention_dropout import code_fingerprints

    report = json.loads(path.read_text())
    if report.get("schema") != "attention-dropout-preflight-v1" or report.get("passed") is not True:
        raise ValueError("successful D1 real-data preflight required")
    _, _, references = reference_identity(reference_dir)
    if (report["controls"]["D1"]["configuration"] != asdict(cfg)
            or report["code_sha256"] != code_fingerprints() or report["references"] != references
            or report["reference_evidence_sha256"] != evidence_fingerprints()):
        raise ValueError("D1 source, reference or queue configuration changed after preflight")
    return report


def comparison_report(exports, draws=2000):
    return norm.comparison_report(exports, draws, labels=LABELS, scope=SCOPE)


def composition_contrasts(pairs):
    return norm.composition_contrasts(pairs, labels=LABELS)


def analyze(cfg, preflight, out):
    return norm.analyze(cfg, preflight, out, labels=LABELS, scope=SCOPE)


def execute(cfg, preflight, out):
    return norm.execute(cfg, preflight, out, analyze_fn=analyze,
                        queue_schema="attention-dropout-queue-v1")


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
    cfg = attention_dropout_config(reference, args.tag_prefix)
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
