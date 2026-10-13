"""Disposable I1/I0 real train/valid checks; no formal training dispatched."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools.checks import check_global_norm as norm
from tools.checks.check_attention_dropout import code_fingerprints as dropout_fingerprints
from tools.eval.lg_structure_diagnosis import dump
from utils.local_structure_experiment import DEFAULT_REFERENCE
from utils.paths import file_sha256


def code_fingerprints():
    return {**dropout_fingerprints(), **{name: file_sha256(ROOT / name) for name in
        ("utils/attention_projection_experiment.py", "tools/checks/check_attention_projection.py",
         "tools/eval/attention_projection_compare.py")}}


def check(out, reference_dir, tag_prefix):
    if out.exists():
        raise FileExistsError(f"refusing existing projection preflight: {out}")
    started = time.monotonic()
    reports = []
    with tempfile.TemporaryDirectory(prefix="attention_projection_preflight_", dir="/tmp") as name:
        for route in ("I1", "I0"):
            path = Path(name) / f"{route}.json"
            norm.check(path, reference_dir, tag_prefix, route=route)
            reports.append(json.loads(path.read_text()))
            gc.collect()
            torch.cuda.empty_cache()
    shared = ("references", "reference_evidence_sha256", "code_sha256", "data_contract", "learning_rate_35")
    if any(reports[0][key] != reports[1][key] for key in shared):
        raise ValueError("source, historical references or data changed between projection probes")
    entries = {route: report["controls"][route] for route, report in zip(("I1", "I0"), reports)}
    left, right = (entries[route]["initialization"] for route in ("I1", "I0"))
    if (left["state_sha256"] != right["state_sha256"]
            or left["retained_state_sha256"] != right["retained_state_sha256"]
            or left["d1_initialization"] != right["d1_initialization"]
            or left["frozen_parameter_names"] or len(right["frozen_parameter_names"]) != 24
            or entries["I1"]["training_start_rng_sha256"] != entries["I0"]["training_start_rng_sha256"]
            or entries["I1"]["data_order"] != entries["I0"]["data_order"]):
        raise ValueError("I1/I0 do not share their complete initial state and training-start protocol")
    report = {"schema": "attention-projection-preflight-v1", "passed": True,
              "scope": "disposable I1/I0 train-batch updates; train/valid only; no formal checkpoints",
              **{key: reports[0][key] for key in shared}, "controls": entries,
              "all_initial_tensors_paired": True, "elapsed_s": time.monotonic() - started}
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, report)
    print(json.dumps({"passed": True, "routes": ["I1", "I0"],
                      "initial_tensors": left["n_tensors"], "parameters": left["parameter_count"],
                      "trainable_parameters": {route: entry["initialization"]["trainable_parameter_count"]
                                               for route, entry in entries.items()},
                      "elapsed_s": report["elapsed_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.reference, args.tag_prefix)


if __name__ == "__main__":
    main()
