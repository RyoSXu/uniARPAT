"""Confirm warmed local CUDA backward timings after the acceptance run.

Same four structure-only profiles, untrained weights and FP32; three warmups,
five measurements. No optimizer or updates. Original raw timings are retained.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.periodic_message_gpu_check import (
    GPUBudget, ROUTES, benchmark_samples, make_batch, measured_forward, module_digest, verify_cpu_evidence,
)
from tools.eval.periodic_message_high_order_compare import create_models
from tools.eval.periodic_message_module_check import aligned_local_models, sha256, write_json
from utils.periodic_geometry import build_periodic_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--acceptance-dir", type=Path, default=ROOT / "results/periodic_message_gpu_q1_r6_20261003")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    prior = json.loads((ROOT / "results/periodic_message_precheck_q1_r6_20261003/manifest.json").read_text())
    accepted = json.loads((args.acceptance_dir / "manifest.json").read_text())
    assert accepted["status"] == "complete" and accepted["verdict"] == "pass"
    for name, digest in accepted["source_sha256"].items():
        assert sha256(ROOT / name) == digest
    verify_cpu_evidence(ROOT / "results/periodic_message_modules_q1_r6_20261003_v2")
    for name, digest in accepted["input_sha256"].items():
        assert sha256(ROOT / "data/train4ARPAT" / name) == digest
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(8 * 2**30 / total)
    references, entry = create_models(prior, torch.float32)
    cpu_models = aligned_local_models(references, torch.float32)
    models = {route: copy.deepcopy(model).cuda() for route, model in cpu_models.items()}
    before = {route: module_digest(model) for route, model in models.items()}
    for route in ROUTES:
        assert before[route] == accepted["states"][str(torch.float32)]["initial"][f"cuda/{route}"]
    args.out_dir.mkdir(parents=True)
    rows = []
    with GPUBudget(90., 2048., 8.) as budget:
        for label, selected in benchmark_samples(prior["selected_samples"]).items():
            pos, mask, ids = make_batch(ROOT / "data/train4ARPAT", selected)
            records = build_periodic_records(pos, mask, 6.).to("cuda")
            with torch.no_grad():
                h = entry(ids, mask).cuda()
            for route in ROUTES:
                for _ in range(3):
                    budget.check()
                    measured_forward(models[route], h, records, mask.cuda(), True)
                times = []
                finite = True
                for _ in range(5):
                    budget.check()
                    duration, good, active = measured_forward(models[route], h, records, mask.cuda(), True)
                    times.append(duration)
                    finite = finite and good and active
                rows.append({"batch_label": label, "route": route, "pass": finite,
                             "seconds": times, "median_seconds": statistics.median(times),
                             "min_seconds": min(times), "max_seconds": max(times)})
        budget.check()
        after = {route: module_digest(model) for route, model in models.items()}
        assert before == after
        report = {"status": "complete", "verdict": "pass" if all(row["pass"] for row in rows) else "fail",
                  "warmup": 3, "repeats": 5, "device": "cuda", "dtype": "torch.float32", "rows": rows,
                  "elapsed_seconds": __import__("time").perf_counter()-budget.start,
                  "peak_rss_MiB": budget.peak_mib, "gpu_peak_allocated_MiB": budget.peak_allocated/2**20,
                  "gpu_peak_reserved_MiB": budget.peak_reserved/2**20, "parameters_unchanged": before == after,
                  "optimizer_created": False, "parameter_updates": 0, "labels_loaded": False,
                  "environment": accepted["environment"], "initial_state_sha256": before,
                  "acceptance_manifest_sha256": sha256(args.acceptance_dir / "manifest.json"),
                  "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))},
                  "original_timings_retained": True, "limits": "short warmed local backward confirmation only"}
    write_json(args.out_dir / "summary.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
