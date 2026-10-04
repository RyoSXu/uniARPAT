"""CPU acceptance of extracted A/B+ modules against frozen prototype evidence.

Uses only the existing 16 train/valid structures and shared untrained weights.
No labels, checkpoint, optimizer, parameter updates or GPU execution. This is
a migration/interface check, not a new training or performance comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np  # Initialize MKL before Torch in the local CPU environment.
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.periodic_messages import PeriodicLocalMessage
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.eval.periodic_message_high_order_compare import create_models, load_sample
from tools.eval.periodic_message_precheck import BudgetMonitor, compare_features, feature_error, state_digest
from tools.eval.periodic_message_prototypes import geometry_from_pos
from utils.periodic_geometry import PeriodicNeighborRecords, build_periodic_records

SOURCE_FILES = (
    "model/__init__.py", "model/periodic_messages.py", "utils/periodic_geometry.py",
    "tests/test_periodic_message_modules.py", "tools/eval/periodic_message_module_check.py",
)
TEST_FILES = (
    "test_periodic_neighbor_radius_audit.py", "test_periodic_geometry_acceptance.py",
    "test_periodic_message_prototypes.py", "test_periodic_message_precheck.py",
    "test_periodic_message_high_order.py", "test_periodic_message_modules.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def aligned_local_models(references, dtype):
    """Migration oracle initialization only, not a new training seed protocol."""
    models = {route: PeriodicLocalMessage(route).to(dtype).eval() for route in ("A", "B+")}
    for route, model in models.items():
        model.load_state_dict(references[route].state_dict(), strict=True)
    return models


def frozen_hashes(prior_directory):
    original = json.loads((prior_directory / "manifest.json").read_text())
    enhanced_dir = ROOT / "results/periodic_message_high_order_q1_r6_20261003_v2"
    enhanced = json.loads((enhanced_dir / "manifest.json").read_text())
    sources = original["source_sha256"] | enhanced["source_sha256"] | enhanced["old_source_sha256"]
    for relative, digest in sources.items():
        if sha256(ROOT / relative) != digest:
            raise AssertionError(f"frozen source changed: {relative}")
    artifacts = {}
    for directory, manifest in ((prior_directory, original), (enhanced_dir, enhanced)):
        artifacts[str(directory.relative_to(ROOT) / "manifest.json")] = sha256(directory / "manifest.json")
        for name, digest in manifest["artifacts_sha256"].items():
            path = directory / name
            if sha256(path) != digest:
                raise AssertionError(f"frozen artifact changed: {path}")
            artifacts[str(path.relative_to(ROOT))] = digest
    return original, sources, artifacts


def run_regressions(directory):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.discover(str(ROOT / "tests"), pattern=name) for name in TEST_FILES)
    started = time.perf_counter()
    with (directory / "regressions.txt").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {"tests": result.testsRun, "passed": result.wasSuccessful(),
            "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "seconds": time.perf_counter() - started,
            "files": list(TEST_FILES)}


def acceptance(args):
    prior, old_sources, old_artifacts = frozen_hashes(args.precheck_dir)
    for relative, digest in prior["input_sha256"].items():
        if sha256(args.data_dir / relative) != digest:
            raise AssertionError(f"input differs from frozen precheck: {relative}")
    args.out_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    rows = []
    states = {}
    try:
        with BudgetMonitor(args.budget_seconds, args.memory_mib) as budget:
            for dtype in (torch.float32, torch.float64):
                budget.check()
                references, entry = create_models(prior, dtype)
                models = aligned_local_models(references, dtype)
                initial = {route: state_digest(model) for route, model in models.items()}
                for sample in prior["selected_samples"]:
                    budget.check()
                    pos, mask, ids = load_sample(args.data_dir, sample, dtype)
                    old_geometry = geometry_from_pos(pos, mask)
                    records = build_periodic_records(pos, mask, 6.)
                    equal = all(torch.equal(getattr(records, name), old_geometry.edges[name])
                                for name in ("batch", "dst", "src", "shifts", "distances"))
                    equal = equal and torch.equal(records.vectors, old_geometry.vectors)
                    equal = equal and records.distances.numel() == sample["n_edges"]
                    context = {"split": sample["split"], "index": sample["index"], "id": sample["id"], "dtype": str(dtype)}
                    rows.append(context | {"category": "physical_records", "pass": equal,
                                           "records": records.distances.numel(),
                                           "max_vector_error_A": float((records.vectors - old_geometry.vectors).abs().max())})
                    with torch.no_grad():
                        h = entry(ids, mask)
                        for route, model in models.items():
                            budget.check()
                            original = references[route](h, mask, old_geometry, True)
                            actual = model(h, records, mask, return_debug=True)
                            metrics = compare_features(original, actual, dtype, "direct")
                            count_equal = all(torch.equal(original[1][name], actual[1][name])
                                              for name in ("degree", "effective_count", "pair_count"))
                            angular_error = 0.
                            if route == "B+":
                                for old, new in zip(original[1]["angular_invariants"], actual[1]["angular_invariants"]):
                                    measurement = feature_error(old, new, (2e-5, 2e-5) if dtype == torch.float32 else (2e-10, 2e-10))
                                    angular_error = max(angular_error, measurement["max_abs_error"])
                                    metrics["pass"] = metrics["pass"] and measurement["pass"]
                            metrics["pass"] = metrics["pass"] and count_equal
                            rows.append(context | {"category": "prototype_equivalence", "route": route,
                                                   "counts_equal": count_equal, "angular_max_abs_error": angular_error} | metrics)
                            if sample["reason"] == "pairs_quantile_0.1":
                                for name, q, _ in e3_operations():
                                    if name == "translation":
                                        continue
                                    from dataclasses import replace
                                    changed = replace(records, vectors=records.vectors @ torch.tensor(q, dtype=dtype))
                                    transformed = model(h, changed, mask, return_debug=True)
                                    metric = compare_features(actual, transformed, dtype, "direct", orthogonal=q)
                                    rows.append(context | {"category": "local_orthogonal", "route": route, "operation": name} | metric)
                final = {route: state_digest(model) for route, model in models.items()}
                if initial != final:
                    raise AssertionError("local weights changed during acceptance")
                states[str(dtype)] = {"initial": initial, "final": final}
            budget.check()
            regressions = run_regressions(args.out_dir)
            budget.check()
            _, final_sources, final_artifacts = frozen_hashes(args.precheck_dir)
            if final_sources != old_sources or final_artifacts != old_artifacts:
                raise AssertionError("frozen evidence changed during acceptance")
            summary = {"status": "complete", "verdict": "pass" if all(row["pass"] for row in rows) and regressions["passed"] else "fail",
                       "cases": len(rows), "nonpassing_case_ids": [i for i, row in enumerate(rows) if not row["pass"]],
                       "prototype_max_abs_error": max(row["max_abs_error"] for row in rows if row["category"] == "prototype_equivalence"),
                       "geometry_max_vector_error_A": max(row["max_vector_error_A"] for row in rows if row["category"] == "physical_records"),
                       "local_orthogonal_max_abs_error": max(row["max_abs_error"] for row in rows if row["category"] == "local_orthogonal"),
                       "regressions": regressions, "elapsed_seconds": time.perf_counter() - budget.start,
                       "peak_rss_MiB": budget.peak_mib}
    finally:
        torch.set_default_dtype(previous_dtype)
    write_json(args.out_dir / "cases.json", rows)
    write_json(args.out_dir / "summary.json", summary)
    manifest = {"status": "complete", "verdict": summary["verdict"], "device": "cpu", "radius_A": 6.,
                "splits": ["train", "valid"], "selected_samples": prior["selected_samples"],
                "labels_loaded": False, "checkpoint_loaded": False, "optimizer_created": False,
                "production_model_executed": False, "parameter_updates": 0,
                "initialization_scope": "explicit copy of frozen untrained local weights; not a training protocol",
                "parameter_counts": {"A": 261568, "B+": 184772}, "states": states,
                "input_sha256": prior["input_sha256"], "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
                "frozen_source_sha256": old_sources, "frozen_artifact_sha256": old_artifacts,
                "artifacts_sha256": {name: sha256(args.out_dir / name) for name in ("cases.json", "summary.json", "regressions.txt")},
                "limits": ["CPU migration/interface evidence only", "no GPU/AMP or training benchmark", "no whole-model E(3) acceptance"]}
    write_json(args.out_dir / "manifest.json", manifest)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/train4ARPAT")
    parser.add_argument("--precheck-dir", type=Path, default=ROOT / "results/periodic_message_precheck_q1_r6_20261003")
    parser.add_argument("--budget-seconds", type=float, default=180.)
    parser.add_argument("--memory-mib", type=float, default=2048.)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    if any(not math.isfinite(value) or value <= 0 for value in (args.budget_seconds, args.memory_mib)):
        raise ValueError("positive finite acceptance budgets required")
    result = acceptance(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
