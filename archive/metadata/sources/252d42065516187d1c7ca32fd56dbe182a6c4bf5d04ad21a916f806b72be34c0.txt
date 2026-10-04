"""Reconcile saved GPU acceptance and timing evidence without neural execution."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.periodic_message_module_check import sha256, write_json

CATEGORIES = {
    "gpu_geometry": 44, "cpu_gpu_features": 88, "gpu_rebuilt_features": 88,
    "gpu_orthogonal": 440, "gpu_padding": 80, "gpu_representation": 48,
    "cpu_gpu_gradients": 12, "gpu_translation": 64, "gpu_zp_pipeline": 2,
    "batch_preparation": 4, "benchmark": 96,
}
DTYPES = ("torch.float32", "torch.float64")
ROUTES = ("A", "B+")


def verify(directory, confirmation_dir, cpu_dir, data_dir):
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    cpu = json.loads((cpu_dir / "manifest.json").read_text())
    assert manifest["status"] == summary["status"] == cpu["status"] == "complete"
    assert manifest["verdict"] == summary["verdict"] == cpu["verdict"] == "pass"
    assert manifest["cpu_manifest_sha256"] == sha256(cpu_dir / "manifest.json")
    assert manifest["selected_samples"] == cpu["selected_samples"] and len(manifest["selected_samples"]) == 16
    assert manifest["core_source_sha256"] == cpu["source_sha256"]
    assert manifest["radius_A"] == 6. and manifest["splits"] == ["train", "valid"]
    for flag in ("labels_loaded", "checkpoint_loaded", "optimizer_created", "production_model_executed"):
        assert manifest[flag] is False
    assert manifest["parameter_updates"] == 0
    assert manifest["environment"]["cpu_threads"] == 1
    assert not manifest["environment"]["amp"] and not manifest["environment"]["tf32"]
    for mapping in (manifest["source_sha256"], manifest["core_source_sha256"],
                    cpu["frozen_source_sha256"], cpu["frozen_artifact_sha256"]):
        for relative, digest in mapping.items():
            assert sha256(ROOT / relative) == digest, relative
    for relative, digest in manifest["input_sha256"].items():
        assert sha256(data_dir / relative) == digest, relative
    for folder, saved in ((directory, manifest), (cpu_dir, cpu)):
        for name, digest in saved["artifacts_sha256"].items():
            assert sha256(folder / name) == digest, name
    for dtype in DTYPES:
        state = manifest["states"][dtype]
        assert state["initial"] == state["final"]
        for route in ROUTES:
            assert state["initial"][f"cpu/{route}"] == state["initial"][f"cuda/{route}"]

    rows = json.loads((directory / "cases.json").read_text())
    assert rows == [json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()]
    with (directory / "cases.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(rows) == len(table) == summary["cases"] == 966
    assert [row["case_id"] for row in rows] == list(range(966))
    csv_cells = 0
    for row, saved in zip(rows, table):
        for key, value in saved.items():
            original = row.get(key)
            expected = "" if original is None else json.dumps(original) if isinstance(original, (dict, list)) else str(original)
            assert value == expected, (row["case_id"], key)
            csv_cells += 1
        assert row["status"] == "pass"
        # Validate each metric independently, even if a recorder's combined
        # boolean could be overwritten when feature/geometry dictionaries merge.
        if "max_normalized_error" in row:
            assert row["max_normalized_error"] <= 1.
            assert row["counts_equal"]
            # All observed angular errors also pass this stricter absolute check.
            assert row["angular_max_abs_error"] <= row["feature_atol"]
        if "geometry_status" in row:
            assert row["geometry_status"] == "pass"
            assert row["boundary_missing"] == row["boundary_extra"] == 0
        if "max_distance_error_A" in row:
            tolerance = 1e-4 if row["dtype"] == DTYPES[0] else 1e-6
            assert row["max_distance_error_A"] <= tolerance and row["max_vector_error_A"] <= tolerance
        if row["category"] == "cpu_gpu_gradients":
            limit = manifest["gradient_tolerances"][row["dtype"]]
            assert row["gradient_tensors"] == len(row["tensors"])
            assert row["global_relative_l2"] <= limit["relative_l2"]
            assert row["max_relative_l2_above_floor"] <= limit["relative_l2"]
            for tensor in row["tensors"].values():
                assert tensor["pass"] and tensor["max_normalized_error"] <= 1.
                if tensor["above_absolute_floor"]:
                    assert tensor["active"] and tensor["relative_l2"] <= limit["relative_l2"]
        if row["category"] == "benchmark":
            assert row["finite_gradients"] and row["active_gradients"] and row["seconds"] > 0.
        if row["category"] == "gpu_zp_pipeline":
            assert all(row[key] for key in ("registered", "finite_gradients", "all_parameter_tensors_active", "parameters_unchanged"))
    categories = Counter(row["category"] for row in rows)
    assert dict(categories) == summary["categories"] == CATEGORIES
    assert summary["status_counts"] == {"pass": 966} and not summary["nonpassing_case_ids"] and summary["error"] is None

    def selected(category, sample=None, dtype=None, route=None):
        return [row for row in rows if row["category"] == category
                and (sample is None or (row.get("split"), row.get("index")) == (sample["split"], sample["index"]))
                and (dtype is None or row.get("dtype") == dtype)
                and (route is None or row.get("route") == route)]

    for sample in manifest["selected_samples"]:
        for dtype in DTYPES:
            geometry = selected("gpu_geometry", sample, dtype)
            assert len(geometry) == 1 and geometry[0]["matched_records"] == sample["n_edges"]
            for route in ROUTES:
                for category in ("cpu_gpu_features", "gpu_rebuilt_features", "gpu_translation", "gpu_padding"):
                    assert len(selected(category, sample, dtype, route)) == 1
                assert len(selected("gpu_orthogonal", sample, dtype, route)) == 5
                if sample["reason"] == "pairs_quantile_0.5":
                    assert len(selected("gpu_representation", sample, dtype, route)) == 4
                    assert len(selected("cpu_gpu_gradients", sample, dtype, route)) == 1

    recomputed = {}
    assert manifest["benchmark_repeats"] == 3 and manifest["benchmark_warmup"] == 1
    for label, samples in manifest["benchmark_batches"].items():
        preparation = [row for row in selected("batch_preparation") if row["batch_label"] == label]
        assert len(preparation) == 1
        preparation = preparation[0]
        assert preparation["samples"] == samples and preparation["batch_size"] == len(samples)
        for field, sample_key in (("n_atoms", "n_atoms"), ("n_edges", "n_edges"), ("angle_pairs", "potential_angle_pairs")):
            assert preparation[field] == sum(sample[sample_key] for sample in samples)
        group = recomputed.setdefault(label, {})
        for route in ROUTES:
            arm = group.setdefault(route, {})
            for device in ("cpu", "cuda"):
                for backward in (False, True):
                    timings = [row for row in selected("benchmark", route=route)
                               if row["batch_label"] == label and row["device"] == device and row["backward"] == backward]
                    assert sorted(row["repetition"] for row in timings) == [0, 1, 2]
                    assert all(row["samples"] == samples and row["dtype"] == DTYPES[0] for row in timings)
                    values = [row["seconds"] for row in timings]
                    key = f"{device}/{'forward_backward' if backward else 'forward'}"
                    arm[key] = {"seconds": values, "median_seconds": statistics.median(values)}
                    if device == "cuda" and backward:
                        for metric in ("allocated", "reserved"):
                            arm[f"gpu_peak_{metric}_MiB"] = max(row[f"peak_{metric}_MiB"] for row in timings)
            arm["cpu_over_gpu_forward_backward"] = arm["cpu/forward_backward"]["median_seconds"] / arm["cuda/forward_backward"]["median_seconds"]
    assert recomputed == summary["comparison"]
    assert summary["elapsed_seconds"] <= manifest["budget_seconds"]
    assert summary["peak_rss_MiB"] <= manifest["cpu_memory_MiB"]
    assert summary["gpu_peak_allocated_MiB"] <= summary["gpu_peak_reserved_MiB"] <= manifest["gpu_memory_GiB"] * 1024

    confirmation = json.loads((confirmation_dir / "summary.json").read_text())
    assert confirmation["status"] == "complete" and confirmation["verdict"] == "pass"
    assert confirmation["acceptance_manifest_sha256"] == sha256(directory / "manifest.json")
    assert confirmation["warmup"] == 3 and confirmation["repeats"] == 5
    assert confirmation["dtype"] == DTYPES[0] and confirmation["device"] == "cuda"
    assert confirmation["environment"] == manifest["environment"]
    assert confirmation["parameters_unchanged"] and confirmation["original_timings_retained"]
    assert not confirmation["optimizer_created"] and not confirmation["labels_loaded"] and confirmation["parameter_updates"] == 0
    for name, digest in confirmation["source_sha256"].items():
        assert sha256(ROOT / name) == digest
    for route in ROUTES:
        assert confirmation["initial_state_sha256"][route] == manifest["states"][DTYPES[0]]["initial"][f"cuda/{route}"]
    assert {(row["batch_label"], row["route"]) for row in confirmation["rows"]} == {
        (label, route) for label in manifest["benchmark_batches"] for route in ROUTES}
    assert len(confirmation["rows"]) == 8
    for row in confirmation["rows"]:
        assert row["pass"] and len(row["seconds"]) == 5 and all(value > 0. for value in row["seconds"])
        assert row["median_seconds"] == statistics.median(row["seconds"])
        assert row["min_seconds"] == min(row["seconds"]) and row["max_seconds"] == max(row["seconds"])
    assert confirmation["elapsed_seconds"] <= 90 and confirmation["peak_rss_MiB"] <= 2048
    assert confirmation["gpu_peak_allocated_MiB"] <= confirmation["gpu_peak_reserved_MiB"] <= 8192
    return {"status": "pass", "cases": len(rows), "categories": dict(categories),
            "csv_cells_checked": csv_cells, "confirmation_profiles": 8,
            "main_manifest_sha256": sha256(directory / "manifest.json"),
            "confirmation_summary_sha256": sha256(confirmation_dir / "summary.json"),
            "verifier_source_sha256": sha256(Path(__file__)),
            "scope": "saved evidence consistency; no neural rerun or full-model/accuracy claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--cpu-dir", type=Path, default=ROOT / "results/periodic_message_modules_q1_r6_20261003_v2")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/train4ARPAT")
    args = parser.parse_args(argv)
    target = args.out_dir / "verification.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    result = verify(args.out_dir, args.confirmation_dir, args.cpu_dir, args.data_dir)
    write_json(target, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
