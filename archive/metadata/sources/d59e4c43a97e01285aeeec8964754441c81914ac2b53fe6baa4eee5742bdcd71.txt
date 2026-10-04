"""Independently reconcile frozen A/B/B+ evidence; does not rerun model forwards."""

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

from tools.eval.element_identity_preflight import sha256_file
from tools.eval.periodic_geometry_acceptance import write_json


def verify(directory, rss_directory, data_dir, prior_directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    prior = json.loads((prior_directory/"manifest.json").read_text())
    assert manifest["status"] == summary["status"] == "complete"
    assert manifest["verdict"] == summary["verdict"] == "pass"
    assert manifest["prior_manifest_sha256"] == sha256_file(prior_directory/"manifest.json")
    assert manifest["selected_samples"] == prior["selected_samples"]
    assert manifest["splits"] == ["train", "valid"] and manifest["radius_A"] == 6.
    for flag in ("labels_loaded", "checkpoint_loaded", "optimizer_created", "production_model_executed"):
        assert manifest[flag] is False
    assert manifest["parameter_updates"] == 0 and manifest["device"] == "cpu"
    assert manifest["initial_state_sha256"] == manifest["final_state_sha256"]
    assert manifest["parameter_counts"] == {"A": 261568, "B": 180676, "B+": 184772}
    for dtype in ("torch.float32", "torch.float64"):
        for route, original in (("A", "invariant"), ("B", "equivariant"), ("zp", "zp")):
            assert manifest["initial_state_sha256"][f"{dtype}/{route}"] == prior["initial_state_sha256"][f"{dtype}/{original}"]
    for mapping in (manifest["source_sha256"], manifest["old_source_sha256"]):
        for relative, digest in mapping.items():
            assert sha256_file(ROOT/relative) == digest, relative
    for relative, digest in manifest["input_sha256"].items():
        assert sha256_file(data_dir/relative) == digest, relative
    for name, digest in manifest["artifacts_sha256"].items():
        assert sha256_file(directory/name) == digest, name
    for name, digest in prior["artifacts_sha256"].items():
        assert sha256_file(prior_directory/name) == digest, f"old artifact changed: {name}"
    rows = json.loads((directory/"cases.json").read_text())
    assert rows == [json.loads(line) for line in (directory/"cases.jsonl").read_text().splitlines()]
    with (directory/"cases.csv").open() as stream:
        table = list(csv.DictReader(stream))
    # The mixed-padding synthetic case has batch=2; the reused physical-rebuild
    # helper only rebuilds batch=1, so six hypothetical rebuilds are excluded.
    assert len(rows) == len(table) == summary["cases"] == 994
    assert [r["case_id"] for r in rows] == list(range(994))
    csv_cells = 0
    for row, saved in zip(rows, table):
        for key, value in saved.items():
            original = row.get(key)
            expected = "" if original is None else json.dumps(original) if isinstance(original, (dict, list)) else str(original)
            assert value == expected, (row["case_id"], key)
            csv_cells += 1
    categories = Counter(r["category"] for r in rows)
    assert categories == {"fixed_radial_angle": 60, "equal_low_order_moments": 60, "high_order_ablation": 20,
                          "synthetic_boundary": 36, "e3_direct": 280, "e3_rebuilt": 62, "representation": 152,
                          "supercell": 4, "real_shared_e3": 96, "high_order_invariant_e3": 32,
                          "real_shear_response": 48, "resources": 144}
    assert not summary["nonpassing_case_ids"] and all(r["status"] == "pass" for r in rows)
    groups = {}
    for row in rows:
        key = f"{row['category']}/{row['route']}/{row['dtype']}"
        group = groups.setdefault(key, {"cases": 0, "pass": 0, "fail": 0, "boundary_limited": 0,
                                        "max_abs_error": 0., "max_normalized_error": 0.})
        group["cases"] += 1
        group["pass"] += 1
        for name in ("max_abs_error", "max_normalized_error"):
            group[name] = max(group[name], row.get(name, 0.))
        if "geometry_status" in row:
            assert row["geometry_status"] == "pass" and row["boundary_missing"] == row["boundary_extra"] == 0
        if "max_normalized_error" in row:
            assert row["max_normalized_error"] <= 1
        if row["category"] == "fixed_radial_angle":
            assert row["identity_control_error"] == 0 and row["radial_fields_bitwise_identical"]
    assert groups == summary["groups"]
    for sample in manifest["selected_samples"]:
        def selected(category, route="B+"):
            return [r for r in rows if r["category"] == category and r["route"] == route
                    and r.get("split") == sample["split"] and r.get("index") == sample["index"]]
        for dtype in ("torch.float32", "torch.float64"):
            assert len([r for r in selected("e3_direct") if r["dtype"] == dtype]) == 5
            assert len([r for r in selected("e3_rebuilt") if r["dtype"] == dtype]) == 1
            assert len([r for r in selected("representation") if r["dtype"] == dtype]) == 4
        for route in ("A", "B", "B+"):
            records = selected("resources", route)
            assert sorted(r["repetition"] for r in records) == [0, 1, 2]
            assert all(r["finite_gradients"] and r["parameters_unchanged"] and r["both_blocks_active"]
                       and not r["inactive_parameter_tensors"] for r in records)
    for route in ("A", "B", "B+"):
        controls = [r for r in rows if r["category"] == "equal_low_order_moments" and r["route"] == route
                    and r["dtype"] == "torch.float64" and r["cutoff_mode"] == "one_ablation"]
        assert sorted(r["seed"] for r in controls) == sorted(manifest["control_seeds"])
        recomputed = {"resolved_seeds": sum(r["response_resolved"] for r in controls), "seeds": 5,
                      "max_abs_change_median": statistics.median(r["max_scalar_output_change"] for r in controls)}
        assert recomputed == summary["comparison"][route]["controlled_counterexample"]
        assert recomputed["resolved_seeds"] == (0 if route == "B" else 5)
        for metric in ("forward_seconds", "forward_plus_backward_seconds"):
            medians = []
            for sample in manifest["selected_samples"]:
                values = [r[metric] for r in rows if r["category"] == "resources" and r["route"] == route
                          and r["split"] == sample["split"] and r["index"] == sample["index"]]
                medians.append(statistics.median(values))
            expected = {"median": statistics.median(medians), "max": max(medians), "sum": sum(medians)}
            assert expected == summary["comparison"][route][metric]
        values = [r["relative_l2_change"] for r in rows if r["category"] == "real_shear_response" and r["route"] == route]
        assert {"median": statistics.median(values), "min": min(values), "max": max(values)} == summary["comparison"][route]["real_shear_relative_l2"]
    runtime = summary["elapsed_seconds"]
    isolated = {}
    for route, name in (("A", "A"), ("B", "B"), ("B+", "Bplus")):
        folder = rss_directory/name
        saved = json.loads((folder/"manifest.json").read_text())
        result = json.loads((folder/"summary.json").read_text())
        assert saved["status"] == "complete" and result["verdict"] == "pass" and result["route"] == route
        assert saved["parameter_updates"] == 0 and saved["optimizer_created"] is False and saved["labels_loaded"] is False
        assert saved["prior_manifest_sha256"] == manifest["prior_manifest_sha256"]
        assert result["parameters_unchanged"] and result["finite_gradients"] and not result["inactive_parameter_tensors"]
        assert result["peak_rss_MiB"] <= 2048 and result["elapsed_seconds"] <= 45
        for relative, digest in saved["source_sha256"].items():
            assert sha256_file(ROOT/relative) == digest
        assert sha256_file(folder/"summary.json") == saved["artifacts_sha256"]["summary.json"]
        isolated[route] = {"peak_rss_MiB": result["peak_rss_MiB"], "manifest_sha256": sha256_file(folder/"manifest.json")}
        runtime += result["elapsed_seconds"]
    assert summary["peak_rss_MiB"] <= 2048 and summary["elapsed_seconds"] <= 750 and runtime < 900
    return {"status": "pass", "cases": len(rows), "categories": dict(categories), "csv_cells_checked": csv_cells,
            "main_and_rss_elapsed_seconds": runtime, "isolated": isolated,
            "main_manifest_sha256": sha256_file(directory/"manifest.json"), "verifier_source_sha256": sha256_file(Path(__file__)),
            "scope": "saved evidence consistency; not a neural rerun, arbitrary-environment proof or DOS evaluation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rss-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/train4ARPAT")
    parser.add_argument("--precheck-dir", type=Path, default=ROOT/"results/periodic_message_precheck_q1_r6_20261003")
    args = parser.parse_args(argv)
    target = args.out_dir/"verification.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    result = verify(args.out_dir, args.rss_dir, args.data_dir, args.precheck_dir)
    write_json(target, result)
    print(json.dumps({key: result[key] for key in ("status", "cases", "csv_cells_checked", "main_and_rss_elapsed_seconds")}))


if __name__ == "__main__":
    main()
