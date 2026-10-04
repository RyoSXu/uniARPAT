"""Reconcile saved precheck evidence without rerunning models or training."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.element_identity_preflight import sha256_file
from tools.eval.periodic_geometry_acceptance import write_json


def verify(directory, controls, data_dir):
    manifest = json.loads((directory/"manifest.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    assert manifest["status"] == summary["status"] == "complete"
    assert manifest["verdict"] == summary["verdict"] == "pass"
    assert manifest["splits"] == ["train", "valid"] and manifest["radius_A"] == 6.
    for flag in ("labels_loaded", "checkpoint_loaded", "optimizer_created", "production_model_executed"):
        assert manifest[flag] is False
    assert manifest["parameter_updates"] == 0
    assert manifest["initial_state_sha256"] == manifest["final_state_sha256"]
    assert manifest["parameter_counts"] == {"invariant": 261568, "equivariant": 180676}
    for relative, digest in manifest["source_sha256"].items():
        assert sha256_file(ROOT/relative) == digest, relative
    for relative, digest in manifest["input_sha256"].items():
        assert sha256_file(data_dir/relative) == digest, relative
    for relative, digest in manifest["prior_artifact_sha256"].items():
        assert sha256_file(ROOT/relative) == digest, relative
    for name, digest in manifest["artifacts_sha256"].items():
        assert sha256_file(directory/name) == digest, name
    rows = json.loads((directory/"cases.json").read_text())
    lines = [json.loads(line) for line in (directory/"cases.jsonl").read_text().splitlines()]
    assert rows == lines
    with (directory/"cases.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert len(rows) == len(table) == summary["cases"] == 840
    assert [row["case_id"] for row in rows] == list(range(840))
    assert [int(row["case_id"]) for row in table] == list(range(840))
    assert all(row["status"] == "pass" for row in rows+table)
    assert summary["nonpassing_case_ids"] == []
    categories = Counter(row["category"] for row in rows)
    assert categories == {"e3_direct": 400, "e3_rebuilt": 84, "representation": 272, "supercell": 8,
                          "resources": 32, "synthetic_boundary": 24, "angle_response": 8,
                          "structural_response": 8, "geometry_vjp": 4}
    recomputed = {}
    for row in rows:
        assert row["dtype"] in ("torch.float32", "torch.float64")
        assert row["route"] in ("invariant", "equivariant")
        key = f"{row['category']}/{row['route']}/{row['dtype']}"
        group = recomputed.setdefault(key, {"cases": 0, "pass": 0, "fail": 0, "boundary_limited": 0,
                                            "max_abs_error": 0., "max_normalized_error": 0.})
        group["cases"] += 1
        group["pass"] += 1
        for metric in ("max_abs_error", "max_normalized_error"):
            group[metric] = max(group[metric], row.get(metric, 0.))
        if "max_normalized_error" in row:
            assert 0 <= row["max_normalized_error"] <= 1
        if "geometry_status" in row:
            assert row["geometry_status"] == "pass" and row["boundary_missing"] == row["boundary_extra"] == 0
        if row["category"] == "resources":
            assert row["finite_gradients"] and row["both_blocks_active"] and row["parameters_unchanged"]
            assert row["inactive_parameter_tensors"] == []
        if row["category"] == "geometry_vjp":
            assert min(row["distance_gradient_max_abs"], row["direction_gradient_max_abs"]) > 1e-12
    assert recomputed == summary["groups"]
    samples = json.loads((directory/"samples.json").read_text())
    assert samples == manifest["selected_samples"] and len(samples) == 16
    assert Counter(row["split"] for row in samples) == {"train": 8, "valid": 8}
    assert len({(row["split"], row["index"]) for row in samples}) == 16
    for sample in samples:
        subset = [row for row in rows if row.get("split") == sample["split"] and row.get("index") == sample["index"]]
        assert all(row["id"] == sample["id"] and row["n_edges"] == sample["n_edges"]
                   and row["potential_angle_pairs"] == sample["potential_angle_pairs"] for row in subset)
        for dtype in ("torch.float32", "torch.float64"):
            for route in ("invariant", "equivariant"):
                own = [row for row in subset if row["dtype"] == dtype and row["route"] == route]
                assert Counter(row["category"] for row in own)["e3_direct"] == 5
                assert Counter(row["category"] for row in own)["e3_rebuilt"] == 1
                assert Counter(row["category"] for row in own)["representation"] == 4
    control_rows, control_hashes, runtime = {}, {}, summary["elapsed_seconds"]
    protected_manifest = sha256_file(directory/"manifest.json")
    for name in ("angles.json", "invariant_resource.json", "equivariant_resource.json"):
        path = controls/name
        data = json.loads(path.read_text())
        assert data["precheck_manifest_sha256"] == protected_manifest
        assert data["optimizer_created"] is False and data["parameter_updates"] == 0
        for relative, digest in data["source_sha256"].items():
            assert sha256_file(ROOT/relative) == digest
        assert data["peak_rss_MiB"] < 2048
        runtime += data["elapsed_seconds"]
        control_hashes[name] = sha256_file(path)
        control_rows[name] = data["result"]
    angles = control_rows["angles.json"]
    assert len(angles) == 10 and all(row["status"] == "pass" for row in angles)
    for row in angles:
        if row["category"] == "fixed_radial_angle_response":
            assert row["radial_fields_bitwise_identical"] and row["degree_identical"]
            assert row["identity_control_error"] == 0 and row["parameters_unchanged"]
            if row["distance_A"] == 5.9:
                assert not row["response_resolved"]
    blind = [row for row in angles if row["category"] == "equal_low_order_moments"]
    assert len(blind) == 2
    assert next(row for row in blind if row["route"] == "invariant")["max_scalar_output_change"] > 1e-8
    assert next(row for row in blind if row["route"] == "equivariant")["max_scalar_output_change"] < 2e-10
    assert summary["peak_rss_MiB"] < 2048 and runtime < 840
    return {"status": "pass", "cases": 840, "control_cases": 10, "isolated_resource_cases": 2,
            "main_manifest_sha256": protected_manifest, "main_case_categories": dict(categories),
            "control_artifact_sha256": control_hashes, "main_and_control_seconds": runtime,
            "checks": ["input/source/prior/output hashes", "JSON/JSONL/CSV identity and counts",
                       "independent summary/group reconciliation", "all 16 samples have expected transform coverage",
                       "parameter/buffer hashes unchanged", "all real backward parameter tensors active and finite",
                       "angle identity controls, near-cutoff weakness, low-order blind spot", "resource limits"],
            "scope": "artifact and protocol consistency only; does not rerun neural forwards or prove DOS accuracy",
            "verifier_source_sha256": sha256_file(Path(__file__)),
            "additional_test_source_sha256": sha256_file(ROOT/"tests/test_periodic_message_precheck.py")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/train4ARPAT")
    args = parser.parse_args(argv)
    path = args.precheck_dir/"verification.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    result = verify(args.precheck_dir, args.controls_dir, args.data_dir)
    write_json(path, result)
    print(json.dumps({key: result[key] for key in ("status", "cases", "control_cases", "main_and_control_seconds")}))


if __name__ == "__main__":
    main()
