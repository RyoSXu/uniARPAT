"""Frozen A/B/B+ CPU comparison: geometry, expressivity controls and resource cost.

Q1 train/valid structures only. No spectra, checkpoint, optimizer, training or
production encoder. Old files/results are checked by hash and never rewritten.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import importlib.metadata
import itertools
import json
from pathlib import Path
import resource
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.element_identity_preflight import sha256_file
from tools.eval.periodic_geometry_acceptance import e3_operations, pack_cell, write_json
from tools.eval.periodic_message_controls import recreate_probes
from tools.eval.periodic_message_high_order import HighOrderMessageProbe
from tools.eval.periodic_message_precheck import (
    BudgetMonitor, FEATURE_TOLERANCES, Recorder, SEED, aggregate, compare_features,
    feature_error, inspect_case, local_forward, measure_resources, state_digest,
    supercell_geometry, synthetic_cases, synthetic_star,
)
from tools.eval.periodic_message_prototypes import ProbeZP, PeriodicMessageProbe, geometry_from_pos, prepare_geometry
from utils.relative_features import build_cell_from_lattice

ROUTES = ("A", "B", "B+")
CONTROL_SEEDS = (SEED, 42, 7, 1234, 314159)
REPEATS = 3
FEATURE_RESPONSE_THRESHOLDS = {"torch.float32": 1e-8, "torch.float64": 1e-10}


def create_models(manifest, dtype, seed=SEED):
    """Float64 master; original weights replayed, only 4096 new B+ weights drawn."""
    if seed == SEED:
        base, entry = recreate_probes(manifest, torch.float64)
    else:
        torch.manual_seed(seed)
        entry = ProbeZP().eval()
        for p in entry.parameters():
            p.requires_grad_(False)
        torch.manual_seed(seed+1)
        base = {"invariant": PeriodicMessageProbe("invariant").eval()}
        torch.manual_seed(seed+2)
        base["equivariant"] = PeriodicMessageProbe("equivariant").eval()
        for name in ("local_input", "local_output"):
            getattr(base["equivariant"], name).load_state_dict(getattr(base["invariant"], name).state_dict())
    torch.manual_seed(seed+3)
    enhanced = HighOrderMessageProbe(base["equivariant"]).eval()
    models = {"A": base["invariant"], "B": base["equivariant"], "B+": enhanced}
    return {route: copy.deepcopy(model).to(dtype).eval() for route, model in models.items()}, copy.deepcopy(entry).to(dtype)


def fixed_radial_pair(first, changed, mask):
    """Change directions only; IDs, distances, RBF, cutoff and counts stay exact."""
    for key in ("batch", "dst", "src", "shifts"):
        if not torch.equal(first.edges[key], changed.edges[key]):
            raise AssertionError("angle control changed the graph")
    controlled = prepare_geometry(dict(changed.edges, distances=first.edges["distances"]),
                                  changed.unit*first.edges["distances"][:, None], mask)
    for name in ("radial", "cutoff", "degree", "effective_count", "pair_count"):
        if not torch.equal(getattr(first, name), getattr(controlled, name)):
            raise AssertionError(f"angle control changed {name}")
    return controlled


def low_moment_pair(dtype, cutoff_one=False):
    """The previous pure-star 12-neighbor counterexample, also with native cutoff."""
    mask = torch.zeros(1, 13, dtype=torch.bool)
    ids = torch.full((1, 13), 6, dtype=torch.long)
    geometries = []
    for ratio in (1.5, (1+np.sqrt(5))/2):
        points = np.array([[0, a, b*ratio] for a, b in itertools.product((-1, 1), repeat=2)])
        u = np.concatenate((points, points[:, [2, 0, 1]], points[:, [1, 2, 0]]))/np.sqrt(1+ratio**2)
        pos = pack_cell(np.eye(3)*30, np.r_[np.zeros((1, 3)), u*5.85]/30+.5, dtype)
        geometry = geometry_from_pos(pos, mask)
        if geometry.degree.tolist() != [[12]+[1]*12]:
            raise AssertionError("counterexample must remain a pure star")
        if geometries:
            geometry = fixed_radial_pair(geometries[0], geometry, mask)
        geometries.append(geometry)
    if cutoff_one:
        geometries = [replace(g, cutoff=torch.ones_like(g.cutoff), effective_count=g.degree.to(dtype)) for g in geometries]
    return geometries, mask, ids


def response_metrics(first, second):
    difference = second-first
    baseline_norm = float(first.norm())
    return {"max_scalar_output_change": float(difference.abs().max()) if first.numel() else 0.,
            "relative_l2_change": float(difference.norm())/max(baseline_norm, 1e-12),
            "baseline_l2_norm": baseline_norm}


def angle_controls(recorder, manifest):
    for seed in CONTROL_SEEDS:
        for dtype in (torch.float32, torch.float64):
            models, entry = create_models(manifest, dtype, seed)
            for distance in (5., 5.9):
                pos, mask, ids = synthetic_star(False, distance, dtype)
                other_pos, _, _ = synthetic_star(True, distance, dtype)
                first = geometry_from_pos(pos, mask)
                second = fixed_radial_pair(first, geometry_from_pos(other_pos, mask), mask)
                atoms = entry(ids, mask)
                for route, model in models.items():
                    one = local_forward(model, atoms, mask, first)
                    two = local_forward(model, atoms, mask, second)
                    identity = local_forward(model, atoms, mask, first)
                    metrics = response_metrics(one[0][0, 0], two[0][0, 0])
                    threshold = FEATURE_RESPONSE_THRESHOLDS[str(dtype)]
                    recorder.add("fixed_radial_angle", {"seed": seed, "route": route, "dtype": str(dtype), "distance_A": distance},
                                 metrics | {"identity_control_error": float((one[0]-identity[0]).abs().max()),
                                            "radial_fields_bitwise_identical": True,
                                            "response_threshold": threshold,
                                            "response_resolved": metrics["max_scalar_output_change"] > threshold,
                                            "pass": bool(torch.equal(one[0], identity[0]))})
            for cutoff_one in (False, True):
                geometries, mask, ids = low_moment_pair(dtype, cutoff_one)
                atoms = entry(ids, mask)
                outputs = {route: [local_forward(model, atoms, mask, g) for g in geometries] for route, model in models.items()}
                for route in ROUTES:
                    metrics = response_metrics(outputs[route][0][0][0, 0], outputs[route][1][0][0, 0])
                    threshold = FEATURE_RESPONSE_THRESHOLDS[str(dtype)]
                    expected = not cutoff_one or (metrics["max_scalar_output_change"] > threshold if route != "B"
                                                  else metrics["max_scalar_output_change"] < (2e-6 if dtype == torch.float32 else 2e-10))
                    recorder.add("equal_low_order_moments", {"seed": seed, "route": route, "dtype": str(dtype),
                                                             "cutoff_mode": "one_ablation" if cutoff_one else "native", "distance_A": 5.85},
                                 metrics | {"response_threshold": threshold, "response_resolved": metrics["max_scalar_output_change"] > threshold,
                                            "radial_fields_bitwise_identical": True, "center_degree": 12,
                                            "expected_blindness": route == "B" and cutoff_one, "pass": expected})
                # Disable only the added inputs: the old B response must be recovered.
                models["B+"].include_high_order = False
                ablated = [local_forward(models["B+"], atoms, mask, g) for g in geometries]
                metrics = compare_features(outputs["B"][0], ablated[0], dtype, "direct")
                recovered_delta = float((ablated[0][0][0, 0]-ablated[1][0][0, 0]).abs().max())
                recorder.add("high_order_ablation", {"seed": seed, "route": "B+", "dtype": str(dtype),
                                                      "cutoff_mode": "one_ablation" if cutoff_one else "native"},
                             metrics | {"ablated_output_change": recovered_delta,
                                        "original_B_output_change": float((outputs["B"][0][0][0, 0]-outputs["B"][1][0][0, 0]).abs().max())})
                models["B+"].include_high_order = True


def shape_summary(rows):
    result = {}
    for route in ROUTES:
        selected = [r for r in rows if r["category"] == "resources" and r["route"] == route]
        by_sample = {}
        for row in selected:
            by_sample.setdefault((row["split"], row["index"]), []).append(row)
        metrics = {}
        for name in ("forward_seconds", "forward_plus_backward_seconds"):
            medians = [float(np.median([row[name] for row in group])) for group in by_sample.values()]
            metrics[name] = {"median": float(np.median(medians)), "max": max(medians), "sum": sum(medians)}
        deformed = [r for r in rows if r["category"] == "real_shear_response" and r["route"] == route]
        metrics["real_shear_relative_l2"] = {"median": float(np.median([r["relative_l2_change"] for r in deformed])),
                                             "min": min(r["relative_l2_change"] for r in deformed),
                                             "max": max(r["relative_l2_change"] for r in deformed)}
        controls = [r for r in rows if r["category"] == "equal_low_order_moments" and r["route"] == route
                    and r["dtype"] == "torch.float64" and r["cutoff_mode"] == "one_ablation"]
        metrics["controlled_counterexample"] = {"resolved_seeds": sum(r["response_resolved"] for r in controls),
                                                "seeds": len(controls),
                                                "max_abs_change_median": float(np.median([r["max_scalar_output_change"] for r in controls]))}
        result[route] = metrics
    return result


def load_sample(data_dir, sample, dtype):
    split, index = sample["split"], sample["index"]
    positions = np.load(data_dir/split/f"positions_{split}.npy", mmap_mode="r").reshape(-1, 82, 3)
    elements = np.load(data_dir/split/f"elements_{split}.npy", mmap_mode="r")[:, 2:]
    pos = torch.tensor(positions[index:index+1], dtype=dtype)
    ids = torch.tensor(elements[index:index+1], dtype=torch.long)
    return pos, ids == 0, ids


def sheared_geometry(pos, mask, amount=.02):
    """Physically deform the cell at fixed fractions; rebuild the full neighborhood."""
    basis, fractions = build_cell_from_lattice(pos.double())
    shear = np.array([[1., amount, 0.], [0., 1., 0.], [0., 0., 1.]])
    changed = pack_cell(basis[0].numpy()@shear, fractions[0].numpy(), pos.dtype)
    return geometry_from_pos(changed, mask)


def main_comparison(args, prior, budget):
    args.out_dir.mkdir(parents=True)
    recorder = Recorder(args.out_dir, budget)
    templates, _ = create_models(prior, torch.float64)
    models, entries, initial = {}, {}, {}
    for dtype in (torch.float32, torch.float64):
        models[dtype], entries[dtype] = create_models(prior, dtype)
        initial.update({f"{dtype}/{route}": state_digest(model) for route, model in models[dtype].items()})
        initial[f"{dtype}/zp"] = state_digest(entries[dtype])
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_version": "Q1", "splits": ["train", "valid"], "device": "cpu", "compute_threads": 1,
                "radius_A": 6., "labels_loaded": False, "checkpoint_loaded": False, "optimizer_created": False,
                "parameter_updates": 0, "production_model_executed": False,
                "parameter_counts": {route: sum(p.numel() for p in model.parameters()) for route, model in templates.items()},
                "control_seeds": list(CONTROL_SEEDS), "resource_repeats": REPEATS,
                "feature_tolerances": FEATURE_TOLERANCES, "response_thresholds": FEATURE_RESPONSE_THRESHOLDS,
                "budget_seconds": args.budget_seconds, "memory_limit_MiB": 2048,
                "torch": torch.__version__, "e3nn": importlib.metadata.version("e3nn"),
                "selected_samples": prior["selected_samples"], "initial_state_sha256": initial,
                "prior_manifest_sha256": sha256_file(args.precheck_dir/"manifest.json"),
                "input_sha256": prior["input_sha256"],
                "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
                    Path(__file__), ROOT/"tools/eval/periodic_message_high_order.py",
                    ROOT/"tests/test_periodic_message_high_order.py")},
                "old_source_sha256": prior["source_sha256"] | {"tools/eval/periodic_message_controls.py": sha256_file(ROOT/"tools/eval/periodic_message_controls.py")},
                "protocol": {"Bplus_e3": "full previous 16-real/two-dtype local protocol including representation and max-pair supercells",
                             "A_B_e3": "new synthetic two-dtype and real float32 rotation/reflection; old same-weight complete protocol also preserved",
                             "resources": "one warm-up then 3 round-robin no-update measurements per route/sample, float32; geometry excluded",
                             "shear": "physical volume-preserving xy shear 0.02, full neighbor rebuild, float32; not pure-angle control",
                             "angles": "5 prespecified seeds, 2 dtypes, exact fixed-radial controls; native vs cutoff=1 ablation distinguished",
                             "initialization": "all existing B weights copied; only appended 16x128 weights per block new, shared ZP and projections"}}
    write_json(args.out_dir/"manifest.json", manifest)
    error = None
    try:
        angle_controls(recorder, prior)
        print(f"angle controls: {len(recorder.rows)} cases", flush=True)
        for dtype in (torch.float32, torch.float64):
            for name, pos, mask, ids in synthetic_cases(dtype):
                geometry = geometry_from_pos(pos, mask)
                atoms = entries[dtype](ids, mask)
                for route, model in models[dtype].items():
                    output, debug = local_forward(model, atoms, mask, geometry)
                    arrays = [output]+debug["states"]+debug.get("angular_invariants", [])
                    recorder.add("synthetic_boundary", {"route": route, "dtype": str(dtype), "sample": name},
                                 {"pass": all(bool(torch.isfinite(x).all() and (x[mask] == 0).all()) for x in arrays)})
                if name in ("cubic_self_images", "skew", "mixed_padding", "empty_neighborhood"):
                    inspect_case(recorder, models[dtype], atoms, pos, mask, geometry,
                                 {"sample": name, "dtype": str(dtype)}, representations=name == "skew")
        for sample in prior["selected_samples"]:
            for dtype in (torch.float32, torch.float64):
                budget.check()
                pos, mask, ids = load_sample(args.data_dir, sample, dtype)
                start = time.perf_counter()
                geometry = geometry_from_pos(pos, mask)
                geometry_seconds = time.perf_counter()-start
                if len(geometry.cutoff) != sample["n_edges"] or int(geometry.pair_count.sum()) != sample["potential_angle_pairs"]:
                    raise AssertionError("selected structure cost changed")
                atoms = entries[dtype](ids, mask)
                context = sample | {"dtype": str(dtype)}
                base = inspect_case(recorder, {"B+": models[dtype]["B+"]}, atoms, pos, mask, geometry, context)["B+"]
                if sample["reason"] == "max_pairs":
                    doubled, new_mask, new_ids, original = supercell_geometry(pos, mask, ids)
                    expected = (base[0][:, original], {"states": [x[:, original] for x in base[1]["states"]]})
                    actual = local_forward(models[dtype]["B+"], entries[dtype](new_ids, new_mask), new_mask, doubled)
                    recorder.add("supercell", context | {"route": "B+", "operation": "2x1x1"},
                                 compare_features(expected, actual, dtype, "rebuilt") | {"records_ratio": len(doubled.cutoff)/len(geometry.cutoff)})
                if dtype != torch.float32:
                    continue
                references = {route: local_forward(model, atoms, mask, geometry) for route, model in models[dtype].items()}
                for name, q, _ in e3_operations():
                    if name not in ("rotation_1", "reflection"):
                        continue
                    changed = prepare_geometry(geometry.edges, geometry.vectors@torch.tensor(q, dtype=dtype), mask)
                    for route, model in models[dtype].items():
                        other = local_forward(model, atoms, mask, changed)
                        recorder.add("real_shared_e3", context | {"route": route, "operation": name},
                                     compare_features(references[route], other, dtype, "direct", orthogonal=q))
                        if route == "B+":
                            errors = [feature_error(x, y, FEATURE_TOLERANCES[str(dtype)]["direct"])
                                      for x, y in zip(references[route][1]["angular_invariants"], other[1]["angular_invariants"])]
                            recorder.add("high_order_invariant_e3", context | {"route": route, "operation": name},
                                         {"max_abs_error": max(x["max_abs_error"] for x in errors), "pass": all(x["pass"] for x in errors)})
                deformed = sheared_geometry(pos, mask)
                for route, model in models[dtype].items():
                    output = local_forward(model, atoms, mask, deformed)[0]
                    recorder.add("real_shear_response", context | {"route": route},
                                 response_metrics(references[route][0][~mask], output[~mask]) |
                                 {"edges_before": len(geometry.cutoff), "edges_after": len(deformed.cutoff),
                                  "degree_change_max": int((geometry.degree-deformed.degree).abs().max()), "pass": bool(torch.isfinite(output).all())})
                # Rotate execution order across rounds; no simultaneous CPU measurements.
                for repetition in range(REPEATS):
                    order = ROUTES[repetition:]+ROUTES[:repetition]
                    for route in order:
                        metrics = measure_resources(models[dtype][route], atoms, mask, geometry, budget)
                        if metrics["inactive_parameter_tensors"]:
                            metrics["pass"] = False
                        recorder.add("resources", context | {"route": route, "repetition": repetition},
                                     metrics | {"geometry_seconds": geometry_seconds})
            print(f"{sample['split']} {sample['index']}: E={sample['n_edges']} P={sample['potential_angle_pairs']}", flush=True)
        final = {f"{dtype}/{route}": state_digest(model) for dtype, group in models.items() for route, model in group.items()}
        final.update({f"{dtype}/zp": state_digest(entries[dtype]) for dtype in models})
        if initial != final:
            raise AssertionError("fixed parameters/buffers changed")
        manifest["final_state_sha256"] = final
        budget.check()
    except (Exception, KeyboardInterrupt) as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        recorder.close()
    failures = [r["case_id"] for r in recorder.rows if r["status"] != "pass"]
    comparison = shape_summary(recorder.rows) if error is None else {}
    summary = {"status": "complete" if error is None else "incomplete", "verdict": "pass" if error is None and not failures else "needs_changes",
               "cases": len(recorder.rows), "nonpassing_case_ids": failures, "groups": aggregate(recorder.rows),
               "comparison": comparison, "error": error,
               "scope": "untrained local atom features; magnitude is not DOS accuracy, no GPU/full model evidence"}
    write_json(args.out_dir/"summary.json", summary)
    manifest.update({"status": summary["status"], "verdict": summary["verdict"], "error": error})
    return manifest, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck-dir", type=Path, default=ROOT/"results/periodic_message_precheck_q1_r6_20261003")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/train4ARPAT")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("main", "rss-A", "rss-B", "rss-Bplus"), default="main")
    parser.add_argument("--budget-seconds", type=float, default=750.)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    if not 0 < args.budget_seconds <= (750 if args.mode == "main" else 45):
        parser.error("caps: main 750s; isolated RSS 45s each; cumulative target 900s incl. tests")
    prior = json.loads((args.precheck_dir/"manifest.json").read_text())
    if prior["status"] != "complete":
        raise ValueError("old precheck incomplete")
    for relative, digest in prior["source_sha256"].items():
        if sha256_file(ROOT/relative) != digest:
            raise ValueError(f"old source changed: {relative}")
    for relative, digest in prior["input_sha256"].items():
        if sha256_file(args.data_dir/relative) != digest:
            raise ValueError(f"Q1 data changed: {relative}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    with BudgetMonitor(args.budget_seconds, 2048.) as budget:
        if args.mode == "main":
            manifest, summary = main_comparison(args, prior, budget)
        else:
            route = {"rss-A": "A", "rss-B": "B", "rss-Bplus": "B+"}[args.mode]
            sample = next(r for r in prior["selected_samples"] if r["split"] == "train" and r["reason"] == "max_pairs")
            models, entry = create_models(prior, torch.float32)
            pos, mask, ids = load_sample(args.data_dir, sample, torch.float32)
            geometry = geometry_from_pos(pos, mask)
            atoms = entry(ids, mask)
            local_forward(models[route], atoms, mask, geometry)
            metrics = measure_resources(models[route], atoms, mask, geometry, budget)
            args.out_dir.mkdir(parents=True)
            summary = {"route": route, "sample": sample, **metrics,
                       "verdict": "pass" if metrics["pass"] and not metrics["inactive_parameter_tensors"] else "needs_changes",
                       "rss_scope": "fresh process, all 3 probes constructed; only selected route forward/backward"}
            manifest = {"status": "complete", "mode": args.mode, "prior_manifest_sha256": sha256_file(args.precheck_dir/"manifest.json"),
                        "labels_loaded": False, "optimizer_created": False, "parameter_updates": 0,
                        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (Path(__file__), ROOT/"tools/eval/periodic_message_high_order.py")}}
        elapsed = time.perf_counter()-start
        for document in (manifest, summary):
            document.update({"elapsed_seconds": elapsed, "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                             "sampled_peak_rss_MiB": budget.peak_mib})
    write_json(args.out_dir/"summary.json", summary)
    manifest["artifacts_sha256"] = {p.name: sha256_file(p) for p in args.out_dir.iterdir() if p.is_file() and p.name != "manifest.json"}
    write_json(args.out_dir/"manifest.json", manifest)
    print(json.dumps({"verdict": summary["verdict"], "cases": summary.get("cases"), "elapsed_seconds": elapsed,
                      "peak_rss_MiB": summary["peak_rss_MiB"], "error": summary.get("error")}), flush=True)
    return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
