"""Bounded CUDA acceptance/CPU comparison of independent periodic A/B+ modules.

Frozen untrained local weights, Q1 train/valid structures, explicit 6 Angstrom.
No labels, checkpoint, optimizer, updates, AMP or production DOS execution.
Timings describe local modules and selected batches, not full training speed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.periodic_geometry_acceptance import compare_records, e3_operations
from tools.eval.periodic_message_high_order_compare import create_models, load_sample
from tools.eval.periodic_message_module_check import aligned_local_models, frozen_hashes, sha256, write_json
from tools.eval.periodic_message_precheck import (
    BudgetMonitor, FEATURE_TOLERANCES, Recorder, changed_geometry, compare_features,
    feature_error, synthetic_cases,
)
from tools.eval.periodic_message_prototypes import geometry_from_pos
from utils.periodic_geometry import PeriodicNeighborRecords, build_periodic_records
from utils.relative_features import build_cell_from_lattice

ROUTES = ("A", "B+")
DTYPES = (torch.float32, torch.float64)
REPEATS = 3
WARMUP = 1
GRAD_TOLERANCES = {
    torch.float32: {"atol": 2e-6, "rtol": 2e-3, "relative_l2": 3e-3},
    torch.float64: {"atol": 2e-11, "rtol": 1e-8, "relative_l2": 1e-8},
}
SOURCE_FILES = ("tools/eval/periodic_message_gpu_check.py", "tests/test_periodic_message_gpu_check.py")


class GPUBudget(BudgetMonitor):
    def __init__(self, seconds, memory_mib, gpu_gib):
        super().__init__(seconds, memory_mib)
        self.gpu_limit = gpu_gib * 2**30
        self.peak_allocated = 0
        self.peak_reserved = 0

    def check(self):
        super().check()
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())
        if self.peak_reserved > self.gpu_limit:
            raise RuntimeError("CUDA allocator exceeds the GPU memory budget")


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def module_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def record_arrays(records):
    return {"keys": np.column_stack([getattr(records, name).cpu().numpy()
                                     for name in ("batch", "dst", "src")] + [records.shifts.cpu().numpy()]),
            "r": records.vectors.detach().double().cpu().numpy(),
            "d": records.distances.detach().double().cpu().numpy()}


def geometry_metrics(reference, actual, dtype):
    result = compare_records(record_arrays(reference), record_arrays(actual),
                             radius=reference.radius, tolerance=1e-4 if dtype == torch.float32 else 1e-6)
    status = result.pop("status")
    return result | {"geometry_status": status, "pass": status == "pass"}


def compare_local(reference, actual, dtype, mode="direct", orthogonal=None, order=None):
    reference, actual = cpu_tree(reference), cpu_tree(actual)
    if len(reference[1]["states"]) != len(actual[1]["states"]):
        raise ValueError("different number of debug states")
    metrics = compare_features(reference, actual, dtype, mode, orthogonal=orthogonal, order=order)
    tolerance = FEATURE_TOLERANCES[str(dtype)][mode]
    angular_error = 0.
    left = reference[1].get("angular_invariants", [])
    right = actual[1].get("angular_invariants", [])
    if len(left) != len(right):
        raise ValueError("different number of high-order statistics")
    for expected, current in zip(left, right):
        if order is not None:
            current = current[:, order]
        measurement = feature_error(expected, current, tolerance)
        metrics["pass"] = metrics["pass"] and measurement["pass"]
        angular_error = max(angular_error, measurement["max_abs_error"])
    counts_equal = True
    for name in ("degree", "pair_count"):
        current = actual[1][name]
        if order is not None:
            current = current[:, order]
        counts_equal = counts_equal and torch.equal(reference[1][name], current)
    metrics["pass"] = metrics["pass"] and counts_equal
    return metrics | {"counts_equal": counts_equal, "angular_max_abs_error": angular_error}


def compare_gradients(reference, actual, dtype):
    """Elementwise and norm checks, distinguishing roundoff from active gradients."""
    if set(reference) != set(actual):
        raise ValueError("gradient names differ")
    limit = GRAD_TOLERANCES[dtype]
    results = {}
    total_difference, total_reference = 0., 0.
    for name, expected in reference.items():
        current = actual[name]
        expected, current = expected.detach().double().cpu(), current.detach().double().cpu()
        measurement = feature_error(expected, current, (limit["atol"], limit["rtol"]))
        scale = float(expected.norm())
        relative = float((expected-current).norm()) / max(scale, 1e-30)
        active = bool(current.count_nonzero()) and bool(expected.count_nonzero())
        above_floor = scale > limit["atol"] * math.sqrt(expected.numel())
        total_difference += float((expected-current).square().sum())
        total_reference += float(expected.square().sum())
        results[name] = measurement | {"relative_l2": relative, "active": active,
                                      "above_absolute_floor": above_floor,
                                      "pass": measurement["pass"] and (not above_floor or (relative <= limit["relative_l2"] and active))}
    global_relative = math.sqrt(total_difference) / max(math.sqrt(total_reference), 1e-30)
    return {"pass": all(result["pass"] for result in results.values()) and global_relative <= limit["relative_l2"],
            "max_abs_error": max(result["max_abs_error"] for result in results.values()),
            "global_relative_l2": global_relative,
            "max_relative_l2_above_floor": max((result["relative_l2"] for result in results.values() if result["above_absolute_floor"]), default=0.),
            "tensors_below_absolute_floor": [name for name, result in results.items() if not result["above_absolute_floor"]],
            "gradient_tensors": len(results), "tensors": results}


def local_gradients(model, h, records, mask):
    source = h.detach().clone().requires_grad_()
    vectors = records.vectors.detach().clone().requires_grad_()
    distances = records.distances.detach().clone().requires_grad_()
    differentiable = replace(records, vectors=vectors, distances=distances)
    output = model(source, differentiable, mask)
    names, parameters = zip(*model.named_parameters())
    values = torch.autograd.grad(output[~mask].square().mean(), (source, vectors, distances) + parameters)
    return {name: value.detach().cpu() for name, value in zip(("input", "vectors", "distances") + names, values)}


def make_batch(data_dir, samples, dtype=torch.float32):
    if not samples:
        raise ValueError("batch must contain samples")
    loaded = [load_sample(data_dir, sample, dtype) for sample in samples]
    return tuple(torch.cat([value[i] for value in loaded], dim=0) for i in range(3))


def benchmark_samples(samples):
    """Fixed structure-only batches before seeing timing: B=1/4/8 plus dense B=1."""
    def selected(split, reason):
        return next(sample for sample in samples if sample["split"] == split and sample["reason"] == reason)
    typical = [selected(split, reason) for reason in ("pairs_quantile_0.5", "pairs_quantile_0.9", "pairs_quantile_0.1", "min_atoms")
               for split in ("train", "valid")]
    return {**{f"typical_b{size}": typical[:size] for size in (1, 4, 8)},
            "dense_b1": [selected("train", "max_pairs")]}


def measured_forward(model, h, records, mask, backward=False):
    """Same synchronized wall-clock boundary on CPU/GPU; no optimizer."""
    cuda = h.device.type == "cuda"
    source = h.detach().clone().requires_grad_(True) if backward else h
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    if backward:
        output = model(source, records, mask)
        gradients = torch.autograd.grad(output.square().mean(), tuple(model.parameters()) + (source,))
    else:
        with torch.no_grad():
            output = model(source, records, mask)
        gradients = ()
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    finite = bool(torch.isfinite(output).all()) and all(bool(torch.isfinite(g).all()) for g in gradients)
    # Individual channels can correctly have zero gradients by symmetry. Require
    # input and both blocks to participate; numerical tests compare every tensor.
    if gradients:
        named = dict(zip((name for name, _ in model.named_parameters()), gradients[:-1]))
        active = bool(gradients[-1].count_nonzero()) and all(
            any(bool(g.count_nonzero()) for name, g in named.items() if name.startswith(f"blocks.{index}.")) for index in (0, 1))
    else:
        active = True
    return elapsed, finite, active


def numeric_checks(recorder, models, gpu_models, entry, dtype, samples, data_dir):
    real = [("real", sample, *load_sample(data_dir, sample, dtype)) for sample in samples]
    synthetic = [(name, None, pos, mask, ids) for name, pos, mask, ids in synthetic_cases(dtype)]
    for name, sample, pos, mask, ids in synthetic + real:
        recorder.budget.check()
        context = {"sample": name, "dtype": str(dtype)}
        if sample:
            context |= {"split": sample["split"], "index": sample["index"], "id": sample["id"]}
        records = build_periodic_records(pos, mask, 6.)
        gpu_records = records.to("cuda")
        gpu_mask = mask.cuda()
        rebuilt = build_periodic_records(pos.cuda(), gpu_mask, 6.)
        recorder.add("gpu_geometry", context, geometry_metrics(records, rebuilt, dtype))
        with torch.no_grad():
            h = entry(ids, mask)
            gh = h.cuda()
            bases = {}
            for route in ROUTES:
                reference = models[route](h, records, mask, return_debug=True)
                baseline = gpu_models[route](gh, gpu_records, gpu_mask, return_debug=True)
                bases[route] = baseline
                recorder.add("cpu_gpu_features", context | {"route": route}, compare_local(reference, baseline, dtype))
                actual = gpu_models[route](gh, rebuilt, gpu_mask, return_debug=True)
                recorder.add("gpu_rebuilt_features", context | {"route": route}, compare_local(baseline, actual, dtype, "rebuilt"))
                for operation, q, _ in e3_operations():
                    if operation == "translation":
                        continue
                    transformed = replace(gpu_records, vectors=gpu_records.vectors @ torch.tensor(q, dtype=dtype, device="cuda"))
                    result = gpu_models[route](gh, transformed, gpu_mask, return_debug=True)
                    recorder.add("gpu_orthogonal", context | {"route": route, "operation": operation},
                                 compare_local(baseline, result, dtype, orthogonal=q))
                if h[~mask].numel():
                    padded = gh.clone()
                    padded[gpu_mask] = float("nan")
                    output = gpu_models[route](padded, gpu_records, gpu_mask)
                    good = bool(torch.isfinite(output).all()) and bool((output[gpu_mask] == 0).all())
                    recorder.add("gpu_padding", context | {"route": route}, {"pass": good})
            if sample:
                basis, _ = build_cell_from_lattice(pos.double())
                translated = pos.clone()
                delta = torch.tensor(e3_operations()[0][2], dtype=torch.float64) @ torch.linalg.inv(basis[0])
                translated[:, 2:][~mask] += delta.to(dtype)
                other = build_periodic_records(translated.cuda(), gpu_mask, 6.)
                metrics = geometry_metrics(records, other, dtype)
                for route in ROUTES:
                    actual = gpu_models[route](gh, other, gpu_mask, return_debug=True)
                    recorder.add("gpu_translation", context | {"route": route},
                                 compare_local(bases[route], actual, dtype, "rebuilt") | metrics)
            # Re-expression is separate from E(3), on the synthetic skew and
            # two fixed real median-cost samples. No CPU-neural rerun needed.
            if name == "skew" or (sample and sample["reason"] == "pairs_quantile_0.5"):
                probe_geometry = geometry_from_pos(pos, mask)
                for kind in ("permutation", "integer_images", "basis_swap", "basis_shear"):
                    old_geometry, order, geometry = changed_geometry(pos, mask, probe_geometry, kind)
                    other_mask = mask if order is None else mask[:, order]
                    other_h = gh if order is None else gh[:, order]
                    physical = PeriodicNeighborRecords.from_edges(old_geometry.edges, old_geometry.vectors, other_mask, 6.).to("cuda")
                    for route in ROUTES:
                        actual = gpu_models[route](other_h, physical, other_mask.cuda(), return_debug=True)
                        metric = compare_local(bases[route], actual, dtype, "rebuilt", order=None if order is None else np.argsort(order))
                        recorder.add("gpu_representation", context | {"route": route, "operation": kind},
                                     metric | {"geometry_status": geometry["status"], "boundary_missing": geometry["boundary_missing"],
                                               "boundary_extra": geometry["boundary_extra"]})
        if name == "skew" or (sample and sample["reason"] == "pairs_quantile_0.5"):
            for route in ROUTES:
                recorder.budget.check()
                expected = local_gradients(models[route], h, records, mask)
                actual = local_gradients(gpu_models[route], gh, gpu_records, gpu_mask)
                recorder.add("cpu_gpu_gradients", context | {"route": route}, compare_gradients(expected, actual, dtype))
        recorder.budget.check()
        if sample:
            print(json.dumps({"stage": "numeric", **context, "cases": len(recorder.rows)}), flush=True)


def pipeline_checks(recorder, gpu_models, entry, sample, data_dir):
    pos, mask, ids = load_sample(data_dir, sample, torch.float32)
    records = build_periodic_records(pos, mask, 6.).to("cuda")
    for route in ROUTES:
        recorder.budget.check()
        pipeline = nn.ModuleDict({"zp": copy.deepcopy(entry).cuda().requires_grad_(True),
                                  "local": copy.deepcopy(gpu_models[route]),
                                  "downstream": nn.Linear(512, 3).to(device="cuda", dtype=torch.float32)})
        before = module_digest(pipeline)
        h = pipeline["zp"](ids.cuda(), mask.cuda())
        output = pipeline["downstream"](pipeline["local"](h, records, mask.cuda()))
        parameters = tuple(pipeline.parameters())
        gradients = torch.autograd.grad(output[~mask.cuda()].square().mean(), parameters)
        registered = len(parameters) == sum(len(tuple(component.parameters())) for component in pipeline.values())
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        active = all(bool(gradient.count_nonzero()) for gradient in gradients)
        recorder.add("gpu_zp_pipeline", {"route": route, "dtype": str(torch.float32)},
                     {"pass": registered and finite and active and before == module_digest(pipeline),
                      "parameter_tensors": len(parameters), "registered": registered,
                      "finite_gradients": finite, "all_parameter_tensors_active": active, "parameters_unchanged": before == module_digest(pipeline)})
        del pipeline, h, output, gradients


def benchmark(recorder, models, gpu_models, entry, samples, data_dir):
    definitions = benchmark_samples(samples)
    for label, selected in definitions.items():
        recorder.budget.check()
        pos, mask, ids = make_batch(data_dir, selected)
        started = time.perf_counter()
        records = build_periodic_records(pos, mask, 6.)
        geometry_seconds = time.perf_counter()-started
        with torch.no_grad():
            h = entry(ids, mask)
        torch.cuda.synchronize()
        started = time.perf_counter()
        gpu_records, gh, gpu_mask = records.to("cuda"), h.cuda(), mask.cuda()
        torch.cuda.synchronize()
        transfer_seconds = time.perf_counter()-started
        expected_edges = sum(sample["n_edges"] for sample in selected)
        context = {"batch_label": label, "batch_size": len(selected), "samples": selected,
                   "n_edges": len(records.distances), "n_atoms": int((~mask).sum()),
                   "angle_pairs": sum(sample["potential_angle_pairs"] for sample in selected),
                   "dtype": str(torch.float32)}
        recorder.add("batch_preparation", context, {"pass": len(records.distances) == expected_edges,
                                                   "cpu_geometry_seconds": geometry_seconds,
                                                   "h2d_and_record_validation_seconds": transfer_seconds})
        # Both devices/routes use the same batches and immutable parameter values.
        # Warmups precede measurements; device order alternates between batches.
        devices = ("cpu", "cuda") if len(selected) in (1, 8) else ("cuda", "cpu")
        for device in devices:
            for route in ROUTES:
                model, source, geometry, supplied_mask = (models[route], h, records, mask) if device == "cpu" else (gpu_models[route], gh, gpu_records, gpu_mask)
                for backward in (False, True):
                    for _ in range(WARMUP):
                        recorder.budget.check()
                        measured_forward(model, source, geometry, supplied_mask, backward)
                    for repetition in range(REPEATS):
                        recorder.budget.check()
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        seconds, finite, active = measured_forward(model, source, geometry, supplied_mask, backward)
                        recorder.budget.check()
                        extra = {"peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
                                 "peak_reserved_MiB": torch.cuda.max_memory_reserved()/2**20} if device == "cuda" else {}
                        recorder.add("benchmark", context | {"device": device, "route": route, "backward": backward, "repetition": repetition},
                                     {"pass": finite and active, "seconds": seconds,
                                      "finite_gradients": finite, "active_gradients": active, **extra})
        print(json.dumps({"stage": "benchmark", "batch": label, "batch_size": len(selected), "cases": len(recorder.rows)}), flush=True)
    return definitions


def summarize_benchmark(rows):
    result = {}
    for row in rows:
        if row["category"] != "benchmark":
            continue
        group = result.setdefault(row["batch_label"], {})
        arm = group.setdefault(row["route"], {})
        key = f"{row['device']}/{'forward_backward' if row['backward'] else 'forward'}"
        arm.setdefault(key, []).append(row["seconds"])
        if row["device"] == "cuda" and row["backward"]:
            arm["gpu_peak_allocated_MiB"] = max(arm.get("gpu_peak_allocated_MiB", 0.), row["peak_allocated_MiB"])
            arm["gpu_peak_reserved_MiB"] = max(arm.get("gpu_peak_reserved_MiB", 0.), row["peak_reserved_MiB"])
    for group in result.values():
        for arm in group.values():
            for key, value in list(arm.items()):
                if isinstance(value, list):
                    arm[key] = {"median_seconds": statistics.median(value), "seconds": value}
            arm["cpu_over_gpu_forward_backward"] = arm["cpu/forward_backward"]["median_seconds"] / arm["cuda/forward_backward"]["median_seconds"]
    return result


def verify_cpu_evidence(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["verdict"] != "pass":
        raise ValueError("completed CPU module acceptance required")
    for mapping in (manifest["source_sha256"], manifest["frozen_source_sha256"], manifest["frozen_artifact_sha256"]):
        for relative, digest in mapping.items():
            if sha256(ROOT / relative) != digest:
                raise AssertionError(f"protected source/artifact changed: {relative}")
    for name, digest in manifest["artifacts_sha256"].items():
        if sha256(directory / name) != digest:
            raise AssertionError(f"CPU acceptance artifact changed: {name}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/train4ARPAT")
    parser.add_argument("--cpu-dir", type=Path, default=ROOT / "results/periodic_message_modules_q1_r6_20261003_v2")
    parser.add_argument("--precheck-dir", type=Path, default=ROOT / "results/periodic_message_precheck_q1_r6_20261003")
    parser.add_argument("--budget-seconds", type=float, default=720.)
    parser.add_argument("--memory-mib", type=float, default=2048.)
    parser.add_argument("--gpu-memory-gib", type=float, default=8.)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this acceptance")
    if any(not math.isfinite(v) or v <= 0 for v in (args.budget_seconds, args.memory_mib, args.gpu_memory_gib)):
        raise ValueError("positive finite budgets required")
    protected = verify_cpu_evidence(args.cpu_dir)
    prior, _, _ = frozen_hashes(args.precheck_dir)
    for relative, digest in prior["input_sha256"].items():
        if sha256(args.data_dir / relative) != digest:
            raise AssertionError(f"input hash differs: {relative}")
    properties = torch.cuda.get_device_properties(0)
    fraction = args.gpu_memory_gib * 2**30 / properties.total_memory
    if fraction > 1:
        raise ValueError("GPU budget exceeds physical memory")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.out_dir.mkdir(parents=True)
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "e3nn": importlib.metadata.version("e3nn"), "gpu": properties.name,
                   "gpu_total_GiB": properties.total_memory/2**30, "cpu_threads": 1,
                   "tf32": False, "amp": False}
    manifest = {"status": "running", "radius_A": 6., "splits": ["train", "valid"],
                "labels_loaded": False, "checkpoint_loaded": False, "optimizer_created": False,
                "production_model_executed": False, "parameter_updates": 0,
                "environment": environment, "selected_samples": prior["selected_samples"],
                "input_sha256": prior["input_sha256"], "cpu_manifest_sha256": sha256(args.cpu_dir / "manifest.json"),
                "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
                "core_source_sha256": protected["source_sha256"],
                "feature_tolerances": FEATURE_TOLERANCES,
                "gradient_tolerances": {str(k): v for k, v in GRAD_TOLERANCES.items()},
                "benchmark_repeats": REPEATS, "benchmark_warmup": WARMUP,
                "budget_seconds": args.budget_seconds, "cpu_memory_MiB": args.memory_mib,
                "gpu_memory_GiB": args.gpu_memory_gib,
                "scope": "local modules/record builder; finite CUDA evidence, no full-model or DOS accuracy conclusion"}
    write_json(args.out_dir / "manifest.json", manifest)
    error, states, definitions = None, {}, {}
    with GPUBudget(args.budget_seconds, args.memory_mib, args.gpu_memory_gib) as budget:
        recorder = Recorder(args.out_dir, budget)
        try:
            for dtype in DTYPES:
                references, entry = create_models(prior, dtype)
                models = aligned_local_models(references, dtype)
                gpu_models = {route: copy.deepcopy(model).cuda() for route, model in models.items()}
                initial = {f"{device}/{route}": module_digest(model)
                           for device, group in (("cpu", models), ("cuda", gpu_models)) for route, model in group.items()}
                numeric_checks(recorder, models, gpu_models, entry, dtype, prior["selected_samples"], args.data_dir)
                if dtype == torch.float32:
                    median = next(sample for sample in prior["selected_samples"] if sample["split"] == "train" and sample["reason"] == "pairs_quantile_0.5")
                    pipeline_checks(recorder, gpu_models, entry, median, args.data_dir)
                    definitions = benchmark(recorder, models, gpu_models, entry, prior["selected_samples"], args.data_dir)
                final = {f"{device}/{route}": module_digest(model)
                         for device, group in (("cpu", models), ("cuda", gpu_models)) for route, model in group.items()}
                if initial != final:
                    raise AssertionError("CPU/GPU weights changed")
                states[str(dtype)] = {"initial": initial, "final": final}
                del references, entry, models, gpu_models
                torch.cuda.empty_cache()
            budget.check()
            verify_cpu_evidence(args.cpu_dir)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        finally:
            torch.cuda.synchronize()
            budget.peak_allocated = max(budget.peak_allocated, torch.cuda.max_memory_allocated())
            budget.peak_reserved = max(budget.peak_reserved, torch.cuda.max_memory_reserved())
            recorder.close()
            torch.set_default_dtype(previous_dtype)
        counts = Counter(row["status"] for row in recorder.rows)
        verdict = "incomplete" if error else "fail" if counts["fail"] else "boundary_limited" if counts["boundary_limited"] else "pass"
        summary = {"status": "incomplete" if error else "complete", "verdict": verdict,
                   "cases": len(recorder.rows), "status_counts": dict(counts),
                   "categories": dict(Counter(row["category"] for row in recorder.rows)),
                   "nonpassing_case_ids": [row["case_id"] for row in recorder.rows if row["status"] != "pass"],
                   "error": error, "elapsed_seconds": time.perf_counter()-budget.start,
                   "peak_rss_MiB": budget.peak_mib, "gpu_peak_allocated_MiB": budget.peak_allocated/2**20,
                   "gpu_peak_reserved_MiB": budget.peak_reserved/2**20,
                   "comparison": summarize_benchmark(recorder.rows) if not error else {}}
    write_json(args.out_dir / "summary.json", summary)
    manifest |= {"status": summary["status"], "verdict": verdict, "states": states,
                 "benchmark_batches": definitions,
                 "artifacts_sha256": {name: sha256(args.out_dir / name) for name in ("summary.json", "cases.json", "cases.jsonl", "cases.csv")}}
    write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
