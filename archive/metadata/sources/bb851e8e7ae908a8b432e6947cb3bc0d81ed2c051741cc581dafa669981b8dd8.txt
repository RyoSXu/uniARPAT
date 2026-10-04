"""Bounded CPU precheck of two untrained local message candidates, Q1 train/valid.

No DOS labels, checkpoints, optimizer, training loop, or production-model edits.
Finite local evidence is distinct from a whole-model E(3) guarantee or DOS gain.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.element_identity_preflight import code_version, sha256_file
from tools.eval.periodic_geometry_acceptance import (
    TOLERANCES as GEOMETRY_TOLERANCES, compare_records, e3_operations, pack_cell, write_json,
)
from tools.eval.periodic_message_prototypes import (
    EDGE_IRREPS, STATE_IRREPS, PeriodicMessageProbe, ProbeZP, geometry_from_pos, prepare_geometry,
)
from tools.eval.periodic_neighbor_radius_audit import SPLITS, load_structures
from utils.relative_features import build_cell_from_lattice

SEED = 20261003
# Fixed before running neural features; these are dimensionless feature tests,
# not the Angstrom tolerances of the earlier geometry acceptance.
FEATURE_TOLERANCES = {
    "torch.float32": {"direct": (2e-5, 2e-5), "rebuilt": (1e-4, 1e-4)},
    "torch.float64": {"direct": (2e-10, 2e-10), "rebuilt": (2e-6, 2e-6)},
}


class BudgetExceeded(RuntimeError):
    pass


class BudgetMonitor:
    """Sample RSS and enforce a computation budget at bounded Python boundaries."""

    def __init__(self, seconds, memory_mib):
        self.seconds, self.memory_mib = seconds, memory_mib
        self.start = time.perf_counter()
        self.peak_mib = 0.
        self.reason = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def sample(self):
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        current = pages*os.sysconf("SC_PAGE_SIZE")/2**20
        self.peak_mib = max(self.peak_mib, current)
        if current > self.memory_mib:
            self.reason = f"RSS {current:.1f} MiB exceeds {self.memory_mib} MiB"
        if time.perf_counter()-self.start > self.seconds:
            self.reason = f"compute wall time exceeds {self.seconds} seconds"
        return current

    def _watch(self):
        while not self.stop.wait(.02):
            self.sample()

    def check(self):
        self.sample()
        if self.reason:
            raise BudgetExceeded(self.reason)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def feature_error(expected, actual, tolerance):
    if expected.shape != actual.shape or not torch.isfinite(actual).all():
        raise AssertionError("non-finite feature or mismatched shape")
    if not expected.numel():
        return {"max_abs_error": 0., "max_normalized_error": 0., "pass": True}
    atol, rtol = tolerance
    difference = (expected-actual).abs()
    error = float(difference.max())
    normalized = float((difference/(atol+rtol*expected.abs())).max())
    return {"max_abs_error": error, "max_normalized_error": normalized, "pass": normalized <= 1.}


def record_arrays(geometry):
    edges = geometry.edges
    return {"keys": np.column_stack([edges[key].numpy() for key in ("batch", "dst", "src")]
                                     + [edges["shifts"].numpy()]),
            "d": edges["distances"].double().detach().numpy(),
            "r": geometry.vectors.double().detach().numpy()}


def compare_features(reference, actual, dtype, mode, orthogonal=None, order=None):
    tolerance = FEATURE_TOLERANCES[str(dtype)][mode]
    output, debug = actual
    expected, baseline = reference
    if order is not None:
        output = output[:, order]
    measurements = [feature_error(expected, output, tolerance)]
    representation = None
    if orthogonal is not None:
        representation = STATE_IRREPS.D_from_matrix(
            torch.as_tensor(orthogonal.T.copy(), dtype=torch.float64)).to(dtype)
    for previous, current in zip(baseline["states"], debug["states"]):
        if order is not None:
            current = current[:, order]
        if previous.shape[-1] == 108 and representation is not None:
            previous = previous @ representation.T
        measurements.append(feature_error(previous, current, tolerance))
    return {"max_abs_error": max(x["max_abs_error"] for x in measurements),
            "max_normalized_error": max(x["max_normalized_error"] for x in measurements),
            "checked_tensors": len(measurements), "pass": all(x["pass"] for x in measurements),
            "feature_atol": tolerance[0], "feature_rtol": tolerance[1]}


def selection(data, prior_directory):
    """Exactly eight per split, chosen only by structure and audited cost."""
    path = prior_directory/"structures.csv.gz"
    manifest = json.loads((prior_directory/"manifest.json").read_text())
    if manifest["status"] != "complete" or sha256_file(path) != manifest["artifacts_sha256"][path.name]:
        raise ValueError("prior structure audit is not complete or hash differs")
    with gzip.open(path, "rt") as stream:
        rows = [row for row in csv.DictReader(stream) if float(row["radius_A"]) == 6.]
    selected = []
    for split in SPLITS:
        sub = sorted((row for row in rows if row["split"] == split), key=lambda row: int(row["index"]))
        pairs = np.array([int(row["potential_angle_pairs"]) for row in sub])
        atoms = np.array([int(row["n_atoms"]) for row in sub])
        degree = np.array([int(row["max_degree"]) for row in sub])
        smin = data[split][3]
        candidates = [(int(np.argmax(pairs)), "max_pairs"), (int(np.argmax(degree)), "max_degree"),
                      (int(np.argmin(smin)), "min_cell_singular_value"),
                      (int(np.argmin(atoms)), "min_atoms"), (int(np.argmax(atoms)), "max_atoms")]
        ordered = np.argsort(pairs, kind="stable")
        candidates += [(int(ordered[int(q*(len(ordered)-1))]), f"pairs_quantile_{q}")
                       for q in (.1, .5, .9, .99)]
        candidates += [(int(index), "even_spaced_fallback")
                       for index in np.linspace(0, len(sub)-1, 16, dtype=int)]
        indices = []
        for index, reason in candidates:
            if index not in indices:
                indices.append(index)
                row = sub[index]
                if int(row["index"]) != index or str(data[split][2][index]) != row["id"]:
                    raise ValueError("prior row ID disagrees with current structure")
                selected.append({"split": split, "index": index, "id": row["id"], "reason": reason,
                                 "n_atoms": int(row["n_atoms"]), "n_edges": int(row["n_edges"]),
                                 "max_degree": int(row["max_degree"]),
                                 "potential_angle_pairs": int(row["potential_angle_pairs"])})
            if len(indices) == 8:
                break
        assert len(indices) == 8
    return selected, {str(path.relative_to(ROOT)): sha256_file(path)}


def frame_geometry(pos, mask, physical_basis, fractions):
    changed_pos = pack_cell(physical_basis, fractions, pos.dtype)
    geometry = geometry_from_pos(changed_pos, mask)
    canonical = build_cell_from_lattice(changed_pos)[0][0].double().numpy()
    frame = np.linalg.solve(canonical, physical_basis)
    if np.max(np.abs(frame.T@frame-np.eye(3))) > 2e-5:
        raise AssertionError("rebuilt frame is not orthogonal within geometry tolerance")
    vectors = geometry.vectors @ torch.as_tensor(frame, dtype=pos.dtype)
    return prepare_geometry(geometry.edges, vectors, mask), frame


def changed_geometry(pos, mask, geometry, kind, orthogonal=None, translation=None):
    """Rebuild real coordinates or alternate cell representations, lifting frames."""
    basis, fractions = build_cell_from_lattice(pos)
    basis, fractions = basis[0].double().numpy(), fractions[0].double().numpy()
    order = None
    if kind == "permutation":
        order = np.random.default_rng(SEED).permutation(mask.shape[1])
        p = torch.cat((pos[:, :2], pos[:, 2:][:, order]), 1)
        actual = geometry_from_pos(p, mask[:, order])
        mapped = record_arrays(actual)
        mapped["keys"][:, 1:3] = order[mapped["keys"][:, 1:3]]
    elif kind == "integer_images":
        offsets = np.random.default_rng(SEED).integers(-2, 3, size=fractions.shape)
        p = pos.clone()
        p[:, 2:] += torch.tensor(offsets, dtype=pos.dtype)
        actual = geometry_from_pos(p, mask)
        mapped = record_arrays(actual)
        keys = mapped["keys"]
        keys[:, 3:] += offsets[keys[:, 2]]-offsets[keys[:, 1]]
    elif kind in ("basis_swap", "basis_shear"):
        transform = {"basis_swap": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                     "basis_shear": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])}[kind]
        actual, _ = frame_geometry(pos, mask, transform@basis, fractions@np.linalg.inv(transform))
        mapped = record_arrays(actual)
        mapped["keys"][:, 3:] = mapped["keys"][:, 3:] @ transform
    elif kind == "physical_e3":
        changed_basis = basis@orthogonal
        cartesian = fractions@basis@orthogonal+translation
        actual, _ = frame_geometry(pos, mask, changed_basis, cartesian@np.linalg.inv(changed_basis))
        mapped = record_arrays(actual)
    else:
        raise ValueError(kind)
    expected = record_arrays(geometry)
    if kind == "physical_e3":
        expected["r"] = expected["r"]@orthogonal
    metrics = compare_records(expected, mapped, tolerance=GEOMETRY_TOLERANCES[pos.dtype])
    return actual, order, metrics


def supercell_geometry(pos, mask, elements):
    basis, fractions = build_cell_from_lattice(pos)
    basis, fractions = basis[0].double().numpy(), fractions[0].double().numpy()
    slots = np.flatnonzero(~mask[0].numpy())
    original = np.tile(slots, 2)
    replicas = np.repeat([[0, 0, 0], [1, 0, 0]], len(slots), axis=0)
    transform = np.diag([2, 1, 1])
    changed_mask = torch.zeros(1, len(original), dtype=torch.bool)
    g, _ = frame_geometry(pos, changed_mask, transform@basis,
                          (fractions[original]+replicas)@np.linalg.inv(transform))
    return g, changed_mask, elements[:, original], original


def synthetic_star(skew, distance, dtype):
    axes = np.eye(3)
    if skew:
        axes[2] = [.25, .25, np.sqrt(.875)]
    directions = np.r_[axes, -axes]
    cartesian = np.r_[np.zeros((1, 3)), distance*directions]
    pos = pack_cell(np.eye(3)*30, cartesian/30+.5, dtype)
    return pos, torch.zeros(1, 7, dtype=torch.bool), torch.full((1, 7), 6, dtype=torch.long)


def synthetic_cases(dtype):
    cubic = pack_cell(np.eye(3)*3, np.array([[0., 0., 0.]]), dtype)
    basis = np.array([[4.1, 0., 0.], [-1.2, 5.3, 0.], [.7, .9, 6.2]])
    skew = pack_cell(basis, np.array([[.13, .27, .39], [.57, .63, .79], [.89, .11, .43]]), dtype)
    mixed = skew.repeat(2, 1, 1)
    mixed_mask = torch.tensor([[False, True, False], [True, True, True]])
    mixed[0, 3] = float("nan")
    mixed[1, 2:] = float("nan")
    sparse = pack_cell(np.eye(3)*30, np.array([[0., 0., 0.]]), dtype)
    return [("cubic_self_images", cubic, torch.zeros(1, 1, dtype=torch.bool), torch.tensor([[14]])),
            ("skew", skew, torch.zeros(1, 3, dtype=torch.bool), torch.tensor([[6, 14, 8]])),
            ("mixed_padding", mixed, mixed_mask, torch.tensor([[6, 0, 8], [0, 0, 0]])),
            ("empty_neighborhood", sparse, torch.zeros(1, 1, dtype=torch.bool), torch.tensor([[6]])),
            ("all_padding", sparse, torch.ones(1, 1, dtype=torch.bool), torch.tensor([[0]])),
            ("zero_slots", sparse[:, :2], torch.zeros(1, 0, dtype=torch.bool), torch.zeros(1, 0, dtype=torch.long))]


class Recorder:
    def __init__(self, directory, budget):
        self.directory, self.budget = directory, budget
        self.rows = []
        self.stream = (directory/"cases.jsonl").open("w")

    def add(self, category, context, result):
        self.budget.check()
        result = dict(result)
        passed = bool(result.pop("pass", True))
        limited = result.get("geometry_status") == "boundary_limited"
        row = {"case_id": len(self.rows), "category": category, **context,
               "status": "fail" if not passed else "boundary_limited" if limited else "pass", **result}
        self.rows.append(row)
        self.stream.write(json.dumps(row, allow_nan=False)+"\n")
        self.stream.flush()

    def close(self):
        self.stream.close()
        write_json(self.directory/"cases.json", self.rows)
        fields = sorted(set().union(*(row.keys() for row in self.rows))) if self.rows else ["case_id"]
        with (self.directory/"cases.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})


def local_forward(model, atoms, mask, geometry):
    with torch.no_grad():
        return model(atoms, mask, geometry, True)


def geometric_metrics(metrics):
    return {"geometry_status": metrics["status"],
            "boundary_missing": metrics["boundary_missing"], "boundary_extra": metrics["boundary_extra"],
            "geometry_distance_error_A": metrics["max_distance_error_A"],
            "geometry_vector_error_A": metrics["max_vector_error_A"]}


def inspect_case(recorder, models, atoms, pos, mask, geometry, context, representations=True):
    """Actual component forwards in external frames; permutation is separate."""
    dtype = pos.dtype
    base = {route: local_forward(model, atoms, mask, geometry) for route, model in models.items()}
    operations = e3_operations()
    for name, q, _ in operations:
        if name == "translation":
            continue  # Physical translation is rebuilt below, not a no-op test.
        recorder.budget.check()
        changed = prepare_geometry(geometry.edges, geometry.vectors@torch.tensor(q, dtype=dtype), mask)
        for route, model in models.items():
            result = compare_features(base[route], local_forward(model, atoms, mask, changed),
                                      dtype, "direct", orthogonal=q)
            recorder.add("e3_direct", context | {"route": route, "operation": name}, result)
    if pos.shape[0] == 1 and mask.shape[1]:
        kinds = ["translation"]
        if context.get("sample") == "skew":
            kinds += ["rotation_1", "reflection"]
        for name, q, t in operations:
            if name not in kinds:
                continue
            changed, _, metrics = changed_geometry(pos, mask, geometry, "physical_e3", q, t)
            for route, model in models.items():
                result = compare_features(base[route], local_forward(model, atoms, mask, changed),
                                          dtype, "rebuilt", orthogonal=q)
                recorder.add("e3_rebuilt", context | {"route": route, "operation": name},
                             result | geometric_metrics(metrics))
    if representations:
        for kind in ("permutation", "integer_images", "basis_swap", "basis_shear"):
            recorder.budget.check()
            changed, order, metrics = changed_geometry(pos, mask, geometry, kind)
            if order is None:
                other_atoms, other_mask, inverse = atoms, mask, None
            else:
                other_atoms, other_mask = atoms[:, order], mask[:, order]
                inverse = np.argsort(order)
            for route, model in models.items():
                result = compare_features(base[route], local_forward(model, other_atoms, other_mask, changed),
                                          dtype, "rebuilt", order=inverse)
                recorder.add("representation", context | {"route": route, "operation": kind},
                             result | geometric_metrics(metrics))
    return base


def measure_resources(model, atoms, mask, geometry, budget):
    """One warmed forward and backward; no parameter update or optimizer."""
    budget.check()
    before = state_digest(model)
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    start = time.perf_counter()
    output = local_forward(model, atoms, mask, geometry)[0]
    forward_seconds = time.perf_counter()-start
    del output
    budget.check()
    source = atoms.detach().clone().requires_grad_()
    parameters = tuple(model.parameters())
    start = time.perf_counter()
    output = model(source, mask, geometry)
    grad_forward_seconds = time.perf_counter()-start
    start = time.perf_counter()
    gradients = torch.autograd.grad(output.square().mean(), parameters+(source,), allow_unused=True)
    backward_seconds = time.perf_counter()-start
    budget.check()
    finite = all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
    active = [name for (name, _), gradient in zip(model.named_parameters(), gradients)
              if gradient is not None and gradient.count_nonzero()]
    inactive = [name for (name, _), gradient in zip(model.named_parameters(), gradients)
                if gradient is None or not gradient.count_nonzero()]
    unchanged = before == state_digest(model)
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    blocks_active = all(any(name.startswith(f"blocks.{i}.") for name in active) for i in range(2))
    return {"forward_seconds": forward_seconds, "grad_forward_seconds": grad_forward_seconds,
            "backward_seconds": backward_seconds, "forward_plus_backward_seconds": grad_forward_seconds+backward_seconds,
            "process_peak_rss_MiB": peak_after, "prior_process_peak_rss_MiB": peak_before,
            "rss_sampled_peak_MiB": budget.peak_mib, "finite_gradients": bool(finite),
            "parameters_unchanged": unchanged, "active_parameter_tensors": active,
            "inactive_parameter_tensors": inactive, "both_blocks_active": blocks_active,
            "pass": bool(finite and unchanged and blocks_active and gradients[-1] is not None)}


def gradient_and_response_cases(recorder, models, entry, dtype):
    """Same-element angle counterexample, radial change, true count change."""
    for distance in (5., 5.9):
        pos, mask, elements = synthetic_star(False, distance, dtype)
        changed, _, _ = synthetic_star(True, distance, dtype)
        g, other = geometry_from_pos(pos, mask), geometry_from_pos(changed, mask)
        with torch.no_grad():
            atoms = entry(elements, mask)
        for route, model in models.items():
            baseline = local_forward(model, atoms, mask, g)
            output, debug = local_forward(model, atoms, mask, other)
            delta = float((baseline[0][0, 0]-output[0, 0]).abs().max())
            state_delta = float((baseline[1]["states"][-1][0, 0]-debug["states"][-1][0, 0]).abs().max())
            central = [x.unit[x.edges["dst"] == 0].double() for x in (g, other)]
            moment = [float((u.T@u-2*torch.eye(3, dtype=torch.float64)).square().sum()) for u in central]
            stable = distance == 5.
            threshold = 1e-8 if dtype == torch.float32 else 1e-10
            recorder.add("angle_response", {"route": route, "dtype": str(dtype), "distance_A": distance},
                         {"max_scalar_output_change": delta, "max_last_state_change": state_delta,
                          "vector_sum_norms": [float(u.sum(0).norm()) for u in central],
                          "quadrupole_norm_sq": moment, "center_degrees": [int(x.degree[0, 0]) for x in (g, other)],
                          "response_threshold": threshold, "response_resolved": delta > threshold,
                          "pass": not stable or delta > threshold})
    pos, mask, elements = synthetic_star(True, 5., dtype)
    g = geometry_from_pos(pos, mask)
    with torch.no_grad():
        atoms = entry(elements, mask)
    radial, _, _ = synthetic_star(True, 5.2, dtype)
    fewer = mask.clone()
    fewer[0, -1] = True
    for name, p, m in (("distance", radial, mask), ("neighbor_count", pos, fewer)):
        geometry = geometry_from_pos(p, m)
        for route, model in models.items():
            first = local_forward(model, atoms, mask, g)[0][0, 0]
            second = local_forward(model, atoms, m, geometry)[0][0, 0]
            delta = float((first-second).abs().max())
            recorder.add("structural_response", {"route": route, "dtype": str(dtype), "operation": name},
                         {"max_scalar_output_change": delta, "center_degree_before": int(g.degree[0, 0]),
                          "center_degree_after": int(geometry.degree[0, 0]), "pass": delta > 1e-8})
    # VJP through the actual distance and angular entrances, with graph IDs fixed.
    for route, model in models.items():
        vectors = g.vectors.detach().clone().requires_grad_()
        distances = g.edges["distances"].detach().clone().requires_grad_()
        edges = dict(g.edges, distances=distances)
        differentiable = prepare_geometry(edges, vectors, mask)
        loss = model(atoms, mask, differentiable).square().mean()
        radial_gradient, direction_gradient = torch.autograd.grad(loss, (distances, vectors), allow_unused=True)
        radial_max = 0. if radial_gradient is None else float(radial_gradient.abs().max())
        direction_max = 0. if direction_gradient is None else float(direction_gradient.abs().max())
        finite = all(x is not None and torch.isfinite(x).all() for x in (radial_gradient, direction_gradient))
        recorder.add("geometry_vjp", {"route": route, "dtype": str(dtype)},
                     {"distance_gradient_max_abs": radial_max, "direction_gradient_max_abs": direction_max,
                      "pass": bool(finite and radial_max > 1e-12 and direction_max > 1e-12)})


def aggregate(rows):
    groups = {}
    for row in rows:
        key = f"{row['category']}/{row.get('route', 'shared')}/{row.get('dtype', 'shared')}"
        group = groups.setdefault(key, {"cases": 0, "pass": 0, "fail": 0, "boundary_limited": 0,
                                        "max_abs_error": 0., "max_normalized_error": 0.})
        group["cases"] += 1
        group[row["status"]] += 1
        for metric in ("max_abs_error", "max_normalized_error"):
            group[metric] = max(group[metric], row.get(metric, 0.))
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/train4ARPAT")
    parser.add_argument("--radius-audit-dir", type=Path, default=ROOT/"results/periodic_neighbor_radius_q1_20261003")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=840.)
    parser.add_argument("--memory-mib", type=float, default=2048.)
    args = parser.parse_args(argv)
    if not 0 < args.budget_seconds <= 840 or not 0 < args.memory_mib <= 2048:
        parser.error("precheck caps: 840 seconds plus test/verification reserve, RSS 2048 MiB")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)  # e3nn rotation matrices also use this precision.
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    args.out_dir.mkdir(parents=True)
    sources = [Path(__file__), ROOT/"tools/eval/periodic_message_prototypes.py",
               ROOT/"tests/test_periodic_message_prototypes.py", ROOT/"tools/eval/periodic_geometry_acceptance.py",
               ROOT/"tools/eval/periodic_neighbor_radius_audit.py", ROOT/"model/periodic_manybody.py",
               ROOT/"utils/g2_periodic_edges.py", ROOT/"utils/relative_features.py",
               ROOT/"tools/eval/element_identity_preflight.py", ROOT/"tools/eval/g2_edge_audit.py"]
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "code_version": code_version(), "data_version": "Q1", "splits": list(SPLITS),
                "labels_loaded": False, "checkpoint_loaded": False, "optimizer_created": False,
                "parameter_updates": 0, "production_model_executed": False,
                "prototype_executed": True, "device": "cpu", "compute_threads": 1,
                "rss_sampling_threads": 1, "radius_A": 6., "strict_cutoff": True, "seed": SEED,
                "torch": torch.__version__, "e3nn": importlib.metadata.version("e3nn"),
                "feature_tolerances": FEATURE_TOLERANCES,
                "budget_seconds": args.budget_seconds, "memory_limit_MiB": args.memory_mib,
                "memory_scope": "process RSS, not per-route allocator attribution; 20ms sampled plus ru_maxrss",
                "resource_scope": "one warmed float32 forward/backward per real sample; no optimizer step",
                "initialization": "shared untrained Embedding->LayerNorm->Linear ZP; float64 master weights cast to each dtype",
                "pair_policy": "all unordered pairs, 4096-pair chunks; non-reentrant recomputation in backward",
                "e3_scope": "direct Cartesian local frames; physical translations rebuilt for all real inputs, rotation/reflection also rebuilt for synthetic skew",
                "guarantee_scope": "finite local prototype evidence; no full DOS-model guarantee or prediction gain",
                "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}}
    write_json(args.out_dir/"manifest.json", manifest)
    error = None
    models, initial, entry_by_dtype = {}, {}, {}
    with BudgetMonitor(args.budget_seconds, args.memory_mib) as budget:
        recorder = Recorder(args.out_dir, budget)
        try:
            data, input_hashes = load_structures(args.data_dir)
            samples, prior_hashes = selection(data, args.radius_audit_dir)
            write_json(args.out_dir/"samples.json", samples)
            elements = {split: np.load(args.data_dir/split/f"elements_{split}.npy")[:, 2:] for split in SPLITS}
            torch.manual_seed(SEED)
            entry = ProbeZP().eval()
            for parameter in entry.parameters():
                parameter.requires_grad_(False)
            torch.manual_seed(SEED+1)
            templates = {"invariant": PeriodicMessageProbe("invariant")}
            torch.manual_seed(SEED+2)
            templates["equivariant"] = PeriodicMessageProbe("equivariant")
            for name in ("local_input", "local_output"):
                getattr(templates["equivariant"], name).load_state_dict(getattr(templates["invariant"], name).state_dict())
            for dtype in (torch.float32, torch.float64):
                entry_by_dtype[dtype] = copy.deepcopy(entry).to(dtype)
                models[dtype] = {route: copy.deepcopy(model).to(dtype).eval() for route, model in templates.items()}
                initial.update({f"{dtype}/{route}": state_digest(model) for route, model in models[dtype].items()})
                initial[f"{dtype}/zp"] = state_digest(entry_by_dtype[dtype])
            manifest.update({"input_sha256": input_hashes, "prior_artifact_sha256": prior_hashes,
                             "selected_samples": samples, "initial_state_sha256": initial,
                             "parameter_counts": {route: sum(p.numel() for p in model.parameters())
                                                  for route, model in templates.items()},
                             "construction_and_loading_seconds": time.perf_counter()-start})
            write_json(args.out_dir/"manifest.json", manifest)
            print("protocol fixed: 16 real structures, 2 dtypes, 2 routes; synthetic checks first", flush=True)
            for dtype in (torch.float32, torch.float64):
                for name, pos, mask, atom_ids in synthetic_cases(dtype):
                    budget.check()
                    geometry = geometry_from_pos(pos, mask)
                    with torch.no_grad():
                        atoms = entry_by_dtype[dtype](atom_ids, mask)
                    context = {"sample": name, "dtype": str(dtype)}
                    for route, model in models[dtype].items():
                        output, debug = local_forward(model, atoms, mask, geometry)
                        ok = bool(torch.isfinite(output).all() and (output[mask] == 0).all())
                        ok = ok and all(torch.isfinite(x).all() and (x[mask] == 0).all() for x in debug["states"])
                        recorder.add("synthetic_boundary", context | {"route": route},
                                     {"n_edges": len(geometry.cutoff), "max_degree": int(geometry.degree.max()) if mask.numel() else 0,
                                      "potential_angle_pairs": int(geometry.pair_count.sum()), "pass": ok})
                    if name in ("cubic_self_images", "skew", "mixed_padding", "empty_neighborhood"):
                        inspect_case(recorder, models[dtype], atoms, pos, mask, geometry, context,
                                     representations=name == "skew")
                gradient_and_response_cases(recorder, models[dtype], entry_by_dtype[dtype], dtype)
            print(f"synthetic complete: {len(recorder.rows)} cases", flush=True)
            for sample in samples:
                split, index = sample["split"], sample["index"]
                for dtype in (torch.float32, torch.float64):
                    budget.check()
                    pos = torch.from_numpy(data[split][0][index:index+1]).to(dtype)
                    mask = torch.from_numpy(data[split][1][index:index+1])
                    atom_ids = torch.from_numpy(elements[split][index:index+1]).long()
                    geometry_start = time.perf_counter()
                    geometry = geometry_from_pos(pos, mask)
                    geometry_seconds = time.perf_counter()-geometry_start
                    assert len(geometry.cutoff) == sample["n_edges"], "count differs from full radius audit"
                    assert int(geometry.pair_count.sum()) == sample["potential_angle_pairs"]
                    with torch.no_grad():
                        atoms = entry_by_dtype[dtype](atom_ids, mask)
                    context = {"split": split, "index": index, "id": sample["id"], "dtype": str(dtype),
                               "n_atoms": sample["n_atoms"], "n_edges": sample["n_edges"],
                               "potential_angle_pairs": sample["potential_angle_pairs"]}
                    inspect_case(recorder, models[dtype], atoms, pos, mask, geometry, context)
                    if dtype == torch.float32:
                        for route, model in models[dtype].items():
                            measurements = measure_resources(model, atoms, mask, geometry, budget)
                            recorder.add("resources", context | {"route": route},
                                         measurements | {"geometry_seconds": geometry_seconds})
                    if sample["reason"] == "max_pairs":
                        doubled, changed_mask, changed_ids, original = supercell_geometry(pos, mask, atom_ids)
                        assert len(doubled.cutoff) == 2*len(geometry.cutoff)
                        with torch.no_grad():
                            changed_atoms = entry_by_dtype[dtype](changed_ids, changed_mask)
                        for route, model in models[dtype].items():
                            output, debug = local_forward(model, atoms, mask, geometry)
                            expected = (output[:, original], {"states": [x[:, original] for x in debug["states"]]})
                            actual = local_forward(model, changed_atoms, changed_mask, doubled)
                            metrics = compare_features(expected, actual, dtype, "rebuilt")
                            recorder.add("supercell", context | {"route": route, "operation": "2x1x1"},
                                         metrics | {"supercell_records": len(doubled.cutoff),
                                                    "receiver_copies": 2})
                print(f"{split} {index} {sample['id']}: both dtypes/routes checked; E={sample['n_edges']} P={sample['potential_angle_pairs']}", flush=True)
            final = {f"{dtype}/{route}": state_digest(model) for dtype, group in models.items()
                     for route, model in group.items()}
            final.update({f"{dtype}/zp": state_digest(entry_by_dtype[dtype]) for dtype in models})
            assert initial == final, "parameters or buffers mutated"
            manifest["final_state_sha256"] = final
            budget.check()
        except (Exception, KeyboardInterrupt) as caught:
            error = f"{type(caught).__name__}: {caught}"
            manifest["error"] = error
        finally:
            recorder.close()
    groups = aggregate(recorder.rows)
    failures = [row["case_id"] for row in recorder.rows if row["status"] != "pass"]
    elapsed = time.perf_counter()-start
    complete = error is None
    summary = {"status": "complete" if complete else "incomplete",
               "verdict": "pass" if complete and not failures else "needs_changes" if complete else "insufficient_evidence",
               "cases": len(recorder.rows), "nonpassing_case_ids": failures, "groups": groups,
               "elapsed_seconds": elapsed, "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               "sampled_peak_rss_MiB": budget.peak_mib, "error": error,
               "scope": "local prototypes only; no spectral accuracy, full-model invariance, GPU, optimizer or parameter updates"}
    write_json(args.out_dir/"summary.json", summary)
    manifest.update({"status": summary["status"], "verdict": summary["verdict"],
                     "elapsed_seconds": elapsed, "peak_rss_MiB": summary["peak_rss_MiB"],
                     "artifacts_sha256": {path.name: sha256_file(path) for path in sorted(args.out_dir.iterdir())
                                           if path.is_file() and path.name != "manifest.json"}})
    write_json(args.out_dir/"manifest.json", manifest)
    print(json.dumps({key: summary[key] for key in ("verdict", "cases", "elapsed_seconds", "peak_rss_MiB", "error")}), flush=True)
    return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
