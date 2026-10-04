"""Q1 train/valid structure-only radius audit; no labels or model execution.

Enumerate at the largest candidate radius once, then filter exact records for
each smaller radius. Adaptive larger references only resolve the first three
distance shells; shell coverage is reported with right censoring, not inferred
from an empty candidate neighborhood. Historical G2 inputs remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.element_identity_preflight import code_version, sha256_file
from tools.eval.g2_edge_audit import _load_split, _quantiles
from utils.g2_periodic_edges import build_g2_edges, g2_indegrees
from utils.relative_features import build_cell_from_lattice

SPLITS = ("train", "valid")


def positive_values(values, name):
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name} must be positive and finite")
    if list(values) != sorted(set(values)):
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    return values


def validate_inputs(pos, mask):
    if pos.ndim != 3 or pos.shape[-1] != 3 or mask.shape != pos[:, 2:].shape[:2]:
        raise ValueError("lattice, atom slots and mask shapes do not align")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if not torch.isfinite(pos[:, :2]).all() or not torch.isfinite(pos[:, 2:][~mask]).all():
        raise ValueError("non-finite lattice or valid atom coordinates")
    angles = pos[:, 1]
    if not ((angles > 0) & (angles < 180)).all():
        raise ValueError("lattice angles must lie strictly between 0 and 180 degrees")
    cell, _ = build_cell_from_lattice(pos)
    smin = torch.linalg.svdvals(cell).amin(dim=-1)
    if not torch.isfinite(smin).all() or (smin <= 1e-6).any():
        raise ValueError("non-finite or degenerate reconstructed cell")
    return smin


def shell_bounds(sorted_distances, tolerance):
    """Three anchored tolerance groups, without chaining close distances."""
    starts = np.full(3, np.nan)
    upper = np.full(3, np.nan)
    cursor = 0
    for shell in range(3):
        if cursor == len(sorted_distances):
            break
        starts[shell] = sorted_distances[cursor]
        end = np.searchsorted(sorted_distances, starts[shell] + tolerance, side="right")
        upper[shell] = sorted_distances[end - 1]
        cursor = end
    return starts, upper


def _distance_lists(edges, slots):
    dst = edges["dst"].numpy()
    distance = edges["distances"].numpy()
    if not np.isfinite(distance).all() or (distance <= 0).any():
        raise ValueError("non-finite or zero-distance interatomic record; audit stops")
    return [np.sort(distance[dst == slot]) for slot in slots]


def inspect_crystal(pos, mask, radii, tolerances, max_reference):
    radii = positive_values(radii, "radii")
    tolerances = positive_values(tolerances, "shell tolerances")
    if not math.isfinite(max_reference) or max_reference < radii[-1]:
        raise ValueError("maximum reference radius must be finite and cover all candidates")
    validate_inputs(pos, mask)
    if pos.shape[0] != 1:
        raise ValueError("inspect_crystal expects one crystal")
    slots = torch.where(~mask[0])[0].numpy()
    start_time = time.perf_counter()
    base = build_g2_edges(pos, mask, r_cut=radii[-1])
    base_seconds = time.perf_counter() - start_time
    distance_lists = _distance_lists(base, slots)
    reference = radii[-1]
    while True:
        bounds = np.full((len(tolerances), len(slots), 3), np.nan)
        censored = np.ones(bounds.shape, dtype=bool)
        for t, tolerance in enumerate(tolerances):
            for atom, distances in enumerate(distance_lists):
                starts, upper = shell_bounds(distances, tolerance)
                complete = np.isfinite(starts) & (starts + tolerance < reference)
                bounds[t, atom] = np.where(complete, upper, np.nan)
                censored[t, atom] = ~complete
        if not censored.any() or reference >= max_reference:
            break
        reference = min(max_reference, reference * 1.5)
        reference_edges = build_g2_edges(pos, mask, r_cut=reference)
        distance_lists = _distance_lists(reference_edges, slots)

    degrees = np.zeros((len(slots), len(radii)), dtype=np.int64)
    self_count = np.zeros(len(radii), dtype=np.int64)
    boundary_count = np.zeros(len(radii), dtype=np.int64)
    for column, radius in enumerate(radii):
        keep = base["distances"] < radius
        degrees[:, column] = g2_indegrees(
            base["batch"][keep], base["dst"][keep], 1, mask.shape[1])[0, slots].numpy()
        self_count[column] = int((keep & (base["src"] == base["dst"])).sum())
        boundary_count[column] = int(((base["distances"] - radius).abs() <= 1e-5).sum())
    return {
        "slots": slots, "degrees": degrees, "shell_bounds": bounds,
        "censored": censored, "reference_radius": reference,
        "base_seconds": base_seconds, "total_seconds": time.perf_counter() - start_time,
        "self_count": self_count, "boundary_count": boundary_count,
    }


def load_structures(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    loaded, identity = {}, {}
    for split in SPLITS:
        elements, positions = _load_split(str(data_dir), split)
        ids = np.load(data_dir / split / f"{split}_index.npy")
        if len(elements) != manifest["splits"][split]["n"] or len(ids) != len(elements):
            raise ValueError(f"{split}: sample count disagrees with Q1 manifest")
        mask = elements[:, 2:] == 0
        if (mask.all(axis=1)).any():
            raise ValueError(f"{split}: a structure has no valid atoms")
        smin = validate_inputs(torch.from_numpy(positions).float(), torch.from_numpy(mask))
        for filename in (f"elements_{split}.npy", f"positions_{split}.npy", f"{split}_index.npy"):
            relative = f"{split}/{filename}"
            digest = sha256_file(data_dir / relative)
            if digest[:16] != manifest["files"][relative]["sha"]:
                raise ValueError(f"{relative}: hash disagrees with Q1 manifest")
            identity[relative] = digest
        loaded[split] = (positions, mask, ids, smin.numpy())
    identity["manifest.json"] = sha256_file(data_dir / "manifest.json")
    return loaded, identity


def audit_split(split, data, radii, tolerances, max_reference, atom_writer, structure_writer):
    positions, masks, ids, smin = data
    degree_blocks, shell_blocks, rows = [], [], []
    for index, (position, mask) in enumerate(zip(positions, masks)):
        try:
            result = inspect_crystal(torch.from_numpy(position[None]).float(),
                                     torch.from_numpy(mask[None]), radii, tolerances, max_reference)
        except (ValueError, AssertionError, RuntimeError) as error:
            raise ValueError(f"{split} index={index} id={ids[index]}: {error}") from error
        degrees, bounds = result["degrees"], result["shell_bounds"]
        count = len(degrees)
        coverage = bounds[..., None] < np.asarray(radii)
        edges = degrees.sum(axis=0)
        pairs = (degrees * (degrees - 1) // 2).sum(axis=0)
        covered = coverage.sum(axis=1)
        zero = (degrees == 0).sum(axis=0)
        rows.append({"n_atoms": count, "edges": edges, "max_degree": degrees.max(axis=0),
                     "mean_degree": degrees.mean(axis=0), "pairs": pairs, "covered": covered,
                     "zero": zero, "self": result["self_count"], "boundary": result["boundary_count"],
                     "reference": result["reference_radius"], "base_seconds": result["base_seconds"],
                     "total_seconds": result["total_seconds"]})
        degree_blocks.append(degrees)
        shell_blocks.append(bounds)
        for atom, slot in enumerate(result["slots"]):
            atom_writer.writerow([split, index, str(ids[index]), int(slot), *degrees[atom].tolist(),
                                  *bounds[:, atom].ravel().tolist()])
        for column, radius in enumerate(radii):
            structure_writer.writerow([split, index, str(ids[index]), radius, count, int(edges[column]),
                                       float(degrees[:, column].mean()), int(degrees[:, column].max()),
                                       int(zero[column]), int(pairs[column]), int(result["self_count"][column]),
                                       int(result["boundary_count"][column]), *covered[:, :, column].ravel().tolist(),
                                       result["reference_radius"], result["base_seconds"], result["total_seconds"]])
        if (index + 1) % 1000 == 0 or index + 1 == len(positions):
            print(f"{split}: {index+1}/{len(positions)}", flush=True)
    return np.concatenate(degree_blocks), np.concatenate(shell_blocks, axis=1), rows


def summarize(split, radii, tolerances, degrees, bounds, rows):
    summaries = []
    atom_counts = np.array([row["n_atoms"] for row in rows])
    for column, radius in enumerate(radii):
        edge_counts = np.array([row["edges"][column] for row in rows])
        for t, tolerance in enumerate(tolerances):
            entry = {"split": split, "radius_A": radius, "shell_tolerance_A": tolerance,
                     "structures": len(rows), "atoms": len(degrees), "total_edges": int(edge_counts.sum()),
                     "degree": _quantiles(degrees[:, column]), "edges_per_structure": _quantiles(edge_counts),
                     "mean_degree_per_structure": _quantiles([row["mean_degree"][column] for row in rows]),
                     "zero_neighbor_atoms": int((degrees[:, column] == 0).sum()),
                     "structures_with_zero_neighbor_atom": int(sum(row["zero"][column] > 0 for row in rows)),
                     "potential_unordered_angle_pairs": int(sum(row["pairs"][column] for row in rows)),
                     "cross_cell_self_records": int(sum(row["self"][column] for row in rows)),
                     "near_cutoff_records_1e-5A": int(sum(row["boundary"][column] for row in rows)),
                     "six_field_storage_bytes_float32_int64": int(edge_counts.sum() * 64)}
            for shell in range(3):
                entry[f"shell{shell+1}_atoms_covered_fraction"] = float((bounds[t, :, shell] < radius).mean())
                entry[f"shell{shell+1}_structures_fully_covered_fraction"] = float(np.mean([
                    row["covered"][t, shell, column] == n for row, n in zip(rows, atom_counts)]))
                entry[f"shell{shell+1}_reference_censored_atoms"] = int(np.isnan(bounds[t, :, shell]).sum())
            summaries.append(entry)
    return summaries


def benchmark(split, data, rows, radii, per_split, repeats, writer):
    positions, masks, ids, smin = data
    atom_counts = (~masks).sum(axis=1)
    selected = np.unique(np.r_[np.linspace(0, len(positions)-1, min(per_split, len(positions)), dtype=int),
                               np.argmax(atom_counts), np.argmin(smin),
                               np.argmax([row["edges"][-1] for row in rows])])
    build_g2_edges(torch.from_numpy(positions[selected[0]:selected[0]+1]).float(),
                   torch.from_numpy(masks[selected[0]:selected[0]+1]), r_cut=radii[-1])
    totals = {str(radius): [] for radius in radii}
    for index in selected:
        pos, mask = torch.from_numpy(positions[index:index+1]).float(), torch.from_numpy(masks[index:index+1])
        for column, radius in enumerate(radii):
            timings = []
            for _ in range(repeats):
                start = time.perf_counter()
                edges = build_g2_edges(pos, mask, r_cut=radius)
                timings.append(time.perf_counter() - start)
            if len(edges["distances"]) != rows[index]["edges"][column]:
                raise ValueError(f"{split}/{index}/R={radius}: direct versus filtered edge counts disagree")
            median = float(np.median(timings))
            totals[str(radius)].append(median)
            writer.writerow([split, int(index), str(ids[index]), radius, int(atom_counts[index]),
                             len(edges["distances"]), repeats, median, min(timings), max(timings)])
    return {"indices": selected.tolist(), "seconds_per_structure": {
        radius: _quantiles(seconds) for radius, seconds in totals.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/train4ARPAT")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--radii", type=float, nargs="+", required=True)
    parser.add_argument("--shell-tolerances", type=float, nargs="+", default=[0.001, 0.01])
    parser.add_argument("--max-reference-radius", type=float, default=32.0)
    parser.add_argument("--benchmark-per-split", type=int, default=64)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    args = parser.parse_args(argv)
    radii = positive_values(args.radii, "radii")
    tolerances = positive_values(args.shell_tolerances, "shell tolerances")
    if not math.isfinite(args.max_reference_radius) or args.max_reference_radius < radii[-1]:
        parser.error("maximum reference radius must be finite and cover all candidates")
    if args.benchmark_per_split <= 0 or args.benchmark_repeats <= 0:
        parser.error("benchmark size and repeats must be positive")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {args.out_dir}")
    torch.set_num_threads(1)
    start = time.perf_counter()
    loaded, input_hashes = load_structures(args.data_dir)
    args.out_dir.mkdir(parents=True)
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_version": "Q1", "splits": list(SPLITS), "labels_loaded": False,
                "model_executed": False, "radii_A": radii, "shell_tolerances_A": tolerances,
                "max_reference_radius_A": args.max_reference_radius, "torch": torch.__version__,
                "dtype": "float32", "device": "cpu", "threads": 1, "code_version": code_version(),
                "input_sha256": input_hashes, "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path)
                    for path in (Path(__file__), ROOT / "utils/g2_periodic_edges.py",
                                 ROOT / "utils/relative_features.py", ROOT / "tools/eval/g2_edge_audit.py",
                                 ROOT / "tools/eval/element_identity_preflight.py")},
                "shell_definition": "sorted distances, groups anchored at first distance with absolute tolerance; no chaining",
                "coverage_definition": "all records in the shell satisfy strict d<R; censored shells are not certified covered",
                "cost_scope": "CPU edge enumeration only; angle pairs and storage are structural estimates, not model cost"}
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    summaries, benchmarks, diagnostics = [], {}, {}
    with gzip.open(args.out_dir / "atoms.csv.gz", "wt", newline="") as atoms_file, \
         gzip.open(args.out_dir / "structures.csv.gz", "wt", newline="") as structures_file, \
         (args.out_dir / "cpu_benchmark.csv").open("w", newline="") as benchmark_file:
        atom_writer, structure_writer, benchmark_writer = map(csv.writer, (atoms_file, structures_file, benchmark_file))
        atom_writer.writerow(["split", "index", "id", "slot", *[f"degree_R{r}" for r in radii],
                              *[f"shell{s}_upper_tol{t}_A" for t in tolerances for s in range(1, 4)]])
        structure_writer.writerow(["split", "index", "id", "radius_A", "n_atoms", "n_edges", "mean_degree",
                                   "max_degree", "zero_neighbor_atoms", "potential_angle_pairs", "self_records",
                                   "near_cutoff_records_1e-5A", *[f"shell{s}_covered_atoms_tol{t}" for t in tolerances for s in range(1,4)],
                                   "reference_radius_A", "max_candidate_enumeration_seconds", "audit_seconds"])
        benchmark_writer.writerow(["split", "index", "id", "radius_A", "n_atoms", "n_edges", "repeats",
                                   "median_seconds", "min_seconds", "max_seconds"])
        for split, data in loaded.items():
            degrees, bounds, rows = audit_split(split, data, radii, tolerances, args.max_reference_radius,
                                                atom_writer, structure_writer)
            summaries.extend(summarize(split, radii, tolerances, degrees, bounds, rows))
            diagnostics[split] = {"s_min_A": _quantiles(data[3]),
                                  "first_shell_upper_A": _quantiles(bounds[0,:,0][np.isfinite(bounds[0,:,0])]),
                                  "reference_radius_A": _quantiles([row["reference"] for row in rows]),
                                  "adaptive_reference_structures": sum(row["reference"] > radii[-1] for row in rows),
                                  "max_candidate_enumeration_seconds": sum(row["base_seconds"] for row in rows),
                                  "audit_seconds": sum(row["total_seconds"] for row in rows)}
            benchmarks[split] = benchmark(split, data, rows, radii, args.benchmark_per_split,
                                           args.benchmark_repeats, benchmark_writer)
    result = {"summary": summaries, "benchmark": benchmarks, "diagnostics": diagnostics}
    (args.out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    flat = [{key: value for key, value in entry.items() if not isinstance(value, dict)} |
            {f"{key}_{stat}": value for key in ("degree", "edges_per_structure", "mean_degree_per_structure")
             for stat, value in entry[key].items()} for entry in summaries]
    with (args.out_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    manifest.update(status="complete", elapsed_seconds=time.perf_counter()-start,
                    peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    structures={split: len(data[0]) for split, data in loaded.items()},
                    artifacts_sha256={path.name: sha256_file(path) for path in args.out_dir.iterdir()
                                      if path.name != "manifest.json"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"completed in {manifest['elapsed_seconds']:.1f}s; outputs: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
