"""CPU geometry acceptance at an explicit cutoff; no spectra or model execution.

Full train/valid record contracts are followed by fixed-sample independent
enumeration, physical E(3) transformations, and separate representation checks.
Cartesian frames are explicit in this diagnostic, outside the production API.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
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

from model.periodic_manybody import periodic_edge_vectors
from tools.eval.element_identity_preflight import code_version, sha256_file
from tools.eval.periodic_neighbor_radius_audit import (
    SPLITS, load_structures, positive_values, validate_inputs,
)
from utils.g2_periodic_edges import build_g2_edges, g2_indegrees
from utils.relative_features import build_cell_from_lattice

# Fixed before execution. The float64 frame tolerance also accounts for the
# existing cell reconstruction's sin(gamma)+1e-8 stabilizer, not just roundoff.
TOLERANCES = {torch.float32: 1e-4, torch.float64: 1e-6}
FRAME_TOLERANCE = 2e-5


def max_abs(array):
    return float(np.max(np.abs(array))) if array.size else 0.0


def row_tokens(keys):
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    return keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()


def checked_records(pos, mask, radius=6.0):
    """Reuse existing functions and certify all six fields, including empty ones."""
    positive_values([radius], "radius")
    validate_inputs(pos, mask)
    edges = build_g2_edges(pos, mask, r_cut=radius)
    vectors = periodic_edge_vectors(pos, edges)
    size = len(edges["distances"])
    for name in ("batch", "dst", "src"):
        assert edges[name].shape == (size,) and edges[name].dtype == torch.int64
    assert edges["shifts"].shape == (size, 3) and edges["shifts"].dtype == torch.int64
    assert edges["distances"].shape == (size,) and edges["distances"].dtype == pos.dtype
    assert vectors.shape == (size, 3) and vectors.dtype == pos.dtype
    keys = np.column_stack([edges[name].numpy() for name in ("batch", "dst", "src")]
                           + [edges["shifts"].numpy()]).astype(np.int64)
    distances, vectors = edges["distances"].double().numpy(), vectors.double().numpy()
    assert np.isfinite(distances).all() and np.isfinite(vectors).all()
    assert (distances > 0).all(), "zero-distance distinct atoms require input rejection"
    assert (distances < radius).all(), "strict cutoff violated"
    B, L = mask.shape
    if size:
        assert ((keys[:, 0] >= 0) & (keys[:, 0] < B)).all()
        assert ((keys[:, 1:3] >= 0) & (keys[:, 1:3] < L)).all()
        mask_array = mask.numpy()
        assert not mask_array[keys[:, 0], keys[:, 1]].any()
        assert not mask_array[keys[:, 0], keys[:, 2]].any()
        assert not ((keys[:, 1] == keys[:, 2]) & (keys[:, 3:] == 0).all(axis=1)).any()
    tokens = row_tokens(keys)
    assert len(np.unique(tokens)) == size, "duplicate periodic record"
    reverse = keys.copy()
    reverse[:, 1:3] = keys[:, [2, 1]]
    reverse[:, 3:] *= -1
    _, forward_indices, reverse_indices = np.intersect1d(
        tokens, row_tokens(reverse), return_indices=True)
    assert len(forward_indices) == size, "missing reverse record"
    reverse_vector_error = max_abs(vectors[forward_indices] + vectors[reverse_indices])
    reverse_distance_error = max_abs(distances[forward_indices] - distances[reverse_indices])
    norm_error = max_abs(np.linalg.norm(vectors, axis=1) - distances)
    cell, frac = build_cell_from_lattice(pos)
    cell_array, frac_array = cell.double().numpy(), frac.double().numpy()
    displacement = (frac_array[keys[:, 0], keys[:, 2]] - frac_array[keys[:, 0], keys[:, 1]]
                    + keys[:, 3:])
    independent_vectors = np.einsum("ei,eij->ej", displacement, cell_array[keys[:, 0]])
    formula_error = max_abs(vectors - independent_vectors)
    tolerance = TOLERANCES[pos.dtype]
    assert max(reverse_vector_error, reverse_distance_error, norm_error, formula_error) <= tolerance, \
        f"record geometry error exceeds {tolerance} A"
    degrees = g2_indegrees(edges["batch"], edges["dst"], B, L).numpy()
    assert int(degrees.sum()) == size and not degrees[mask.numpy()].any()
    metrics = {"n_edges": size, "n_atoms": int((~mask).sum()),
               "zero_neighbor_atoms": int(((degrees == 0) & ~mask.numpy()).sum()),
               "max_degree": int(degrees.max()) if degrees.size else 0,
               "cross_cell_self_records": int((keys[:, 1] == keys[:, 2]).sum()),
               "near_cutoff_records": int((np.abs(distances-radius) <= tolerance).sum()),
               "max_reverse_vector_error_A": reverse_vector_error,
               "max_reverse_distance_error_A": reverse_distance_error,
               "max_norm_error_A": norm_error, "max_formula_error_A": formula_error}
    return {"keys": keys, "d": distances, "r": vectors, "cell": cell_array,
            "frac": frac_array, "metrics": metrics}


def independent_reference(cell, fractions, mask, radius=6.0):
    """Independent float64 enumeration with reciprocal-column component bounds.

    For ||r||<R, |(r A^-1)_k|<R ||(A^-1)[:,k]||. Wrapping each atom to [0,1)
    gives |delta_k|<1, so a separate integer bound per axis is sufficient.
    This does not use the production singular-value bound or pair centering.
    """
    blocks, vector_blocks = [], []
    for batch, (basis, frac, padding) in enumerate(zip(cell, fractions, mask)):
        slots = np.flatnonzero(~padding)
        if not len(slots):
            continue
        original = frac[slots].astype(np.float64)
        offsets = np.floor(original).astype(np.int64)
        wrapped = original - offsets
        limits = np.ceil(radius*np.linalg.norm(np.linalg.inv(basis), axis=0)+1).astype(int)
        shifts = np.array(list(itertools.product(*[range(-k, k+1) for k in limits])), dtype=np.int64)
        delta = wrapped[None, :, :] - wrapped[:, None, :]
        cartesian = delta @ basis
        for start in range(0, len(shifts), 64):
            translations = shifts[start:start+64]
            candidates = cartesian[:, :, None, :] + (translations @ basis)[None, None, :, :]
            keep = np.linalg.norm(candidates, axis=-1) < radius
            for zero in np.flatnonzero((translations == 0).all(axis=1)):
                keep[np.arange(len(slots)), np.arange(len(slots)), zero] = False
            dst, src, image = np.where(keep)
            true_shifts = translations[image] + offsets[dst] - offsets[src]
            blocks.append(np.column_stack([np.full(len(dst), batch), slots[dst], slots[src], true_shifts]))
            vector_blocks.append(candidates[dst, src, image])
    keys = np.concatenate(blocks).astype(np.int64) if blocks else np.empty((0, 6), dtype=np.int64)
    vectors = np.concatenate(vector_blocks) if vector_blocks else np.empty((0, 3))
    return {"keys": keys, "r": vectors, "d": np.linalg.norm(vectors, axis=1)}


def compare_records(reference, actual, radius=6.0, tolerance=1e-4):
    """Align exact record IDs; report cutoff ambiguity instead of dropping it."""
    assert len(np.unique(row_tokens(actual["keys"]))) == len(actual["keys"]), "duplicate mapped record"
    _, left, right = np.intersect1d(row_tokens(reference["keys"]), row_tokens(actual["keys"]),
                                  return_indices=True)
    missing = np.setdiff1d(np.arange(len(reference["keys"])), left)
    extra = np.setdiff1d(np.arange(len(actual["keys"])), right)
    stable_missing = missing[np.abs(reference["d"][missing]-radius) > tolerance]
    stable_extra = extra[np.abs(actual["d"][extra]-radius) > tolerance]
    assert not len(stable_missing) and not len(stable_extra), \
        f"non-boundary record mismatch: missing={len(stable_missing)}, extra={len(stable_extra)}"
    distance_error = max_abs(reference["d"][left]-actual["d"][right])
    vector_error = max_abs(reference["r"][left]-actual["r"][right])
    assert max(distance_error, vector_error) <= tolerance, \
        f"matched-record error: distance={distance_error}, vector={vector_error}, tolerance={tolerance}"
    return {"status": "boundary_limited" if len(missing)+len(extra) else "pass",
            "matched_records": len(left), "boundary_missing": len(missing), "boundary_extra": len(extra),
            "max_distance_error_A": distance_error, "max_vector_error_A": vector_error}


def pack_cell(cell, fractions, dtype):
    """Encode an explicit physical row-basis in the existing scalar pos format."""
    lengths = np.linalg.norm(cell, axis=1)
    angles = [np.degrees(np.arccos(np.clip(np.dot(cell[i], cell[j])/(lengths[i]*lengths[j]), -1, 1)))
              for i, j in ((1, 2), (0, 2), (0, 1))]
    pos = torch.zeros(1, len(fractions)+2, 3, dtype=dtype)
    pos[0, 0] = torch.tensor([lengths[0], lengths[1], 1/lengths[2]], dtype=dtype)
    pos[0, 1] = torch.tensor(angles, dtype=dtype)
    pos[0, 2:] = torch.as_tensor(fractions, dtype=dtype)
    return pos


def lift_frame(records, physical_cell):
    """Map canonical vectors into the independently specified physical frame."""
    frame = np.linalg.solve(records["cell"][0], physical_cell)
    error = max_abs(frame.T @ frame - np.eye(3))
    assert error <= FRAME_TOLERANCE, f"cell encoding frame is not orthogonal: {error}"
    lifted = dict(records, r=records["r"] @ frame)
    return lifted, error


def e3_operations():
    rng = np.random.default_rng(20261003)
    rotations = []
    for _ in range(2):
        matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrix[:, 0] *= np.linalg.det(matrix)
        rotations.append(matrix)
    return [("translation", np.eye(3), np.array([.713, -.227, .391])),
            ("rotation_1", rotations[0], np.zeros(3)),
            ("rotation_2", rotations[1], np.zeros(3)),
            ("reflection", np.diag([-1., 1., 1.]), np.zeros(3)),
            ("inversion", -np.eye(3), np.zeros(3)),
            ("improper_with_translation", -rotations[0], np.array([-.311, .619, .173]))]


def physical_e3_case(pos, mask, baseline, orthogonal, translation, radius):
    """Transform both lattice and Cartesian atoms, re-encode, then lift vectors."""
    assert max_abs(orthogonal.T @ orthogonal - np.eye(3)) < 1e-12
    basis, fractions = baseline["cell"][0], baseline["frac"][0]
    physical_basis = basis @ orthogonal
    cartesian = (fractions @ basis) @ orthogonal + translation
    changed_fractions = cartesian @ np.linalg.inv(physical_basis)
    offsets = np.zeros_like(fractions, dtype=np.int64)
    slots = ~mask[0].numpy()
    offsets[slots] = -np.floor(changed_fractions[slots]).astype(np.int64)
    changed_fractions[slots] += offsets[slots]
    changed = checked_records(pack_cell(physical_basis, changed_fractions, pos.dtype), mask, radius)
    lifted, frame_error = lift_frame(changed, physical_basis)
    mapped = lifted["keys"].copy()
    mapped[:, 3:] += offsets[mapped[:, 2]] - offsets[mapped[:, 1]]
    expected = dict(baseline, r=baseline["r"] @ orthogonal)
    return compare_records(expected, dict(lifted, keys=mapped), radius, TOLERANCES[pos.dtype]) | {
        "frame_orthogonality_error": frame_error, "det_Q": float(np.linalg.det(orthogonal))}


def representation_case(pos, mask, baseline, kind, radius=6.0):
    """Atom order, integer coordinate images, and det +/-1 lattice re-expression."""
    fractions, basis = baseline["frac"][0], baseline["cell"][0]
    rng = np.random.default_rng(17)
    frame_error = 0.0
    if kind == "permutation":
        order = rng.permutation(len(fractions))
        changed_pos = torch.cat([pos[:, :2], pos[:, 2:][:, order]], dim=1)
        changed = checked_records(changed_pos, mask[:, order], radius)
        mapped = changed["keys"].copy()
        mapped[:, 1:3] = order[mapped[:, 1:3]]
    elif kind == "integer_images":
        offsets = rng.integers(-2, 3, size=fractions.shape)
        changed_pos = pos.clone()
        changed_pos[:, 2:] += torch.as_tensor(offsets, dtype=pos.dtype)
        changed = checked_records(changed_pos, mask, radius)
        mapped = changed["keys"].copy()
        mapped[:, 3:] += offsets[mapped[:, 2]] - offsets[mapped[:, 1]]
    else:
        transform = {"basis_swap": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                     "basis_shear": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])}[kind]
        changed_fractions = fractions @ np.linalg.inv(transform)
        offsets = np.zeros_like(fractions, dtype=np.int64)
        slots = ~mask[0].numpy()
        offsets[slots] = -np.floor(changed_fractions[slots]).astype(np.int64)
        changed_fractions[slots] += offsets[slots]
        physical_basis = transform @ basis
        changed = checked_records(pack_cell(physical_basis, changed_fractions, pos.dtype), mask, radius)
        changed, frame_error = lift_frame(changed, physical_basis)
        mapped = changed["keys"].copy()
        mapped[:, 3:] = (mapped[:, 3:] + offsets[mapped[:, 2]] - offsets[mapped[:, 1]]) @ transform
    return compare_records(baseline, dict(changed, keys=mapped), radius, TOLERANCES[pos.dtype]) | {
        "frame_orthogonality_error": frame_error}


def supercell_cases(pos, mask, baseline, radius=6.0):
    """A 2x1x1 supercell: compare each receiver copy's complete local records."""
    slots = np.flatnonzero(~mask[0].numpy())
    transform = np.diag([2, 1, 1])
    replicas = np.repeat([[0, 0, 0], [1, 0, 0]], len(slots), axis=0)
    original = np.tile(slots, 2)
    fractions = (baseline["frac"][0, original] + replicas) @ np.linalg.inv(transform)
    physical_basis = transform @ baseline["cell"][0]
    changed = checked_records(pack_cell(physical_basis, fractions, pos.dtype),
                              torch.zeros(1, len(original), dtype=torch.bool), radius)
    changed, frame_error = lift_frame(changed, physical_basis)
    results = []
    for copy in range(2):
        select = (changed["keys"][:, 1] // len(slots)) == copy
        mapped = changed["keys"][select].copy()
        dst, src = mapped[:, 1].copy(), mapped[:, 2].copy()
        mapped[:, 3:] = mapped[:, 3:] @ transform + replicas[src] - replicas[dst]
        mapped[:, 1], mapped[:, 2] = original[dst], original[src]
        actual = {"keys": mapped, "d": changed["d"][select], "r": changed["r"][select]}
        results.append(compare_records(baseline, actual, radius, TOLERANCES[pos.dtype]) | {
            "copy": copy, "supercell_total_records": len(changed["keys"]),
            "frame_orthogonality_error": frame_error})
    return results


def prior_counts(directory, radius):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["splits"] == list(SPLITS)
    path = directory / "structures.csv.gz"
    assert sha256_file(path) == manifest["artifacts_sha256"][path.name]
    counts, old_zero = {}, {split: [] for split in SPLITS}
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            split, index = row["split"], int(row["index"])
            if float(row["radius_A"]) == radius:
                counts[split, index] = int(row["n_edges"])
            if float(row["radius_A"]) == 5.5 and int(row["zero_neighbor_atoms"]):
                old_zero[split].append(index)
    return counts, old_zero, {str(path.relative_to(ROOT)): sha256_file(path)}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/train4ARPAT")
    parser.add_argument("--radius-audit-dir", type=Path, default=ROOT / "results/periodic_neighbor_radius_q1_20261003")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--samples-per-split", type=int, default=64)
    args = parser.parse_args(argv)
    positive_values([args.radius], "radius")
    if args.samples_per_split <= 0:
        parser.error("sample size must be positive")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    torch.set_num_threads(1)
    start = time.perf_counter()
    loaded, input_hashes = load_structures(args.data_dir)
    counts, old_zero, prior_hashes = prior_counts(args.radius_audit_dir, args.radius)
    sources = [Path(__file__), ROOT / "tests/test_periodic_geometry_acceptance.py",
               ROOT / "tools/eval/periodic_neighbor_radius_audit.py", ROOT / "model/periodic_manybody.py",
               ROOT / "utils/g2_periodic_edges.py", ROOT / "utils/relative_features.py",
               ROOT / "tools/eval/g2_edge_audit.py", ROOT / "tools/eval/element_identity_preflight.py"]
    args.out_dir.mkdir(parents=True)
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "code_version": code_version(), "data_version": "Q1", "splits": list(SPLITS),
                "labels_loaded": False, "model_executed": False, "training_started": False,
                "radius_A": args.radius, "device": "cpu", "threads": 1, "torch": torch.__version__,
                "tolerance_A": {str(dtype): value for dtype, value in TOLERANCES.items()},
                "frame_tolerance": FRAME_TOLERANCE, "input_sha256": input_hashes,
                "prior_artifact_sha256": prior_hashes,
                "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
                "oracle": "float64, per-axis reciprocal-column bound, per-atom wrapping; no production K",
                "boundary_policy": "strict d<R unchanged; unmatched records within dtype tolerance of R are separately limited",
                "frame_policy": "physical A->A Q, x->x Q+t; scalar encoding; independently supplied canonical-to-physical frame",
                "guarantee_scope": "finite geometry evidence; no full-model E(3) or architecture guarantee"}
    write_json(args.out_dir / "manifest.json", manifest)
    full, sample_indices, supercell_indices = {}, {}, {}
    fields = ["split", "index", "id", "n_edges", "n_atoms", "zero_neighbor_atoms", "max_degree",
              "cross_cell_self_records", "near_cutoff_records", "max_reverse_vector_error_A",
              "max_reverse_distance_error_A", "max_norm_error_A", "max_formula_error_A"]
    with (args.out_dir / "full_record_contracts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, (positions, masks, ids, smin) in loaded.items():
            metrics = []
            for index in range(len(positions)):
                try:
                    records = checked_records(torch.from_numpy(positions[index:index+1]).float(),
                                              torch.from_numpy(masks[index:index+1]), args.radius)
                    assert records["metrics"]["n_edges"] == counts[split, index], "prior radius count mismatch"
                except (AssertionError, ValueError, RuntimeError) as error:
                    write_json(args.out_dir / "failure.json", {"split": split, "index": index,
                               "id": str(ids[index]), "stage": "full_contracts", "error": str(error)})
                    raise
                metrics.append(records["metrics"])
                writer.writerow({"split": split, "index": index, "id": str(ids[index]), **records["metrics"]})
                if (index+1) % 1000 == 0 or index+1 == len(positions):
                    print(f"record contracts {split}: {index+1}/{len(positions)}", flush=True)
            full[split] = {"structures": len(metrics), "total_edges": sum(m["n_edges"] for m in metrics),
                           "near_cutoff_records": sum(m["near_cutoff_records"] for m in metrics),
                           "zero_neighbor_atoms": sum(m["zero_neighbor_atoms"] for m in metrics),
                           **{field: max(m[field] for m in metrics) for field in fields if field.startswith("max_")}}
            cell, _ = build_cell_from_lattice(torch.from_numpy(positions).float())
            singular = torch.linalg.svdvals(cell).numpy()
            critical = [int(np.argmax((~masks).sum(axis=1))), int(np.argmin(smin)),
                        int(np.argmax(singular[:, 0]/singular[:, -1])),
                        int(np.argmax([m["n_edges"] for m in metrics])),
                        int(np.argmax([m["near_cutoff_records"] for m in metrics])),
                        int(np.argmin((~masks).sum(axis=1)))]
            sample_indices[split] = np.unique(np.r_[np.linspace(0, len(positions)-1,
                min(args.samples_per_split, len(positions)), dtype=int), critical, old_zero[split]]).tolist()
            supercell_indices[split] = sorted(set(critical[:4] + [int(np.argmin((~masks).sum(axis=1)))]))
    cases = []

    def execute(split, index, identity, dtype, group, name, operation):
        row = {"split": split, "index": index, "id": str(identity), "dtype": str(dtype),
               "group": group, "case": name}
        try:
            row.update(operation())
        except (AssertionError, ValueError, RuntimeError) as error:
            row.update(status="fail", error=str(error))
        cases.append(row)

    for split, indices in sample_indices.items():
        positions, masks, ids, _ = loaded[split]
        for number, index in enumerate(indices):
            mask = torch.from_numpy(masks[index:index+1])
            baselines = {}
            for dtype in (torch.float32, torch.float64):
                pos = torch.from_numpy(positions[index:index+1]).to(dtype)
                baseline = checked_records(pos, mask, args.radius)
                baselines[dtype] = baseline
                execute(split, index, ids[index], dtype, "enumeration", "independent_reference",
                        lambda: compare_records(independent_reference(baseline["cell"], baseline["frac"],
                            mask.numpy(), args.radius), baseline, args.radius, TOLERANCES[dtype]))
                for name, orthogonal, translation in e3_operations():
                    execute(split, index, ids[index], dtype, "E3", name,
                            lambda: physical_e3_case(pos, mask, baseline, orthogonal, translation, args.radius))
                for kind in ("permutation", "integer_images", "basis_swap", "basis_shear"):
                    execute(split, index, ids[index], dtype, "representation", kind,
                            lambda: representation_case(pos, mask, baseline, kind, args.radius))
                if index in supercell_indices[split]:
                    for copy in range(2):
                        execute(split, index, ids[index], dtype, "supercell", f"receiver_copy_{copy}",
                                lambda: supercell_cases(pos, mask, baseline, args.radius)[copy])
            execute(split, index, ids[index], torch.float32, "precision", "float32_vs_float64",
                    lambda: compare_records(baselines[torch.float64], baselines[torch.float32],
                                            args.radius, TOLERANCES[torch.float32]))
            if (number+1) % 10 == 0 or number+1 == len(indices):
                print(f"geometry sample {split}: {number+1}/{len(indices)}; failures="
                      f"{sum(row['status']=='fail' for row in cases)}", flush=True)
    write_json(args.out_dir / "cases.json", cases)
    case_fields = sorted(set().union(*(row.keys() for row in cases)))
    with (args.out_dir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_fields)
        writer.writeheader()
        writer.writerows(cases)
    groups = {}
    for group in sorted(set(row["group"] for row in cases)):
        rows = [row for row in cases if row["group"] == group]
        groups[group] = {"cases": len(rows), "failed": sum(row["status"] == "fail" for row in rows),
                         "boundary_limited": sum(row["status"] == "boundary_limited" for row in rows),
                         **{field: max(row.get(field, 0.0) for row in rows) for field in
                            ("max_distance_error_A", "max_vector_error_A", "frame_orthogonality_error")},
                         "boundary_missing": sum(row.get("boundary_missing", 0) for row in rows),
                         "boundary_extra": sum(row.get("boundary_extra", 0) for row in rows)}
    failed = [row for row in cases if row["status"] == "fail"]
    limited = [row for row in cases if row["status"] == "boundary_limited"]
    summary = {"outcome": "needs_changes" if failed else "boundary_limited" if limited else "pass",
               "full_record_contracts": full, "sample_indices": sample_indices,
               "supercell_indices": supercell_indices, "groups": groups,
               "failed_cases": failed, "boundary_limited_cases": limited,
               "model_invariance_verified": False, "architecture_guarantee_established": False}
    write_json(args.out_dir / "summary.json", summary)
    manifest.update(status="complete", outcome=summary["outcome"], elapsed_seconds=time.perf_counter()-start,
                    peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    artifacts_sha256={path.name: sha256_file(path) for path in args.out_dir.iterdir()
                                      if path.name != "manifest.json"})
    write_json(args.out_dir / "manifest.json", manifest)
    print(f"completed: {summary['outcome']}; {len(cases)} cases; {manifest['elapsed_seconds']:.1f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
