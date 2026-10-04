"""Follow-up controls using the exact frozen weights of a completed precheck.

Angle comparisons hold radial fields fixed. Resource comparisons run one route
in a fresh CPU process, separating RSS from the other route's earlier peak.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.element_identity_preflight import sha256_file
from tools.eval.periodic_geometry_acceptance import pack_cell, write_json
from tools.eval.periodic_message_precheck import (
    BudgetMonitor, SEED, local_forward, measure_resources, state_digest, synthetic_star,
)
from tools.eval.periodic_message_prototypes import PeriodicMessageProbe, ProbeZP, geometry_from_pos, prepare_geometry


def recreate_probes(manifest, dtype):
    """Replay documented seeds; require exact parameter/buffer hashes."""
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
    models = {route: copy.deepcopy(model).to(dtype).eval() for route, model in templates.items()}
    entry = copy.deepcopy(entry).to(dtype)
    for route, model in models.items():
        assert state_digest(model) == manifest["initial_state_sha256"][f"{dtype}/{route}"]
    assert state_digest(entry) == manifest["initial_state_sha256"][f"{dtype}/zp"]
    return models, entry


def angle_control(manifest, budget):
    rows = []
    for dtype in (torch.float32, torch.float64):
        models, entry = recreate_probes(manifest, dtype)
        for distance in (5., 5.9):
            pos, mask, elements = synthetic_star(False, distance, dtype)
            changed, _, _ = synthetic_star(True, distance, dtype)
            first, raw = geometry_from_pos(pos, mask), geometry_from_pos(changed, mask)
            for key in ("batch", "dst", "src", "shifts"):
                assert torch.equal(first.edges[key], raw.edges[key])
            # Directions change, while radial fields, graph IDs and counts are
            # exactly shared. Rescaling preserves unit directions and d=||r||.
            edges = dict(raw.edges, distances=first.edges["distances"])
            controlled = prepare_geometry(edges, raw.unit*first.edges["distances"][:, None], mask)
            assert torch.equal(first.radial, controlled.radial)
            assert torch.equal(first.cutoff, controlled.cutoff)
            assert torch.equal(first.degree, controlled.degree)
            identity = prepare_geometry(edges, first.vectors, mask)
            with torch.no_grad():
                atoms = entry(elements, mask)
            for route, model in models.items():
                budget.check()
                before = state_digest(model)
                baseline, bdebug = local_forward(model, atoms, mask, first)
                output, debug = local_forward(model, atoms, mask, controlled)
                equal = local_forward(model, atoms, mask, identity)[0]
                delta = float((baseline[0, 0]-output[0, 0]).abs().max())
                identity_error = float((baseline-equal).abs().max())
                threshold = 1e-8 if dtype == torch.float32 else 1e-10
                state = debug["states"][-1]
                previous = bdebug["states"][-1]
                scalar_delta = float((previous[0, 0, :64]-state[0, 0, :64]).abs().max())
                direction_delta = float((previous[0, 0, 64:]-state[0, 0, 64:]).abs().max()) if route == "equivariant" else None
                unchanged = before == state_digest(model)
                rows.append({"category": "fixed_radial_angle_response", "route": route, "dtype": str(dtype), "distance_A": distance,
                             "raw_radial_distance_max_difference_A": float((first.edges["distances"]-raw.edges["distances"]).abs().max()),
                             "radial_fields_bitwise_identical": True, "degree_identical": True,
                             "max_scalar_output_change": delta, "last_scalar_state_change": scalar_delta,
                             "last_direction_state_change": direction_delta,
                             "identity_control_error": identity_error, "parameters_unchanged": unchanged,
                             "response_threshold": threshold, "response_resolved": delta > threshold,
                             "status": "pass" if identity_error == 0 and unchanged and (distance != 5. or delta > threshold) else "fail"})
    # Two different inversion-symmetric 12-direction sets with equal l<=2
    # moments. All leaf pairs stay beyond 6 A, so both graphs are pure stars.
    # A cutoff=1 ablation separates the representational limit from suppression
    # of very weak near-cutoff angular content. This is not an adopted formula.
    dtype = torch.float64
    models, entry = recreate_probes(manifest, dtype)
    geometries, moments, fourth = [], [], []
    mask = torch.zeros(1, 13, dtype=torch.bool)
    elements = torch.full((1, 13), 6, dtype=torch.long)
    for ratio in (1.5, (1+np.sqrt(5))/2):
        points = np.array([[0, a, b*ratio] for a, b in itertools.product((-1, 1), repeat=2)])
        directions = np.concatenate((points, points[:, [2, 0, 1]], points[:, [1, 2, 0]]))/np.sqrt(1+ratio**2)
        fractions = np.r_[np.zeros((1, 3)), directions*5.85]/30+.5
        g = geometry_from_pos(pack_cell(np.eye(3)*30, fractions, dtype), mask)
        assert g.degree.tolist() == [[12]+[1]*12]
        if geometries:
            first = geometries[0]
            for key in ("batch", "dst", "src", "shifts"):
                assert torch.equal(first.edges[key], g.edges[key])
            g = prepare_geometry(dict(g.edges, distances=first.edges["distances"]),
                                 g.unit*first.edges["distances"][:, None], mask)
        geometries.append(replace(g, cutoff=torch.ones_like(g.cutoff),
                                  effective_count=g.degree.to(dtype)))
        central = g.unit[g.edges["dst"] == 0]
        moments.append({"vector_sum_norm": float(central.sum(0).norm()),
                        "quadrupole_norm_sq": float((central.T@central-4*torch.eye(3)).square().sum())})
        q = central@central.T
        fourth.append(float(q.pow(4)[torch.triu_indices(12, 12, offset=1).unbind()].sum()))
    with torch.no_grad():
        atoms = entry(elements, mask)
    for route, model in models.items():
        budget.check()
        first = local_forward(model, atoms, mask, geometries[0])
        second = local_forward(model, atoms, mask, geometries[1])
        delta = float((first[0][0, 0]-second[0][0, 0]).abs().max())
        rows.append({"category": "equal_low_order_moments", "route": route, "dtype": str(dtype),
                     "distance_A": 5.85, "cutoff_override": "one, controlled ablation only",
                     "center_degree": 12, "low_order_moments": moments, "angle_fourth_sums": fourth,
                     "max_scalar_output_change": delta,
                     "status": "pass" if (delta > 1e-8 if route == "invariant" else delta < 2e-10) else "fail",
                     "interpretation": "A resolves this constructed pair; this l<=2 B cannot resolve it in the pure-star setting"})
    return rows


def resource_control(manifest, route, data_dir, budget):
    dtype = torch.float32
    models, entry = recreate_probes(manifest, dtype)
    sample = next(row for row in manifest["selected_samples"] if row["split"] == "train" and row["reason"] == "max_pairs")
    index = sample["index"]
    for key in ("elements_train.npy", "positions_train.npy"):
        relative = f"train/{key}"
        assert sha256_file(data_dir/relative) == manifest["input_sha256"][relative]
    position = np.load(data_dir/"train/positions_train.npy", mmap_mode="r").reshape(-1, 82, 3)
    elements = np.load(data_dir/"train/elements_train.npy", mmap_mode="r")[:, 2:]
    pos = torch.tensor(position[index:index+1], dtype=dtype)
    atom_ids = torch.tensor(elements[index:index+1], dtype=torch.long)
    mask = atom_ids == 0
    geometry = geometry_from_pos(pos, mask)
    assert len(geometry.cutoff) == sample["n_edges"]
    with torch.no_grad():
        atoms = entry(atom_ids, mask)
    model = models[route]
    local_forward(model, atoms, mask, geometry)  # One unrecorded warm-up.
    metrics = measure_resources(model, atoms, mask, geometry, budget)
    return {"route": route, "dtype": str(dtype), "sample": sample, **metrics,
            "status": "pass" if metrics["pass"] else "fail",
            "rss_scope": "fresh process, shared library/probe construction included; this route alone runs backward"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/train4ARPAT")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("angles", "invariant", "equivariant"), required=True)
    parser.add_argument("--budget-seconds", type=float, default=100.)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if not 0 < args.budget_seconds <= 100:
        parser.error("control budget must be in (0,100] seconds")
    manifest_path = args.precheck_dir/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete", "wait for the precheck to finish"
    for relative, digest in manifest["source_sha256"].items():
        assert sha256_file(ROOT/relative) == digest, f"precheck source changed: {relative}"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    with BudgetMonitor(args.budget_seconds, 2048.) as budget:
        result = angle_control(manifest, budget) if args.mode == "angles" else resource_control(manifest, args.mode, args.data_dir, budget)
        budget.check()
    output = {"mode": args.mode, "precheck_manifest_sha256": sha256_file(manifest_path),
              "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__))},
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "sampled_peak_rss_MiB": budget.peak_mib,
              "optimizer_created": False, "parameter_updates": 0, "result": result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, output)
    print(json.dumps({key: output[key] for key in ("mode", "elapsed_seconds", "peak_rss_MiB")}), flush=True)
    passed = all(row["status"] == "pass" for row in result) if isinstance(result, list) else result["status"] == "pass"
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
