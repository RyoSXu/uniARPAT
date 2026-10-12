"""Hard-pair attribution for the C1 small-sample spectral-difference fit.

Phase ``inputs`` (CPU) re-verifies data correspondence (material ID, full
structure, spectra labels, normalization, energy grid), records label
provenance (missing fields stay "unknown"), and compares the periodic
neighbor / element / distance information that C1 actually consumes for the
two hard pairs and two matched reference pairs of the 2026-10-12 run.

Phase ``model`` (GPU) isolates a faithful reproduction of that disposable
600-step fitting run (same selection, init, losses and budget), saves the
diagnostic weights the original run discarded, checks the reproduced
trajectory against the archived trace, and attributes pair differences
through local update / global layer / decoder tokens / final output plus the
per-loss gradient geometry on local/global/decoder parameter groups.

No production checkpoint, model formula, data split or Git state is modified.
Train-only, seed 42. Pooled summaries never replace the saved full tensors.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scipy.optimize import linear_sum_assignment

from model.losses import edos_pair_contrast_loss, sumnorm_klw_loss
from tools.eval.lg_structure_diagnosis import pair_errors
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position
from tools.eval.lg_valid_export import load_run, select_best
from tools.eval.local_structure_small_fit import (
    evaluate, new_model, select_pairs)
from tools.train import configure_local_global_execution, setup_ablation_seed
from utils.bucket_batch import crystal_collate
from utils.builder import ConfigBuilder
from utils.crystal import CrystalGeometry, atom_padding_mask
from utils.paths import file_sha256
from utils.periodic_geometry import build_periodic_records
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash

DEFAULT_RUN = ROOT / "results/runs/ablation_m1_lsc_attrib20261009_k2_g6_c1_staged_s42_e35"
DEFAULT_PAIRS = ROOT / "results/runs/stage_readout_diagnosis_c1c2_20261010/pairs_train_absolute.csv"
ARCHIVED_FIT = ROOT / "results/runs/local_structure_small_fit_c1_s42_20261012"
SEED = 42
LOCAL_RADIUS = 6.0
FOCUS = [
    ("mp-aaaaaafe", "mp-aaacevps", "hard_H1"),
    ("mp-aaabipws", "mp-aaabjbnl", "hard_H2"),
    ("mp-aaabijzd", "mp-aaacfxzp", "ref_R1"),
    ("mp-aaaaaity", "mp-aaabkqhp", "ref_R2"),
]
RAW_ROOT = Path("/root/home/newstudy/getdata/raw")
PARAM_GROUPS = ("atom_entry", "local", "global", "decoder", "head")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def material_records(row_index, dataset):
    """Full stored structure and labels for one train row.

    Dataset rows map to packed store rows through ``structure_indices`` (the
    train view drops two quality-isolated materials); labels are stored in
    dataset-row order while structures live in store-row order.
    """
    store_row = int(dataset.structure_indices[row_index])
    numbers, geometry = dataset.crystals[store_row]
    cell = geometry.cell.numpy().astype(np.float64)
    frac = geometry.frac_coords.numpy().astype(np.float64)
    numbers = numbers.numpy().astype(int)
    lengths = np.linalg.norm(cell, axis=1)
    angles = [
        float(np.degrees(np.arccos(np.clip(
            np.dot(cell[a], cell[b]) / (lengths[a] * lengths[b]), -1, 1))))
        for a, b in ((1, 2), (0, 2), (0, 1))
    ]
    return {
        "train_row": int(row_index),
        "store_row": store_row,
        "material_id": str(dataset.material_ids[row_index]),
        "n_atoms": int(len(numbers)),
        "atomic_numbers": numbers,
        "cell": cell,
        "frac_coords": frac,
        "lattice_lengths": lengths.tolist(),
        "lattice_angles_deg": angles,
        "volume_A3": float(abs(np.linalg.det(cell))),
        "edos_target": dataset.edos_tgtdos[row_index].numpy().astype(np.float64),
        "phdos_target": dataset.phdos_tgtdos[row_index].numpy().astype(np.float64),
        "edos_mask": dataset.edos_mask[row_index].numpy().astype(bool),
        "phdos_mask": dataset.phdos_mask[row_index].numpy().astype(bool),
        "nvalence": None if dataset.nvalence is None else float(dataset.nvalence[row_index]),
    }


def edge_arrays(numbers, cell, frac, radius=LOCAL_RADIUS):
    """The periodic neighbor records C1 consumes, as raw physical fields."""
    geometry = CrystalGeometry(torch.tensor(cell, dtype=torch.float32).unsqueeze(0),
                               torch.tensor(frac, dtype=torch.float32).unsqueeze(0))
    mask = torch.zeros(1, len(numbers), dtype=torch.bool)
    records = build_periodic_records(geometry, mask, radius=radius)
    dst = records.dst.numpy().astype(int)
    src = records.src.numpy().astype(int)
    z = np.asarray(numbers, dtype=int)
    order = np.lexsort((records.distances.numpy(), z[src], z[dst]))
    return {
        "z_dst": z[dst][order],
        "z_src": z[src][order],
        "distances": records.distances.numpy()[order],
        "shifts": records.shifts.numpy()[order],
        "vectors": records.vectors.numpy()[order],
        "n_edges": int(len(order)),
    }


def fingerprint(arrays):
    rounded = np.column_stack((arrays["z_dst"], arrays["z_src"],
                               np.round(arrays["distances"], 4)))
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(rounded).tobytes())
    return digest.hexdigest()


def compare_edges(left, right):
    """Permutation-invariant comparison of two C1 edge multisets."""
    result = {
        "edge_count_left": left["n_edges"], "edge_count_right": right["n_edges"],
        "multiset_equal": None, "max_distance_delta": None,
        "pooled_hist_tv": None, "per_element_pair": {},
    }
    if left["n_edges"] == right["n_edges"]:
        same = (np.array_equal(left["z_dst"], right["z_dst"])
                and np.array_equal(left["z_src"], right["z_src"]))
        delta = float(np.abs(left["distances"] - right["distances"]).max()) \
            if left["n_edges"] else 0.0
        result["multiset_equal"] = bool(same and delta <= 2e-4)
        result["max_distance_delta"] = delta
        if same:
            result["mean_abs_distance_delta"] = float(
                np.abs(left["distances"] - right["distances"]).mean())
    else:
        result["multiset_equal"] = False
    bins = np.arange(0.0, LOCAL_RADIUS + 0.1, 0.1)
    groups = sorted(set(zip(left["z_dst"], left["z_src"]))
                    | set(zip(right["z_dst"], right["z_src"])))
    for key in groups:
        d_left = left["distances"][(left["z_dst"] == key[0]) & (left["z_src"] == key[1])]
        d_right = right["distances"][(right["z_dst"] == key[0]) & (right["z_src"] == key[1])]
        h_left = np.histogram(d_left, bins=bins)[0].astype(float)
        h_right = np.histogram(d_right, bins=bins)[0].astype(float)
        p = h_left / max(h_left.sum(), 1.0)
        q = h_right / max(h_right.sum(), 1.0)
        result["per_element_pair"][f"Z{key[0]}->Z{key[1]}"] = {
            "n_left": int(len(d_left)), "n_right": int(len(d_right)),
            "hist_tv": float(0.5 * np.abs(p - q).sum()),
            "mean_distance_left": float(d_left.mean()) if len(d_left) else None,
            "mean_distance_right": float(d_right.mean()) if len(d_right) else None,
        }
    h_left = np.histogram(left["distances"], bins=bins)[0].astype(float)
    h_right = np.histogram(right["distances"], bins=bins)[0].astype(float)
    p = h_left / max(h_left.sum(), 1.0)
    q = h_right / max(h_right.sum(), 1.0)
    result["pooled_hist_tv"] = float(0.5 * np.abs(p - q).sum())
    return result


def raw_dos_task(mpid, dos_task_id):
    """Locate the label's own DOS task record and its embedded structure."""
    out = {"mpid": mpid, "dos_task_id": dos_task_id, "found": False}
    with open(RAW_ROOT / "mp_edos_raw.jsonl") as handle:
        for line in handle:
            if dos_task_id not in line:
                continue
            rec = json.loads(line)
            if rec.get("dos_task_id") != dos_task_id:
                continue
            item = rec["edos_raw"][0]
            lattice = np.asarray(item["structure"]["lattice"]["matrix"], dtype=float)
            species, coords = [], []
            for site in item["structure"]["sites"]:
                species.append(site["species"][0]["element"])
                coords.append(site["abc"])
            out.update({
                "found": True,
                "run_type": item.get("run_type"),
                "spin_up_densities_present": item.get("spin_up_densities") is not None,
                "spin_down_densities_present": item.get("spin_down_densities") is not None,
                "efermi": item.get("efermi"),
                "label_cell": lattice,
                "label_frac": np.asarray(coords, dtype=float),
                "label_species": species,
            })
            return out
    return out


def raw_phonon_record(mpid):
    with open(RAW_ROOT / "mp_phdos_raw.jsonl") as handle:
        for line in handle:
            if mpid not in line:
                continue
            rec = json.loads(line)
            if rec.get("mpid") == mpid:
                items = rec.get("phdos_raw") or []
                methods = sorted({item.get("method") for item in items})
                return {"found": True, "methods": methods, "n_records": len(items)}
    return {"found": False}


def _task_index_row(task_index, identifier):
    """Per-DOS-task structure metadata from the frozen delta index."""
    rows = task_index[task_index.identifier == identifier]
    if len(rows) != 1:
        return {"found": False, "rows": int(len(rows))}
    row = rows.iloc[0]
    return {"found": True, "nsites": int(row.nsites), "volume_A3": float(row.volume),
            "species": str(row.species), "run_type": row.run_type}


def provenance_rows(mids):
    """Available label provenance; anything absent stays explicitly unknown."""
    table = pd.read_parquet(RAW_ROOT / "v2_processed.parquet")
    task_index = pd.read_parquet(RAW_ROOT / "delta_edos_index.parquet")
    summaries, dos_tasks = {}, {}
    with open(RAW_ROOT / "mp_increment_raw.jsonl") as handle:
        for line in handle:
            if not any(mid in line for mid in mids):
                continue
            rec = json.loads(line)
            if rec.get("mpid") in mids:
                summaries[rec["mpid"]] = rec
    out = {}
    for mid in mids:
        rows = table[table.mpid == mid]
        if len(rows) != 1:
            out[mid] = {"known": False, "reason": f"parquet rows={len(rows)}"}
            continue
        row = rows.iloc[0]
        task = raw_dos_task(mid, row.edos_ref.split(":")[-1])
        summary = summaries.get(mid, {})
        mp_summary = summary.get("mp_summary") or {}
        sym = mp_summary.get("symmetry") or {}
        rec = {
            "known": True,
            "edos_source": row.src_edos,
            "edos_task_ref": row.edos_ref,
            "phdos_source": row.src_ph,
            "phdos_task_ref": row.ph_ref,
            "phdos_weight": float(row.ph_weight),
            "struct_ref": row.struct_ref,
            "provenance": json.loads(row.prov),
            "split": row.split,
            "space_group_number": int(row.sg),
            "crystal_system": row.crystal,
            "nsites_prim": int(row.nsites_prim),
            "truncated_flag": bool(row.truncated),
            "band_gap": mp_summary.get("band_gap"),
            "efermi_summary": mp_summary.get("efermi"),
            "energy_above_hull": mp_summary.get("energy_above_hull"),
            "formation_energy_per_atom": mp_summary.get("formation_energy_per_atom"),
            "space_group_symbol": sym.get("symbol"),
            "mp_db_version": (summary.get("provenance") or {}).get("mp_db"),
            "fetched_at": (summary.get("provenance") or {}).get("fetched_at"),
            "edos_task": {k: v for k, v in task.items()
                          if k not in ("label_cell", "label_frac", "label_species")},
            "edos_task_index": _task_index_row(task_index, row.edos_ref.split(":")[-1]),
            "phdos_task": raw_phonon_record(mid),
        }
        rec["magnetic_state"] = "unknown: no local metadata field"
        rec["calculation_conditions"] = {
            "run_type": task.get("run_type",
                                 rec["edos_task_index"].get("run_type")),
            "spin_up_densities_present": task.get("spin_up_densities_present"),
            "spin_down_densities_present": task.get("spin_down_densities_present"),
            "kpoints": "unknown", "cutoff": "unknown", "smearing": "unknown",
            "functional_detail": "unknown beyond run_type",
            "relaxation_state": "unknown",
        }
        out[mid] = rec
        dos_tasks[mid] = task
    return out, dos_tasks


def load_context(args):
    saved, params, contract = load_run(args.run_dir)
    if params.get("local_route") != "C1" or saved["cli"]["norm"] != "sumnorm":
        raise ValueError("hard-pair diagnosis requires the existing C1 SumNorm run")
    checkpoint_path = args.run_dir / "checkpoint_best.pth"
    picked = select_best(args.run_dir, contract["selection_metric"], saved)
    weights = picked["checkpoint"]["model"]
    epoch = int(picked["checkpoint"]["epoch"])
    del picked
    dataset = ConfigBuilder(**deepcopy(saved["config"])).get_dataset(
        split="train", dos_minmax=True, dos_sumnorm=True)
    if dataset.augment:
        raise ValueError("diagnosis requires immutable structures")
    frame = pd.read_csv(args.pair_table)
    selected, eligibility = select_pairs(frame, dataset.atom_counts,
                                         args.pairs, args.seed, args.max_atoms)
    return saved, params, contract, weights, epoch, dataset, selected, eligibility


def focus_table(selected, archived_pairs):
    """Resolve the four focus pairs against the archived per-pair fit results."""
    rows = []
    for left, right, tag in FOCUS:
        match = selected[(selected.left_id == left) & (selected.right_id == right)]
        if len(match) != 1:
            raise ValueError(f"focus pair not in locked selection: {left}/{right}")
        fit = archived_pairs[(archived_pairs.left_id == left)
                             & (archived_pairs.right_id == right)]
        if len(fit) != 1:
            raise ValueError(f"focus pair missing archived fit row: {left}/{right}")
        row = match.iloc[0]
        fit_row = fit.iloc[0]
        rows.append({
            "tag": tag, "left_id": left, "right_id": right,
            "left_row": int(row.i), "right_row": int(row.j),
            "composition": str(row.composition),
            "target_tv": float(fit_row.target_tv),
            "control_error": float(fit_row.control_error),
            "control_skill": float(fit_row.control_skill),
            "candidate_error": float(fit_row.candidate_error),
            "candidate_skill": float(fit_row.candidate_skill),
            "pooled_radial_proxy_gap": float(fit_row.pooled_radial_proxy_gap),
        })
    return pd.DataFrame(rows)


def run_inputs(args):
    out = args.out
    if out.exists():
        raise FileExistsError(f"refusing existing output directory: {out}")
    out.mkdir(parents=True)
    start = time.monotonic()
    _saved, _params, _contract, weights, epoch, dataset, selected, eligibility = load_context(args)
    immutable = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in (args.run_dir / "checkpoint_best.pth", args.run_dir / "config_used.yaml",
                  args.pair_table, ARCHIVED_FIT / "selected_train_pairs.csv",
                  ARCHIVED_FIT / "pair_fit_with_geometry.csv", ARCHIVED_FIT / "targets.npz")
    }
    archived_selected = pd.read_csv(ARCHIVED_FIT / "selected_train_pairs.csv")
    selection_equal = selected.equals(archived_selected)
    if not selection_equal:
        raise ValueError("re-derived selection differs from the archived 2026-10-12 selection")
    archived_pairs = pd.read_csv(ARCHIVED_FIT / "pair_fit_with_geometry.csv")
    focus = focus_table(selected, archived_pairs)
    focus.to_csv(out / "focus_pairs.csv", index=False)

    rows = selected.i.tolist() + selected.j.tolist()
    ids = [str(dataset.material_ids[i]) for i in rows]
    if ids != selected.left_id.tolist() + selected.right_id.tolist():
        raise ValueError("pair materials do not match the current train partition")
    keys = composition_keys_by_position(dataset, rows)["absolute"]
    batch = crystal_collate([dataset[i] for i in rows])
    archived_targets = np.load(ARCHIVED_FIT / "targets.npz")
    target_deltas = {
        "material_ids_equal": bool(np.array_equal(
            archived_targets["material_ids"], np.asarray(ids))),
        "edos_max_abs_delta": float(np.abs(archived_targets["edos"] - batch[2].numpy()).max()),
        "phdos_max_abs_delta": float(np.abs(archived_targets["phdos"] - batch[3].numpy()).max()),
    }

    # Energy grid and normalization correspondence.
    grids = json.loads((ROOT / "data/reference/grids.json").read_text())
    edos_x = batch[15].numpy() if len(batch) > 15 and batch[15] is not None else None
    phdos_x = batch[16].numpy() if len(batch) > 16 and batch[16] is not None else None

    def grid_check(values, name):
        """Batch bin centers must equal the midpoints of the stored grid edges."""
        if values is None:
            return {"name": name, "present": False, "reference_format": "bin edges"}
        edges = np.asarray(grids[name], dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        deltas = np.abs(values - centers.reshape(1, -1))
        return {"name": name, "present": True, "reference_format": "bin edges",
                "edge_bins": int(len(edges)), "bin_centers": int(len(centers)),
                "batch_bins": int(values.shape[-1]),
                "all_batch_rows_identical": bool(
                    np.allclose(values, values[:1], atol=0, rtol=0)),
                "max_abs_delta_vs_edge_midpoints": float(deltas.max())}

    materials = []
    for idx in rows:
        rec = material_records(idx, dataset)
        materials.append(rec)
    by_id = {rec["material_id"]: rec for rec in materials}

    normalization = {
        "mode": "SumNorm: target /= row sum; dataset sumnorm path",
        "edos_sum_min": float(min(rec["edos_target"].sum() for rec in materials)),
        "edos_sum_max": float(max(rec["edos_target"].sum() for rec in materials)),
        "phdos_sum_min": float(min(rec["phdos_target"].sum() for rec in materials)),
        "phdos_sum_max": float(max(rec["phdos_target"].sum() for rec in materials)),
        "edos_negative_bins": int(sum(int((rec["edos_target"] < 0).sum()) for rec in materials)),
        "phdos_negative_bins": int(sum(int((rec["phdos_target"] < 0).sum()) for rec in materials)),
    }

    mids = {rec["material_id"] for rec in materials}
    provenance, dos_tasks = provenance_rows(mids)

    material_rows, edges, fingerprints = [], {}, {}
    for rec in materials:
        mid = rec["material_id"]
        edges[mid] = edge_arrays(rec["atomic_numbers"], rec["cell"], rec["frac_coords"])
        fingerprints[mid] = fingerprint(edges[mid])
        # Atom permutation must leave the C1 input multiset unchanged.
        perm = np.random.default_rng(SEED).permutation(rec["n_atoms"])
        perm_edges = edge_arrays(rec["atomic_numbers"][perm], rec["cell"],
                                 rec["frac_coords"][perm])
        perm_ok = fingerprint(perm_edges) == fingerprints[mid]
        # Equivalent cell: unimodular basis change keeps the physical lattice.
        basis = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cell2 = basis @ rec["cell"]
        frac2 = rec["frac_coords"] @ np.linalg.inv(basis)
        cell_edges = edge_arrays(rec["atomic_numbers"], cell2, frac2)
        cell_ok = compare_edges(edges[mid], cell_edges)["multiset_equal"]
        # The DOS task's own structure vs the stored training structure.
        task = dos_tasks.get(mid, {})
        label_check = {"label_structure_found": bool(task.get("found"))}
        if task.get("found"):
            z_lookup = {symbol: number for number, symbol in
                        ((1, "H"), (3, "Li"), (4, "Be"), (5, "B"), (6, "C"), (7, "N"),
                         (8, "O"), (9, "F"), (11, "Na"), (12, "Mg"), (13, "Al"),
                         (14, "Si"), (15, "P"), (16, "S"), (17, "Cl"), (19, "K"),
                         (20, "Ca"), (22, "Ti"), (23, "V"), (24, "Cr"), (25, "Mn"),
                         (26, "Fe"), (27, "Co"), (28, "Ni"), (29, "Cu"), (30, "Zn"),
                         (31, "Ga"), (32, "Ge"), (33, "As"), (34, "Se"), (35, "Br"),
                         (37, "Rb"), (38, "Sr"), (39, "Y"), (40, "Zr"), (41, "Nb"),
                         (42, "Mo"), (44, "Ru"), (45, "Rh"), (46, "Pd"), (47, "Ag"),
                         (48, "Cd"), (49, "In"), (50, "Sn"), (51, "Sb"), (52, "Te"),
                         (53, "I"), (55, "Cs"), (56, "Ba"), (57, "La"), (59, "Pr"),
                         (73, "Ta"), (74, "W"), (75, "Re"), (78, "Pt"), (79, "Au"),
                         (82, "Pb"), (83, "Bi"))}
            label_z = np.asarray([z_lookup[s] for s in task["label_species"]], dtype=int)
            label_edges = edge_arrays(label_z, task["label_cell"], task["label_frac"])
            label_check.update({
                "label_n_atoms": int(len(label_z)),
                "stored_n_atoms": rec["n_atoms"],
                "label_volume_A3": float(abs(np.linalg.det(task["label_cell"]))),
                "stored_volume_A3": rec["volume_A3"],
                "volume_per_atom_delta": float(
                    abs(abs(np.linalg.det(task["label_cell"])) / len(label_z)
                        - rec["volume_A3"] / rec["n_atoms"])),
                "edge_multiset_equal_to_label_structure":
                    compare_edges(edges[mid], label_edges)["multiset_equal"],
            })
        index_row = provenance.get(mid, {}).get("edos_task_index") or {}
        if index_row.get("found"):
            label_check.update({
                "task_index_found": True,
                "task_index_nsites": index_row["nsites"],
                "task_index_volume_A3": index_row["volume_A3"],
                "task_index_species": index_row["species"],
                "nsites_match_task_index": bool(index_row["nsites"] == rec["n_atoms"]),
                "volume_per_atom_delta_vs_task_index": float(abs(
                    index_row["volume_A3"] / index_row["nsites"]
                    - rec["volume_A3"] / rec["n_atoms"])),
            })
        else:
            label_check["task_index_found"] = False
        material_rows.append({
            "material_id": mid, "train_row": rec["train_row"],
            "store_row": rec["store_row"],
            "n_atoms": rec["n_atoms"],
            "composition_key": repr(keys[rows.index(rec["train_row"])]),
            "lattice_lengths": rec["lattice_lengths"],
            "lattice_angles_deg": rec["lattice_angles_deg"],
            "volume_A3": rec["volume_A3"],
            "edos_raw_sum": float(rec["edos_target"].sum()),
            "edos_mask_covered_bins": int(rec["edos_mask"].sum()),
            "phdos_mask_covered_bins": int(rec["phdos_mask"].sum()),
            "nvalence": rec["nvalence"],
            "c1_edge_count": edges[mid]["n_edges"],
            "c1_fingerprint": fingerprints[mid],
            "permutation_invariant": bool(perm_ok),
            "equivalent_cell_invariant": bool(cell_ok),
            "label_structure_check": label_check,
            "provenance": provenance.get(mid),
        })
    pd.DataFrame([{
        "material_id": row["material_id"], "train_row": row["train_row"],
        "n_atoms": row["n_atoms"], "volume_A3": row["volume_A3"],
        "c1_edge_count": row["c1_edge_count"], "c1_fingerprint": row["c1_fingerprint"],
        "permutation_invariant": row["permutation_invariant"],
        "equivalent_cell_invariant": row["equivalent_cell_invariant"],
    } for row in material_rows]).to_csv(out / "materials.csv", index=False)

    pair_rows = []
    for row in focus.to_dict("records"):
        left, right = by_id[row["left_id"]], by_id[row["right_id"]]
        cmp_edges = compare_edges(edges[row["left_id"]], edges[row["right_id"]])
        pair_rows.append({
            "tag": row["tag"], "left_id": row["left_id"], "right_id": row["right_id"],
            "target_tv": row["target_tv"],
            "control_skill": row["control_skill"], "candidate_skill": row["candidate_skill"],
            "pooled_radial_proxy_gap": row["pooled_radial_proxy_gap"],
            "c1_multiset_equal": cmp_edges["multiset_equal"],
            "c1_max_distance_delta": cmp_edges["max_distance_delta"],
            "c1_mean_abs_distance_delta": cmp_edges.get("mean_abs_distance_delta"),
            "c1_pooled_hist_tv": cmp_edges["pooled_hist_tv"],
            "edge_count_left": cmp_edges["edge_count_left"],
            "edge_count_right": cmp_edges["edge_count_right"],
            "per_element_pair": cmp_edges["per_element_pair"],
        })
    pair_frame = pd.DataFrame([{k: v for k, v in row.items() if k != "per_element_pair"}
                              for row in pair_rows])
    pair_frame.to_csv(out / "pairs_input.csv", index=False)

    np.savez_compressed(
        out / "edge_fingerprints.npz",
        **{f"{mid}__{key}": value for mid, edge in edges.items()
           for key, value in edge.items()})

    report = {
        "scope": "train-only input/provenance audit of 4 focus pairs; no model update",
        "seed": SEED, "run_dir": str(args.run_dir.relative_to(ROOT)),
        "source_epoch": epoch, "initial_state_hash": state_hash(weights),
        "selection": {**eligibility, "pairs": args.pairs,
                      "equals_archived_selection": selection_equal},
        "target_recheck": target_deltas,
        "normalization": normalization,
        "energy_grid": {
            "reference_keys": ["E0", "P0"],
            "edos_grid": grid_check(edos_x, "E0"),
            "phdos_grid": grid_check(phdos_x, "P0"),
        },
        "focus_pairs": focus.to_dict("records"),
        "pairs_input": pair_rows,
        "materials": material_rows,
        "unknown_fields": [
            "magnetic_state for all focus materials",
            "kpoints/cutoff/smearing/functional detail/relaxation state for all focus materials",
        ],
        "immutable_inputs": immutable,
        "artifacts": {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
        "wall_s": time.monotonic() - start,
    }
    write_json(out / "inputs_report.json", report)
    print(json.dumps({"phase": "inputs", "selection_equal": selection_equal,
                      "target_recheck": target_deltas,
                      "wall_s": report["wall_s"]}), flush=True)


# ---------------------------------------------------------------------------
# Phase model: isolated fit reproduction, stage attribution, gradient geometry
# ---------------------------------------------------------------------------

def param_group_map(transformer):
    """Assign every transformer parameter to exactly one reported group."""
    groups = {name: [] for name in PARAM_GROUPS}
    for name, parameter in transformer.named_parameters():
        top = name.split(".")[0]
        if top in ("tok_emb", "atom_norm", "atom_proj"):
            group = "atom_entry"
        elif name.startswith("encoder.local_layers"):
            group = "local"
        elif name.startswith("encoder.global_layers"):
            group = "global"
        elif top == "decoder" or top in ("edos_query_embed", "edos_tgt",
                                         "phdos_query_embed", "phdos_tgt",
                                         "edos_trunk", "phdos_trunk"):
            group = "decoder"
        elif top in ("edos_out_head", "phdos_out_head", "eta_head"):
            group = "head"
        else:
            raise ValueError(f"unassigned parameter: {name}")
        groups[group].append((name, parameter))
    sizes = {key: len(value) for key, value in groups.items()}
    if any(size == 0 for size in sizes.values()):
        raise ValueError(f"empty parameter group: {sizes}")
    return groups


class FullStageExtractor:
    """Read-only hooks keeping full atom states and per-energy decoder tokens."""

    def __init__(self, transformer):
        encoder = transformer.encoder
        if getattr(encoder, "local_layers", None) is None:
            raise ValueError("stage capture requires the local/global encoder")
        picked = {}
        for kind, index in encoder.execution_order:
            picked[kind] = index
        self.stage_modules = {
            "atom_local_last": encoder.local_layers[picked["local"]],
            "atom_global_last": encoder.global_layers[picked["global"]],
        }
        self.features = {"atom_local_last": [], "atom_global_last": [],
                         "decoder_edos_tokens": [], "decoder_phdos_tokens": []}
        self.mask = None
        self.decoder_call = 0
        self.hooks = [transformer.register_forward_pre_hook(self._begin),
                      transformer.decoder.register_forward_hook(self._decoder)]
        for name, module in self.stage_modules.items():
            self.hooks.append(module.register_forward_hook(self._atom(name)))

    def _begin(self, module, args):
        self.mask = atom_padding_mask(args[2], args[1])
        self.decoder_call = 0

    def _atom(self, name):
        def hook(module, args, output):
            self.features[name].append(output.detach().float().cpu().numpy())
        return hook

    def _decoder(self, module, args, output):
        value = output[0]
        if value.dim() != 3:
            raise ValueError("decoder output must keep its energy positions")
        key = "decoder_edos_tokens" if self.decoder_call == 0 else "decoder_phdos_tokens"
        self.features[key].append(value.detach().float().cpu().numpy())
        self.decoder_call += 1

    def pop_batch(self):
        stacked = {name: np.concatenate(chunks, axis=0)
                   for name, chunks in self.features.items()}
        for chunks in self.features.values():
            chunks.clear()
        mask = self.mask.detach().cpu().numpy().astype(bool)
        self.mask, self.decoder_call = None, 0
        return stacked, mask

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def atom_set_diff(left, right):
    """Compare full atom-state sets using the decoder's per-feature RMS units.

    The original diagnostic used a Euclidean atom distance with a per-feature
    RMS denominator, inflating relative atom distances by sqrt(d). Historical
    evidence remains archived; newly computed distances use consistent units.
    A uniform scaling of the cost preserves the Hungarian assignment.
    """
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (left.ndim != 2 or right.ndim != 2 or not len(left) or not len(right)
            or left.shape[1] != right.shape[1] or left.shape[1] == 0):
        raise ValueError("atom states require nonempty [N,d] arrays with equal feature width")
    cost = np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).mean(axis=-1))
    rows, cols = linear_sum_assignment(cost)
    matched = cost[rows, cols]
    scale = float(np.sqrt(0.5 * ((left ** 2).mean() + (right ** 2).mean())))
    pooled_left = np.concatenate((left.mean(0), np.sqrt((left ** 2).mean(0))))
    pooled_right = np.concatenate((right.mean(0), np.sqrt((right ** 2).mean(0))))
    return {
        "n_left": int(left.shape[0]), "n_right": int(right.shape[0]),
        "matched_rms": float(np.sqrt((matched ** 2).mean())),
        "matched_max": float(matched.max()),
        "state_scale_rms": scale,
        "relative_matched_rms": float(np.sqrt((matched ** 2).mean()) / max(scale, 1e-12)),
        "chamfer_mean": float(0.5 * (cost.min(1).mean() + cost.min(0).mean())),
        "pooled_mean_rms_diff": float(np.sqrt(((pooled_left - pooled_right) ** 2).mean())),
        "distance_units": "per-feature-rms-v2",
    }


def token_diff(left, right):
    """Aligned per-energy token comparison for decoder outputs [E,d]."""
    diff = left - right
    scale = float(np.sqrt(0.5 * ((left ** 2).mean() + (right ** 2).mean())))
    rms = float(np.sqrt((diff ** 2).mean()))
    return {
        "rms_diff": rms, "token_scale_rms": scale,
        "relative_rms": float(rms / max(scale, 1e-12)),
        "per_energy_rms": np.sqrt((diff ** 2).mean(-1)),
    }


def spectrum_diff(pred_left, pred_right, target_left, target_right):
    predicted = pred_left.astype(np.float64) - pred_right
    actual = target_left.astype(np.float64) - target_right
    amplitude = float(0.5 * np.abs(actual).sum())
    error = float(0.5 * np.abs(predicted - actual).sum())
    denominator = float(np.linalg.norm(actual) * np.linalg.norm(predicted))
    return {
        "target_diff_tv": amplitude,
        "predicted_diff_tv": float(0.5 * np.abs(predicted).sum()),
        "difference_error": error,
        "skill": float(1 - error / amplitude) if amplitude > 1e-10 else None,
        "direction_cosine": float(predicted @ actual / denominator)
        if denominator > 1e-12 else None,
        "per_bin_abs_error": np.abs(predicted - actual),
    }


def capture_stages(model, pair_batches, full_ids):
    """Forward the focus pairs and keep full stage tensors per material."""
    transformer = model.model["transformer"]
    was_training = transformer.training
    transformer.eval()
    out = {}
    try:
        with torch.no_grad():
            for tag, pair_batch in pair_batches.items():
                extractor = FullStageExtractor(transformer)
                data = model.data_preprocess(pair_batch)
                outputs = transformer(data[0], data[2], data[1], data[-2], data[-1])
                outputs = model.fp32_outputs(outputs)
                feats, mask = extractor.pop_batch()
                extractor.close()
                edos = outputs["edos"].squeeze(1).float().softmax(-1).cpu().numpy()
                phdos = outputs["phdos"].squeeze(1).float().softmax(-1).cpu().numpy()
                targets_e = data[3].cpu().numpy()
                targets_p = data[4].cpu().numpy()
                per_material = {}
                for slot in range(mask.shape[0]):
                    valid = ~mask[slot]
                    per_material[str(full_ids[tag][slot])] = {
                        "atom_local_last": feats["atom_local_last"][slot][valid],
                        "atom_global_last": feats["atom_global_last"][slot][valid],
                        "decoder_edos_tokens": feats["decoder_edos_tokens"][slot],
                        "decoder_phdos_tokens": feats["decoder_phdos_tokens"][slot],
                        "edos_pred": edos[slot], "phdos_pred": phdos[slot],
                        "edos_target": targets_e[slot], "phdos_target": targets_p[slot],
                    }
                out[tag] = per_material
    finally:
        transformer.train(was_training)
    return out


def pair_stage_diffs(per_material, left_id, right_id):
    left, right = per_material[left_id], per_material[right_id]
    record = {}
    for stage in ("atom_local_last", "atom_global_last"):
        record[stage] = atom_set_diff(left[stage], right[stage])
    for stage in ("decoder_edos_tokens", "decoder_phdos_tokens"):
        diff = token_diff(left[stage], right[stage])
        record[stage] = {k: v for k, v in diff.items() if k != "per_energy_rms"}
        record[stage]["per_energy_rms_profile"] = diff["per_energy_rms"]
    record["output_edos"] = spectrum_diff(
        left["edos_pred"], right["edos_pred"], left["edos_target"], right["edos_target"])
    record["output_phdos"] = spectrum_diff(
        left["phdos_pred"], right["phdos_pred"], left["phdos_target"], right["phdos_target"])
    return record


def gradient_geometry(model, batch, groups, pair_lambda):
    """Loss-wise gradient magnitude/direction per parameter group, eval mode."""
    transformer = model.model["transformer"]
    was_training = transformer.training
    transformer.eval()
    try:
        data = model.data_preprocess(batch)
        with torch.enable_grad():
            outputs = transformer(data[0], data[2], data[1], data[-2], data[-1])
        outputs = model.fp32_outputs(outputs)
        edos = outputs["edos"]
        if edos.dim() == 3:
            edos = edos.squeeze(1)
        phdos = outputs["phdos"]
        if phdos.dim() == 3:
            phdos = phdos.squeeze(1)
        loss_e = sumnorm_klw_loss(edos, data[3], data[13], model.use_mask,
                                  model.w_w1, model.w_huber, model.huber_delta).mean()
        loss_p = sumnorm_klw_loss(phdos, data[4], data[14], model.use_mask,
                                  model.w_w1, model.w_huber, model.huber_delta).mean()
        half = edos.shape[0] // 2
        if half == 0 or edos.shape[0] != 2 * half:
            raise ValueError("gradient geometry needs an even A-pair/B-pair batch")
        loss_pair = edos_pair_contrast_loss(edos[:half], edos[half:],
                                            data[3][:half], data[3][half:])
        flat = [(name, p) for items in groups.values() for name, p in items]
        params = [p for _, p in flat]
        grads = {}
        for loss_name, loss in (("edos_base", loss_e), ("phdos_base", loss_p),
                                ("pair", loss_pair)):
            grads[loss_name] = torch.autograd.grad(
                loss, params, retain_graph=True, allow_unused=True)
        index, pos = {}, 0
        for group_name, items in groups.items():
            index[group_name] = list(range(pos, pos + len(items)))
            pos += len(items)

        def flatten(loss_name, group_name):
            pieces = []
            for i in index[group_name]:
                grad = grads[loss_name][i]
                # Unreached parameters count as exact zeros so every loss's
                # group vector keeps the same layout for dot products.
                pieces.append(torch.zeros_like(params[i]).reshape(-1) if grad is None
                              else grad.reshape(-1))
            return torch.cat(pieces) if pieces else None

        def describe(group_name):
            vecs = {name: flatten(name, group_name) for name in grads}
            entry = {}
            for name, vec in vecs.items():
                entry[f"{name}_grad_norm"] = float(vec.norm()) if vec is not None else 0.0
                entry[f"{name}_tensors_present"] = int(sum(
                    1 for i in index[group_name] if grads[name][i] is not None))
            for left, right in (("edos_base", "pair"), ("edos_base", "phdos_base"),
                                ("pair", "phdos_base")):
                a, b = vecs[left], vecs[right]
                if a is None or b is None or float(a.norm()) == 0 or float(b.norm()) == 0:
                    entry[f"cos_{left}_{right}"] = None
                else:
                    entry[f"cos_{left}_{right}"] = float(
                        (a @ b) / (a.norm() * b.norm()))
            base = entry["edos_base_grad_norm"]
            entry["pair_over_edos_grad_norm"] = (
                float(entry["pair_grad_norm"] / base) if base > 0 else None)
            entry["pair_weighted_grad_norm"] = float(pair_lambda * entry["pair_grad_norm"])
            entry["pair_weighted_over_edos_grad_norm"] = (
                float(pair_lambda * entry["pair_grad_norm"] / base) if base > 0 else None)
            return entry

        result = {
            "batch_materials": int(edos.shape[0]),
            "loss_values": {"edos_base": float(loss_e), "phdos_base": float(loss_p),
                            "pair": float(loss_pair)},
            "pair_lambda": float(pair_lambda),
            "groups": {name: describe(name) for name in PARAM_GROUPS},
            "all_parameters": describe_all(grads, params),
        }
        return result
    finally:
        transformer.train(was_training)


def describe_all(grads, params):
    """Whole-model gradient geometry, to anchor the group-level numbers."""
    def flat_all(loss_name):
        pieces = [torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                  for g, p in zip(grads[loss_name], params)]
        return torch.cat(pieces) if pieces else None

    vecs = {name: flat_all(name) for name in grads}
    entry = {f"{name}_grad_norm": float(vec.norm()) if vec is not None else 0.0
             for name, vec in vecs.items()}
    for left, right in (("edos_base", "pair"), ("edos_base", "phdos_base"),
                        ("pair", "phdos_base")):
        a, b = vecs[left], vecs[right]
        entry[f"cos_{left}_{right}"] = (
            float((a @ b) / (a.norm() * b.norm()))
            if a is not None and b is not None
            and float(a.norm()) > 0 and float(b.norm()) > 0 else None)
    return entry


def reproduce_arm(saved, weights, arm, batch, keys, ids, pair_budget,
                  calibration_lambda, steps, eval_every, out, initial_hash, args_ns):
    """Faithful 600-step rerun of the discarded disposable fit, weights saved."""
    model = new_model(saved, weights, arm, args_ns)
    model.pair_lambda = calibration_lambda
    model.pair_aux_calibrated = True
    transformer = model.model["transformer"]
    if state_hash(transformer.state_dict()) != initial_hash:
        raise ValueError("reproduction arm did not start from the source weights")
    setup_ablation_seed(SEED)
    rng_start = rng_digest(rng_snapshot())
    baseline, _, _, _ = evaluate(model, batch, keys, ids)
    trace = [{"step": 0, **baseline}]
    arm_start = time.monotonic()
    transformer.train()
    final_pairs = final_edos = final_phdos = metrics = None
    for step in range(1, steps + 1):
        values = model.train_one_step(batch, step, pair_batch=batch,
                                      pair_budget=pair_budget)
        if not all(np.isfinite(x) for x in values.values()):
            raise ValueError("non-finite fitting loss")
        if step % eval_every == 0 or step == steps:
            metrics, final_pairs, final_edos, final_phdos = evaluate(
                model, batch, keys, ids)
            trace.append({"step": step, "loss": values["loss"], **metrics})
            print(json.dumps({"arm": arm, "step": step,
                              "pair_error_mean": metrics["pair_error_mean"],
                              "edos_tv": metrics["individual_edos_tv_mean"]}),
                  flush=True)
    pd.DataFrame(trace).to_csv(out / f"trace_repro_{arm}.csv", index=False)
    final_pairs.to_csv(out / f"pairs_repro_{arm}_s{steps}.csv", index=False)
    np.savez_compressed(out / f"predictions_repro_{arm}_s{steps}.npz",
                        edos=final_edos, phdos=final_phdos)
    final_hash = state_hash(transformer.state_dict())
    torch.save({"arm": arm, "step": steps, "seed": SEED,
                "initial_state_hash": initial_hash, "final_state_hash": final_hash,
                "pair_lambda": calibration_lambda,
                "model": {k: v.cpu() for k, v in transformer.state_dict().items()}},
               out / f"diagnostic_state_{arm}_s{steps}.pth")
    entry = {"initial": baseline, "final": metrics,
             "rng_start": rng_start, "rng_end": rng_digest(rng_snapshot()),
             "final_state_hash": final_hash,
             "state_file": f"diagnostic_state_{arm}_s{steps}.pth",
             "wall_s": time.monotonic() - arm_start}
    del model, transformer
    gc.collect()
    torch.cuda.empty_cache()
    return entry, pd.DataFrame(trace)


def compare_trace(arm, frame):
    archived = pd.read_csv(ARCHIVED_FIT / f"trace_{arm}.csv")
    merged = archived.merge(frame, on="step", suffixes=("_archive", "_repro"))
    columns = {}
    for column in archived.columns:
        if column == "step":
            continue
        left = merged[f"{column}_archive"]
        right = merged[f"{column}_repro"]
        if column == "fit_demonstrated":
            columns[column] = {"exact_equal": bool(
                (left.astype(str) == right.astype(str)).all())}
            continue
        left = left.astype(float)
        right = right.astype(float)
        delta = (left - right).abs()
        finite = delta[np.isfinite(delta)]
        columns[column] = {
            "max_abs_delta": float(finite.max()) if len(finite) else None,
            "final_abs_delta": float(delta.iloc[-1]) if np.isfinite(delta.iloc[-1]) else None,
        }
    return {"arm": arm, "steps_compared": int(len(merged)),
            "columns": columns}


def run_model(args):
    out = args.out
    if not (out / "inputs_report.json").is_file():
        raise FileNotFoundError("run --phase inputs first; the model phase only "
                                "extends its evidence directory")
    if not torch.cuda.is_available() or args.seed != SEED:
        raise ValueError("the model phase requires CUDA and seed 42")
    if args.steps != 600 or args.eval_every != 100:
        raise ValueError("the reproduction is validated only for the archived "
                         "600-step / eval-every-100 budget")
    torch.set_num_threads(1)
    setup_ablation_seed(SEED)
    start = time.monotonic()
    args_ns = SimpleNamespace(pair_ratio=1.0, lr=5e-5)
    saved, _params, _contract, weights, epoch, dataset, selected, _eligibility = load_context(args)
    rows = selected.i.tolist() + selected.j.tolist()
    ids = [str(dataset.material_ids[i]) for i in rows]
    keys = composition_keys_by_position(dataset, rows)["absolute"]
    batch = crystal_collate([dataset[i] for i in rows])
    pair_budget = int(len(rows) * batch[0].shape[1] ** 2)
    initial_hash = state_hash(weights)
    archived_report = json.loads((ARCHIVED_FIT / "report.json").read_text())
    if initial_hash != archived_report["initial_state_hash"]:
        raise ValueError("source weights differ from the archived 2026-10-12 run")

    focus = pd.read_csv(out / "focus_pairs.csv")
    pair_batches, pair_ids = {}, {}
    for row in focus.to_dict("records"):
        pair_batches[row["tag"]] = crystal_collate(
            [dataset[row["left_row"]], dataset[row["right_row"]]])
        pair_ids[row["tag"]] = [row["left_id"], row["right_id"]]

    report = {
        "scope": "isolated train-only reproduction of the 2026-10-12 disposable "
                 "600-step C1 fit; diagnostic weights saved; no valid/test",
        "seed": SEED, "source_epoch": epoch, "initial_state_hash": initial_hash,
        "reproduced": "same selection, init, losses, optimizer, 600-step budget as "
                      "tools/eval/local_structure_small_fit.py",
        "device": torch.cuda.get_device_name(),
        "arms": {}, "complete": False,
    }
    if args.reuse_states:
        previous = json.loads((out / "model_report.json").read_text()) \
            if (out / "model_report.json").is_file() else {}
        if not previous.get("arms") or any(
                "state_file" not in previous["arms"].get(arm, {})
                for arm in ("control", "candidate")):
            raise ValueError("no completed reproduction artifacts to reuse")
        for arm in ("control", "candidate"):
            if not (out / previous["arms"][arm]["state_file"]).is_file():
                raise FileNotFoundError(f"missing saved diagnostic state: {arm}")
        report["arms"] = previous["arms"]
        report["calibration"] = previous["calibration"]
        report["reproduction_reused_from_prior_run"] = True
        pair_lambda = float(previous["calibration"]["lambda_used_for_reproduction"])
    else:
        calibration_model = new_model(saved, weights, "candidate", args_ns)
        calibration_model._calibrate_edos_pair_weight(batch)
        calibration = dict(calibration_model.pair_aux_calibration)
        del calibration_model
        gc.collect()
        torch.cuda.empty_cache()
        calibration["lambda_delta_vs_archive"] = float(abs(
            calibration["lambda"] - archived_report["calibration"]["lambda"]))
        calibration["reproduced_with_archived_lambda"] = True
        report["calibration"] = calibration
        # Independent recalibration is subject to FP32 GPU nondeterminism at the
        # 1e-6 level; the reproduction uses the archived weight exactly.
        if calibration["lambda_delta_vs_archive"] > 1e-4:
            raise ValueError("re-calibrated pair weight differs from the archived run")
        pair_lambda = float(archived_report["calibration"]["lambda"])
        calibration["lambda_used_for_reproduction"] = pair_lambda

        for arm in ("control", "candidate"):
            entry, frame = reproduce_arm(
                saved, weights, arm, batch, keys, ids, pair_budget, pair_lambda,
                args.steps, args.eval_every, out, initial_hash, args_ns)
            entry["trace_vs_archive"] = compare_trace(arm, frame)
            entry["final_state_hash_equals_archive"] = bool(
                entry["final_state_hash"] == archived_report["arms"][arm]["final_state_hash"])
            report["arms"][arm] = entry
            write_json(out / "model_report.json", report)
        if report["arms"]["control"]["rng_start"] != report["arms"]["candidate"]["rng_start"]:
            raise ValueError("reproduction arm starting RNG states differ")
        if report["arms"]["control"]["rng_end"] != report["arms"]["candidate"]["rng_end"]:
            raise ValueError("reproduction arms consumed different dropout RNG")

    # --- stage attribution and gradient geometry at three states ---
    states = {"start": {"weights": weights, "source": "original checkpoint best ep34"}}
    for arm in ("candidate", "control"):
        saved_state = torch.load(out / report["arms"][arm]["state_file"],
                                 map_location="cpu")
        states[f"{arm}_s600"] = {
            "weights": saved_state["model"],
            "source": f"isolated reproduction of {arm} arm after 600 steps",
        }
    stage_report, gradient_report, tensors = {}, {}, {}
    for state_name, state in states.items():
        model = new_model(saved, state["weights"], "candidate", args_ns)
        model.pair_lambda = pair_lambda
        model.pair_aux_calibrated = True
        captured = capture_stages(model, pair_batches, pair_ids)
        stage_report[state_name] = {"source": state["source"], "pairs": {}}
        for row in focus.to_dict("records"):
            tag = row["tag"]
            diffs = pair_stage_diffs(captured[tag], row["left_id"], row["right_id"])
            profile = {}
            for stage in ("decoder_edos_tokens", "decoder_phdos_tokens"):
                profile[stage] = diffs[stage].pop("per_energy_rms_profile")
            stage_report[state_name]["pairs"][tag] = diffs
            for mid in (row["left_id"], row["right_id"]):
                for key, value in captured[tag][mid].items():
                    tensors[f"{state_name}__{mid}__{key}"] = value
            for stage, values in profile.items():
                tensors[f"{state_name}__{tag}__{stage}__per_energy_rms"] = values
        gradient_report[state_name] = {"source": state["source"], "full_batch": {},
                                       "pairs": {}}
        gradient_report[state_name]["full_batch"] = gradient_geometry(
            model, batch, param_group_map(model.model["transformer"]), pair_lambda)
        for tag, pair_batch in pair_batches.items():
            gradient_report[state_name]["pairs"][tag] = gradient_geometry(
                model, pair_batch, param_group_map(model.model["transformer"]),
                pair_lambda)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    np.savez_compressed(out / "stage_tensors.npz", **tensors)

    report["stage_diffs"] = stage_report
    report["gradients"] = gradient_report
    report["group_definition"] = {
        "atom_entry": "tok_emb/atom_norm/atom_proj (ZP element entrance)",
        "local": "encoder.local_layers.* (C1 radial local update)",
        "global": "encoder.global_layers.* (attention + FFN)",
        "decoder": "decoder.* plus edos/phdos query, tgt and trunk parameters",
        "head": "edos_out_head/phdos_out_head/eta_head",
    }
    report["complete"] = True
    report["wall_s"] = time.monotonic() - start
    report["artifacts"] = {p.name: file_sha256(p) for p in sorted(out.iterdir())
                           if p.is_file() and p.name != "model_report.json"}
    write_json(out / "model_report.json", report)
    print(json.dumps({"complete": True, "wall_s": report["wall_s"],
                      "arms": {arm: report["arms"][arm]["trace_vs_archive"]["columns"]["pair_error_mean"]
                               for arm in report["arms"]}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("inputs", "model"), required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--pair-table", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--reuse-states", action="store_true",
                        help="skip the fit reproduction and analyze saved states")
    args = parser.parse_args()
    if args.phase == "inputs":
        run_inputs(args)
    else:
        run_model(args)


if __name__ == "__main__":
    main()
