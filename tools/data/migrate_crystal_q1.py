"""Build a new complete FP32 crystal cache; never overwrite historical Q1.

Source: frozen Q1 membership/order/labels + A4 MP standardized primitive cells.
Only atom-truncated records need raw MP structures. No downloads or training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch
from pymatgen.core import Lattice, Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from utils.crystal import CRYSTAL_SCHEMA
from utils.crystal_cache import PackedCrystalStore
from utils.paths import file_sha256
from utils.relative_features import build_cell_from_lattice


def restore_structures(raw_path, wanted):
    restored = {}
    with Path(raw_path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["mpid"] not in wanted:
                continue
            if row["mpid"] in restored:
                raise ValueError("duplicate MP raw structure ID")
            data = row["mp_summary"]["structure"]
            structure = Structure(Lattice(data["lattice"]["matrix"]),
                                  [s["label"] for s in data["sites"]],
                                  [s["abc"] for s in data["sites"]])
            restored[row["mpid"]] = SpacegroupAnalyzer(
                structure, symprec=0.1).get_primitive_standard_structure()
    if set(restored) != set(wanted):
        raise ValueError("raw structures missing for truncated crystal records")
    return restored


def restore_atom_fields(row, old_numbers, old_frac, structure):
    """Check historical prefix/basis, preserve it exactly, append full tails."""
    numbers = np.asarray(structure.atomic_numbers, dtype=np.uint8)
    frac = np.asarray(structure.frac_coords, dtype=np.float32)
    if len(numbers) != row["nsites_prim"] or not np.array_equal(numbers[:80], old_numbers):
        raise ValueError(f"primitive atom identity/count mismatch: {row['mpid']}")
    a, b, c = structure.lattice.abc
    parameters = np.asarray([[a, b, 1 / c], structure.lattice.angles], dtype=np.float32)
    old_parameters = np.asarray(row["positions"], dtype=np.float32).reshape(-1, 3)[:2]
    if not np.array_equal(parameters, old_parameters):
        raise ValueError(f"primitive lattice basis mismatch: {row['mpid']}")
    delta = frac[:80].astype(np.float64) - old_frac.astype(np.float64)
    delta -= np.round(delta)
    if np.max(np.abs(delta)) > 1e-6:
        raise ValueError(f"primitive coordinate prefix mismatch: {row['mpid']}")
    frac[:80] = old_frac
    return numbers, frac


def migrate_q1(legacy, raw, out):
    legacy, raw, out = Path(legacy), Path(raw), Path(out)
    if out.exists():
        raise FileExistsError(f"refusing existing migration target: {out}")
    rows = pq.read_table(raw / "v2_processed.parquet", columns=[
        "mpid", "nsites_prim", "positions", "elements", "struct_ref", "split"]).to_pylist()
    by_id = {r["mpid"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate source material ID")
    split_ids = {s: np.load(legacy / s / f"{s}_index.npy", allow_pickle=False)
                 for s in ("train", "valid", "test")}
    all_ids = set(str(i) for ids in split_ids.values() for i in ids)
    exclusions = set(json.loads((ROOT / "data/reference/q1_excluded_records.json").read_text())["mpids"])
    if all_ids != set(by_id) - exclusions or sum(map(len, split_ids.values())) != len(all_ids):
        raise ValueError("source membership is not frozen Q1")
    wanted = {i for i in all_ids if by_id[i]["nsites_prim"] > 80}
    if any(by_id[i]["struct_ref"] != f"mp_increment_raw.jsonl:{i}" for i in wanted):
        raise ValueError("unsupported raw structure provenance")
    restored = restore_structures(raw / "mp_increment_raw.jsonl", wanted)
    valence = {r["mpid"]: r for r in pq.read_table(
        ROOT / "data/reference/edos_valence_by_material.parquet").to_pylist()}
    truncation = {r["mpid"]: r["trunc"] for r in pq.read_table(
        ROOT / "data/reference/edos_truncation_by_material.parquet").to_pylist()}
    zval = json.loads((ROOT / "data/reference/valence_by_element.json").read_text())
    manifest = {
        "schema": CRYSTAL_SCHEMA, "label_pool": "Q1", "grid": "E0/P0",
        "cell_convention": "row lattice vectors, angstrom; legacy canonical frame",
        "frac_convention": "dimensionless, same primitive basis and atom order",
        "primitive_policy": {"source": "MP", "symprec": 0.1},
        "geometry_precision": "float32", "atom_cap": None,
        "legacy_cell_conversion": "one-time build_cell_from_lattice; source positions FP32",
        "sources": {}, "splits": {}, "files": {}, "repairs": [],
        "valence_policy": "frozen audit, except atom-truncated composition fallbacks repaired",
        "quality_review": [],
        "training_exclusions": [],
        "quality_policy": "existing Q1 C_full/N_val<0.5; corrected N_val applied to new training view",
    }
    def source(path):
        manifest["sources"][str(path.resolve())] = file_sha256(path)
    for path in (legacy / "manifest.json", raw / "v2_processed.parquet",
                 raw / "mp_increment_raw.jsonl", ROOT / "data/reference/valence_by_element.json",
                 ROOT / "data/reference/edos_valence_by_material.parquet",
                 ROOT / "data/reference/edos_truncation_by_material.parquet",
                 ROOT / "data/reference/q1_excluded_records.json", Path(__file__)):
        source(path)
    out.mkdir(parents=True)
    for split, ids in split_ids.items():
        directory = out / split
        directory.mkdir()
        old_dir = legacy / split
        old_el = np.load(old_dir / f"elements_{split}.npy", allow_pickle=False)
        old_pos = np.load(old_dir / f"positions_{split}.npy", allow_pickle=False).reshape(len(ids), -1, 3)
        old_nv = np.load(old_dir / f"nvalence_{split}.npy", allow_pickle=False)
        numbers, fractions, counts = [], [], []
        new_nv = old_nv.astype(np.float32)
        for index, material_id in enumerate(ids):
            material_id = str(material_id)
            row = by_id[material_id]
            if row["split"] != split or not np.array_equal(old_el[index], row["elements"]) \
                    or not np.array_equal(old_pos[index].reshape(-1), np.asarray(row["positions"], dtype=np.float32)):
                raise ValueError("Q1 row/order/source structure mismatch")
            if float(old_nv[index]) != valence[material_id]["N_val"]:
                raise ValueError("frozen valence audit differs from legacy cache")
            prefix = old_el[index, 2:][old_el[index, 2:] != 0].astype(np.uint8)
            n = len(prefix)
            frac = old_pos[index, 2:2 + n].copy()
            if material_id in restored:
                nums, frac = restore_atom_fields(row, prefix, frac, restored[material_id])
                repair = {"material_id": material_id, "split": split, "old_atoms": n,
                          "n_atoms": len(nums), "source": row["struct_ref"]}
                audit = valence[material_id]
                if audit["method"].startswith("fallback:"):
                    zvalues = np.array([zval[Element.from_Z(int(z)).symbol]["zval"] for z in nums])
                    if audit["N_val"] != float(zvalues[:80].sum()):
                        raise ValueError("fallback audit does not equal truncated composition")
                    new_nv[index] = float(zvalues.sum())
                    repair["valence"] = {"old": audit["N_val"], "new": float(new_nv[index]),
                                         "method": "complete_composition_fallback", "old_method": audit["method"]}
                    corrected_trunc = truncation[material_id] * audit["N_val"] / float(new_nv[index])
                    repair["corrected_truncation_ratio"] = corrected_trunc
                    if corrected_trunc < 0.5:
                        manifest["quality_review"].append({"material_id": material_id,
                            "reason": "corrected full-composition N_val gives truncation<0.5",
                            "trunc": corrected_trunc})
                        if split == "train":
                            manifest["training_exclusions"].append(material_id)
                manifest["repairs"].append(repair)
            else:
                nums = prefix
            if len(nums) != row["nsites_prim"]:
                raise ValueError("incomplete structure remains after migration")
            numbers.append(nums); fractions.append(frac); counts.append(len(nums))
        cell, _ = build_cell_from_lattice(torch.from_numpy(old_pos).float())
        arrays = {
            f"cell_{split}.npy": cell.numpy(),
            f"atomic_numbers_{split}.npy": np.concatenate(numbers),
            f"frac_coords_{split}.npy": np.concatenate(fractions).astype(np.float32),
            f"n_atoms_{split}.npy": np.asarray(counts, dtype=np.int32),
            f"atom_offsets_{split}.npy": np.r_[0, np.cumsum(counts, dtype=np.int64)],
            f"nvalence_{split}.npy": new_nv, f"{split}_index.npy": ids,
        }
        for prefix in ("edos_tgtdos", "phdos_tgtdos", "edos_mask", "phdos_mask"):
            array = np.load(old_dir / f"{prefix}_{split}.npy", allow_pickle=False)
            arrays[f"{prefix}_{split}.npy"] = array.astype(np.bool_ if "mask" in prefix else np.float32)
        for filename, array in arrays.items():
            path = directory / filename
            np.save(path, array, allow_pickle=False)
            manifest["files"][f"{split}/{filename}"] = {"shape": list(array.shape),
                "dtype": str(array.dtype), "sha256": file_sha256(path)}
        for filename in (f"elements_{split}.npy", f"positions_{split}.npy", f"nvalence_{split}.npy",
                         f"{split}_index.npy", f"edos_tgtdos_{split}.npy", f"phdos_tgtdos_{split}.npy",
                         f"edos_mask_{split}.npy", f"phdos_mask_{split}.npy"):
            source(old_dir / filename)
        manifest["splits"][split] = {"n": len(ids), "atoms": sum(counts), "max_atoms": max(counts),
            "structure_repairs": sum(str(i) in wanted for i in ids),
            "training_n": len(ids) - sum(str(i) in manifest["training_exclusions"] for i in ids)}
        print(f"{split}: {len(ids)} structures, {sum(counts)} complete atoms, max={max(counts)}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for split in split_ids:
        PackedCrystalStore(out, split)
    print(f"migrated {len(manifest['repairs'])} truncated structures; quality review={len(manifest['quality_review'])}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    migrate_q1(args.legacy, args.raw, args.out)


if __name__ == "__main__":
    main()
