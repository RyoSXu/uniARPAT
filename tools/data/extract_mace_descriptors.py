"""Extract frozen per-atom MACE invariant descriptors for the repaired Q1 crystals.

Design 2026-11-08 (MACE frozen-feature reference). Runs in a separate Python
environment with mace-torch (it conflicts with the training environment), so it
imports nothing from the repository. For every split it reads the packed crystal
arrays, builds periodic ASE structures in the stored atom order and writes the
L=0 (rotation-invariant) node features of every interaction layer:

    <out>/<split>.npy   float32 [total atoms of split, D], same order as atomic_numbers_<split>.npy
    <out>/manifest.json model/data fingerprints, layer layout, train-only standardization
                        statistics and the invariance self-checks

Standardization is not applied to the arrays; the loader applies the frozen
train mean/std recorded in the manifest. Test descriptors are produced so the
input contract covers every split, never for evaluation.

Usage (MACE environment):
    python tools/data/extract_mace_descriptors.py --model <mace.model> \
        --data data/crystal_q1_v1 --out data/external/mace_mp0b3_medium_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = "external-atom-features-v1"
SPLITS = ("train", "valid", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(root, split):
    directory = Path(root) / split
    arrays = {name: np.load(directory / f"{name}_{split}.npy", allow_pickle=False)
              for name in ("cell", "frac_coords", "atomic_numbers", "atom_offsets", "n_atoms")}
    return arrays


def structure(arrays, index):
    from ase import Atoms

    start, end = arrays["atom_offsets"][index:index + 2]
    return Atoms(numbers=arrays["atomic_numbers"][start:end].astype(int),
                 cell=arrays["cell"][index].astype(np.float64),
                 scaled_positions=arrays["frac_coords"][start:end].astype(np.float64), pbc=True)


def rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def self_checks(calc, arrays, describe, count=8):
    """Rotation invariance and 2x1x1 supercell consistency on the first structures."""
    rot, sup = [], []
    for index in range(min(count, len(arrays["n_atoms"]))):
        atoms = structure(arrays, index)
        base = describe(atoms)
        turned = atoms.copy()
        turned.set_cell(atoms.cell.array @ rotation(index).T, scale_atoms=True)
        rot.append(float(np.abs(describe(turned) - base).max()))
        big = atoms.repeat((2, 1, 1))
        sup.append(float(np.abs(describe(big) - np.concatenate([base, base])).max()))
    return {"structures": len(rot), "rotation_max_abs": max(rot), "supercell_2x1x1_max_abs": max(sup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    args = parser.parse_args()

    import mace
    import torch
    from mace.calculators import MACECalculator

    out = Path(args.out)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite existing features: {out}")
    out.mkdir(parents=True, exist_ok=True)
    calc = MACECalculator(model_paths=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    layers = int(model.num_interactions)

    def describe(atoms):
        return np.asarray(calc.get_descriptors(atoms, invariants_only=True, num_layers=-1), dtype=np.float64)

    data_manifest = Path(args.data) / "manifest.json"
    result = {"schema": SCHEMA, "source": "MACE invariant node descriptors (L=0 channels of every interaction layer)",
              "model": {"path": str(Path(args.model).resolve()), "sha256": sha256(args.model),
                        "name": Path(args.model).stem, "license": "MIT (mace-foundations/mace-mp-0)",
                        "training_data": "MPtrj (PBE/PBE+U energies, forces, stresses)",
                        "num_interactions": layers, "r_max": float(model.r_max),
                        "hidden_irreps": str(model.products[0].linear.irreps_out)},
              "software": {"mace_torch": getattr(mace, "__version__", "unknown"), "torch": torch.__version__,
                           "dtype": args.dtype, "device": args.device},
              "data": {"root": str(Path(args.data).resolve()), "manifest_sha256": sha256(data_manifest)},
              "splits": {}}
    train_stats = None
    for split in SPLITS:
        arrays = load_split(args.data, split)
        size = len(arrays["n_atoms"])
        chunks, started = [], time.time()
        for index in range(size):
            atoms = structure(arrays, index)
            values = describe(atoms)
            if values.shape[0] != len(atoms) or not np.isfinite(values).all():
                raise ValueError(f"{split}[{index}]: bad descriptor block {values.shape}")
            chunks.append(values)
            if index % 2000 == 0:
                print(f"{split} {index}/{size} {time.time() - started:.0f}s", flush=True)
        features = np.concatenate(chunks)
        if features.shape[0] != len(arrays["atomic_numbers"]):
            raise ValueError(f"{split}: atom count mismatch")
        path = out / f"{split}.npy"
        np.save(path, features.astype(np.float32))
        if split == "train":
            mean, std = features.mean(0), features.std(0)
            train_stats = {"mean": mean.tolist(), "std": std.tolist(),
                           "std_floor": 1e-6, "near_constant_channels": int((std < 1e-6).sum())}
        result["splits"][split] = {"materials": size, "atoms": int(features.shape[0]), "dim": int(features.shape[1]),
                                   "sha256": sha256(path), "seconds": round(time.time() - started, 1)}
        print(split, result["splits"][split], flush=True)
    dim = result["splits"]["train"]["dim"]
    if dim % layers:
        raise ValueError("descriptor width is not divisible by the interaction count")
    result["layout"] = {"dim": dim, "layers": layers, "per_layer": dim // layers,
                        "order": "interaction layer 1 scalars, then layer 2 scalars"}
    result["standardization"] = {"fit": "train split atoms only", **train_stats}
    result["self_checks"] = self_checks(calc, load_split(args.data, "valid"), describe)
    (out / "manifest.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "standardization"}, indent=1))


if __name__ == "__main__":
    main()
