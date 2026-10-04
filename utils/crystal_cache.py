"""Read the versioned packed crystal cache without atom padding or truncation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from utils.crystal import CRYSTAL_SCHEMA, CrystalGeometry
from utils.paths import file_sha256


def crystal_data_contract(root, *, normalization, pair_budget=None):
    """Bind new training/checkpoints to a verified cache revision and loss mode."""
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if "schema" not in manifest:
        return None
    if manifest["schema"] != CRYSTAL_SCHEMA:
        raise ValueError("unsupported crystal schema")
    for name, entry in manifest["files"].items():
        if file_sha256(root / name) != entry["sha256"]:
            raise ValueError(f"crystal cache digest mismatch: {name}")
    return {"schema": CRYSTAL_SCHEMA, "manifest_sha256": file_sha256(path),
            "normalization": normalization, "pair_budget": pair_budget,
            "automatic_pair_budget": "max(effective_batch_size * 6400, largest_crystal_atoms**2)"}


class PackedCrystalStore:
    def __init__(self, root, split):
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest.get("schema") != CRYSTAL_SCHEMA:
            raise ValueError("unsupported crystal cache schema")
        directory = root / split
        arrays = {}
        for name, dtype in (("cell", np.float32), ("frac_coords", np.float32),
                            ("atomic_numbers", np.uint8), ("atom_offsets", np.int64),
                            ("n_atoms", np.int32)):
            array = np.load(directory / f"{name}_{split}.npy", allow_pickle=False)
            if array.dtype != dtype:
                raise ValueError(f"{name}: expected {np.dtype(dtype)}, got {array.dtype}")
            arrays[name] = array
        cell, frac, numbers, offsets, counts = (arrays[name] for name in (
            "cell", "frac_coords", "atomic_numbers", "atom_offsets", "n_atoms"))
        self.material_ids = np.load(directory / f"{split}_index.npy", allow_pickle=False)
        size = len(counts)
        if cell.shape != (size, 3, 3) or counts.shape != (size,):
            raise ValueError("cell/count row alignment failure")
        if offsets.shape != (size + 1,) or offsets[0] != 0 or (counts <= 0).any():
            raise ValueError("invalid atom offsets/counts")
        if not np.array_equal(np.diff(offsets), counts) or offsets[-1] != len(numbers):
            raise ValueError("offsets disagree with complete atom counts")
        if frac.shape != (len(numbers), 3) or numbers.ndim != 1:
            raise ValueError("packed atom fields disagree")
        if self.material_ids.shape != (size,) or len(set(self.material_ids)) != size:
            raise ValueError("invalid material ID alignment")
        if size != self.manifest["splits"][split]["n"]:
            raise ValueError("manifest row count differs")
        if not np.isfinite(cell).all() or not np.isfinite(frac).all():
            raise ValueError("non-finite packed geometry")
        if (numbers == 0).any() or (numbers > 118).any():
            raise ValueError("packed atoms cannot contain padding or sentinels")
        if (np.linalg.svd(cell, compute_uv=False).min(axis=-1) <= 1e-6).any():
            raise ValueError("degenerate packed crystal cell")
        self.cell = torch.from_numpy(cell)
        self.frac_coords = torch.from_numpy(frac)
        self.atomic_numbers = torch.from_numpy(numbers).long()
        self.offsets = offsets
        self.n_atoms = torch.from_numpy(counts).long()

    def __len__(self):
        return len(self.n_atoms)

    def __getitem__(self, index):
        start, end = self.offsets[index:index + 2]
        return (self.atomic_numbers[start:end],
                CrystalGeometry(self.cell[index], self.frac_coords[start:end]))
