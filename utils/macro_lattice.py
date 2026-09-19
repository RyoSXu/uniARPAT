"""CIF-only macro lattice features used by the optional E10 branch."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from utils.relative_features import build_cell_from_lattice


_TABLE_PATH = os.path.join(os.path.dirname(__file__), "periodic_table_v2.csv")


def raw_atomic_mass_table() -> torch.Tensor:
    """Return a Z-indexed raw atomic-mass table, with padding Z=0 fixed to zero."""
    masses = pd.read_csv(_TABLE_PATH)["AtomicMass"].to_numpy(dtype=np.float32)
    if not np.isfinite(masses).all():
        raise ValueError("E10 requires finite raw atomic masses for every supported element")
    return torch.from_numpy(np.concatenate(([0.0], masses)))


def macro_lattice_features(pos: torch.Tensor, atom_idx: torch.Tensor,
                           mask_atom: torch.Tensor, atomic_masses: torch.Tensor,
                           mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Return standardized [log(V/N), log(total_atomic_mass/V)] for each crystal.

    ``pos`` follows the repository's lattice-plus-fractional-coordinate contract.
    Atom counts and masses exclude padding, so padding length cannot alter either feature.
    """
    cell, _ = build_cell_from_lattice(pos)
    volume = torch.linalg.det(cell).abs().clamp_min(1e-12)
    valid = ~mask_atom
    n_atom = valid.sum(dim=-1).to(dtype=pos.dtype).clamp_min(1.0)
    mass = atomic_masses.to(device=atom_idx.device, dtype=pos.dtype)[atom_idx]
    total_mass = (mass * valid.to(dtype=pos.dtype)).sum(dim=-1).clamp_min(1e-12)
    raw = torch.stack((torch.log(volume / n_atom), torch.log(total_mass / volume)), dim=-1)
    return (raw - mean.to(device=raw.device, dtype=raw.dtype)) / std.to(device=raw.device, dtype=raw.dtype)
