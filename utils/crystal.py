"""Explicit crystal geometry: row lattice vectors in Å and fractional atoms.

No lattice tokens, atom-count cap, or Cartesian-frame canonicalization is
performed here. FP64 is supported for numerical reference checks; production
cache arrays use FP32. A negative determinant is a valid reflected frame.
"""
from __future__ import annotations

from typing import NamedTuple

import torch


CRYSTAL_SCHEMA = "crystal-q1-v1"


class CrystalGeometry(NamedTuple):
    cell: torch.Tensor
    frac_coords: torch.Tensor

    def to(self, *args, **kwargs):
        return CrystalGeometry(self.cell.to(*args, **kwargs),
                               self.frac_coords.to(*args, **kwargs))

    def new_empty(self, *args, **kwargs):
        """Allocate geometry temporaries for historical helper compatibility."""
        return self.cell.new_empty(*args, **kwargs)

    @property
    def device(self):
        return self.cell.device

    @property
    def dtype(self):
        return self.cell.dtype


def validate_crystal_geometry(geometry: CrystalGeometry,
                              padding_mask: torch.Tensor) -> None:
    cell, frac = geometry
    if frac.ndim != 3 or frac.shape[-1] != 3:
        raise ValueError("fractional coordinates must have shape [B,L,3]")
    if cell.shape != (frac.shape[0], 3, 3):
        raise ValueError("row lattice vectors must have shape [B,3,3]")
    if cell.dtype not in (torch.float32, torch.float64) or frac.dtype != cell.dtype:
        raise ValueError("cell and coordinates require the same FP32/FP64 dtype")
    if padding_mask.shape != frac.shape[:2] or padding_mask.dtype != torch.bool:
        raise ValueError("boolean padding mask must align with atom slots")
    if frac.device != cell.device or padding_mask.device != cell.device:
        raise ValueError("cell, coordinates and mask must share a device")
    with torch.no_grad():
        if not torch.isfinite(cell).all() or not torch.isfinite(frac[~padding_mask]).all():
            raise ValueError("non-finite cell or valid fractional coordinates")
        if (torch.linalg.svdvals(cell).amin(-1) <= 1e-6).any():
            raise ValueError("degenerate crystal cell")


def atom_padding_mask(geometry, mask: torch.Tensor) -> torch.Tensor:
    """Select actual atom slots at the explicit legacy compatibility boundary."""
    if isinstance(geometry, CrystalGeometry):
        if mask.shape != geometry.frac_coords.shape[:2]:
            raise ValueError("atom mask and fractional coordinates differ")
        return mask
    return mask[:, 2:2 + geometry.shape[1] - 2]


def structure_to_crystal_inputs(structure, *, primitive: bool = True):
    """Convert an ordered CIF/Structure without clipping any atoms.

    Production uses the historical MP standard primitive policy, symprec=0.1,
    and the same canonical Cartesian frame as the migrated Q1 cache.
    ``primitive=False`` preserves an explicitly supplied matrix/frame instead.
    This converter does not authenticate or apply a historical checkpoint.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    if not structure.is_ordered or len(structure) == 0:
        raise ValueError("a nonempty ordered structure is required")
    if primitive:
        structure = SpacegroupAnalyzer(structure, symprec=0.1).get_primitive_standard_structure()
    numbers = torch.tensor(structure.atomic_numbers, dtype=torch.long)
    if ((numbers < 1) | (numbers > 118)).any():
        raise ValueError("atomic numbers must lie in [1,118]")
    geometry = CrystalGeometry(torch.tensor(structure.lattice.matrix, dtype=torch.float32),
                               torch.tensor(structure.frac_coords, dtype=torch.float32))
    if primitive:
        # One-time import conversion, matching A4's FP32 scalar quantization.
        # Native geometry consumers never reconstruct or recanonicalize cell.
        from utils.relative_features import build_cell_from_lattice
        a, b, c = structure.lattice.abc
        parameters = torch.tensor([[[a, b, 1 / c], structure.lattice.angles]], dtype=torch.float32)
        cell, _ = build_cell_from_lattice(parameters)
        geometry = CrystalGeometry(cell[0], geometry.frac_coords)
    validate_crystal_geometry(CrystalGeometry(geometry.cell[None], geometry.frac_coords[None]),
                              torch.zeros(1, len(numbers), dtype=torch.bool))
    return numbers, geometry
