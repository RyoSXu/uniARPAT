"""Physical periodic records, independent of atom features and model code.

All distances/vectors are in Angstrom. Vectors point from the receiver to the
sender image. The radius is explicit; enumeration retains every image with
d < radius and excludes only the zero-image self record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from utils.g2_periodic_edges import build_g2_edges
from utils.relative_features import build_cell_from_lattice


def _validate_radius(radius: float) -> None:
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")


def validate_periodic_inputs(pos: torch.Tensor, padding_mask: torch.Tensor) -> None:
    """Validate the existing (a,b,1/c), angles, fractional-coordinate format."""
    if pos.ndim != 3 or pos.shape[1] < 2 or pos.shape[-1] != 3:
        raise ValueError("positions must have shape [B,L+2,3]")
    if pos.dtype not in (torch.float32, torch.float64):
        raise ValueError("float32 or float64 positions required")
    if padding_mask.shape != pos[:, 2:].shape[:2] or padding_mask.dtype != torch.bool:
        raise ValueError("boolean padding mask must align with atom slots")
    if padding_mask.device != pos.device:
        raise ValueError("positions and padding mask must share a device")
    with torch.no_grad():
        if not torch.isfinite(pos[:, :2]).all() or not torch.isfinite(pos[:, 2:][~padding_mask]).all():
            raise ValueError("non-finite lattice or valid atom coordinates")
        if not (pos[:, 0] > 0).all():
            raise ValueError("a, b and inverse c must be positive")
        if not ((pos[:, 1] > 0) & (pos[:, 1] < 180)).all():
            raise ValueError("lattice angles must lie strictly between 0 and 180 degrees")
        cell, _ = build_cell_from_lattice(pos)
        smin = torch.linalg.svdvals(cell).amin(dim=-1)
        if not torch.isfinite(smin).all() or (smin <= 1e-6).any():
            raise ValueError("non-finite or degenerate reconstructed cell")


def periodic_edge_vectors(pos: torch.Tensor, edges: dict) -> torch.Tensor:
    """Reconstruct r=(f_src-f_dst+T)A in the same order as the edge IDs.

    Extracted from the historical Encoder helper; its old definition remains
    frozen for earlier evidence. This utility does not import that Encoder.
    """
    cell, frac = build_cell_from_lattice(pos)
    batch, dst, src = edges["batch"], edges["dst"], edges["src"]
    if batch.numel() == 0:
        return pos.new_empty((0, 3))
    frac_delta = frac[batch, src] - frac[batch, dst]
    frac_delta = frac_delta + edges["shifts"].to(frac_delta.dtype)
    return torch.bmm(frac_delta.unsqueeze(1), cell[batch]).squeeze(1)


@dataclass(frozen=True)
class PeriodicNeighborRecords:
    """Six physical fields plus radius and a snapshot of the atom-slot mask.

    The mask/radius are context metadata, not edge features. Use ``from_edges``
    for supplied Cartesian frames and ``build_periodic_records`` for dataset
    positions. Supplied records are checked for alignment, not independently
    enumerated for completeness. Tensor fields must not be mutated in place.
    """

    batch: torch.Tensor
    dst: torch.Tensor
    src: torch.Tensor
    shifts: torch.Tensor
    vectors: torch.Tensor
    distances: torch.Tensor
    radius: float
    padding_mask: torch.Tensor

    def __post_init__(self) -> None:
        _validate_radius(self.radius)
        mask, vectors, distances = self.padding_mask, self.vectors, self.distances
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("boolean [B,L] padding mask required")
        if distances.ndim != 1 or distances.dtype not in (torch.float32, torch.float64):
            raise ValueError("float32 or float64 [E] distances required")
        size = distances.numel()
        if vectors.shape != (size, 3) or vectors.dtype != distances.dtype:
            raise ValueError("vectors must align with distances in shape and dtype")
        for name in ("batch", "dst", "src"):
            value = getattr(self, name)
            if value.shape != (size,) or value.dtype != torch.long:
                raise ValueError(f"invalid {name} record indices")
        if self.shifts.shape != (size, 3) or self.shifts.dtype != torch.long:
            raise ValueError("invalid integer image shifts")
        for name in ("batch", "dst", "src", "shifts", "vectors", "padding_mask"):
            if getattr(self, name).device != distances.device:
                raise ValueError("all record fields and mask must share a device")
        with torch.no_grad():
            norm = vectors.norm(dim=-1)
            if not torch.isfinite(distances).all() or not torch.isfinite(vectors).all():
                raise ValueError("non-finite record geometry")
            if (distances <= 0).any() or (norm <= 0).any():
                raise ValueError("zero-length record has undefined direction")
            if (distances >= self.radius).any():
                raise ValueError("record violates strict cutoff")
            tolerance = 1e-4 if vectors.dtype == torch.float32 else 1e-6
            if not torch.allclose(norm, distances, atol=tolerance, rtol=0):
                raise ValueError("vector norm and record distance disagree")
            B, L = mask.shape
            if size:
                if ((self.batch < 0) | (self.batch >= B) | (self.dst < 0) |
                    (self.dst >= L) | (self.src < 0) | (self.src >= L)).any():
                    raise ValueError("record index out of bounds")
                if mask[self.batch, self.dst].any() or mask[self.batch, self.src].any():
                    raise ValueError("padding occurs in record")
                if ((self.dst == self.src) & (self.shifts == 0).all(-1)).any():
                    raise ValueError("zero-image self record is forbidden")
        # Do not retain a caller-owned mutable mask as the layout contract.
        object.__setattr__(self, "padding_mask", mask.clone())

    @classmethod
    def from_edges(cls, edges: dict, vectors: torch.Tensor,
                   padding_mask: torch.Tensor, radius: float) -> PeriodicNeighborRecords:
        """Adapt audited edge IDs/d into six-field physical records; no RBF/SH."""
        if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
            raise ValueError("boolean [B,L] padding mask required")
        if "L" in edges and edges["L"] != padding_mask.shape[-1]:
            raise ValueError("edge slot count differs from padding mask")
        return cls(*(edges[name] for name in ("batch", "dst", "src", "shifts")),
                   vectors, edges["distances"], radius, padding_mask)

    def to(self, device=None, dtype=None) -> PeriodicNeighborRecords:
        """Move records with the model, preserving integer IDs and gradients."""
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise ValueError("records support float32 and float64")
        values = {name: getattr(self, name).to(device=device)
                  for name in ("batch", "dst", "src", "shifts", "padding_mask")}
        values.update({name: getattr(self, name).to(device=device, dtype=dtype)
                       for name in ("vectors", "distances")})
        return replace(self, **values)


def build_periodic_records(pos: torch.Tensor, padding_mask: torch.Tensor,
                           radius: float) -> PeriodicNeighborRecords:
    """Validate inputs, reuse complete image enumeration, and reconstruct r.

    Edge selection and the enumerated distances use the existing no-grad
    builder. This does not promise differentiability through neighbor changes.
    ``from_edges`` also accepts differentiable r/d when supplied by a caller.
    """
    _validate_radius(radius)
    validate_periodic_inputs(pos, padding_mask)
    edges = build_g2_edges(pos, padding_mask, r_cut=radius)
    return PeriodicNeighborRecords.from_edges(
        edges, periodic_edge_vectors(pos, edges), padding_mask, radius)
