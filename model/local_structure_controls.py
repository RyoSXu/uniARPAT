"""C0 pointwise and C1 distance-only controls for the accepted scalar backbone.

C0 accepts no physical records. C1 uses the existing all-image records, RBF,
cutoff and counts, without angular features or directional hidden channels.
Both calls keep the 512 -> 64 -> 512 projections and two scalar residual blocks.
"""
from __future__ import annotations

import torch
from torch import nn

from model.periodic_messages import (
    _record_context, count_features, mlp, prepare_shared_message_geometry,
)
from utils.periodic_geometry import PeriodicNeighborRecords

CONTROL_ROUTES = ("C0", "C1")


class PointwiseControlBlock(nn.Module):
    """An atom-only residual block, with 51,985 effective trainable parameters."""

    def __init__(self):
        super().__init__()
        # Widths keep the two-stage residual organization and almost match C1's
        # 51,968 parameters. Every parameter participates in the forward pass.
        self.point_mlp = nn.Sequential(nn.Linear(64, 208), nn.SiLU(), nn.Linear(208, 64))
        self.update = nn.Sequential(nn.Linear(128, 129), nn.SiLU(), nn.Linear(129, 64))
        self.norm = nn.LayerNorm(64)

    def forward(self, scalar):
        message = self.point_mlp(scalar)
        return self.norm(scalar + self.update(torch.cat((scalar, message), dim=-1)))


class RadialControlBlock(nn.Module):
    """Scalar neighbor content, with the accepted cutoff and sqrt-degree sum."""

    def __init__(self):
        super().__init__()
        self.edge_mlp = mlp(144, 64)
        self.update = mlp(130, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, scalar, geometry):
        flat = scalar.reshape(-1, 64)
        edge = self.edge_mlp(torch.cat((flat[geometry.receiver], flat[geometry.sender],
                                        geometry.radial), dim=-1))
        radial = torch.zeros_like(flat).index_add(
            0, geometry.receiver, geometry.cutoff[:, None] * edge).view_as(scalar)
        radial = radial / geometry.degree.to(scalar.dtype).clamp_min(1).sqrt()[..., None]
        update = self.update(torch.cat((scalar, radial, count_features(geometry, scalar.dtype)), -1))
        return self.norm(scalar + update)


class ControlLocalMessage(nn.Module):
    """Same outer scalar interface as A/B+, with control-specific message content."""

    def __init__(self, route: str, d_model: int = 512):
        super().__init__()
        if route not in CONTROL_ROUTES:
            raise ValueError("structure control route must be C0 or C1")
        self.route, self.d_model = route, d_model
        self.local_input = nn.Linear(d_model, 64)
        self.local_output = nn.Linear(64, d_model)
        block = PointwiseControlBlock if route == "C0" else RadialControlBlock
        self.blocks = nn.ModuleList(block() for _ in range(2))

    def forward(self, h, records, padding_mask, *, prepared_geometry=None):
        if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
            raise ValueError("boolean [B,L] padding mask required")
        if h.shape != (*padding_mask.shape, self.d_model):
            raise ValueError("scalar atom shape does not match the control atom layout")
        if h.device != padding_mask.device or h.device != self.local_input.weight.device:
            raise ValueError("atom features, mask and module must share a device")
        if h.dtype != self.local_input.weight.dtype or not torch.isfinite(h[~padding_mask]).all():
            raise ValueError("finite scalars with the module dtype required on valid atom slots")
        geometry = None
        if self.route == "C0":
            if records is not None or prepared_geometry is not None:
                raise ValueError("C0 must receive no physical records or prepared geometry")
        else:
            if not isinstance(records, PeriodicNeighborRecords):
                raise TypeError("C1 requires physical PeriodicNeighborRecords")
            if (records.vectors.device != h.device or records.vectors.dtype != h.dtype
                    or not torch.equal(padding_mask, records.padding_mask)):
                raise ValueError("physical records must match the control atom layout/device/dtype")
            geometry = prepared_geometry
            if geometry is None:
                geometry = prepare_shared_message_geometry(records, "C1")
            if geometry.record_context != _record_context(records) or geometry.prepared_route != "C1":
                raise ValueError("prepared geometry differs from C1 physical records")
        scalar = self.local_input(h.masked_fill(padding_mask[..., None], 0))
        scalar = scalar.masked_fill(padding_mask[..., None], 0)
        for block in self.blocks:
            scalar = (block(scalar) if geometry is None else block(scalar, geometry))
            scalar = scalar.masked_fill(padding_mask[..., None], 0)
        return self.local_output(scalar).masked_fill(padding_mask[..., None], 0)
