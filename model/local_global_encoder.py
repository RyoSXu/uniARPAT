"""Periodic local geometry updates and invariant global atom mixing.

Physical records enter only local messages. Every external state is an even
scalar; the global branch has no coordinate, direction or geometric-score API.
One local call retains the accepted two internal blocks and scalar readout.
"""
from __future__ import annotations

import torch
from torch import nn

from model.periodic_messages import PeriodicLocalMessage
from utils.periodic_geometry import PeriodicNeighborRecords

LOCAL_ROUTES = ("A", "B+")
LOCAL_LAYOUTS = ("staged", "alternating")
LOCAL_RADIUS = 6.0


class LocalResidualLayer(nn.Module):
    """Keep the full-width atom state around a pre-normalized local branch."""

    def __init__(self, route: str, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Full-model A keeps angle activations instead of recomputing them.
        # This execution choice adds no parameters or checkpoint keys.
        self.message = PeriodicLocalMessage(route, d_model=d_model, recompute_angles=False)

    def forward(self, h, records, padding_mask):
        branch = self.message(self.norm(h), records, padding_mask)
        return (h + branch).masked_fill(padding_mask[..., None], 0)


class ScalarGlobalLayer(nn.Module):
    """Learned Q/K/V attention and pointwise FFN, with separate pre-norm skips."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, activation: str):
        super().__init__()
        if activation not in ("gelu", "relu", "leaky_relu"):
            raise ValueError("local/global FFN supports gelu, relu or leaky_relu")
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        nonlinearity = {"gelu": nn.GELU, "relu": nn.ReLU,
                        "leaky_relu": nn.LeakyReLU}[activation]
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nonlinearity(),
                                 nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, h, padding_mask):
        h = h.masked_fill(padding_mask[..., None], 0)
        # Empty rows never enter an all-masked softmax. Keep their atom states zero.
        active = (~padding_mask).any(-1)
        if not active.any():
            return h
        values = h[active]
        mask = padding_mask[active]
        normalized = self.norm_attn(values).masked_fill(mask[..., None], 0)
        message = self.attn(normalized, normalized, normalized,
                            key_padding_mask=mask, need_weights=False)[0]
        values = (values + self.dropout_attn(message)).masked_fill(mask[..., None], 0)
        values = (values + self.dropout_ffn(self.ffn(self.norm_ffn(values)))).masked_fill(
            mask[..., None], 0)
        return h.index_copy(0, active.nonzero(as_tuple=True)[0], values)


class LocalGlobalEncoder(nn.Module):
    """K independent local calls and K independent global attention/FFN updates.

    staged: L1 ... LK G1 ... GK; alternating: L1 G1 ... LK GK.
    The same module/key ordering in both layouts permits matched initialization.
    K=1 is supported for smoke checks; it gives the same sequence in both layouts.
    """

    def __init__(self, route: str, layout: str, depth: int, d_model: int = 512,
                 nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.05,
                 activation: str = "gelu"):
        super().__init__()
        if route not in LOCAL_ROUTES or layout not in LOCAL_LAYOUTS:
            raise ValueError("explicit A/B+ route and staged/alternating layout required")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("local_depth must be a positive integer")
        self.route, self.layout, self.depth = route, layout, depth
        self.d_model = d_model
        self.local_layers = nn.ModuleList(LocalResidualLayer(route, d_model) for _ in range(depth))
        self.global_layers = nn.ModuleList(
            ScalarGlobalLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(depth))

    @property
    def update_counts(self):
        return {"local_calls": self.depth, "internal_local_blocks": 2 * self.depth,
                "global_attention": self.depth, "global_ffn": self.depth}

    @property
    def execution_order(self):
        if self.layout == "staged":
            return tuple((kind, i) for kind in ("local", "global") for i in range(self.depth))
        return tuple((kind, i) for i in range(self.depth) for kind in ("local", "global"))

    def forward(self, h: torch.Tensor, records: PeriodicNeighborRecords,
                padding_mask: torch.Tensor):
        if h.shape != (*padding_mask.shape, self.d_model):
            raise ValueError("atom state does not match the local/global atom layout")
        if padding_mask.dtype != torch.bool or padding_mask.device != h.device:
            raise ValueError("boolean atom mask must share the state device")
        if records.radius != LOCAL_RADIUS:
            raise ValueError("current local/global design requires radius 6.0 Angstrom")
        h = h.masked_fill(padding_mask[..., None], 0)
        for kind, i in self.execution_order:
            if kind == "local":
                h = self.local_layers[i](h, records, padding_mask)
            else:
                h = self.global_layers[i](h, padding_mask)
        return h
