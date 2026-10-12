"""Periodic local geometry updates and invariant global atom mixing.

Physical records enter only local messages. Every external state is an even
scalar; the global branch has no coordinate, direction or geometric-score API.
One local call retains the accepted two internal blocks and scalar readout.
"""
from __future__ import annotations

import torch
from torch import nn

from model.periodic_messages import PeriodicLocalMessage, prepare_shared_message_geometry
from model.local_structure_controls import CONTROL_ROUTES, ControlLocalMessage
from utils.periodic_geometry import PeriodicNeighborRecords

LOCAL_ROUTES = ("A", "B+", *CONTROL_ROUTES)
LOCAL_LAYOUTS = ("staged", "alternating")
LOCAL_RADIUS = 6.0
LOCAL_PAIR_CHUNK = 65536


class LocalResidualLayer(nn.Module):
    """Keep the full-width atom state around a pre-normalized local branch."""

    def __init__(self, route: str, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Full-model A keeps angle activations instead of recomputing them.
        # This execution choice adds no parameters or checkpoint keys.
        self.message = (ControlLocalMessage(route, d_model=d_model) if route in CONTROL_ROUTES
                        else PeriodicLocalMessage(route, d_model=d_model, pair_chunk=LOCAL_PAIR_CHUNK,
                                                  recompute_angles=False))

    def forward(self, h, records, padding_mask, prepared_geometry=None):
        normalized = self.norm(h)
        if prepared_geometry is None:
            branch = self.message(normalized, records, padding_mask)
        else:
            branch = self.message(normalized, records, padding_mask,
                                  prepared_geometry=prepared_geometry)
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
    """Independent local calls and independent global attention/FFN updates.

    staged: L1 ... LK G1 ... GG; alternating: L1 G1 ... LK GK G(K+1) ... GG,
    where the interleaved part covers ``min(K, G)`` pairs and any remaining
    update of the longer branch runs afterwards. The historical meaning is
    ``global_depth = local_depth``, so both layouts and every old checkpoint are
    unchanged. The same module/key ordering in every layout permits matched
    initialization. K=1 is supported for smoke checks.
    """

    def __init__(self, route: str, layout: str, depth: int, d_model: int = 512,
                 nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.05,
                 activation: str = "gelu", global_depth: int | None = None):
        super().__init__()
        if route not in LOCAL_ROUTES or layout not in LOCAL_LAYOUTS:
            raise ValueError("explicit local route and staged/alternating layout required")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("local_depth must be a positive integer")
        if global_depth is None:
            global_depth = depth
        if isinstance(global_depth, bool) or not isinstance(global_depth, int) or global_depth < 1:
            raise ValueError("global_depth must be a positive integer")
        self.route, self.layout, self.depth = route, layout, depth
        self.global_depth = global_depth
        self.d_model = d_model
        self.shared_geometry = route != "C0"
        self.local_layers = nn.ModuleList(LocalResidualLayer(route, d_model) for _ in range(depth))
        self.global_layers = nn.ModuleList(
            ScalarGlobalLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(global_depth))

    @property
    def update_counts(self):
        return {"local_calls": self.depth, "internal_local_blocks": 2 * self.depth,
                "global_attention": self.global_depth, "global_ffn": self.global_depth}

    @property
    def execution_order(self):
        if self.layout == "staged":
            return tuple((kind, i) for kind, count in
                         (("local", self.depth), ("global", self.global_depth))
                         for i in range(count))
        paired = min(self.depth, self.global_depth)
        order = [(kind, i) for i in range(paired) for kind in ("local", "global")]
        order.extend(("local", i) for i in range(paired, self.depth))
        order.extend(("global", i) for i in range(paired, self.global_depth))
        return tuple(order)

    def forward(self, h: torch.Tensor, records: PeriodicNeighborRecords | None,
                padding_mask: torch.Tensor, *, prepared_geometry=None):
        if h.shape != (*padding_mask.shape, self.d_model):
            raise ValueError("atom state does not match the local/global atom layout")
        if padding_mask.dtype != torch.bool or padding_mask.device != h.device:
            raise ValueError("boolean atom mask must share the state device")
        if self.route == "C0":
            if records is not None or prepared_geometry is not None:
                raise ValueError("C0 must receive no geometry")
        elif records is None or records.radius != LOCAL_RADIUS:
            raise ValueError("current local/global design requires radius 6.0 Angstrom")
        if prepared_geometry is None and self.shared_geometry:
            prepared_geometry = prepare_shared_message_geometry(records, self.route)
        h = h.masked_fill(padding_mask[..., None], 0)
        for kind, i in self.execution_order:
            if kind == "local":
                h = self.local_layers[i](h, records, padding_mask, prepared_geometry)
            else:
                h = self.global_layers[i](h, padding_mask)
        return h
