"""Independent A/B+ local messages with an invariant scalar interface.

Input: scalar atom features, physical records, padding mask. Output: invariant
scalar atom features of the same shape. No atom initialization, global Encoder,
DOS head, experiment configuration or training runner is imported here.

The accepted prototype formulas/channels are retained. The historical CPU
implementations remain frozen numerical references, not dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from utils.periodic_neighbors import (
    g2_indegrees, g2_quintic_cutoff, g2_rbf_features, g2_rbf_params,
)
from utils.periodic_geometry import PeriodicNeighborRecords

SCALAR_CHANNELS = 64
RADIAL_CHANNELS = 16
STATE_IRREPS = o3.Irreps("64x0e + 8x1o + 4x2e")
EDGE_IRREPS = o3.Irreps.spherical_harmonics(2)
HIGH_ORDERS = (3, 4)
MOMENT_CHANNELS = 8
EXTRA_SCALARS = len(HIGH_ORDERS) * MOMENT_CHANNELS


@dataclass(frozen=True)
class _MessageGeometry:
    """Derived features owned by local messages, not the record contract."""

    unit: torch.Tensor
    radial: torch.Tensor
    cutoff: torch.Tensor
    degree: torch.Tensor
    effective_count: torch.Tensor
    pair_count: torch.Tensor
    receiver: torch.Tensor
    sender: torch.Tensor
    order: torch.Tensor
    pairs: tuple[torch.Tensor, torch.Tensor] | None = None
    angular_basis: torch.Tensor | None = None
    pair_cutoff: torch.Tensor | None = None
    pair_receiver: torch.Tensor | None = None
    harmonics: torch.Tensor | None = None
    record_context: tuple | None = None
    prepared_route: str | None = None

    @property
    def groups(self):
        """Explicit groups for diagnostics; normal forwards need no CPU list."""
        counts = torch.unique_consecutive(self.receiver[self.order], return_counts=True)[1]
        return tuple(self.order.split(counts.tolist())) if self.order.numel() else ()


def _record_context(records):
    fields = (records.batch, records.dst, records.src, records.shifts,
              records.vectors, records.distances, records.padding_mask)
    return (records.radius, tuple(records.padding_mask.shape),
            tuple((t.data_ptr(), None if t.is_inference() else t._version,
                   t.requires_grad, t.dtype, t.device) for t in fields))


def _prepare_message_geometry(records: PeriodicNeighborRecords, *, angle_pairs=False,
                              directions=True) -> _MessageGeometry:
    B, L = records.padding_mask.shape
    vectors, distances = records.vectors, records.distances
    unit = vectors / vectors.norm(dim=-1)[:, None] if directions else vectors.new_empty((0, 3))
    centers, width = g2_rbf_params(records.radius, RADIAL_CHANNELS,
                                 device=vectors.device, dtype=vectors.dtype)
    radial = g2_rbf_features(distances, records.radius, centers, width)
    cut = g2_quintic_cutoff(distances, records.radius)
    degree = g2_indegrees(records.batch, records.dst, B, L)
    receiver, sender = records.batch * L + records.dst, records.batch * L + records.src
    effective = vectors.new_zeros(B * L).index_add(0, receiver, cut).view(B, L)
    order = torch.argsort(receiver, stable=True)
    geometry = _MessageGeometry(unit, radial, cut, degree, effective,
                                degree * (degree - 1) // 2, receiver, sender, order,
                                record_context=_record_context(records))
    if angle_pairs:
        geometry = replace(geometry, pairs=_unordered_pairs(geometry))
    return geometry


def prepare_shared_message_geometry(records, route, *, pair_receivers=True):
    """Prepare structure-only features once, retaining any geometry gradients."""
    if route not in ("A", "B+", "C1"):
        raise ValueError("shared geometry requires A, B+ or C1")
    g = _prepare_message_geometry(records, angle_pairs=(route == "A"), directions=(route != "C1"))
    if route == "C1":
        return replace(g, prepared_route=route)
    if route == "A":
        left, right = g.pairs
        q = (g.unit[left] * g.unit[right]).sum(-1).clamp(-1, 1)
        return replace(g, angular_basis=torch.stack((q, q.square(), q**3, q**4), -1),
                       pair_cutoff=g.cutoff[left] * g.cutoff[right],
                       pair_receiver=g.receiver[left] if pair_receivers else None,
                       prepared_route=route)
    harmonics = o3.spherical_harmonics(list(range(5)), g.unit, normalize=False,
                                     normalization="component")
    return replace(g, harmonics=harmonics, prepared_route=route)


def _unordered_pairs(geometry):
    """All same-center pairs in the original upper-triangle record order.

    Invert triangular row offsets in float64, then use integer indices. No
    padded neighbor square, sampled pairs, or per-center GPU launches are used.
    """
    degree = geometry.degree.reshape(-1)
    counts = geometry.pair_count.reshape(-1)
    nodes = torch.repeat_interleave(torch.arange(len(degree), device=degree.device), counts)
    local = torch.arange(len(nodes), device=degree.device) - (counts.cumsum(0) - counts)[nodes]
    widths = 2 * degree[nodes] - 1
    row = ((widths.double() - (widths.square() - 8 * local).double().sqrt()) * .5).floor().long()
    row_start = row * (2 * degree[nodes] - row - 1) // 2
    column = row + 1 + local - row_start
    offset = (degree.cumsum(0) - degree)[nodes]
    return geometry.order[offset + row], geometry.order[offset + column]


def mlp(input_channels, output_channels, hidden_channels=128):
    return nn.Sequential(nn.Linear(input_channels, hidden_channels), nn.SiLU(),
                         nn.Linear(hidden_channels, output_channels))


def count_features(geometry, dtype):
    return torch.stack((geometry.degree.to(dtype).log1p(),
                        geometry.effective_count.to(dtype)), dim=-1)


class InvariantMessageBlock(nn.Module):
    """A: all unordered same-center pairs, exchange-symmetric scalar messages."""

    def __init__(self, pair_chunk=4096, recompute_angles=True, *, scalar_channels=64, hidden_channels=128,
                 norm_first=False):
        super().__init__()
        if not isinstance(pair_chunk, int) or isinstance(pair_chunk, bool) or pair_chunk <= 0:
            raise ValueError("pair chunk must be a positive integer")
        self.pair_chunk = pair_chunk
        self.recompute_angles = recompute_angles
        if not isinstance(norm_first, bool):
            raise ValueError("local norm_first must be boolean")
        self.norm_first = norm_first
        for width in (scalar_channels, hidden_channels):
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError("local widths must be positive integers")
        self.scalar_channels, self.hidden_channels = scalar_channels, hidden_channels
        self.edge_mlp = mlp(2 * scalar_channels + RADIAL_CHANNELS, scalar_channels, hidden_channels)
        self.angle_mlp = mlp(3 * scalar_channels + 2 * RADIAL_CHANNELS + 4, scalar_channels, hidden_channels)
        self.update = mlp(3 * scalar_channels + 2, scalar_channels, hidden_channels)
        self.norm = nn.LayerNorm(scalar_channels)

    def _angle_chunk(self, center_projection, first_projection, second_projection,
                     unit, cut, receiver, left, right, angular=None, pair_cutoff=None,
                     pair_receiver=None):
        if angular is None:
            q = (unit[left] * unit[right]).sum(-1).clamp(-1, 1)
            angular = torch.stack((q, q.square(), q**3, q**4), dim=-1)
        centers = receiver[left] if pair_receiver is None else pair_receiver
        angle_start = 3 * self.scalar_channels + 2 * RADIAL_CHANNELS
        shared = center_projection[centers] + F.linear(angular, self.angle_mlp[0].weight[:, angle_start:])
        forward = shared + first_projection[left] + second_projection[right]
        backward = shared + first_projection[right] + second_projection[left]
        # The nonlinearity stays after the complete original affine sum.
        message = (self.angle_mlp[2](self.angle_mlp[1](forward))
                   + self.angle_mlp[2](self.angle_mlp[1](backward))) * .5
        cutoff = cut[left] * cut[right] if pair_cutoff is None else pair_cutoff
        message = message * cutoff[:, None]
        return message.new_zeros((len(center_projection), self.scalar_channels)).index_add(
            0, centers, message)

    def forward(self, scalar, geometry):
        residual = scalar
        if self.norm_first:
            # Normalize the complete message/update branch, retaining the raw
            # scalar state on the internal identity connection.
            scalar = self.norm(scalar)
        shape = scalar.shape
        channels = self.scalar_channels
        flat = scalar.reshape(-1, channels)
        g = geometry
        edge_input = torch.cat((flat[g.receiver], flat[g.sender], g.radial), -1)
        message = self.edge_mlp(edge_input) * g.cutoff[:, None]
        radial = flat.new_zeros(flat.shape).index_add(0, g.receiver, message)
        radial = radial.view(shape) / g.degree.to(scalar.dtype).clamp(min=1).sqrt()[..., None]
        angular = flat.new_zeros(flat.shape)
        left_all, right_all = g.pairs if g.pairs is not None else _unordered_pairs(g)
        if left_all.numel():
            weight, bias = self.angle_mlp[0].weight, self.angle_mlp[0].bias
            radial_start = 3 * channels
            center = F.linear(flat, weight[:, :channels], bias)
            first = (F.linear(flat, weight[:, channels:2*channels])[g.sender]
                     + F.linear(g.radial, weight[:, radial_start:radial_start+RADIAL_CHANNELS]))
            second = (F.linear(flat, weight[:, 2*channels:3*channels])[g.sender]
                      + F.linear(g.radial, weight[:, radial_start+RADIAL_CHANNELS:radial_start+2*RADIAL_CHANNELS]))
            for start in range(0, len(left_all), self.pair_chunk):
                left, right = left_all[start:start+self.pair_chunk], right_all[start:start+self.pair_chunk]
                arguments = (center, first, second, g.unit, g.cutoff, g.receiver, left, right)
                if g.angular_basis is not None:
                    part = slice(start, start+self.pair_chunk)
                    arguments += (g.angular_basis[part], g.pair_cutoff[part],
                                  g.pair_receiver[part] if g.pair_receiver is not None else None)
                if self.recompute_angles and torch.is_grad_enabled():
                    pooled = checkpoint(self._angle_chunk, *arguments, use_reentrant=False,
                                        preserve_rng_state=False)
                else:
                    pooled = self._angle_chunk(*arguments)
                angular = angular + pooled
        angular = angular.view(shape) / g.pair_count.to(scalar.dtype).clamp(min=1).sqrt()[..., None]
        update_input = torch.cat((scalar, radial, angular, count_features(g, scalar.dtype)), -1)
        updated = residual + self.update(update_input)
        return updated if self.norm_first else self.norm(updated)


def high_order_pair_features(geometry, weights, harmonics):
    """B+ weighted P3/P4 pair sums via SH moments, without pair enumeration.

    Component-normalized Y_l gives ||sum wY_l||²/(2l+1)-sum w²
    = 2 sum_{e<f} w_e w_f P_l(u_e dot u_f). All images remain distinct.
    Weights include the cutoff; empty and single-edge centers return zero.
    """
    shape = geometry.degree.shape
    nodes = geometry.degree.numel()
    if weights.shape != (len(geometry.cutoff), MOMENT_CHANNELS):
        raise ValueError("eight weights required per periodic record")
    if harmonics.shape != (len(geometry.cutoff), 16):
        raise ValueError("l=3,4 harmonics must align with records")
    moments = weights.new_zeros(nodes, MOMENT_CHANNELS, 16).index_add(
        0, geometry.receiver, weights[..., None] * harmonics[:, None, :])
    diagonal = weights.new_zeros(nodes, MOMENT_CHANNELS).index_add(
        0, geometry.receiver, weights.square())
    pair_count = geometry.pair_count.reshape(-1)
    denominator = 2 * pair_count.to(weights.dtype).clamp(min=1).sqrt()
    features, offset = [], 0
    for order in HIGH_ORDERS:
        width = 2 * order + 1
        power = moments[..., offset:offset+width].square().sum(-1) / width
        pair_sum = (power - diagonal) / denominator[:, None]
        features.append(pair_sum.masked_fill(pair_count[:, None] == 0, 0))
        offset += width
    return torch.stack(features, dim=-1).reshape(*shape, EXTRA_SCALARS)


class HighOrderEquivariantBlock(nn.Module):
    """B+: persistent 0e/1o/2e state plus l=3,4 scalar pair-statistic inputs.

    Constructed from its own configuration. Shared-weight copying for a
    controlled A/B experiment belongs to that experiment, not this component.
    """

    def __init__(self, gate_direction=True):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(STATE_IRREPS, EDGE_IRREPS, STATE_IRREPS)
        self.mix = o3.Linear(STATE_IRREPS, STATE_IRREPS, biases=False)
        self.radial_gate = mlp(144, 76)
        self.update = mlp(78 + EXTRA_SCALARS, 64)
        self.norm = nn.LayerNorm(64)
        self.direction_gate = nn.Linear(64, 12) if gate_direction else None

    @staticmethod
    def invariants(state):
        shape = state.shape[:-1]
        vectors = state[..., 64:88].reshape(*shape, 8, 3).square().sum(-1)
        tensors = state[..., 88:].reshape(*shape, 4, 5).square().sum(-1)
        return torch.cat((vectors, tensors), -1)

    def forward(self, state, geometry, harmonics, include_high_order=True):
        shape = state.shape
        flat = state.reshape(-1, 108)
        g = geometry
        gate_input = torch.cat((flat[g.receiver, :64], flat[g.sender, :64], g.radial), -1)
        gates = self.radial_gate(gate_input)
        angular = high_order_pair_features(g, gates[:, 64:72] * g.cutoff[:, None],
                                          harmonics[:, 9:])
        if not include_high_order:
            angular = torch.zeros_like(angular)
        expanded = torch.cat((gates[:, :64], gates[:, 64:72].repeat_interleave(3, -1),
                              gates[:, 72:].repeat_interleave(5, -1)), -1)
        message = self.tp(flat[g.sender], harmonics[:, :9]) * expanded * g.cutoff[:, None]
        aggregate = flat.new_zeros(flat.shape).index_add(0, g.receiver, message).view(shape)
        aggregate = aggregate / g.degree.to(state.dtype).clamp(min=1).sqrt()[..., None]
        mixed = state + self.mix(aggregate)
        original_input = torch.cat((mixed[..., :64], self.invariants(mixed),
                                    count_features(g, state.dtype)), -1)
        scalar = self.norm(mixed[..., :64] + self.update(torch.cat((original_input, angular), -1)))
        direction = mixed[..., 64:]
        if self.direction_gate is not None:
            direction_gates = torch.sigmoid(self.direction_gate(scalar))
            direction = direction * torch.cat(
                (direction_gates[..., :8].repeat_interleave(3, -1),
                 direction_gates[..., 8:].repeat_interleave(5, -1)), -1)
        return torch.cat((scalar, direction), -1), angular


class PeriodicLocalMessage(nn.Module):
    """A/B+ share ``forward(h, records, padding_mask) -> h_local``.

    h/h_local: invariant scalars [B,L,d_model], padding output exactly zero.
    A supports explicit scalar/MLP widths; defaults retain the accepted 64/128
    initialization and parameter keys. B+ keeps its 108-component state.
    The module follows normal PyTorch ``.to(device, dtype)``; records must be
    moved explicitly too. FP32/FP64 are supported; AMP is not yet accepted.
    Local O(3) invariance assumes invariant input h and covariant input vectors;
    it does not establish whole-model spectrum invariance.
    """

    def __init__(self, route: str, d_model: int = 512, pair_chunk: int = 4096,
                 recompute_angles: bool = True, include_high_order: bool = True,
                 *, scalar_channels: int = 64, hidden_channels: int = 128, norm_first: bool = False):
        super().__init__()
        if route not in ("A", "B+"):
            raise ValueError("route must be A or B+")
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        for width in (scalar_channels, hidden_channels):
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError("local widths must be positive integers")
        if route == "B+" and (scalar_channels, hidden_channels) != (64, 128):
            raise ValueError("B+ widths retain the accepted 64/128 configuration")
        if not isinstance(norm_first, bool) or route == "B+" and norm_first:
            raise ValueError("internal pre-norm is supported only for A; norm_first must be boolean")
        self.route, self.d_model = route, d_model
        self.norm_first = norm_first
        self.scalar_channels, self.hidden_channels = scalar_channels, hidden_channels
        self.include_high_order = include_high_order
        self.local_input = nn.Linear(d_model, scalar_channels)
        self.local_output = nn.Linear(scalar_channels, d_model)
        if route == "A":
            self.blocks = nn.ModuleList(InvariantMessageBlock(pair_chunk, recompute_angles,
                scalar_channels=scalar_channels, hidden_channels=hidden_channels,
                norm_first=norm_first) for _ in range(2))
        else:
            self.blocks = nn.ModuleList(HighOrderEquivariantBlock(gate_direction=(i == 0)) for i in range(2))

    def forward(self, h: torch.Tensor, records: PeriodicNeighborRecords,
                padding_mask: torch.Tensor, *, return_debug: bool = False,
                prepared_geometry: _MessageGeometry | None = None):
        if not isinstance(records, PeriodicNeighborRecords):
            raise TypeError("physical PeriodicNeighborRecords required")
        if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
            raise ValueError("boolean [B,L] padding mask required")
        if padding_mask.device != records.padding_mask.device or not torch.equal(padding_mask, records.padding_mask):
            raise ValueError("padding mask differs from record atom-slot layout")
        if h.shape != (*padding_mask.shape, self.d_model) or h.dtype != records.vectors.dtype:
            raise ValueError("scalar atom shape/dtype does not match records")
        if h.device != records.vectors.device or h.device != self.local_input.weight.device:
            raise ValueError("atom features, records and module must share a device")
        if h.dtype != self.local_input.weight.dtype:
            raise ValueError("atom features and module must share a dtype")
        if not torch.isfinite(h[~padding_mask]).all():
            raise ValueError("finite scalars required on valid atom slots")
        if prepared_geometry is None:
            g = _prepare_message_geometry(records, angle_pairs=(self.route == "A"))
        else:
            g = prepared_geometry
            if g.record_context != _record_context(records) or g.prepared_route != self.route:
                raise ValueError("prepared geometry differs from physical records or route")
        atoms = h.masked_fill(padding_mask[..., None], 0)
        scalar = self.local_input(atoms).masked_fill(padding_mask[..., None], 0)
        states, angular_states = [scalar], []
        if self.route == "B+":
            state = torch.cat((scalar, scalar.new_zeros((*padding_mask.shape, 44))), -1)
            states = [state]
            harmonics = g.harmonics
            if harmonics is None:
                harmonics = o3.spherical_harmonics(list(range(5)), g.unit, normalize=False,
                                                 normalization="component")
            for block in self.blocks:
                state, angular = block(state, g, harmonics, self.include_high_order)
                state = state.masked_fill(padding_mask[..., None], 0)
                states.append(state)
                angular_states.append(angular.masked_fill(padding_mask[..., None], 0))
            scalar = state[..., :64]
        else:
            for block in self.blocks:
                scalar = block(scalar, g).masked_fill(padding_mask[..., None], 0)
                states.append(scalar)
        output = self.local_output(scalar).masked_fill(padding_mask[..., None], 0)
        if return_debug:
            debug = {"states": states, "degree": g.degree,
                     "effective_count": g.effective_count, "pair_count": g.pair_count}
            if self.route == "B+":
                debug["angular_invariants"] = angular_states
            return output, debug
        return output
