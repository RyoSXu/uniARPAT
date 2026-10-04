"""Standalone CPU message candidates. Not connected to the DOS model or trainer.

Geometry is supplied separately so arbitrary Cartesian frames can be tested.
The radius is explicit; all periodic records are used, without neighbor pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from e3nn import o3
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.periodic_manybody import periodic_edge_vectors
from tools.eval.periodic_neighbor_radius_audit import validate_inputs
from utils.g2_periodic_edges import (
    build_g2_edges, g2_indegrees, g2_quintic_cutoff, g2_rbf_features, g2_rbf_params,
)

SCALAR_CHANNELS = 64
RADIAL_CHANNELS = 16
STATE_IRREPS = o3.Irreps("64x0e + 8x1o + 4x2e")
EDGE_IRREPS = o3.Irreps.spherical_harmonics(2)


@dataclass(frozen=True)
class ProbeGeometry:
    edges: dict
    vectors: torch.Tensor
    unit: torch.Tensor
    radial: torch.Tensor
    cutoff: torch.Tensor
    degree: torch.Tensor
    effective_count: torch.Tensor
    pair_count: torch.Tensor
    receiver: torch.Tensor
    sender: torch.Tensor
    groups: tuple[torch.Tensor, ...]
    radius: float


def prepare_geometry(edges, vectors, mask, radius=6.0):
    """Check the supplied record/vector interface and prepare shared features.

    A zero-length interatomic record raises an error; it is never silently lost.
    Duplicate or missing records are the geometric caller's contract, already
    audited separately. This function checks alignment, padding and distances.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if mask.ndim != 2 or mask.dtype != torch.bool or mask.device.type != "cpu":
        raise ValueError("CPU boolean [B,L] mask required")
    B, L = mask.shape
    size = edges["distances"].numel()
    if vectors.shape != (size, 3) or vectors.device.type != "cpu":
        raise ValueError("CPU vectors must align with [E,3] records")
    if vectors.dtype not in (torch.float32, torch.float64):
        raise ValueError("float32 or float64 vectors required")
    for key in ("batch", "dst", "src"):
        value = edges[key]
        if value.shape != (size,) or value.dtype != torch.long or value.device.type != "cpu":
            raise ValueError(f"invalid {key} record indices")
    if edges["shifts"].shape != (size, 3) or edges["shifts"].dtype != torch.long:
        raise ValueError("invalid integer image shifts")
    distances = edges["distances"]
    if distances.shape != (size,) or distances.dtype != vectors.dtype:
        raise ValueError("distance dtype/shape differs from vectors")
    norm = vectors.norm(dim=-1)
    with torch.no_grad():
        if not torch.isfinite(distances).all() or not torch.isfinite(vectors).all():
            raise ValueError("non-finite record geometry")
        if (distances <= 0).any() or (norm <= 0).any():
            raise ValueError("zero-length record has undefined direction")
        if (distances >= radius).any():
            raise ValueError("record violates strict cutoff")
        tolerance = 1e-4 if vectors.dtype == torch.float32 else 1e-6
        if not torch.allclose(norm, distances, atol=tolerance, rtol=0):
            raise ValueError("vector norm and record distance disagree")
        eb, dst, src = edges["batch"], edges["dst"], edges["src"]
        if size:
            if ((eb < 0) | (eb >= B) | (dst < 0) | (dst >= L) | (src < 0) | (src >= L)).any():
                raise ValueError("record index out of bounds")
            if mask[eb, dst].any() or mask[eb, src].any():
                raise ValueError("padding occurs in record")
            if ((dst == src) & (edges["shifts"] == 0).all(-1)).any():
                raise ValueError("zero-image self record is forbidden")
    unit = vectors / norm[:, None]
    centers, width = g2_rbf_params(radius, RADIAL_CHANNELS, dtype=vectors.dtype)
    radial = g2_rbf_features(distances, radius, centers, width)
    cut = g2_quintic_cutoff(distances, radius)
    degree = g2_indegrees(eb, dst, B, L)
    receiver, sender = eb * L + dst, eb * L + src
    effective = vectors.new_zeros(B * L).index_add(0, receiver, cut).view(B, L)
    order = torch.argsort(receiver, stable=True)
    counts = torch.unique_consecutive(receiver[order], return_counts=True)[1]
    groups = tuple(order.split(counts.tolist())) if size else ()
    return ProbeGeometry(edges, vectors, unit, radial, cut, degree, effective,
                         degree * (degree-1) // 2, receiver, sender, groups, radius)


def geometry_from_pos(pos, mask, radius=6.0):
    """Reuse the audited builder, with finite input guards at this caller."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    validate_inputs(pos, mask)
    edges = build_g2_edges(pos, mask, r_cut=radius)
    return prepare_geometry(edges, periodic_edge_vectors(pos, edges), mask, radius)


def mlp(input_channels, output_channels):
    return nn.Sequential(nn.Linear(input_channels, 128), nn.SiLU(),
                         nn.Linear(128, output_channels))


def count_features(geometry, dtype):
    return torch.stack((geometry.degree.to(dtype).log1p(),
                        geometry.effective_count.to(dtype)), dim=-1)


class InvariantMessageBlock(nn.Module):
    """All unordered same-center pairs, exchange-symmetric scalar messages."""

    def __init__(self, pair_chunk=4096, recompute_angles=True):
        super().__init__()
        if pair_chunk <= 0:
            raise ValueError("pair chunk must be positive")
        self.pair_chunk = pair_chunk
        self.recompute_angles = recompute_angles
        self.edge_mlp = mlp(144, 64)
        self.angle_mlp = mlp(228, 64)
        self.update = mlp(194, 64)
        self.norm = nn.LayerNorm(64)

    def _angle_sum(self, state, radial, unit, cut, sender, receiver, left, right):
        q = (unit[left] * unit[right]).sum(-1).clamp(-1, 1)
        angular = torch.stack((q, q.square(), q**3, q**4), dim=-1)
        center = state[receiver[left]]
        first, second = state[sender[left]], state[sender[right]]
        forward = torch.cat((center, first, second, radial[left], radial[right], angular), -1)
        backward = torch.cat((center, second, first, radial[right], radial[left], angular), -1)
        message = (self.angle_mlp(forward) + self.angle_mlp(backward)) * .5
        return (message * (cut[left] * cut[right])[:, None]).sum(0)

    def forward(self, scalar, geometry):
        shape = scalar.shape
        flat = scalar.reshape(-1, 64)
        g = geometry
        edge_input = torch.cat((flat[g.receiver], flat[g.sender], g.radial), -1)
        message = self.edge_mlp(edge_input) * g.cutoff[:, None]
        radial = flat.new_zeros(flat.shape).index_add(0, g.receiver, message)
        radial = radial.view(shape) / g.degree.to(scalar.dtype).clamp(min=1).sqrt()[..., None]
        centers, totals = [], []
        for group in g.groups:
            if len(group) < 2:
                continue
            pairs = torch.triu_indices(len(group), len(group), offset=1)
            total = flat.new_zeros(64)
            for start in range(0, pairs.shape[1], self.pair_chunk):
                left = group[pairs[0, start:start+self.pair_chunk]]
                right = group[pairs[1, start:start+self.pair_chunk]]
                arguments = (flat, g.radial, g.unit, g.cutoff, g.sender, g.receiver, left, right)
                if self.recompute_angles and torch.is_grad_enabled():
                    pooled = checkpoint(self._angle_sum, *arguments, use_reentrant=False,
                                        preserve_rng_state=False)
                else:
                    pooled = self._angle_sum(*arguments)
                total = total + pooled
            centers.append(g.receiver[group[0]])
            totals.append(total)
        angular = flat.new_zeros(flat.shape)
        if centers:
            angular = angular.index_add(0, torch.stack(centers), torch.stack(totals))
        angular = angular.view(shape) / g.pair_count.to(scalar.dtype).clamp(min=1).sqrt()[..., None]
        update_input = torch.cat((scalar, radial, angular, count_features(g, scalar.dtype)), -1)
        return self.norm(scalar + self.update(update_input))


class EquivariantMessageBlock(nn.Module):
    """Persistent 0e/1o/2e states; direction norms directly update 0e content."""

    def __init__(self, gate_direction=True):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(STATE_IRREPS, EDGE_IRREPS, STATE_IRREPS)
        self.mix = o3.Linear(STATE_IRREPS, STATE_IRREPS, biases=False)
        self.radial_gate = mlp(144, 76)
        self.update = mlp(78, 64)
        self.norm = nn.LayerNorm(64)
        self.direction_gate = nn.Linear(64, 12) if gate_direction else None

    @staticmethod
    def invariants(state):
        shape = state.shape[:-1]
        vectors = state[..., 64:88].reshape(*shape, 8, 3).square().sum(-1)
        tensors = state[..., 88:].reshape(*shape, 4, 5).square().sum(-1)
        return torch.cat((vectors, tensors), -1)

    def forward(self, state, geometry, harmonics):
        shape = state.shape
        flat = state.reshape(-1, 108)
        g = geometry
        gate_input = torch.cat((flat[g.receiver, :64], flat[g.sender, :64], g.radial), -1)
        gates = self.radial_gate(gate_input)
        gates = torch.cat((gates[:, :64], gates[:, 64:72].repeat_interleave(3, -1),
                           gates[:, 72:].repeat_interleave(5, -1)), -1)
        message = self.tp(flat[g.sender], harmonics) * gates * g.cutoff[:, None]
        aggregate = flat.new_zeros(flat.shape).index_add(0, g.receiver, message).view(shape)
        aggregate = aggregate / g.degree.to(state.dtype).clamp(min=1).sqrt()[..., None]
        mixed = state + self.mix(aggregate)
        scalar_input = torch.cat((mixed[..., :64], self.invariants(mixed),
                                  count_features(g, state.dtype)), -1)
        scalar = self.norm(mixed[..., :64] + self.update(scalar_input))
        direction = mixed[..., 64:]
        if self.direction_gate is not None:
            direction_gates = torch.sigmoid(self.direction_gate(scalar))
            direction = direction * torch.cat(
                (direction_gates[..., :8].repeat_interleave(3, -1),
                 direction_gates[..., 8:].repeat_interleave(5, -1)), -1)
        return torch.cat((scalar, direction), -1)


class PeriodicMessageProbe(nn.Module):
    """Same scalar interface for two blocks of either independent candidate."""

    def __init__(self, route, pair_chunk=4096, recompute_angles=True):
        super().__init__()
        if route not in ("invariant", "equivariant"):
            raise ValueError("unknown prototype route")
        self.route = route
        self.local_input = nn.Linear(512, 64)
        self.local_output = nn.Linear(64, 512)
        if route == "invariant":
            self.blocks = nn.ModuleList(InvariantMessageBlock(pair_chunk, recompute_angles) for _ in range(2))
        else:
            # Only the first gate can affect the next block. The final scalar
            # readout has already occurred, so a last direction gate is dead.
            self.blocks = nn.ModuleList(EquivariantMessageBlock(gate_direction=(i == 0)) for i in range(2))

    def forward(self, atoms, mask, geometry, return_debug=False):
        if atoms.shape != (*mask.shape, 512) or atoms.dtype != geometry.vectors.dtype:
            raise ValueError("scalar atom shape/dtype does not match geometry")
        if atoms.device.type != "cpu" or not torch.isfinite(atoms[~mask]).all():
            raise ValueError("finite CPU scalars required on valid atom slots")
        atoms = atoms.masked_fill(mask[..., None], 0)
        scalar = self.local_input(atoms).masked_fill(mask[..., None], 0)
        states = [scalar]
        if self.route == "equivariant":
            state = torch.cat((scalar, scalar.new_zeros((*mask.shape, 44))), -1)
            states = [state]
            harmonics = o3.spherical_harmonics(EDGE_IRREPS, geometry.unit, normalize=False,
                                               normalization="component")
            for block in self.blocks:
                state = block(state, geometry, harmonics).masked_fill(mask[..., None], 0)
                states.append(state)
            scalar = state[..., :64]
        else:
            for block in self.blocks:
                scalar = block(scalar, geometry).masked_fill(mask[..., None], 0)
                states.append(scalar)
        output = self.local_output(scalar).masked_fill(mask[..., None], 0)
        if return_debug:
            return output, {"states": states, "degree": geometry.degree,
                            "effective_count": geometry.effective_count,
                            "pair_count": geometry.pair_count}
        return output


class ProbeZP(nn.Module):
    """Shared untrained scalar ZP entry; no checkpoint or numeric element props."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(119, 512)
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 512)

    def forward(self, elements, mask):
        return self.projection(self.norm(self.embedding(elements))).masked_fill(mask[..., None], 0)
