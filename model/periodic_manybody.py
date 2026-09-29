"""Periodic equivariant local states with global atom attention for DOS models.

The local branch keeps l=0,1,2 states across three blocks. Two successive
directional messages can therefore turn a bond-angle relation into scalar
atom content. Only the scalar content enters the global attention and decoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from e3nn import o3
from torch import nn

from utils.g2_periodic_edges import build_g2_edges, g2_quintic_cutoff
from utils.relative_features import build_cell_from_lattice


LOCAL_CUTOFF = 4.5
LOCAL_IRREPS = o3.Irreps("48x0e + 12x1o + 6x2e")
EDGE_IRREPS = o3.Irreps.spherical_harmonics(2)
RADIAL_COUNT = 16


def periodic_edge_vectors(pos: torch.Tensor, edges: dict) -> torch.Tensor:
    """Cartesian vectors for the exact (receiver <- sender, image) edge IDs."""
    cell, frac = build_cell_from_lattice(pos)
    batch, dst, src = edges["batch"], edges["dst"], edges["src"]
    if batch.numel() == 0:
        return pos.new_empty((0, 3))
    frac_delta = frac[batch, src] - frac[batch, dst]
    frac_delta = frac_delta + edges["shifts"].to(frac_delta.dtype)
    return torch.bmm(frac_delta.unsqueeze(1), cell[batch]).squeeze(1)


class EquivariantLocalBlock(nn.Module):
    """One sparse radial and spherical tensor-product message block."""

    def __init__(self, irreps: o3.Irreps = LOCAL_IRREPS):
        super().__init__()
        self.irreps = irreps
        # Shared tensor-product paths keep per-edge storage linear in channels.
        # The radial network supplies a separate scalar gate per output irrep.
        self.tp = o3.FullyConnectedTensorProduct(irreps, EDGE_IRREPS, irreps)
        self.radial = nn.Sequential(
            nn.Linear(RADIAL_COUNT, 64), nn.SiLU(), nn.Linear(64, 66),
        )
        self.mix = o3.Linear(irreps, irreps)
        self.scalar_norm = nn.LayerNorm(48)
        self.scalar_ffn = nn.Sequential(nn.Linear(48, 96), nn.SiLU(), nn.Linear(96, 48))
        self.direction_gate = nn.Linear(48, 18)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, state, edges, harmonics, radial, cutoff):
        B, L, D = state.shape
        if edges["batch"].numel():
            src_state = state[edges["batch"], edges["src"]]
            message = self.tp(src_state, harmonics)
            gates = self.radial(radial)
            gate_parts = (gates[:, :48],
                          gates[:, 48:60].repeat_interleave(3, dim=-1),
                          gates[:, 60:66].repeat_interleave(5, dim=-1))
            message = message * torch.cat(gate_parts, dim=-1) * cutoff.unsqueeze(-1)
            linear_index = edges["batch"] * L + edges["dst"]
            agg = torch.zeros(B * L, D, device=state.device, dtype=state.dtype)
            agg.index_add_(0, linear_index, message)
            degree = torch.bincount(linear_index, minlength=B * L).to(state.dtype)
            agg = agg * degree.clamp(min=1).rsqrt().unsqueeze(-1)
            state = state + self.gain * self.mix(agg.view(B, L, D))

        scalar = self.scalar_norm(state[..., :48])
        scalar = scalar + self.scalar_ffn(scalar)
        direction = state[..., 48:]
        gates = torch.sigmoid(self.direction_gate(scalar))
        direction = direction * torch.cat(
            (gates[..., :12].repeat_interleave(3, dim=-1),
             gates[..., 12:].repeat_interleave(5, dim=-1)), dim=-1)
        return torch.cat((scalar, direction), dim=-1)


class GlobalAtomLayer(nn.Module):
    """Standard learned Q/K/V attention on invariant atom states."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, atoms, padding_mask):
        mixed, _ = self.attn(
            atoms, atoms, atoms, key_padding_mask=padding_mask,
            need_weights=False)
        atoms = self.norm1(atoms + self.dropout(mixed))
        return self.norm2(atoms + self.dropout(self.ffn(atoms)))


class PeriodicManyBodyEncoder(nn.Module):
    """Three persistent equivariant blocks interleaved with six global layers."""

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048,
                 dropout=0.05, num_global_layers=6):
        super().__init__()
        if num_global_layers != 6:
            raise ValueError("periodic many-body candidate requires six global layers")
        self.local_input = nn.Linear(d_model, 48)
        self.from_global = nn.ModuleList(nn.Linear(d_model, 48) for _ in range(2))
        self.to_global = nn.ModuleList(nn.Linear(48, d_model) for _ in range(3))
        self.local_blocks = nn.ModuleList(EquivariantLocalBlock() for _ in range(3))
        self.global_layers = nn.ModuleList(
            GlobalAtomLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_global_layers))
        centers = torch.linspace(0.01 * LOCAL_CUTOFF, 0.99 * LOCAL_CUTOFF, RADIAL_COUNT)
        self.register_buffer("radial_centers", centers)
        self.register_buffer("radial_width", centers[1] - centers[0])

    def forward(self, atoms, padding_mask, pos):
        B, L, _ = atoms.shape
        edges = build_g2_edges(pos, padding_mask, r_cut=LOCAL_CUTOFF)
        vectors = periodic_edge_vectors(pos, edges)
        distances = edges["distances"].to(atoms.dtype)
        if vectors.numel():
            harmonics = o3.spherical_harmonics(
                EDGE_IRREPS, vectors, normalize=True, normalization="component")
            radial = torch.exp(-0.5 * (
                (distances[:, None] - self.radial_centers) / self.radial_width) ** 2)
            cutoff = g2_quintic_cutoff(distances, LOCAL_CUTOFF)
        else:
            harmonics = atoms.new_empty((0, EDGE_IRREPS.dim))
            radial = atoms.new_empty((0, RADIAL_COUNT))
            cutoff = atoms.new_empty((0,))

        local_scalar = self.local_input(atoms)
        local_state = torch.cat((local_scalar, atoms.new_zeros((B, L, 66))), dim=-1)
        for index, layer in enumerate(self.global_layers):
            if index < 3:
                if index:
                    local_state = torch.cat((
                        local_state[..., :48] + self.from_global[index - 1](atoms),
                        local_state[..., 48:]), dim=-1)
                local_state = self.local_blocks[index](
                    local_state, edges, harmonics, radial, cutoff)
                atoms = atoms + self.to_global[index](local_state[..., :48])
            atoms = layer(atoms, padding_mask)
        return atoms
