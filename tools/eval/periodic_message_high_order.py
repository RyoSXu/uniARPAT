"""CPU-only B enhancement: weighted l=3,4 pair statistics without pair enumeration.

The old A/B implementation stays unchanged. This module copies an untrained B,
adds 16 invariant inputs per update, and preserves all original weights.
"""

from __future__ import annotations

import copy

import torch
from e3nn import o3
from torch import nn

from tools.eval.periodic_message_prototypes import (
    EquivariantMessageBlock, PeriodicMessageProbe, count_features,
)

HIGH_ORDERS = (3, 4)
MOMENT_CHANNELS = 8
EXTRA_SCALARS = len(HIGH_ORDERS) * MOMENT_CHANNELS


def high_order_pair_features(geometry, weights, harmonics):
    """Return weighted Legendre pair sums using the spherical-harmonic identity.

    With component-normalized Y_l, ||sum wY_l||^2/(2l+1)-sum w^2 equals
    2 sum_{e<f} w_e w_f P_l(u_e dot u_f). Each record/image remains distinct.
    Division by 2 sqrt(max(1,p_i)) matches A's pair-count scaling. Single-edge
    and empty centers return exact zeros. Weights already include the cutoff.
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


class HighOrderEquivariantBlock(EquivariantMessageBlock):
    """B's original persistent state plus l=3,4 scalar pair-statistic inputs."""

    def __init__(self, baseline):
        nn.Module.__init__(self)
        for name in ("tp", "mix", "radial_gate", "update", "norm", "direction_gate"):
            setattr(self, name, copy.deepcopy(getattr(baseline, name)))
        old = self.update[0]
        widened = nn.Linear(old.in_features + EXTRA_SCALARS, old.out_features,
                            device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            widened.weight[:, :old.in_features].copy_(old.weight)
            widened.bias.copy_(old.bias)
        self.update[0] = widened

    def forward(self, state, geometry, harmonics, include_high_order=True):
        shape = state.shape
        flat = state.reshape(-1, 108)
        g = geometry
        gate_input = torch.cat((flat[g.receiver, :64], flat[g.sender, :64], g.radial), -1)
        gates = self.radial_gate(gate_input)
        # Reuse B's eight invariant vector-channel coefficients: no new edge MLP.
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


class HighOrderMessageProbe(PeriodicMessageProbe):
    """Independent B+; original 512/64 projections and B state are copied exactly."""

    def __init__(self, baseline, include_high_order=True):
        nn.Module.__init__(self)
        if baseline.route != "equivariant":
            raise ValueError("B+ requires the existing equivariant B prototype")
        self.route = "equivariant"
        self.include_high_order = include_high_order
        self.local_input = copy.deepcopy(baseline.local_input)
        self.local_output = copy.deepcopy(baseline.local_output)
        self.blocks = nn.ModuleList(HighOrderEquivariantBlock(block) for block in baseline.blocks)

    def forward(self, atoms, mask, geometry, return_debug=False):
        if atoms.shape != (*mask.shape, 512) or atoms.dtype != geometry.vectors.dtype:
            raise ValueError("scalar atom shape/dtype does not match geometry")
        if atoms.device.type != "cpu" or not torch.isfinite(atoms[~mask]).all():
            raise ValueError("finite CPU scalars required on valid atom slots")
        atoms = atoms.masked_fill(mask[..., None], 0)
        scalar = self.local_input(atoms).masked_fill(mask[..., None], 0)
        state = torch.cat((scalar, scalar.new_zeros((*mask.shape, 44))), -1)
        states, angular_states = [state], []
        harmonics = o3.spherical_harmonics(list(range(5)), geometry.unit, normalize=False,
                                         normalization="component")
        for block in self.blocks:
            state, angular = block(state, geometry, harmonics, self.include_high_order)
            state = state.masked_fill(mask[..., None], 0)
            states.append(state)
            angular_states.append(angular.masked_fill(mask[..., None], 0))
        output = self.local_output(state[..., :64]).masked_fill(mask[..., None], 0)
        if return_debug:
            return output, {"states": states, "angular_invariants": angular_states,
                            "degree": geometry.degree, "effective_count": geometry.effective_count,
                            "pair_count": geometry.pair_count}
        return output
