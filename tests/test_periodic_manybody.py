"""Contracts for the proposed periodic local/global encoder."""

import math
import unittest

import torch

from model.periodic_manybody import periodic_edge_vectors
from model.transformer import Transformer
from utils.g2_periodic_edges import build_g2_edges
from utils.relative_features import build_cell_from_lattice


def _pos(lengths, angles, fractions):
    pos = torch.zeros(1, 2 + len(fractions), 3)
    a, b, c = lengths
    pos[0, 0] = torch.tensor([a, b, 1.0 / c])
    pos[0, 1] = torch.tensor(angles)
    pos[0, 2:] = fractions
    return pos


def _model():
    torch.manual_seed(7)
    model = Transformer(
        token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=6, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, scale_mode="eta", use_periodic_manybody=True)
    return model.eval()


def _outputs_close(first, second, atol=2e-4):
    for key in first:
        if torch.is_tensor(first[key]):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=atol)


def test_periodic_vectors_match_graph_distances_and_images():
    frac = torch.tensor([[0.04, 0.10, 0.20], [0.94, 0.80, 0.70]])
    pos = _pos((3.2, 4.0, 5.0), (83.0, 91.0, 106.0), frac)
    edges = build_g2_edges(pos, torch.zeros(1, 2, dtype=torch.bool), r_cut=4.5)
    vectors = periodic_edge_vectors(pos, edges)
    assert vectors.shape == (len(edges["distances"]), 3)
    torch.testing.assert_close(vectors.norm(dim=-1), edges["distances"], rtol=1e-5, atol=1e-5)
    assert (edges["src"] == edges["dst"]).any()  # cross-cell self-images kept


def test_candidate_invariant_to_order_integer_shifts_and_cell_basis():
    model = _model()
    src = torch.tensor([[126, 127, 14, 8, 16, 0]])
    mask = src.eq(0)
    frac = torch.tensor([[0.04, 0.10, 0.20], [0.64, 0.31, 0.70],
                         [0.27, 0.83, 0.45], [0.0, 0.0, 0.0]])
    pos = _pos((4.0, 5.0, 6.0), (80.0, 95.0, 100.0), frac)
    with torch.inference_mode():
        original = model(src, mask, pos)

        order = torch.tensor([2, 0, 1, 3])
        permuted_src = torch.cat((src[:, :2], src[:, 2:][:, order]), dim=1)
        permuted_pos = torch.cat((pos[:, :2], pos[:, 2:][:, order]), dim=1)
        _outputs_close(original, model(permuted_src, permuted_src.eq(0), permuted_pos))

        shifted = pos.clone()
        shifted[0, 2] += torch.tensor([1.0, -1.0, 2.0])
        _outputs_close(original, model(src, mask, shifted))

        # Swapping a and b rotates the internally reconstructed Cartesian frame.
        swapped = _pos((5.0, 4.0, 6.0), (95.0, 80.0, 100.0), frac[:, [1, 0, 2]])
        _outputs_close(original, model(src, mask, swapped))

        cell, _ = build_cell_from_lattice(pos)
        unimodular = torch.tensor([[1.0, 1.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])
        changed_cell = unimodular @ cell[0]
        lengths = [float(v.norm()) for v in changed_cell]
        def angle(a, b):
            return math.degrees(math.acos(float(torch.dot(a, b) / (a.norm() * b.norm()))))
        angles = (angle(changed_cell[1], changed_cell[2]),
                  angle(changed_cell[0], changed_cell[2]),
                  angle(changed_cell[0], changed_cell[1]))
        changed_frac = (frac @ torch.linalg.inv(unimodular)) % 1.0
        changed = _pos(lengths, angles, changed_frac)
        _outputs_close(original, model(src, mask, changed))


def test_directional_paths_reach_scalar_memory_and_dos():
    model = _model().train()
    src = torch.tensor([[126, 127, 14, 8, 16]])
    frac = torch.tensor([[0.04, 0.10, 0.20], [0.64, 0.31, 0.70],
                         [0.27, 0.83, 0.45]])
    pos = _pos((4.0, 5.0, 6.0), (80.0, 95.0, 100.0), frac)
    out = model(src, src.eq(0), pos)
    out["edos"].square().sum().backward()
    block = model.encoder.local_blocks[1]
    grad = block.tp.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    offset = 0
    directional_gradient = 0.0
    for instruction in block.tp.instructions:
        width = math.prod(instruction.path_shape)
        if instruction.i_in1 in (1, 2) and instruction.i_out == 0:
            directional_gradient += float(grad[offset:offset + width].abs().sum())
        offset += width
    assert directional_gradient > 0.0


class PeriodicManyBodyContracts(unittest.TestCase):
    def test_vectors(self):
        test_periodic_vectors_match_graph_distances_and_images()

    def test_invariance(self):
        test_candidate_invariant_to_order_integer_shifts_and_cell_basis()

    def test_directional_gradient(self):
        test_directional_paths_reach_scalar_memory_and_dos()
