"""R1b: Coordinate-generated query unit tests (CPU, deterministic)."""

import unittest
import torch
import torch.nn as nn
import numpy as np

from model.heads import CoordQueryGenerator
from model.transformer import Transformer


def _tiny_m1(r1b_coord=False, d_model=32):
    return Transformer(
        token_num=128, d_model=d_model, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=2, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=False, use_gated_cross_attn=False,
        head_type="legacy", predict_scale=False,
        r1a_point=True, r1b_coord=r1b_coord
    )


def _dummy_batch(B=2, Lp=6):
    src = torch.zeros(B, Lp, dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    src[:, 2:5] = torch.tensor([[14, 8, 14], [6, 14, 6]])
    mask = (src == 0)
    pos = torch.zeros(B, Lp, 3)
    pos[:, 0, :] = torch.tensor([5.0, 5.0, 0.2])
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:, :] = 0.5
    return src, mask, pos


def test_r1b_disabled_path_equivalence():
    """When r1b_coord=False, model uses fixed queries and matches standard M1."""
    torch.manual_seed(42)
    m_default = _tiny_m1(r1b_coord=False)
    torch.manual_seed(42)
    m_base = _tiny_m1(r1b_coord=False)

    assert not m_default.r1b_coord
    assert m_default.q_e is None and m_default.q_p is None
    assert m_default.edos_query_embed is not None
    assert m_default.state_dict().keys() == m_base.state_dict().keys()

    src, mask, pos = _dummy_batch()
    m_default.eval()
    m_base.eval()
    with torch.no_grad():
        out_def = m_default(src, mask, pos)
        out_base = m_base(src, mask, pos)

    assert torch.equal(out_def["edos"], out_base["edos"])
    assert torch.equal(out_def["phdos"], out_base["phdos"])


def test_r1b_arbitrary_query_lengths():
    """R1b model accepts arbitrary query bin lengths for eDOS and phDOS."""
    torch.manual_seed(42)
    m = _tiny_m1(r1b_coord=True)
    m.eval()
    src, mask, pos = _dummy_batch(B=2)

    # Test 1: Non-standard arbitrary lengths (e.g. E=37, P=19)
    ex1 = torch.linspace(-5.0, 5.0, 37)
    px1 = torch.linspace(-200.0, 800.0, 19)
    with torch.no_grad():
        out1 = m(src, mask, pos, edos_x=ex1, phdos_x=px1)
    assert out1["edos"].shape == (2, 37), f"Expected shape (2, 37), got {out1['edos'].shape}"
    assert out1["phdos"].shape == (2, 19), f"Expected shape (2, 19), got {out1['phdos'].shape}"
    assert torch.isfinite(out1["edos"]).all()
    assert torch.isfinite(out1["phdos"]).all()

    # Test 2: Dense fine-grained lengths (e.g. E=120, P=60)
    ex2 = torch.linspace(-6.0, 6.0, 120)
    px2 = torch.linspace(-280.0, 980.0, 60)
    with torch.no_grad():
        out2 = m(src, mask, pos, edos_x=ex2, phdos_x=px2)
    assert out2["edos"].shape == (2, 120)
    assert out2["phdos"].shape == (2, 60)


def test_r1b_same_coordinates_determinism():
    """Passing identical coordinates twice produces bitwise identical predictions."""
    torch.manual_seed(42)
    m = _tiny_m1(r1b_coord=True)
    m.eval()
    src, mask, pos = _dummy_batch()

    ex = torch.linspace(-6.0, 6.0, 16)
    px = torch.linspace(-280.0, 980.0, 8)

    with torch.no_grad():
        out1 = m(src, mask, pos, edos_x=ex, phdos_x=px)
        out2 = m(src, mask, pos, edos_x=ex, phdos_x=px)

    assert torch.equal(out1["edos"], out2["edos"]), "eDOS output is non-deterministic"
    assert torch.equal(out1["phdos"], out2["phdos"]), "phDOS output is non-deterministic"


def test_r1b_coordinate_and_mlp_gradients():
    """Gradients exist and are finite for query MLP parameters and coordinates."""
    torch.manual_seed(42)
    m = _tiny_m1(r1b_coord=True)
    m.train()
    src, mask, pos = _dummy_batch()

    ex = torch.linspace(-6.0, 6.0, 8, requires_grad=True)
    px = torch.linspace(-280.0, 980.0, 4, requires_grad=True)

    out = m(src, mask, pos, edos_x=ex, phdos_x=px)
    loss = out["edos"].square().mean() + out["phdos"].square().mean()
    loss.backward()

    # Verify query generator MLPs receive non-None, finite, non-zero gradients
    for name, p in m.q_e.named_parameters():
        assert p.grad is not None, f"q_e {name} gradient is None"
        assert torch.isfinite(p.grad).all(), f"q_e {name} gradient non-finite"
        assert p.grad.abs().sum().item() > 0, f"q_e {name} gradient is all zeros"

    for name, p in m.q_p.named_parameters():
        assert p.grad is not None, f"q_p {name} gradient is None"
        assert torch.isfinite(p.grad).all(), f"q_p {name} gradient non-finite"
        assert p.grad.abs().sum().item() > 0, f"q_p {name} gradient is all zeros"

    # Verify coordinate input itself receives gradients
    assert ex.grad is not None and torch.isfinite(ex.grad).all()
    assert ex.grad.abs().sum().item() > 0, "Coordinate input ex received no gradients"
    assert px.grad is not None and torch.isfinite(px.grad).all()
    assert px.grad.abs().sum().item() > 0, "Coordinate input px received no gradients"


def test_r1b_e0_p0_cpu_smoke():
    """Real Q1 CPU single-step smoke test with genuine dataset coordinates."""
    from datasets.dataset import Dos_Dataset
    from torch.utils.data import DataLoader

    ds = Dos_Dataset(data_dir="./data/train4ARPAT", split="train")
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    src = batch[0]
    pos = batch[1]
    mask = (src == 0)
    edos_x = batch[15]
    phdos_x = batch[16]

    m = _tiny_m1(r1b_coord=True)
    m.train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)

    optimizer.zero_grad()
    out = m(src, mask, pos, edos_x=edos_x, phdos_x=phdos_x)

    assert out["edos"].shape == (2, edos_x.shape[-1])
    assert out["phdos"].shape == (2, phdos_x.shape[-1])
    assert torch.isfinite(out["edos"]).all(), "NaN/Inf detected in eDOS"
    assert torch.isfinite(out["phdos"]).all(), "NaN/Inf detected in phDOS"

    loss = out["edos"].mean() + out["phdos"].mean()
    loss.backward()

    p_before = [p.clone() for p in m.q_e.parameters()]
    optimizer.step()
    changed = any(not torch.equal(p1, p2) for p1, p2 in zip(p_before, m.q_e.parameters()))
    assert changed, "q_e parameters did not update"


class TestR1bCoord(unittest.TestCase):
    def test_disabled_equivalence(self):
        test_r1b_disabled_path_equivalence()

    def test_arbitrary_query_lengths(self):
        test_r1b_arbitrary_query_lengths()

    def test_same_coordinates_determinism(self):
        test_r1b_same_coordinates_determinism()

    def test_coordinate_and_mlp_gradients(self):
        test_r1b_coordinate_and_mlp_gradients()

    def test_e0_p0_cpu_smoke(self):
        test_r1b_e0_p0_cpu_smoke()


if __name__ == "__main__":
    unittest.main()
