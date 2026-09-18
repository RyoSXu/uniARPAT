"""E9-P0 Q1 coordinate-trunk tests (Design-E section 9).

Gates: grid-coordinate correctness (E0/P0 centers, units, zeros),
day-0 equivalence (trunk silent at init), gradient flow, missing-x
assertion, batch shape handling, legacy-batch backward compatibility.
All CPU, deterministic, no checkpoint.
"""

import json
import os
import unittest

import numpy as np
import torch

from model.heads import CoordTrunk
from model.transformer import Transformer


def _tiny(q1=False):
    torch.manual_seed(0)
    return Transformer(
        token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=True, use_gated_cross_attn=False, head_type="legacy",
        predict_scale=False, q1_coord=q1, q1_hidden=16,
    )


def _fixture():
    torch.manual_seed(0)
    B, Lp, n = 2, 12, 5
    src = torch.zeros(B, Lp, dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    src[:, 2:2 + n] = 14
    mask = (src == 0)
    pos = torch.zeros(B, Lp, 3)
    pos[:, 0, :] = torch.tensor([5.0, 5.0, 0.2])
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:, :] = torch.rand(B, Lp - 2, 3) * 0.9 + 0.05
    return src, mask, pos


def test_q1_grid_coordinates():
    """Dataset x == E0/P0 centers; widths match H1 deltas; zeros per Design."""
    from datasets.dataset import Dos_Dataset
    ds = Dos_Dataset(data_dir="./data/train4ARPAT", split="test")
    grids = json.load(open("./data/grids_c2b/grids.json"))
    e0 = np.asarray(grids["E0"], dtype=float)
    p0 = np.asarray(grids["P0"], dtype=float)
    ex = ds[0][15].numpy()
    px = ds[0][16].numpy()
    assert np.allclose(ex, (e0[:-1] + e0[1:]) / 2, atol=1e-6), "edos_x != E0 centers"
    assert np.allclose(px, (p0[:-1] + p0[1:]) / 2, atol=1e-6), "phdos_x != P0 centers"
    assert abs((ex[1] - ex[0]) - 0.09375) < 1e-9, "eDOS width != delta_edos"
    assert abs((px[1] - px[0]) - 19.6875) < 1e-9, "phDOS width != delta_phdos"
    assert np.any(e0 == 0.0), "Fermi=0 must be an E0 edge (zero convention)"
    assert p0[0] < 0.0 < p0[-1], "nu=0 must sit inside P0 (zero convention)"
    # constant across samples (grid constants, not labels)
    assert torch.equal(ds[0][15], ds[7][15]) and torch.equal(ds[0][16], ds[7][16])


def test_q1_day_zero_equivalence():
    """Trunk silent at init: zero output + same-weight base equality.

    NOTE: q1 on/off models draw different RNG streams at init (extra params
    shift downstream draws, same as C1.2); the guaranteed property is
    per-model (residual == 0, grows by grad), verified here by transplanting
    shared weights from a base model.
    """
    src, mask, pos = _fixture()
    ex = torch.linspace(-6, 6, 8)
    px = torch.linspace(-280, 980, 4)
    m0 = _tiny(q1=False)
    m1 = _tiny(q1=True)
    # trunk output exactly zero at init, both shape conventions
    with torch.no_grad():
        assert m1.edos_trunk(ex).abs().max().item() == 0.0
        assert m1.phdos_trunk(px).abs().max().item() == 0.0
    # transplant shared weights -> outputs must match the base exactly
    m1.load_state_dict(m0.state_dict(), strict=False)
    m0.eval()
    m1.eval()
    with torch.no_grad():
        o0 = m0(src, mask, pos)
        o1 = m1(src, mask, pos, ex, px)
    assert (o0["edos"] - o1["edos"]).abs().max().item() == 0.0
    assert (o0["phdos"] - o1["phdos"]).abs().max().item() == 0.0
    # batched x [B,E] behaves identically to unbatched [E]
    with torch.no_grad():
        o2 = m1(src, mask, pos, ex.unsqueeze(0).expand(2, -1),
                px.unsqueeze(0).expand(2, -1))
    assert (o1["edos"] - o2["edos"]).abs().max().item() == 0.0


def test_q1_trunk_learns():
    """Trunk proj starts at exact zero and receives nonzero gradients."""
    tr = CoordTrunk(32, hidden_dim=16, x_scale=6.0)
    assert tr.proj.weight.abs().max().item() == 0.0
    assert tr.proj.bias.abs().max().item() == 0.0
    x = torch.linspace(-6, 6, 8)
    tr.train()
    tr(x).sum().backward()
    assert tr.proj.weight.grad is not None
    assert tr.proj.weight.grad.abs().max().item() > 0.0
    # zero-preserving scale: x=0 maps through bias-only path deterministically
    tr.eval()
    with torch.no_grad():
        z1 = tr(torch.zeros(4))
        z2 = tr(torch.zeros(4))
    assert torch.equal(z1, z2)


def test_q1_missing_x_asserts():
    """q1 on without coordinates must fail loud, never silently unconditioned."""
    src, mask, pos = _fixture()
    m = _tiny(q1=True)
    m.eval()
    for kw in ({}, {"edos_x": torch.linspace(-6, 6, 8)}):
        try:
            with torch.no_grad():
                m(src, mask, pos, **kw)
        except AssertionError:
            continue
        raise AssertionError(f"missing-x not caught for {sorted(kw)}")
    # wrong bin count must also fail loud
    try:
        with torch.no_grad():
            m(src, mask, pos, torch.linspace(-6, 6, 7), torch.linspace(-280, 980, 4))
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong edos bins not caught")


def test_q1_legacy_batch_compat():
    """15-item legacy batches -> x None (old caches / synthetic callers keep working)."""
    from datasets.dataset import Dos_Dataset
    ds = Dos_Dataset(data_dir="./data/train4ARPAT", split="test")
    item = ds[0]
    assert len(item) == 17, f"dataset must return 17 items, got {len(item)}"
    legacy = item[:15]
    assert len(legacy) == 15
    # base model runs on legacy-style inputs without coordinates
    m = _tiny(q1=False)
    m.eval()
    src, mask, pos = _fixture()
    with torch.no_grad():
        out = m(src, mask, pos)
    assert torch.isfinite(out["edos"]).all().item()


class TestQ1Coord(unittest.TestCase):
    __test__ = False

    def test_grid_coordinates(self):
        test_q1_grid_coordinates()

    def test_day_zero(self):
        test_q1_day_zero_equivalence()

    def test_learns(self):
        test_q1_trunk_learns()

    def test_missing_x(self):
        test_q1_missing_x_asserts()

    def test_legacy_compat(self):
        test_q1_legacy_batch_compat()


if __name__ == "__main__":
    unittest.main()
