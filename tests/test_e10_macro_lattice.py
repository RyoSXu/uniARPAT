"""E10 macro lattice contracts (CPU, deterministic)."""
import math
import unittest

import torch

from model.transformer import Transformer
from utils.macro_lattice import macro_lattice_features, raw_atomic_mass_table


def _batch(B=2, Lp=6):
    src = torch.zeros(B, Lp, dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    src[:, 2:4] = torch.tensor([[1, 8], [14, 8]])
    mask = src.eq(0)
    pos = torch.zeros(B, Lp, 3)
    pos[:, 0, :] = torch.tensor([2.0, 3.0, 0.25])  # c=4, V=24
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:4, :] = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    return src, mask, pos


def _tiny(use_macro_lattice=False):
    return Transformer(
        token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=False, use_gated_cross_attn=False,
        head_type="legacy", predict_scale=False, scale_mode="eta",
        use_macro_lattice=use_macro_lattice,
        macro_lattice_mean=(0.0, 0.0), macro_lattice_std=(1.0, 1.0),
    )


def test_e10_feature_values_and_padding_contract():
    src, mask, pos = _batch()
    masses = raw_atomic_mass_table()
    feat = macro_lattice_features(pos, src[:, 2:], mask[:, 2:], masses,
                                  torch.zeros(2), torch.ones(2))
    expected0 = torch.tensor([
        math.log(12.0),
        math.log((float(masses[1]) + float(masses[8])) / 24.0),
    ])
    assert torch.allclose(feat[0], expected0, atol=1e-6)

    # Padding IDs and coordinates must not influence mass or atom count.
    src2, mask2, pos2 = src.clone(), mask.clone(), pos.clone()
    src2[:, 4:] = 0
    pos2[:, 4:] = 999.0
    feat2 = macro_lattice_features(pos2, src2[:, 2:], mask2[:, 2:], masses,
                                   torch.zeros(2), torch.ones(2))
    assert torch.equal(feat, feat2)
    assert torch.isfinite(feat).all()


def test_e10_disabled_and_initial_enabled_equivalence():
    torch.manual_seed(42)
    base = _tiny(use_macro_lattice=False)
    torch.manual_seed(7)
    macro = _tiny(use_macro_lattice=True)
    macro.load_state_dict(base.state_dict(), strict=False)
    src, mask, pos = _batch()
    base.eval()
    macro.eval()
    with torch.no_grad():
        out_base = base(src, mask, pos)
        out_macro = macro(src, mask, pos)
    assert torch.equal(out_base["edos"], out_macro["edos"])
    assert torch.equal(out_base["phdos"], out_macro["phdos"])
    assert torch.equal(out_base["eta"], out_macro["eta"])
    assert macro.macro_lattice_alpha.item() == 0.0


def test_e10_residual_gradually_unfreezes():
    torch.manual_seed(42)
    model = _tiny(use_macro_lattice=True)
    model.train()
    src, mask, pos = _batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(2):
        optimizer.zero_grad()
        out = model(src, mask, pos)
        loss = out["edos"].square().mean() + out["phdos"].square().mean() + out["eta"].square().mean()
        loss.backward()
        assert model.macro_lattice_alpha.grad is not None
        assert torch.isfinite(model.macro_lattice_alpha.grad)
        assert model.macro_lattice_alpha.grad.abs().item() > 0
        if step == 1:
            grads = [p.grad for p in model.macro_lattice_mlp.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert any(g.abs().sum().item() > 0 for g in grads)
        optimizer.step()


def test_e10_real_q1_cpu_smoke():
    from datasets.dataset import Dos_Dataset
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(Dos_Dataset(data_dir="./data/train4ARPAT", split="train"), batch_size=2)))
    src, pos = batch[0], batch[1]
    model = _tiny(use_macro_lattice=True)
    out = model(src, src.eq(0), pos)
    assert out["edos"].shape == (2, 8)
    assert out["phdos"].shape == (2, 4)
    assert torch.isfinite(out["edos"]).all()
    assert torch.isfinite(out["phdos"]).all()
    assert torch.isfinite(out["eta"]).all()


class TestE10MacroLattice(unittest.TestCase):
    def test_feature_values_and_padding_contract(self):
        test_e10_feature_values_and_padding_contract()

    def test_disabled_and_initial_enabled_equivalence(self):
        test_e10_disabled_and_initial_enabled_equivalence()

    def test_residual_gradually_unfreezes(self):
        test_e10_residual_gradually_unfreezes()

    def test_real_q1_cpu_smoke(self):
        test_e10_real_q1_cpu_smoke()


if __name__ == "__main__":
    unittest.main()
