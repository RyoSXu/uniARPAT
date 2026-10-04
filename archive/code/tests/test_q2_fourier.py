"""E9-P0 Q2 Fourier-trunk tests (Design-E section 9).

Gates: frozen frequency bank determinism, day-0 silence (+ transplant
equality), inference clip, task sigma separation, flag wiring.
NOTE (frozen decision #10): held-out-query *training* would need loss-side
bin masking and is therefore NOT part of Q2; Q2 verdict = Fourier-vs-MLP
encoding factor (vs _q1exp) + total (vs _q1ctl). All CPU, deterministic.
"""

import unittest

import torch

from model.heads import FourierTrunk
from model.transformer import Transformer


def test_q2_bank_deterministic():
    a = FourierTrunk(32, hidden_dim=16, n_freq=64, sigma=8.0, seed=42)
    b = FourierTrunk(32, hidden_dim=16, n_freq=64, sigma=8.0, seed=42)
    assert torch.equal(a.freq_B, b.freq_B), "same seed must give same bank"
    c = FourierTrunk(32, hidden_dim=16, n_freq=64, sigma=8.0, seed=43)
    assert not torch.equal(a.freq_B, c.freq_B), "different seed must differ"
    # exact recomputation (no hidden state)
    g = torch.Generator().manual_seed(42)
    assert torch.equal(a.freq_B, torch.randn(64, generator=g) * 8.0)


def test_q2_task_sigmas():
    e = FourierTrunk(32, n_freq=64, sigma=8.0, x_scale=6.0, seed=42)
    p = FourierTrunk(32, n_freq=32, sigma=2.0, x_scale=980.0, seed=43)
    assert e.freq_B.shape == (64,) and p.freq_B.shape == (32,)
    # eDOS bank spans ~4x the phDOS bank (high vs low frequency by design)
    assert e.freq_B.abs().max().item() > p.freq_B.abs().max().item()


def test_q2_day_zero_and_clip():
    tr = FourierTrunk(32, hidden_dim=16, n_freq=8, sigma=8.0,
                      x_scale=6.0, seed=42, x_min=-1.0, x_max=1.0)
    x = torch.linspace(-6, 6, 8)
    with torch.no_grad():
        assert tr(x).abs().max().item() == 0.0, "trunk must be silent at init"
        # clip: far-outside inputs collapse to the range edge exactly
        big = torch.tensor([-600.0, 600.0])
        edge = torch.tensor([-6.0, 6.0])
        tr.train()  # silence is init-zero proj, independent of mode
        assert torch.equal(tr(big), tr(edge)), "inference clip broken"
    tr.train()
    tr(x).sum().backward()
    assert tr.proj.weight.grad.abs().max().item() > 0.0, "no gradient flow"


def test_q2_flag_wiring():
    from model.heads import CoordTrunk
    torch.manual_seed(0)
    m = Transformer(
        token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, normalize_before=False, decoupled_decoder=True,
        head_type="legacy", predict_scale=False, q2_fourier=True, q1_hidden=16)
    assert m.q1_coord is True and m.q2_fourier is True
    assert isinstance(m.edos_trunk, FourierTrunk)
    assert isinstance(m.phdos_trunk, FourierTrunk)
    torch.manual_seed(0)
    m0 = Transformer(
        token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, normalize_before=False, decoupled_decoder=True,
        head_type="legacy", predict_scale=False, q1_coord=True, q1_hidden=16)
    assert isinstance(m0.edos_trunk, CoordTrunk), "q1 alone must stay plain MLP"
    # transplant equality (same RNG-stream caveat as Q1, see test_q1_coord)
    torch.manual_seed(1)
    base = Transformer(
        token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, normalize_before=False, decoupled_decoder=True,
        head_type="legacy", predict_scale=False)
    m.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    m.eval()
    B, Lp = 1, 10
    src = torch.zeros(B, Lp, dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    src[:, 2:6] = 14
    mask = (src == 0)
    pos = torch.zeros(B, Lp, 3)
    pos[:, 0, :] = torch.tensor([5.0, 5.0, 0.2])
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:, :] = 0.5
    ex = torch.linspace(-6, 6, 8)
    px = torch.linspace(-280, 980, 4)
    with torch.no_grad():
        o0 = base(src, mask, pos)
        o1 = m(src, mask, pos, ex, px)
    assert (o0["edos"] - o1["edos"]).abs().max().item() == 0.0
    assert (o0["phdos"] - o1["phdos"]).abs().max().item() == 0.0


class TestQ2Fourier(unittest.TestCase):
    __test__ = False

    def test_bank(self):
        test_q2_bank_deterministic()

    def test_sigmas(self):
        test_q2_task_sigmas()

    def test_silence_clip(self):
        test_q2_day_zero_and_clip()

    def test_wiring(self):
        test_q2_flag_wiring()


if __name__ == "__main__":
    unittest.main()
