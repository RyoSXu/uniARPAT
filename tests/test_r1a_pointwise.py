"""R1a: Parameter-matched pointwise readout head unit tests (CPU, deterministic)."""

import unittest
import torch
import torch.nn as nn
import numpy as np

from model.heads import CNN, PointwiseMLPHead
from model.transformer import Transformer


def _tiny_m1(r1a_point=False, d_model=32):
    return Transformer(
        token_num=128, d_model=d_model, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=2, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=False, use_gated_cross_attn=False,
        head_type="legacy", predict_scale=False, r1a_point=r1a_point
    )


def _dummy_batch(B=2, Lp=6):
    src = torch.zeros(B, Lp, dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    src[:, 2:5] = torch.tensor([[14, 8, 14], [6, 14, 6]])
    mask = src == 0
    pos = torch.zeros(B, Lp, 3)
    pos[:, 0, :] = torch.tensor([5.0, 5.0, 0.2])
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:, :] = 0.5
    return src, mask, pos


def test_r1a_disabled_path_equivalence():
    """When r1a_point=False (default), model state dict and outputs match exactly."""
    torch.manual_seed(42)
    m_default = _tiny_m1()
    torch.manual_seed(42)
    m_base = _tiny_m1(r1a_point=False)

    assert not m_default.r1a_point
    assert not m_base.r1a_point
    assert m_default.state_dict().keys() == m_base.state_dict().keys()

    src, mask, pos = _dummy_batch()
    m_default.eval()
    m_base.eval()
    with torch.no_grad():
        out_def = m_default(src, mask, pos)
        out_base = m_base(src, mask, pos)

    assert torch.equal(out_def["edos"], out_base["edos"])
    assert torch.equal(out_def["phdos"], out_base["phdos"])


def test_r1a_pointwise_permutation_contract():
    """Pointwise head preserves token-wise permutation equivariance, unlike Conv1d."""
    torch.manual_seed(42)
    d_model = 32
    L = 16
    B = 2
    head = PointwiseMLPHead([d_model, 16, 1])
    head.eval()

    x = torch.randn(B, L, d_model)
    perm = torch.randperm(L)

    with torch.no_grad():
        out_orig = head(x)  # [B, L, 1]
        out_perm = head(x[:, perm, :])  # [B, L, 1]

    # Equivariance contract: f(P x) == P f(x)
    assert torch.allclose(out_perm, out_orig[:, perm, :], atol=1e-6), \
        "Pointwise head violated permutation equivariance contract"

    # In contrast, Conv1d (k=3) mixes neighbor bins and MUST fail permutation equivariance
    cnn = CNN(d_model, d_model * 2, output_dim=1, num_layers=2, kernel_size=3)
    cnn.eval()
    with torch.no_grad():
        cnn_out_orig = cnn(x.transpose(1, 2)).transpose(1, 2)
        cnn_out_perm = cnn(x[:, perm, :].transpose(1, 2)).transpose(1, 2)
    assert not torch.allclose(cnn_out_perm, cnn_out_orig[:, perm, :], atol=1e-3), \
        "Conv1d unexpectedly satisfied token permutation equivariance"


def test_r1a_parameter_matching():
    """Parameter counts match the design specification within < 0.2% difference."""
    # eDOS: 512 -> 3 -> 1 (GELU) -> 1,543 parameters vs baseline 1,537 (+6)
    edos_point = PointwiseMLPHead([512, 3, 1])
    edos_point_params = sum(p.numel() for p in edos_point.parameters())
    assert edos_point_params == 1543, f"eDOS pointwise params {edos_point_params} != 1543"

    edos_conv = CNN(512, 512 * 3, output_dim=1, num_layers=1)
    edos_conv_params = sum(p.numel() for p in edos_conv.parameters())
    assert edos_conv_params == 1537, f"eDOS baseline conv params {edos_conv_params} != 1537"

    # phDOS: 512 -> 2704 -> 2704 -> 2704 -> 2704 -> 2704 -> 1 -> 30,647,137 params
    phdos_point = PointwiseMLPHead([512, 2704, 2704, 2704, 2704, 2704, 1])
    phdos_point_params = sum(p.numel() for p in phdos_point.parameters())
    assert phdos_point_params == 30647137, f"phDOS pointwise params {phdos_point_params} != 30647137"

    phdos_conv = CNN(512, 512 * 3, output_dim=1, num_layers=6)
    phdos_conv_params = sum(p.numel() for p in phdos_conv.parameters())
    assert phdos_conv_params == 30683137, f"phDOS baseline conv params {phdos_conv_params} != 30683137"

    # Capacity difference < 0.2%
    diff_pct = abs(phdos_point_params - phdos_conv_params) / phdos_conv_params
    assert diff_pct < 0.002, f"phDOS capacity difference {diff_pct:.4%} exceeds 0.2%"


def test_r1a_finite_gradients():
    """Gradients flow cleanly through pointwise readout heads and are all finite."""
    torch.manual_seed(42)
    m = _tiny_m1(r1a_point=True)
    m.train()

    src, mask, pos = _dummy_batch()
    out = m(src, mask, pos)

    loss = out["edos"].square().mean() + out["phdos"].square().mean()
    loss.backward()

    for name, p in m.edos_out_head.named_parameters():
        assert p.grad is not None, f"edos_out_head {name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"edos_out_head {name} gradient non-finite"
        assert p.grad.abs().sum().item() > 0, f"edos_out_head {name} gradient is all zeros"

    for name, p in m.phdos_out_head.named_parameters():
        assert p.grad is not None, f"phdos_out_head {name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"phdos_out_head {name} gradient non-finite"
        assert p.grad.abs().sum().item() > 0, f"phdos_out_head {name} gradient is all zeros"


def test_r1a_q1_cpu_smoke():
    """Real Q1 CPU single-step smoke test using genuine dataset item."""
    from datasets.dataset import Dos_Dataset
    from torch.utils.data import DataLoader

    ds = Dos_Dataset(data_dir="./data/train4ARPAT", split="train")
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    m = _tiny_m1(r1a_point=True)
    m.train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)

    src = batch[0]
    pos = batch[1]
    mask = (src == 0)

    optimizer.zero_grad()
    out = m(src, mask, pos)
    assert torch.isfinite(out["edos"]).all(), "eDOS output contains NaN/Inf"
    assert torch.isfinite(out["phdos"]).all(), "phDOS output contains NaN/Inf"

    loss = out["edos"].mean() + out["phdos"].mean()
    loss.backward()

    # Check that parameters update
    p_before = [p.clone() for p in m.edos_out_head.parameters()]
    optimizer.step()
    changed = any(not torch.equal(p1, p2) for p1, p2 in zip(p_before, m.edos_out_head.parameters()))
    assert changed, "Parameters failed to update after optimizer step"


class TestR1aPointwise(unittest.TestCase):
    def test_disabled_equivalence(self):
        test_r1a_disabled_path_equivalence()

    def test_permutation_contract(self):
        test_r1a_pointwise_permutation_contract()

    def test_parameter_matching(self):
        test_r1a_parameter_matching()

    def test_finite_gradients(self):
        test_r1a_finite_gradients()

    def test_q1_cpu_smoke(self):
        test_r1a_q1_cpu_smoke()


if __name__ == "__main__":
    unittest.main()
