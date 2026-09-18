"""C5 token-level decoder MoE gates (all CPU, deterministic)."""

import unittest

import torch

from model.transformer import C5TokenMoEResidual, Transformer


def _tiny(c5_moe=False):
    return Transformer(
        token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=3, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=False, use_gated_cross_attn=False,
        head_type="legacy", predict_scale=False, c5_moe=c5_moe)


def _batch():
    B, Lp = 2, 8
    src = torch.zeros(B, Lp, dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    src[:, 2:6] = torch.tensor([[14, 8, 14, 8], [6, 14, 6, 14]])
    mask = src == 0
    pos = torch.zeros(B, Lp, 3)
    pos[:, 0, :] = torch.tensor([5.0, 5.0, 0.2])
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:, :] = 0.5
    return src, mask, pos


def test_c5_top2_load_balance_and_gate_gradient():
    torch.manual_seed(7)
    moe = C5TokenMoEResidual(32, expert_hidden=16, num_experts=4, top_k=2, dropout=0.0)
    x = torch.randn(3, 5, 32)
    delta, balance, load = moe(x)
    assert delta.abs().max().item() == 0.0, "zero alpha must keep C5 silent"
    assert torch.isfinite(balance).item() and torch.isfinite(load).all().item()
    assert abs(load.sum().item() - 1.0) < 1e-6, "Top-2 dispatch must be normalized"
    assert moe.last_top_indices.shape == (3, 5, 2)
    assert all(torch.unique(pair).numel() == 2 for pair in moe.last_top_indices.reshape(-1, 2))
    balance.backward()
    assert moe.gate.weight.grad is not None and moe.gate.weight.grad.abs().sum().item() > 0.0


def test_c5_transplant_is_exact_then_experts_receive_gradients():
    torch.manual_seed(1)
    base = _tiny(c5_moe=False)
    torch.manual_seed(2)
    c5 = _tiny(c5_moe=True)
    missing, unexpected = c5.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing and all("c5_moe" in key for key in missing)
    base.eval()
    c5.eval()
    src, mask, pos = _batch()
    with torch.no_grad():
        out_base = base(src, mask, pos)
        out_c5 = c5(src, mask, pos)
    assert torch.equal(out_base["edos"], out_c5["edos"])
    assert torch.equal(out_base["phdos"], out_c5["phdos"])

    c5.train()
    optimizer = torch.optim.Adam(c5.parameters(), lr=1e-3)
    # First step trains alpha (and the balancing router); second step reaches experts.
    for _ in range(2):
        optimizer.zero_grad()
        out = c5(src, mask, pos)
        loss = out["edos"].square().mean() + out["phdos"].square().mean() + 0.01 * out["c5_moe_balance"]
        loss.backward()
        optimizer.step()
    moe_layers = [layer.c5_moe for layer in c5.decoder.layers if layer.c5_moe is not None]
    assert len(moe_layers) == 2, "only final two decoder layers may receive C5"
    assert all(layer.alpha.grad is not None and layer.alpha.grad.abs().sum().item() > 0 for layer in moe_layers)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for layer in moe_layers for expert in layer.experts for parameter in expert.parameters()
    ), "experts received no gradients after C5 activation"


def test_c5_parameter_budget_and_m1_only_contract():
    moe = C5TokenMoEResidual(512, expert_hidden=768, num_experts=4, top_k=2, dropout=0.05)
    added = 2 * sum(parameter.numel() for parameter in moe.parameters())
    assert added == 6_305_802
    assert added <= 8_000_000
    try:
        _ = Transformer(
            token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
            num_encoder_layers=1, num_decoder_layers=2, dim_feedforward=64,
            dropout=0.0, decoupled_decoder=True, c5_moe=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("C5 must reject decoupled non-M1 decoders")


class TestC5TokenMoE(unittest.TestCase):
    __test__ = False

    def test_top2(self):
        test_c5_top2_load_balance_and_gate_gradient()

    def test_transplant(self):
        test_c5_transplant_is_exact_then_experts_receive_gradients()

    def test_budget(self):
        test_c5_parameter_budget_and_m1_only_contract()


if __name__ == "__main__":
    unittest.main()
