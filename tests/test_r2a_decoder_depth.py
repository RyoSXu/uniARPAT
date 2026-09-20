"""R2a shared-decoder depth contracts (CPU, deterministic)."""
import logging
import math
import unittest

import torch

from model.transformer import Transformer
from utils.experiment_config import ExperimentConfig


def _batch(B=2, Lp=6):
    src = torch.zeros(B, Lp, dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    src[:, 2:4] = torch.tensor([[14, 8], [13, 8]])
    mask = src.eq(0)
    pos = torch.zeros(B, Lp, 3)
    pos[:, 0, :] = torch.tensor([3.0, 3.0, 0.25])
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:4, :] = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    return src, mask, pos


def _tiny(decoder_layers=None):
    kwargs = dict(
        token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, dim_feedforward=64, dropout=0.0,
        activation="gelu", normalize_before=False, decoupled_decoder=False,
        use_gated_cross_attn=False, head_type="legacy", predict_scale=False,
        scale_mode="eta",
    )
    if decoder_layers is not None:
        kwargs["num_decoder_layers"] = decoder_layers
    return Transformer(**kwargs)


def test_r2a_default_and_explicit_six_are_bitwise_equal():
    """The new runner knob must not perturb frozen B7 behavior at its default."""
    torch.manual_seed(42)
    default = _tiny()
    torch.manual_seed(42)
    explicit = _tiny(decoder_layers=6)
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key in default.state_dict():
        assert torch.equal(default.state_dict()[key], explicit.state_dict()[key]), key
    src, mask, pos = _batch()
    default.eval()
    explicit.eval()
    with torch.no_grad():
        got_default = default(src, mask, pos)
        got_explicit = explicit(src, mask, pos)
    for key in ("edos", "phdos", "eta"):
        assert torch.equal(got_default[key], got_explicit[key]), key


def test_r2a_three_layers_shape_and_gradient_contract():
    torch.manual_seed(42)
    model = _tiny(decoder_layers=3)
    assert len(model.decoder.layers) == 3
    src, mask, pos = _batch()
    model.train()
    out = model(src, mask, pos)
    assert out["edos"].shape == (2, 8)
    assert out["phdos"].shape == (2, 4)
    assert out["eta"].shape == (2, 2)
    assert all(torch.isfinite(out[k]).all() for k in ("edos", "phdos", "eta"))
    loss = out["edos"].square().mean() + out["phdos"].square().mean() + out["eta"].square().mean()
    loss.backward()
    grads = [p.grad for p in model.decoder.layers[-1].parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_r2a_config_default_and_override():
    assert ExperimentConfig().decoder_layers == 6
    cfg = ExperimentConfig(decoder_layers=3)
    assert cfg.decoder_layers == 3


def test_r2a_production_train_one_step_sumnorm_h1():
    """Production dimensions, Q1 batch, SumNorm and H1 must remain intact."""
    from datasets.dataset import Dos_Dataset
    from model.model import basemodel
    from torch.utils.data import DataLoader

    logger = logging.getLogger("r2atest")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    params = dict(
        dos_minmax=True, dos_zscore=False, apply_log=False, scale_factor=1.0,
        loss_form="sumnorm_klw", use_mask=False, lambda_ph=1.0, grad_clip=0.0,
        w_w1=1.0, w_huber=1.0, huber_delta=0.02,
        tv_w=0.0, grad_w=0.0, peak_w=1.0, tail_w=1.0, tail_start=-1,
        scale_sup_w=1.0, eta_sup_w=1.0, delta_edos=0.09375, delta_phdos=19.6875,
        scalar_sup_w=1.0, c5_moe_balance_w=0.01, save_best="balanced_score",
        metrics_list=[],
        sub_model=dict(transformer=dict(
            token_num=118, d_model=512, nhead=8, edos_num=128, phdos_num=64,
            num_encoder_layers=6, num_decoder_layers=3, dim_feedforward=2048,
            dropout=0.05, activation="gelu", normalize_before=False,
            decoupled_decoder=False, use_gated_cross_attn=False,
            head_type="legacy", predict_scale=False, atom_feat_mode="legacy3",
            energy_code="none", scale_mode="eta", scalar_mode="none",
            use_g1=False, use_g2=False)),
        optimizer=dict(transformer=dict(
            type="AdamW", params=dict(lr=5e-5, betas=[0.9, 0.99]))),
        lr_scheduler={},
    )
    torch.manual_seed(42)
    model = basemodel(logger, **params)
    model.to(torch.device("cpu"))
    batch = next(iter(DataLoader(
        Dos_Dataset(data_dir="./data/train4ARPAT", split="train",
                    dos_minmax=True, dos_sumnorm=True), batch_size=2)))
    assert batch[14] is not None, "Q1 nvalence sidecar must be present for H1"
    model.model["transformer"].train()
    result = model.train_one_step(batch, step=0)
    for key, value in result.items():
        assert math.isfinite(value), f"{key} not finite: {value}"
    assert result["loss_eta"] > 0, "H1 eta/gamma supervision must be active"
    grads = [p.grad for p in model.model["transformer"].decoder.layers[-1].parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


class TestR2aDecoderDepth(unittest.TestCase):
    def test_default_parity(self):
        test_r2a_default_and_explicit_six_are_bitwise_equal()

    def test_three_layer_contract(self):
        test_r2a_three_layers_shape_and_gradient_contract()

    def test_config(self):
        test_r2a_config_default_and_override()

    def test_production_smoke(self):
        test_r2a_production_train_one_step_sumnorm_h1()


if __name__ == "__main__":
    unittest.main()
