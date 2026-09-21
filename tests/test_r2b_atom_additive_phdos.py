"""R2b atom-additive fixed-grid phDOS contracts (CPU, deterministic)."""
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


def _tiny(use_atom_additive_phdos=False):
    return Transformer(
        token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=3, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=False, use_gated_cross_attn=False,
        head_type="legacy", predict_scale=False, scale_mode="eta",
        use_atom_additive_phdos=use_atom_additive_phdos,
    )


def test_r2b_off_path_is_bitwise_r2a():
    torch.manual_seed(42)
    default = _tiny()
    torch.manual_seed(42)
    explicit_off = _tiny(use_atom_additive_phdos=False)
    assert default.state_dict().keys() == explicit_off.state_dict().keys()
    for key in default.state_dict():
        assert torch.equal(default.state_dict()[key], explicit_off.state_dict()[key]), key
    assert not hasattr(default, "atom_phdos_head")
    src, mask, pos = _batch()
    default.eval()
    explicit_off.eval()
    with torch.no_grad():
        a, b = default(src, mask, pos), explicit_off(src, mask, pos)
    for key in ("edos", "phdos", "eta"):
        assert torch.equal(a[key], b[key]), key


def test_r2b_additivity_padding_and_permutation_contracts():
    torch.manual_seed(42)
    model = _tiny(use_atom_additive_phdos=True)
    assert not hasattr(model, "phdos_out_head")
    assert not hasattr(model, "phdos_query_embed")
    assert sum(p.numel() for p in model.atom_phdos_head.parameters()) == 32 * 32 + 32 + 32 * 4 + 4
    src, mask, pos = _batch()
    model.eval()
    with torch.no_grad():
        out = model(src, mask, pos)
    contrib = out["atom_phdos_contrib"]
    assert contrib.shape == (2, 4, 4)
    assert (contrib >= 0).all()
    assert torch.equal(contrib[mask[:, 2:]], torch.zeros_like(contrib[mask[:, 2:]]))
    assert torch.allclose(out["phdos"].exp(), contrib.sum(dim=1), atol=1e-6)

    # Padding rows are not keys, senders or contributors.
    src_pad, mask_pad, pos_pad = src.clone(), mask.clone(), pos.clone()
    pos_pad[:, 4:, :] = 999.0
    with torch.no_grad():
        padded = model(src_pad, mask_pad, pos_pad)
    assert torch.allclose(out["phdos"], padded["phdos"], atol=1e-6)

    # Joint atom permutation permutes local contributions and preserves totals.
    perm = torch.tensor([1, 0, 3, 2])
    inv = torch.argsort(perm)
    src_perm, mask_perm, pos_perm = src.clone(), mask.clone(), pos.clone()
    src_perm[:, 2:] = src[:, 2:][:, perm]
    mask_perm[:, 2:] = mask[:, 2:][:, perm]
    pos_perm[:, 2:] = pos[:, 2:][:, perm]
    with torch.no_grad():
        permuted = model(src_perm, mask_perm, pos_perm)
    assert torch.allclose(out["phdos"], permuted["phdos"], atol=1e-5)
    assert torch.allclose(out["atom_phdos_contrib"],
                          permuted["atom_phdos_contrib"][:, inv], atol=1e-5)


def test_r2b_gradients_and_config_contract():
    assert not ExperimentConfig().use_atom_additive_phdos
    assert ExperimentConfig(use_atom_additive_phdos=True).use_atom_additive_phdos
    torch.manual_seed(42)
    model = _tiny(use_atom_additive_phdos=True)
    src, mask, pos = _batch()
    model.train()
    out = model(src, mask, pos)
    loss = out["edos"].square().mean() + out["phdos"].square().mean() + out["eta"].square().mean()
    loss.backward()
    grads = [p.grad for p in model.atom_phdos_head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_r2b_production_train_one_step_sumnorm_h1():
    from datasets.dataset import Dos_Dataset
    from model.model import basemodel
    from torch.utils.data import DataLoader

    logger = logging.getLogger("r2btest")
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
            use_g1=False, use_g2=False, use_atom_additive_phdos=True)),
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
    model.model["transformer"].train()
    result = model.train_one_step(batch, step=0)
    for key, value in result.items():
        assert math.isfinite(value), f"{key} not finite: {value}"
    assert result["loss_eta"] > 0, "H1 eta/gamma supervision must be active"
    grads = [p.grad for p in model.model["transformer"].atom_phdos_head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


class TestR2bAtomAdditivePhDOS(unittest.TestCase):
    def test_off_path(self):
        test_r2b_off_path_is_bitwise_r2a()

    def test_additivity(self):
        test_r2b_additivity_padding_and_permutation_contracts()

    def test_gradients_config(self):
        test_r2b_gradients_and_config_contract()

    def test_production_smoke(self):
        test_r2b_production_train_one_step_sumnorm_h1()


if __name__ == "__main__":
    unittest.main()
