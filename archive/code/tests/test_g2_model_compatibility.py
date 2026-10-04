"""Historical G2 residual/model compatibility; shared geometry fixtures are reused."""

import itertools
import math
import unittest
import torch
from model.transformer import Transformer
from utils.periodic_neighbors import (
    build_g2_edges,
    g2_quintic_cutoff,
    g2_rbf_features,
)
from utils.relative_features import build_cell_from_lattice
from tests.data.test_periodic_neighbors import (
    R_CUT, _pos, _brute_edges, _edge_key_set, _cell_to_lattice_params,
)

def _tiny(d_model=32, nhead=4, use_g2=False):
    torch.manual_seed(0)
    return Transformer(
        token_num=128, d_model=d_model, nhead=nhead, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=True, use_gated_cross_attn=False, head_type="legacy",
        predict_scale=False, use_g2=use_g2,
    )

def _model_inputs_for_fracs(a, b, c, angles, fracs, seed=0):
    """Full-model (src, mask, pos) triple sharing one species draw."""
    torch.manual_seed(seed)
    n = torch.as_tensor(fracs).shape[0] if not torch.is_tensor(fracs) else fracs.shape[0]
    pos = _pos(a, b, c, angles, fracs)
    Lp = n + 2
    src = torch.zeros(1, Lp, dtype=torch.long)
    src[0, 0] = 126
    src[0, 1] = 127
    src[0, 2:] = torch.randint(1, 30, (n,))
    return src, src.eq(0), pos

def test_g2_full_model_translation_invariance():
    # Same lattice/frame, shifted fractional origin: the B7 dense path and
    # the G2 radial path are both blind to the shift, so full readouts agree.
    torch.manual_seed(0)
    fracs = torch.rand(4, 3) * 0.9 + 0.05
    src, m, pos = _model_inputs_for_fracs(
        4.0, 5.0, 6.0, (90.0, 90.0, 90.0), fracs, seed=11)
    pos2 = _pos(4.0, 5.0, 6.0, (90.0, 90.0, 90.0), (fracs + 0.37) % 1.0)
    model = _tiny(use_g2=True)
    model.eval()
    with torch.no_grad():
        out = model(src, m, pos)
        out2 = model(src, m, pos2)
    assert (out["edos"] - out2["edos"]).abs().max().item() < 1e-5
    assert (out["phdos"] - out2["phdos"]).abs().max().item() < 1e-5

def test_g2_module_basis_swap_invariance():
    # Basis swap re-expresses the cell in a rotated frame: the direction-fed
    # B7 backbone legitimately varies (untrained B7 edos diff ~2e-02), so the
    # G2 contract is scoped to the frame-blind radial residual, which agrees.
    from model.transformer import PeriodicEdgeMessage
    torch.manual_seed(0)
    fracs = torch.rand(4, 3) * 0.9 + 0.05
    pos = _pos(4.0, 5.0, 6.0, (80.0, 95.0, 100.0), fracs)
    pos2 = _pos(5.0, 4.0, 6.0, (95.0, 80.0, 100.0), fracs[:, [1, 0, 2]])
    mask = torch.zeros(1, 4, dtype=torch.bool)
    e1 = build_g2_edges(pos, mask)
    e2 = build_g2_edges(pos2, mask)
    torch.manual_seed(0)
    msg = PeriodicEdgeMessage(16, rbf_num=64, r_cut=5.5)
    with torch.no_grad():
        msg.alpha.fill_(1.0)
    msg.eval()
    torch.manual_seed(5)
    h = torch.randn(1, 4, 16)
    with torch.no_grad():
        y1 = msg(h, e1["batch"], e1["dst"], e1["src"], e1["distances"])
        y2 = msg(h, e2["batch"], e2["dst"], e2["src"], e2["distances"])
    assert (y1 - y2).abs().max().item() < 1e-5

def test_g2_module_unimodular_invariance():
    # Same scoping as basis swap: edge distances agree and the radial
    # residual agrees on shared hidden states.
    from model.transformer import PeriodicEdgeMessage
    torch.manual_seed(0)
    fracs = torch.rand(4, 3) * 0.9 + 0.05
    pos = _pos(4.0, 5.0, 6.0, (80.0, 95.0, 100.0), fracs)
    cell, _ = build_cell_from_lattice(pos)
    U = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cell2 = U @ cell[0]
    na, nb, nc, angs = _cell_to_lattice_params(cell2)
    fracs2 = (fracs @ torch.linalg.inv(U)) % 1.0
    pos2 = _pos(na, nb, nc, angs, fracs2)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    e1 = build_g2_edges(pos, mask)
    e2 = build_g2_edges(pos2, mask)
    torch.manual_seed(0)
    msg = PeriodicEdgeMessage(16, rbf_num=64, r_cut=5.5)
    with torch.no_grad():
        msg.alpha.fill_(1.0)
    msg.eval()
    torch.manual_seed(6)
    h = torch.randn(1, 4, 16)
    with torch.no_grad():
        y1 = msg(h, e1["batch"], e1["dst"], e1["src"], e1["distances"])
        y2 = msg(h, e2["batch"], e2["dst"], e2["src"], e2["distances"])
    assert (y1 - y2).abs().max().item() < 1e-5

def test_g2_default_off_and_param_parity():
    torch.manual_seed(0)
    base = _tiny(use_g2=False)
    torch.manual_seed(0)
    g2 = _tiny(use_g2=True)
    nb = sum(p.numel() for p in base.parameters())
    ng = sum(p.numel() for p in g2.parameters())
    assert ng > nb, "G2a must add parameters when enabled"
    assert not hasattr(base.encoder, "g2_msgs"), "off-path must not instantiate G2a"
    assert hasattr(g2.encoder, "g2_msgs") and len(g2.encoder.g2_msgs) == 1
    assert set(base.state_dict().keys()) == set(base.state_dict().keys())
    assert set(base.state_dict().keys()) < set(g2.state_dict().keys())

def test_g2_alpha_zero_bitwise_equivalence():
    torch.manual_seed(42)
    base = _tiny(use_g2=False)
    torch.manual_seed(7)
    g2 = _tiny(use_g2=True)
    g2.load_state_dict(base.state_dict(), strict=False)
    assert float(g2.encoder.g2_msgs[0].alpha.item()) == 0.0
    src = torch.zeros(2, 6, dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    src[:, 2:4] = torch.tensor([[1, 8], [14, 8]])
    mask = src.eq(0)
    pos = torch.zeros(2, 6, 3)
    pos[:, 0, :] = torch.tensor([2.0, 3.0, 0.25])
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:4, :] = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    base.eval()
    g2.eval()
    with torch.no_grad():
        ob = base(src, mask, pos)
        og = g2(src, mask, pos)
    assert torch.equal(ob["edos"], og["edos"]), "alpha=0 eval must be bitwise identical"
    assert torch.equal(ob["phdos"], og["phdos"])

def test_g2_grad_unfreezing():
    torch.manual_seed(0)
    model = _tiny(use_g2=True)
    model.train()
    src = torch.zeros(2, 6, dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    src[:, 2:4] = torch.tensor([[1, 8], [14, 8]])
    mask = src.eq(0)
    pos = torch.zeros(2, 6, 3)
    pos[:, 0, :] = torch.tensor([2.0, 3.0, 0.25])
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:4, :] = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    msg = model.encoder.g2_msgs[0]
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    opt.zero_grad()
    out = model(src, mask, pos)
    (out["edos"].square().mean() + out["phdos"].square().mean()).backward()
    assert msg.alpha.grad is not None and torch.isfinite(msg.alpha.grad).all()
    assert msg.alpha.grad.abs().item() > 0, "alpha must take a finite nonzero first gradient"
    opt.step()
    opt.zero_grad()
    out = model(src, mask, pos)
    (out["edos"].square().mean() + out["phdos"].square().mean()).backward()
    for name, mod in (("W_v", msg.W_v), ("W_g", msg.W_g), ("W_o", msg.W_o)):
        grads = [p.grad for p in mod.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert any(g.abs().sum().item() > 0 for g in grads), f"{name} stayed frozen"

def test_g2_message_permutation_equivariance():
    from model.transformer import PeriodicEdgeMessage
    torch.manual_seed(0)
    B, L, D = 1, 5, 16
    msg = PeriodicEdgeMessage(D, rbf_num=64, r_cut=5.5)
    msg.train()
    h = torch.randn(B, L, D)
    fracs = torch.rand(L, 3) * 0.9 + 0.05
    pos = _pos(5.0, 5.0, 5.0, (90.0, 90.0, 90.0), fracs)
    mask = torch.zeros(1, L, dtype=torch.bool)
    edges = build_g2_edges(pos, mask)
    perm = torch.randperm(L)
    pos_p = _pos(5.0, 5.0, 5.0, (90.0, 90.0, 90.0), fracs[perm])
    edges_p = build_g2_edges(pos_p, mask)
    with torch.no_grad():
        y = msg(h, edges["batch"], edges["dst"], edges["src"], edges["distances"])
        y_p = msg(h[:, perm, :], edges_p["batch"], edges_p["dst"],
                  edges_p["src"], edges_p["distances"])
    inv = torch.empty(L, dtype=torch.long)
    inv[perm] = torch.arange(L)
    # y_p is in permuted order; map back and compare.
    assert (y_p[:, inv, :] - y).abs().max().item() < 1e-5

def test_g2_full_model_permutation_invariance():
    torch.manual_seed(0)
    n = 5
    fracs = torch.rand(n, 3) * 0.9 + 0.05
    perm = torch.randperm(n)
    model = _tiny(use_g2=True)
    model.eval()
    Lp = n + 2
    src = torch.zeros(1, Lp, dtype=torch.long)
    src[0, 0] = 126
    src[0, 1] = 127
    src[0, 2:] = torch.randint(1, 30, (n,))
    src_p = src.clone()
    src_p[0, 2:] = src[0, 2:][perm]
    m = src.eq(0)
    mp = src_p.eq(0)
    pos = torch.zeros(1, Lp, 3)
    pos[0, 0, :] = torch.tensor([5.0, 5.0, 0.2])
    pos[0, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[0, 2:, :] = fracs
    pos_p = pos.clone()
    pos_p[0, 2:, :] = fracs[perm]
    with torch.no_grad():
        out = model(src, m, pos)
        out_p = model(src_p, mp, pos_p)
    assert (out["edos"] - out_p["edos"]).abs().max().item() < 1e-5
    assert (out["phdos"] - out_p["phdos"]).abs().max().item() < 1e-5

def test_g2_real_q1_cpu_smoke():
    from utils.dataset import Dos_Dataset
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(
        Dos_Dataset(data_dir="./data/train4ARPAT", split="train"), batch_size=2)))
    src, pos = batch[0], batch[1]
    model = _tiny(use_g2=True)
    model.eval()
    with torch.no_grad():
        out = model(src, src.eq(0), pos)
    assert out["edos"].shape == (2, 8)
    assert out["phdos"].shape == (2, 4)
    for k in ("edos", "phdos"):
        assert torch.isfinite(out[k]).all(), k
        assert out[k].abs().max().item() > 0, f"{k} degenerate"

def test_g2_production_train_one_step_sumnorm_h1():
    """Gate 6 (CPU part): production M1 recipe + G2 single train_one_step.

    Uses the exact `_g2edge` training path — d_model 512, 6+6 layers,
    SumNorm KL/W1/Huber (`loss_form='sumnorm_klw'`) and the H1 eta/gamma head
    (`scale_mode='eta'`) on a real Q1 train batch — and requires every loss
    key finite with an active (nonzero) H1 `loss_eta`.
    """
    import logging
    import math
    from utils.dataset import Dos_Dataset
    from torch.utils.data import DataLoader
    from model.model import basemodel

    logger = logging.getLogger("g2test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    params = dict(
        dos_minmax=True, dos_zscore=False, apply_log=False, scale_factor=1.0,
        loss_form="sumnorm_klw", use_mask=False, lambda_ph=1.0, grad_clip=0.0,
        w_w1=1.0, w_huber=1.0, huber_delta=0.02,
        tv_w=0.0, grad_w=0.0, peak_w=1.0, tail_w=1.0, tail_start=-1,
        scale_sup_w=1.0, eta_sup_w=1.0, delta_edos=0.09375, delta_phdos=19.6875,
        scalar_sup_w=1.0, c5_moe_balance_w=0.01,
        save_best="balanced_score", metrics_list=[],
        sub_model=dict(transformer=dict(
            token_num=118, d_model=512, nhead=8, edos_num=128, phdos_num=64,
            num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
            dropout=0.05, activation="gelu", normalize_before=False,
            decoupled_decoder=False, use_gated_cross_attn=False,
            head_type="legacy", predict_scale=False, atom_feat_mode="legacy3",
            energy_code="none", scale_mode="eta", scalar_mode="none",
            use_g1=False, use_g2=True, g2_r_cut=5.5)),
        optimizer=dict(transformer=dict(
            type="AdamW", params=dict(lr=5e-5, betas=[0.9, 0.99]))),
        lr_scheduler={},
    )
    torch.manual_seed(42)
    model = basemodel(logger, **params)
    model.to(torch.device("cpu"))
    ds = Dos_Dataset(data_dir="./data/train4ARPAT", split="train",
                     dos_minmax=True, dos_sumnorm=True)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch[14] is not None, "Q1 nvalence sidecar must be present for H1"
    model.model["transformer"].train()
    out = model.train_one_step(batch, step=0)
    for k in ("loss", "loss_edos", "loss_phdos", "loss_eta"):
        assert k in out, f"missing loss key {k}"
        assert math.isfinite(out[k]), f"{k} not finite: {out[k]}"
    for k, v in out.items():
        assert math.isfinite(v), f"{k} not finite: {v}"
    assert out["loss_eta"] > 0, "H1 loss_eta must be active on Q1"
    # The G2 residual must participate in the training path.
    alpha = model.model["transformer"].encoder.g2_msgs[0].alpha
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all(), \
        "G2 alpha took no finite gradient in production train_one_step"

class TestHistoricalG2Model(unittest.TestCase):
    def test_off_parity(self):
        test_g2_default_off_and_param_parity()

    def test_bitwise(self):
        test_g2_alpha_zero_bitwise_equivalence()

    def test_grad(self):
        test_g2_grad_unfreezing()

    def test_msg_equivariance(self):
        test_g2_message_permutation_equivariance()

    def test_model_invariance(self):
        test_g2_full_model_permutation_invariance()

    def test_translation_model(self):
        test_g2_full_model_translation_invariance()

    def test_module_basis_swap(self):
        test_g2_module_basis_swap_invariance()

    def test_module_unimodular(self):
        test_g2_module_unimodular_invariance()

    def test_q1_smoke(self):
        test_g2_real_q1_cpu_smoke()

    def test_production_smoke(self):
        test_g2_production_train_one_step_sumnorm_h1()
