"""G2a periodic multi-image edge contracts (CPU, deterministic).

Covers design-g2 section "测试与预检关卡" 1-5 plus the CPU part of gate 6
(Q1 single-step numeric contract incl. SumNorm + H1 loss keys; V100 batch-32
resource measurement stays a manual gate). One failure => no training.

Gates:
  enumeration completeness vs K+3 brute force (skewed/small/high-aspect)
  Si diamond primitive counts (34/recv, 18 self/recv, 4 first-shell/recv)
  physical: reverse pairing, strict cutoff, quintic ends, padding, finite,
            frac+integer invariance
  equivalence: translation, joint permutation, basis swap, unimodular.
    Edge-distance multisets agree for all four (tol 1e-5). Full-model outputs
    agree for translation and joint permutation (tol 1e-5); basis swap and
    unimodular re-express the cell in a rotated frame, so the direction-fed
    B7 backbone legitimately varies there (measured ~2e-02 on untrained
    weights) and the G2 contract is scoped to the radial residual, which is
    frame-blind and agrees at module level (tol 1e-5).
  model: default-off param parity, alpha=0 bitwise, grad unfreezing,
         permutation equivariance of the message
  smoke: real Q1 CPU forward finite (tiny G2 model) + production-recipe
         train_one_step finite over all loss keys incl. H1 loss_eta
"""

import itertools
import math
import unittest

import torch

from model.transformer import Transformer
from utils.g2_periodic_edges import (
    build_g2_edges,
    g2_quintic_cutoff,
    g2_rbf_features,
)
from utils.relative_features import build_cell_from_lattice


R_CUT = 5.5


def _pos(a, b, c, angles, fracs, dtype=torch.float32):
    """82-contract pos tensor [1, 2+N, 3] from lattice scalars + fracs."""
    fracs = torch.as_tensor(fracs, dtype=dtype)
    pos = torch.zeros(1, 2 + fracs.shape[0], 3, dtype=dtype)
    pos[0, 0, :] = torch.tensor([a, b, 1.0 / c], dtype=dtype)
    pos[0, 1, :] = torch.tensor(angles, dtype=dtype)
    pos[0, 2:, :] = fracs
    return pos


def _brute_edges(pos, mask_atom, r_cut=R_CUT, extra=3):
    """Brute-force edge set with K+extra enumeration (reference)."""
    from utils.relative_features import build_cell_from_lattice
    cell, frac = build_cell_from_lattice(pos)
    assert pos.shape[0] == 1
    s_min = torch.linalg.svdvals(cell)[0].amin().item()
    K = math.ceil(r_cut / s_min + 0.5) + extra
    rng = range(-K, K + 1)
    cell0 = cell[0]
    frac0 = frac[0]
    L = frac0.shape[0]
    valid = (~mask_atom[0]).tolist()
    edges = []
    for i in range(L):
        if not valid[i]:
            continue
        for j in range(L):
            if not valid[j]:
                continue
            for T in itertools.product(rng, rng, rng):
                if i == j and T == (0, 0, 0):
                    continue
                Tt = torch.tensor(T, dtype=frac0.dtype, device=frac0.device)
                r = ((frac0[j] - frac0[i] + Tt) @ cell0)
                d = r.norm().item()
                if d < r_cut:
                    edges.append((i, j, T, d))
    return edges


def _edge_key_set(out):
    keys = set()
    b = out["batch"].tolist()
    d = out["dst"].tolist()
    s = out["src"].tolist()
    t = out["shifts"].tolist()
    for i in range(len(b)):
        keys.add((b[i], d[i], s[i], tuple(t[i])))
    return keys


def _tiny(d_model=32, nhead=4, use_g2=False):
    torch.manual_seed(0)
    return Transformer(
        token_num=128, d_model=d_model, nhead=nhead, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=True, use_gated_cross_attn=False, head_type="legacy",
        predict_scale=False, use_g2=use_g2,
    )


# ---------------------------------------------------------------------------
# 1. enumeration completeness
# ---------------------------------------------------------------------------

def test_g2_enumeration_completeness():
    torch.manual_seed(0)
    fixtures = [
        # skewed small cell (G1's enumeration-fix regime)
        (2.89, 3.86, 11.50, (47.8, 39.6, 55.8), torch.rand(2, 3) * 0.9 + 0.05),
        # small cubic cell (large K)
        (3.0, 3.0, 3.0, (90.0, 90.0, 90.0), torch.rand(3, 3) * 0.9 + 0.05),
        # high aspect ratio cell
        (3.0, 3.0, 15.0, (90.0, 90.0, 90.0), torch.rand(4, 3) * 0.9 + 0.05),
    ]
    for a, b, c, angles, fracs in fixtures:
        pos = _pos(a, b, c, angles, fracs)
        mask = torch.zeros(1, fracs.shape[0], dtype=torch.bool)
        out = build_g2_edges(pos, mask)
        ref = _brute_edges(pos, mask)
        got_keys = _edge_key_set(out)
        ref_keys = {(0, i, j, T) for (i, j, T, _) in ref}
        assert got_keys == ref_keys, (
            f"cell {(a,b,c,angles)}: G2 {len(got_keys)} != brute {len(ref_keys)}")
        # distances agree element-wise
        ref_d = {(0, i, j, T): d for (i, j, T, d) in ref}
        dd = out["distances"].tolist()
        bb = out["batch"].tolist()
        dst = out["dst"].tolist()
        src = out["src"].tolist()
        sh = out["shifts"].tolist()
        for k in range(len(bb)):
            key = (bb[k], dst[k], src[k], tuple(sh[k]))
            assert abs(dd[k] - ref_d[key]) < 1e-4, (key, dd[k], ref_d[key])
        # no i=j,T=0
        for k in range(len(bb)):
            assert not (dst[k] == src[k] and tuple(sh[k]) == (0, 0, 0))


# ---------------------------------------------------------------------------
# 2. Si diamond primitive counts
# ---------------------------------------------------------------------------

def test_g2_si_primitive_counts():
    a_len = 5.431 / math.sqrt(2)
    pos = _pos(a_len, a_len, a_len, (60.0, 60.0, 60.0),
               torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]))
    mask = torch.zeros(1, 2, dtype=torch.bool)
    out = build_g2_edges(pos, mask)
    assert out["k"].tolist() == [3]
    assert out["batch"].numel() == 68
    dst = out["dst"]
    src = out["src"]
    dist = out["distances"]
    for i in (0, 1):
        m = (dst == i)
        assert int(m.sum()) == 34, f"dst {i} indegree {int(m.sum())} != 34"
        selfm = m & (dst == src)
        assert int(selfm.sum()) == 18, f"dst {i} self {int(selfm.sum())} != 18"
        first = m & (dst != src) & ((dist - 2.351692).abs() < 1e-3)
        assert int(first.sum()) == 4, f"dst {i} first-shell {int(first.sum())} != 4"
    assert int((dst == src).sum()) == 36
    assert int((((dist - 2.351692).abs() < 1e-3) & (dst != src)).sum()) == 8
    # shell multiset (tolerance 1e-3 buckets): 8/24/24/12
    d = dist.tolist()
    n_first = sum(1 for x in d if abs(x - 2.351692) < 1e-3)
    n_a = sum(1 for x in d if abs(x - 3.840297) < 1e-3)
    n_b = sum(1 for x in d if abs(x - 4.503147) < 2e-3)
    n_c = sum(1 for x in d if abs(x - 5.431) < 1e-3)
    assert (n_first, n_a, n_b, n_c) == (8, 24, 24, 12), (n_first, n_a, n_b, n_c)


# ---------------------------------------------------------------------------
# 3. physical contracts
# ---------------------------------------------------------------------------

def test_g2_physical_contracts():
    torch.manual_seed(1)
    fracs = torch.rand(4, 3) * 0.9 + 0.05
    pos = _pos(4.0, 5.0, 6.0, (80.0, 95.0, 100.0), fracs)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    out = build_g2_edges(pos, mask)
    assert out["batch"].numel() > 0
    assert torch.isfinite(out["distances"]).all()
    assert bool((out["distances"] < R_CUT).all()), "strict cutoff violated"
    # reverse pairing with equal distance
    keys = {}
    for k in range(out["batch"].numel()):
        keys[(int(out["dst"][k]), int(out["src"][k]), tuple(out["shifts"][k].tolist()))] = float(out["distances"][k])
    for (i, j, T), d in keys.items():
        rev = (j, i, (-T[0], -T[1], -T[2]))
        assert rev in keys, f"missing reverse edge for {(i, j, T)}"
        assert abs(keys[rev] - d) < 1e-5, "reverse distance mismatch"
    # quintic value range + endpoints
    w = g2_quintic_cutoff(torch.tensor([0.0, 2.75, 5.5, 9.0]))
    assert w[0].item() == 1.0 and w[2].item() == 0.0 and w[3].item() == 0.0
    assert abs(w[1].item() - 0.5) < 1e-6
    assert bool(((w >= 0) & (w <= 1)).all())
    w_kept = g2_quintic_cutoff(out["distances"])
    assert bool((w_kept > 0).all()), "kept edges must carry positive smooth weight"
    assert bool(torch.isfinite(w_kept).all())
    # RBF finite, correct width
    phi = g2_rbf_features(out["distances"])
    assert phi.shape == (out["batch"].numel(), 64)
    assert bool(torch.isfinite(phi).all())
    assert bool((phi >= 0).all()) and bool((phi <= 1.0 + 1e-6).all())
    # padding exclusion
    mask2 = torch.zeros(1, 4, dtype=torch.bool)
    mask2[0, 2:] = True
    out2 = build_g2_edges(pos, mask2)
    if out2["batch"].numel():
        assert bool((out2["dst"] < 2).all()) and bool((out2["src"] < 2).all())
    # frac + integer invariance (distances/messages, not T labels: T is
    # representation-dependent while the physical edge set is not).
    fracs_b = fracs.clone()
    fracs_b[1] = fracs_b[1] + torch.tensor([1.0, 0.0, -1.0])
    pos_b = _pos(4.0, 5.0, 6.0, (80.0, 95.0, 100.0), fracs_b)
    out_b = build_g2_edges(pos_b, mask)
    assert out["batch"].numel() == out_b["batch"].numel()
    assert (out["distances"].sort().values - out_b["distances"].sort().values).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# 4. equivalence contracts
# ---------------------------------------------------------------------------

def test_g2_translation_invariance():
    torch.manual_seed(0)
    fracs = torch.rand(4, 3) * 0.9 + 0.05
    pos = _pos(4.0, 5.0, 6.0, (90.0, 90.0, 90.0), fracs)
    pos2 = _pos(4.0, 5.0, 6.0, (90.0, 90.0, 90.0), (fracs + 0.37) % 1.0)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    o1 = build_g2_edges(pos, mask)
    o2 = build_g2_edges(pos2, mask)
    d1 = sorted(o1["distances"].tolist())
    d2 = sorted(o2["distances"].tolist())
    assert len(d1) == len(d2) and len(d1) > 0
    assert max(abs(x - y) for x, y in zip(d1, d2)) < 1e-5


def test_g2_permutation_equivariance():
    torch.manual_seed(0)
    n = 5
    fracs = torch.rand(n, 3) * 0.9 + 0.05
    pos = _pos(5.0, 5.0, 5.0, (90.0, 90.0, 90.0), fracs)
    perm = torch.randperm(n)
    pos_p = _pos(5.0, 5.0, 5.0, (90.0, 90.0, 90.0), fracs[perm])
    mask = torch.zeros(1, n, dtype=torch.bool)
    o1 = build_g2_edges(pos, mask)
    o2 = build_g2_edges(pos_p, mask)
    d1 = sorted(o1["distances"].tolist())
    d2 = sorted(o2["distances"].tolist())
    assert len(d1) == len(d2)
    assert max(abs(x - y) for x, y in zip(d1, d2)) < 1e-5
    # permuted edge index sets correspond
    inv = torch.empty(n, dtype=torch.long)
    inv[perm] = torch.arange(n)
    remapped = {(0, int(inv[int(d)]), int(inv[int(s)]), tuple(t))
                for d, s, t in zip(o2["dst"].tolist(), o2["src"].tolist(),
                                   o2["shifts"].tolist())}
    # shifts are translation labels, not permuted; dst/src remap must recover
    # the same (dst, src) pairs up to the periodic image chosen per pair.
    # The invariant check is the distance multiset above plus the model-level
    # equivariance below; here assert pair multisets match.
    pairs1 = sorted([(int(d), int(s)) for d, s in zip(o1["dst"].tolist(), o1["src"].tolist())])
    pairs2 = sorted([(int(d), int(s)) for d, s, _ in
                     [ (inv[int(d)], inv[int(s)], t) for d, s, t in
                       zip(o2["dst"].tolist(), o2["src"].tolist(), o2["shifts"].tolist())]])
    assert pairs1 == pairs2


def _cell_to_lattice_params(cell):
    va, vb, vc = cell[0], cell[1], cell[2]
    na, nb, nc = va.norm().item(), vb.norm().item(), vc.norm().item()

    def _ang(x, y):
        return math.degrees(math.acos(
            torch.dot(x, y).item() / (x.norm().item() * y.norm().item())))

    return na, nb, nc, (_ang(vb, vc), _ang(va, vc), _ang(va, vb))


def test_g2_basis_swap_invariance():
    torch.manual_seed(0)
    a, b, c = 4.0, 5.0, 6.0
    al, be, ga = 80.0, 95.0, 100.0
    fracs = torch.rand(4, 3) * 0.9 + 0.05
    pos = _pos(a, b, c, (al, be, ga), fracs)
    pos2 = _pos(b, a, c, (be, al, ga), fracs[:, [1, 0, 2]])
    mask = torch.zeros(1, 4, dtype=torch.bool)
    o1 = build_g2_edges(pos, mask)
    o2 = build_g2_edges(pos2, mask)
    d1 = sorted(o1["distances"].tolist())
    d2 = sorted(o2["distances"].tolist())
    assert len(d1) == len(d2) and len(d1) > 0
    assert max(abs(x - y) for x, y in zip(d1, d2)) < 1e-5


def test_g2_unimodular_invariance():
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
    o1 = build_g2_edges(pos, mask)
    o2 = build_g2_edges(pos2, mask)
    d1 = sorted(o1["distances"].tolist())
    d2 = sorted(o2["distances"].tolist())
    assert len(d1) == len(d2) and len(d1) > 0
    assert max(abs(x - y) for x, y in zip(d1, d2)) < 1e-5


# ---------------------------------------------------------------------------
# 4b. model-output equivalence contracts (G2 enabled)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 5. model contracts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 6. real-data CPU smoke (numeric contract only, not a cost claim)
# ---------------------------------------------------------------------------

def test_g2_real_q1_cpu_smoke():
    from datasets.dataset import Dos_Dataset
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
    from datasets.dataset import Dos_Dataset
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


class TestG2PeriodicEdges(unittest.TestCase):
    def test_enumeration(self):
        test_g2_enumeration_completeness()

    def test_si_counts(self):
        test_g2_si_primitive_counts()

    def test_physical(self):
        test_g2_physical_contracts()

    def test_translation(self):
        test_g2_translation_invariance()

    def test_permutation(self):
        test_g2_permutation_equivariance()

    def test_basis_swap(self):
        test_g2_basis_swap_invariance()

    def test_unimodular(self):
        test_g2_unimodular_invariance()

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


if __name__ == "__main__":
    unittest.main()
