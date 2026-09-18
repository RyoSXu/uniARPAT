"""E9-P0 G1 hard gate tests (Design-E section 6).

Five invariance gates + cutoff/top-k contract + enumeration fix, all CPU,
deterministic, no checkpoint/data. One failure => G2/G3 off the table.

Gates:
  test_g1_enumeration_fixes_skewed_cell  (multi-mirror fix; the reason G1 exists)
  test_g1_rotation_invariance            (rigid rotation of the cell)
  test_g1_translation_invariance         (frac shift mod 1)
  test_g1_permutation_equivariance       (atom reorder; graph + encoder level)
  test_g1_periodic_image_invariance      (frac + integer == same site)
  test_g1_cell_basis_swap_invariance     (a<->b relabelling == same crystal)
Contract:
  test_g1_cutoff_topk_contract           (quintic C2 ends, degree<=49, pad rules)
"""

import itertools
import math
import unittest

import torch

from model.transformer import Transformer
from utils.g1_graph import build_g1_graph, quintic_cutoff
from utils.relative_features import build_cell_from_lattice, compute_relative_features


def _pos(a, b, c, angles, fracs):
    """82-contract pos tensor [1, 2+N, 3] from lattice scalars + fracs."""
    fracs = torch.as_tensor(fracs, dtype=torch.float32)
    pos = torch.zeros(1, 2 + fracs.shape[0], 3)
    pos[0, 0, :] = torch.tensor([a, b, 1.0 / c])
    pos[0, 1, :] = torch.tensor(angles, dtype=torch.float32)
    pos[0, 2:, :] = fracs
    return pos


def _brute_min_dist(cell, fi, fj, t_range=2):
    best = float("inf")
    rng = range(-t_range, t_range + 1)
    for T in itertools.product(rng, rng, rng):
        T = torch.tensor(T, dtype=torch.float32)
        best = min(best, (((fj - fi) + T) @ cell).norm().item())
    return best


def _tiny_g1(d_model=32, nhead=4):
    torch.manual_seed(0)
    return Transformer(
        token_num=128, d_model=d_model, nhead=nhead, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=True, use_gated_cross_attn=False, head_type="legacy",
        predict_scale=False, use_g1=True,
    )


# ---------------------------------------------------------------------------
# 1. enumeration fix (the reason G1 exists)
# ---------------------------------------------------------------------------

def test_g1_enumeration_fixes_skewed_cell():
    """Skewed small cell: fractional rounding overestimates by ~1.3A."""
    a, b, c = 2.89, 3.86, 11.50
    angles = (47.8, 39.6, 55.8)
    fi = torch.tensor([0.1, 0.2, 0.3])
    fj = torch.tensor([0.6, 0.7, 0.8])
    pos = _pos(a, b, c, angles, torch.stack([fi, fj]))
    mask = torch.zeros(1, 2, dtype=torch.bool)
    d, _, _, _ = build_g1_graph(pos, mask)
    cell, _ = build_cell_from_lattice(pos)
    expected = _brute_min_dist(cell[0], fi, fj)
    assert abs(d[0, 0, 1].item() - expected) < 1e-4, (
        f"G1 != brute-force minimum: {d[0,0,1].item():.4f} vs {expected:.4f}")
    old, _ = compute_relative_features(pos)
    assert old[0, 0, 1].item() - d[0, 0, 1].item() > 0.3, (
        "fixture decayed: legacy path no longer wrong here, pick a new cell")


# ---------------------------------------------------------------------------
# 2. rotation invariance (rigid rotation of Cartesian cell)
# ---------------------------------------------------------------------------

def test_g1_rotation_invariance():
    torch.manual_seed(0)
    a, b, c, angles = 4.0, 5.0, 6.0, (80.0, 95.0, 100.0)
    fracs = torch.rand(4, 3) * 0.9 + 0.05
    pos = _pos(a, b, c, angles, fracs)
    cell, _ = build_cell_from_lattice(pos)
    # Rigid rotation: 37 deg about z.
    th = math.radians(37.0)
    R = torch.tensor([[math.cos(th), -math.sin(th), 0.0],
                      [math.sin(th), math.cos(th), 0.0],
                      [0.0, 0.0, 1.0]])
    rot = (R @ cell[0].T).T  # rows = rotated lattice vectors
    va, vb, vc = rot[0], rot[1], rot[2]
    na, nb, nc = va.norm().item(), vb.norm().item(), vc.norm().item()
    def _ang(x, y):
        return math.degrees(math.acos(
            torch.dot(x, y).item() / (x.norm().item() * y.norm().item())))
    pos2 = _pos(na, nb, nc, (_ang(vb, vc), _ang(va, vc), _ang(va, vb)), fracs)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    d1, _, adj1, _ = build_g1_graph(pos, mask)
    d2, _, adj2, _ = build_g1_graph(pos2, mask)
    assert (d1 - d2).abs().max().item() < 1e-3, (
        f"rotation changed distances: {(d1-d2).abs().max().item():.2e}")
    assert torch.equal(adj1, adj2), "rotation changed adjacency"


# ---------------------------------------------------------------------------
# 3. translation invariance
# ---------------------------------------------------------------------------

def test_g1_translation_invariance():
    torch.manual_seed(0)
    fracs = torch.rand(4, 3) * 0.9 + 0.05
    pos = _pos(4.0, 5.0, 6.0, (90.0, 90.0, 90.0), fracs)
    pos2 = _pos(4.0, 5.0, 6.0, (90.0, 90.0, 90.0), (fracs + 0.37) % 1.0)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    d1, u1, adj1, s1 = build_g1_graph(pos, mask)
    d2, u2, adj2, s2 = build_g1_graph(pos2, mask)
    assert (d1 - d2).abs().max().item() < 1e-5
    assert torch.equal(adj1, adj2)
    assert (s1 - s2).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 4. permutation equivariance (graph + encoder level)
# ---------------------------------------------------------------------------

def test_g1_permutation_equivariance():
    torch.manual_seed(0)
    n = 5
    fracs = torch.rand(n, 3) * 0.9 + 0.05
    pos = _pos(5.0, 5.0, 5.0, (90.0, 90.0, 90.0), fracs)
    perm = torch.randperm(n)
    pos_p = _pos(5.0, 5.0, 5.0, (90.0, 90.0, 90.0), fracs[perm])
    mask = torch.zeros(1, n, dtype=torch.bool)
    d, _, adj, _ = build_g1_graph(pos, mask)
    dp, _, adjp, _ = build_g1_graph(pos_p, mask)
    assert (d[:, perm][:, :, perm] - dp).abs().max().item() < 1e-5
    assert torch.equal(adj[:, perm][:, :, perm], adjp)

    # Encoder level: outputs permute, pooled global identical.
    model = _tiny_g1()
    model.eval()
    Lp = n + 2
    src = torch.zeros(1, Lp, dtype=torch.long)
    src[0, 0] = 126
    src[0, 1] = 127
    src[0, 2:] = torch.randint(1, 30, (n,))
    src_p = src.clone()
    src_p[0, 2:] = src[0, 2:][perm]
    m = (src == 0)
    mp = (src_p == 0)
    pos_full = torch.zeros(1, Lp, 3)
    pos_full[0, 0, :] = torch.tensor([5.0, 5.0, 0.2])
    pos_full[0, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos_full[0, 2:, :] = fracs
    pos_full_p = pos_full.clone()
    pos_full_p[0, 2:, :] = fracs[perm]
    # NOTE: numeric atom features ride `src`; permutation must move species
    # together with coordinates (joint relabelling), mirroring dataset rows.
    with torch.no_grad():
        enc = model.encoder
        a = torch.cat([model.tok_emb(src[:, 2:]), model.num_emb_encoder(src[:, 2:])], dim=-1)
        a = model.fuse_proj(a)
    _ = a  # wiring sanity (full forward below is the real assertion)
    with torch.no_grad():
        out = model(src, m, pos_full)
        out_p = model(src_p, mp, pos_full_p)
    # Decoder outputs are permutation-invariant (pooled memory only feeds
    # decoders via cross-attention over the set): assert exact invariance.
    assert (out["edos"] - out_p["edos"]).abs().max().item() < 1e-5, (
        "encoder not permutation-invariant at readout")
    assert (out["phdos"] - out_p["phdos"]).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# 5. periodic image invariance (frac + integer == same site)
# ---------------------------------------------------------------------------

def test_g1_periodic_image_invariance():
    torch.manual_seed(0)
    fracs = torch.rand(3, 3) * 0.9 + 0.05
    fracs2 = fracs.clone()
    fracs2[1] = fracs2[1] + torch.tensor([1.0, 0.0, -1.0])  # same site
    pos = _pos(4.5, 5.5, 6.5, (90.0, 90.0, 90.0), fracs)
    pos2 = _pos(4.5, 5.5, 6.5, (90.0, 90.0, 90.0), fracs2)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    d1, _, adj1, _ = build_g1_graph(pos, mask)
    d2, _, adj2, _ = build_g1_graph(pos2, mask)
    assert (d1 - d2).abs().max().item() < 1e-5
    assert torch.equal(adj1, adj2)


# ---------------------------------------------------------------------------
# 6. cell basis swap invariance (a<->b relabelling == same crystal)
# ---------------------------------------------------------------------------

def test_g1_cell_basis_swap_invariance():
    torch.manual_seed(0)
    a, b, c = 4.0, 5.0, 6.0
    al, be, ga = 80.0, 95.0, 100.0
    fracs = torch.rand(4, 3) * 0.9 + 0.05
    pos = _pos(a, b, c, (al, be, ga), fracs)
    # Swap va<->vb: a'<->b, al'<->be, fx'<->fy (see utils/g1_graph docstring).
    fracs2 = fracs[:, [1, 0, 2]]
    pos2 = _pos(b, a, c, (be, al, ga), fracs2)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    d1, _, adj1, _ = build_g1_graph(pos, mask)
    d2, _, adj2, _ = build_g1_graph(pos2, mask)
    assert (d1 - d2).abs().max().item() < 1e-3, (
        f"basis swap changed distances: {(d1-d2).abs().max().item():.2e}")
    assert torch.equal(adj1, adj2), "basis swap changed adjacency"


# ---------------------------------------------------------------------------
# 7. cutoff + top-k + padding contract
# ---------------------------------------------------------------------------

def test_g1_cutoff_topk_contract():
    # Quintic ends: value + vanishing derivatives at 0 and r_cut.
    w = quintic_cutoff(torch.tensor([0.0, 2.75, 5.5, 9.0]))
    assert w[0].item() == 1.0 and w[2].item() == 0.0 and w[3].item() == 0.0
    assert abs(w[1].item() - 0.5) < 1e-6
    h = 1e-3
    d0 = (quintic_cutoff(torch.tensor([h])) - quintic_cutoff(torch.tensor([0.0]))) / h
    d1 = (quintic_cutoff(torch.tensor([5.5])) - quintic_cutoff(torch.tensor([5.5 - h]))) / h
    assert abs(d0.item()) < 1e-2 and abs(d1.item()) < 1e-2, (d0.item(), d1.item())

    # Dense packing: degree capped at 48 + self; far pairs blocked.
    torch.manual_seed(0)
    n = 60
    fracs = torch.rand(n, 3)
    pos = _pos(20.0, 20.0, 20.0, (90.0, 90.0, 90.0), fracs)
    mask = torch.zeros(1, n, dtype=torch.bool)
    d, _, adj, sm = build_g1_graph(pos, mask)
    assert adj[:, list(range(n)), list(range(n))].all().item(), "self must stay"
    assert adj.sum(dim=-1).max().item() <= 49, "degree cap 48+self violated"
    assert ((d > 5.5) & adj).sum().item() == 0, "beyond-cutoff edge allowed"
    assert (sm < 0).sum().item() == 0, "negative cutoff weight"
    assert ((adj) & (sm < 0)).sum().item() == 0

    # Padding: padded keys blocked; padded query rows stay non-empty/finite.
    mask2 = torch.zeros(1, n, dtype=torch.bool)
    mask2[0, 50:] = True
    d2, u2, adj2, sm2 = build_g1_graph(pos, mask2)
    assert (~adj2[:, :, 50:]).all().item(), "padded keys must be blocked"
    assert adj2.any(dim=-1).all().item(), "empty attention row (NaN risk)"
    for t in (d2, u2, sm2):
        assert torch.isfinite(t).all().item()


class TestG1Graph(unittest.TestCase):
    __test__ = False

    def test_enumeration_fix(self):
        test_g1_enumeration_fixes_skewed_cell()

    def test_rotation(self):
        test_g1_rotation_invariance()

    def test_translation(self):
        test_g1_translation_invariance()

    def test_permutation(self):
        test_g1_permutation_equivariance()

    def test_image(self):
        test_g1_periodic_image_invariance()

    def test_basis_swap(self):
        test_g1_cell_basis_swap_invariance()

    def test_contract(self):
        test_g1_cutoff_topk_contract()


if __name__ == "__main__":
    unittest.main()
