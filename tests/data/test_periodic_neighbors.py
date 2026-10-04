"""Current periodic neighbor enumeration and physical geometry contracts."""

import itertools
import math
import unittest
import torch
from utils.periodic_neighbors import (
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

class TestPeriodicNeighbors(unittest.TestCase):
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
