# utils/g2_periodic_edges.py
"""G2a periodic multi-image edge construction (G2a design, single-factor module).

Each cutoff neighbor ``(i <- j, T)`` is kept as an independent directed edge:

    r_{ijT} = (f_j - f_i + T) A,  keep iff ||r|| < R,

excluding only ``i == j, T == 0``. Cross-cell self images (``i == j, T != 0``)
are kept. Padding atoms are neither senders nor receivers.

Enumeration uses a conservative per-sample bound from the cell's minimum
singular value, never the fixed G1 ``[-2, 2]^3`` box and never silent top-k
pruning:

    K = ceil(R / s_min(A) + 0.5),  T_k in [-K, K].

The fractional difference is first centered to the nearest-cell image
``delta = f_j - f_i - round(f_j - f_i)`` so equivalent ``frac + integer``
inputs give identical edge sets.
"""

from __future__ import annotations

import itertools
import math

import torch


G2_R_CUT_DEFAULT = 5.5
G2_RBF_NUM = 64


def g2_quintic_cutoff(distances: torch.Tensor, r_cut: float = G2_R_CUT_DEFAULT) -> torch.Tensor:
    """Quintic smooth cutoff w(r): 1 at 0, 0 at/after r_cut, C2-continuous."""
    x = (distances / r_cut).clamp(0.0, 1.0)
    w = 1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5
    # float32 cancellation near x=1 can yield +-1e-7; weights stay in [0,1].
    return w.clamp(0.0, 1.0)


def g2_rbf_params(r_cut: float = G2_R_CUT_DEFAULT, num: int = G2_RBF_NUM,
                  device=None, dtype=None):
    """Return (centers, width) for the G2a radial RBF expansion."""
    centers = torch.linspace(0.01 * r_cut, 0.99 * r_cut, num,
                             device=device, dtype=dtype)
    width = centers[1] - centers[0]
    return centers, width


def g2_rbf_features(distances: torch.Tensor, r_cut: float = G2_R_CUT_DEFAULT,
                    centers: torch.Tensor | None = None,
                    width: torch.Tensor | float | None = None) -> torch.Tensor:
    """Gaussian RBF expansion of edge distances: [..., E] -> [..., E, 64]."""
    if centers is None or width is None:
        centers, width = g2_rbf_params(r_cut, G2_RBF_NUM,
                                       device=distances.device,
                                       dtype=distances.dtype)
    else:
        centers = centers.to(device=distances.device, dtype=distances.dtype)
        if torch.is_tensor(width):
            width = width.to(device=distances.device, dtype=distances.dtype)
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))


def _cell_s_min(cell: torch.Tensor) -> torch.Tensor:
    """Minimum singular value per sample for the enumeration bound."""
    s = torch.linalg.svdvals(cell)  # [B, 3]
    return s.amin(dim=-1)  # [B]


@torch.no_grad()
def build_g2_edges(pos: torch.Tensor, mask_atom: torch.Tensor,
                   r_cut: float = G2_R_CUT_DEFAULT, chunk_shifts: int = 64):
    """Build exact periodic multi-image edges for a batch.

    Args:
        pos: [B, Lp, 3] lattice + fractional rows (dataset 82-format).
        mask_atom: [B, L] bool, True = padding slot.
        r_cut: interaction cutoff in Angstrom (G2a fixed at 5.5).
        chunk_shifts: number of integer shifts evaluated per chunk.

    Returns:
        dict with ``batch`` [E] long, ``dst`` [E] long, ``src`` [E] long,
        ``shifts`` [E, 3] long, ``distances`` [E] (same dtype as ``pos``),
        ``k`` [B] long (per-sample enumeration half-width), ``L`` int.
    """
    from utils.relative_features import build_cell_from_lattice

    assert pos.dim() == 3 and pos.shape[-1] == 3
    assert mask_atom.dim() == 2
    assert r_cut > 0, f"G2a r_cut must be positive, got {r_cut}"
    cell, frac = build_cell_from_lattice(pos)  # [B,3,3], [B,L,3]
    B, L, _ = frac.shape
    assert mask_atom.shape == (B, L)
    device = frac.device
    dtype = frac.dtype

    s_min = _cell_s_min(cell)  # [B]
    if not torch.isfinite(s_min).all():
        bad = torch.where(~torch.isfinite(s_min))[0].tolist()
        raise ValueError(f"G2a: non-finite cell s_min at samples {bad}; refusing to enumerate")
    if (s_min <= 1e-6).any():
        bad = torch.where(s_min <= 1e-6)[0].tolist()
        raise ValueError(f"G2a: degenerate cell s_min<=1e-6 A at samples {bad}; refusing to enumerate")
    k = torch.ceil(r_cut / s_min + 0.5).to(torch.long)  # [B]

    batch_list, dst_list, src_list, shift_list, dist_list = [], [], [], [], []
    cell_b_list = cell  # keep reference for einsum per sample

    for b in range(B):
        valid = ~mask_atom[b]  # [L]
        valid_idx = torch.where(valid)[0]  # [Nv] global atom ids
        nv = int(valid_idx.numel())
        if nv == 0:
            continue
        frac_v = frac[b][valid]  # [Nv, 3]
        # delta[i, j] = f_j - f_i, centered to the nearest-cell image for a
        # stable enumeration. Stored shifts are the true integers satisfying
        # r = (f_j - f_i + T) A, i.e. T_true = T_centered - round(diff).
        diff = frac_v.unsqueeze(0) - frac_v.unsqueeze(1)  # [Nv, Nv, 3]
        rnd = torch.round(diff)
        delta = diff - rnd
        rnd_int = rnd.to(torch.long)  # [Nv, Nv, 3]
        kb = int(k[b].item())
        rng = range(-kb, kb + 1)
        shifts_all = torch.tensor(list(itertools.product(rng, rng, rng)),
                                  device=device, dtype=dtype)  # [S, 3]
        shifts_int = shifts_all.to(torch.long)
        n_shift = shifts_all.shape[0]
        zero_mask = (shifts_all == 0).all(dim=-1)  # [S]
        has_zero = bool(zero_mask.any())
        zero_pos = int(torch.where(zero_mask)[0][0].item()) if has_zero else -1
        cell_b = cell_b_list[b]  # [3, 3]

        for s in range(0, n_shift, chunk_shifts):
            T = shifts_all[s:s + chunk_shifts]  # [C, 3]
            Tint = shifts_int[s:s + chunk_shifts]  # [C, 3]
            C = T.shape[0]
            # [Nv, Nv, C, 3] candidate fractional displacements.
            cand = delta.unsqueeze(2) + T.view(1, 1, -1, 3)
            cart = torch.einsum("ijcd,dk->ijck", cand, cell_b)  # [Nv,Nv,C,3]
            d = cart.norm(dim=-1)  # [Nv, Nv, C]
            keep = d < r_cut  # strict cutoff
            if has_zero:
                # Exclude only i == j, T == 0 within this chunk.
                c_local = zero_pos - s
                if 0 <= c_local < C:
                    diag = torch.arange(nv, device=device)
                    keep[diag, diag, c_local] = False
            if not bool(keep.any()):
                continue
            ii, jj, cc = torch.where(keep)
            batch_list.append(torch.full((ii.numel(),), b, device=device, dtype=torch.long))
            dst_list.append(valid_idx[ii])
            src_list.append(valid_idx[jj])
            # True periodic image label for r = (f_j - f_i + T) A.
            shift_list.append(Tint[cc] - rnd_int[ii, jj, :])
            dist_list.append(d[ii, jj, cc])

    if batch_list:
        out = {
            "batch": torch.cat(batch_list),
            "dst": torch.cat(dst_list),
            "src": torch.cat(src_list),
            "shifts": torch.cat(shift_list),
            "distances": torch.cat(dist_list),
            "k": k,
            "L": L,
        }
    else:
        out = {
            "batch": torch.zeros((0,), device=device, dtype=torch.long),
            "dst": torch.zeros((0,), device=device, dtype=torch.long),
            "src": torch.zeros((0,), device=device, dtype=torch.long),
            "shifts": torch.zeros((0, 3), device=device, dtype=torch.long),
            "distances": torch.zeros((0,), device=device, dtype=dtype),
            "k": k,
            "L": L,
        }
    return out


@torch.no_grad()
def g2_indegrees(edge_batch: torch.Tensor, edge_dst: torch.Tensor,
                 B: int, L: int) -> torch.Tensor:
    """Per-receiver true indegree [B, L] (padding rows stay zero)."""
    deg = torch.zeros((B, L), device=edge_dst.device, dtype=torch.long)
    if edge_batch.numel():
        lin = edge_batch * L + edge_dst
        cnt = torch.bincount(lin, minlength=B * L).reshape(B, L)
        deg = cnt
    return deg
