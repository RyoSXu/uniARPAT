# utils/g1_graph.py
"""Build sparse, periodic atom graphs for the optional graph encoder.

The graph enumerates integer cell shifts and chooses the shortest Cartesian
image, which is necessary for skewed cells where fractional wrapping is not
the shortest physical displacement. It returns pair distances, directions, an
attention adjacency mask, and smooth cutoff weights. Padded atoms are excluded.
"""
import itertools

import torch

from utils.relative_features import build_cell_from_lattice


def quintic_cutoff(distances: torch.Tensor, r_cut: float = 5.5) -> torch.Tensor:
    """Quintic smooth cutoff w(r): 1 at 0, 0 at/after r_cut, C2-continuous."""
    x = (distances / r_cut).clamp(0.0, 1.0)
    w = 1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5
    # float32 cancellation near x=1 can yield +-1e-7; weights stay in [0,1].
    return w.clamp(0.0, 1.0)


@torch.no_grad()
def build_g1_graph(pos: torch.Tensor, mask_atom: torch.Tensor,
                   r_cut: float = 5.5, max_neighbors: int = 48,
                   t_range: int = 2):
    """Build the G1 sparse periodic graph for a batch.

    Args:
        pos: [B, Lp, 3] lattice + fractional rows (dataset 82-format).
        mask_atom: [B, L] bool, True = padding slot.
        r_cut: interaction cutoff in Å.
        max_neighbors: per-atom neighbor cap, excluding self.
        t_range: half-width of the integer-shift enumeration.

    Returns:
        (distances, unit_dirs, adj, smooth) as documented above.
    """
    cell, frac = build_cell_from_lattice(pos)  # [B,3,3], [B,L,3]
    B, L, _ = frac.shape
    device, dtype = frac.device, frac.dtype
    rng = range(-t_range, t_range + 1)
    shifts = torch.tensor(list(itertools.product(rng, rng, rng)),
                          device=device, dtype=dtype)  # [S,3], S=125
    n_shift = shifts.shape[0]

    diff0 = frac.unsqueeze(2) - frac.unsqueeze(1)  # [B,L,L,3]
    best_d = torch.full((B, L, L), float("inf"), device=device, dtype=dtype)
    best_v = torch.zeros((B, L, L, 3), device=device, dtype=dtype)

    # Chunked enumeration: 5 x 25-shift broadcasts (~60MB transient at
    # B=32/L=80) instead of 125 serial einsums or one 125-wide tensor.
    chunk = 25
    for s in range(0, n_shift, chunk):
        T = shifts[s:s + chunk]  # [C,3]
        # [B,L,L,C,3] = diff0[...,None,:] + T; then @ cell^T.
        cand = diff0.unsqueeze(3) + T.view(1, 1, 1, -1, 3)
        cart = torch.einsum("bijcd,bdk->bijck", cand, cell)  # [B,L,L,C,3]
        d = cart.norm(dim=-1)  # [B,L,L,C]
        d_min, idx = d.min(dim=-1)  # [B,L,L]
        upd = d_min < best_d
        best_d = torch.where(upd, d_min, best_d)
        best_v = torch.where(
            upd.unsqueeze(-1),
            cart.gather(3, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 3)).squeeze(3),
            best_v,
        )

    distances = best_d
    unit_dirs = best_v / (distances.unsqueeze(-1) + 1e-8)
    # Self pairs: exact zero direction (avoid 0/eps noise).
    eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    unit_dirs = torch.where(eye.unsqueeze(-1), torch.zeros_like(unit_dirs), unit_dirs)
    distances = torch.where(eye, torch.zeros_like(distances), distances)

    valid = ~mask_atom  # [B,L] True = real atom
    valid_j = valid.unsqueeze(1).expand(-1, L, -1)  # [B,L,L] key validity
    # Strict <: at exactly r_cut the quintic weight is 0 == blocked, so the
    # continuous choice is blocked (allowed always implies smooth > 0).
    adj = (distances < r_cut) & valid_j
    # Self always allowed (keeps rows non-empty; identity path).
    adj = adj | eye
    # Padded query rows: geometry is garbage (frac=0 slots) -> attend to all
    # valid keys instead; outputs are discarded downstream via mask pooling.
    pad_row = mask_atom.unsqueeze(2).expand(-1, -1, L)  # [B,L,L] query padded
    adj = torch.where(pad_row, valid_j, adj)
    # Degenerate all-padded batch guard (never happens; keeps softmax safe).
    empty = ~adj.any(dim=-1, keepdim=True).expand(-1, -1, L)
    adj = torch.where(empty, eye, adj)

    # Top-k cap: keep self + nearest max_neighbors allowed (degree<=49).
    k = min(max_neighbors + 1, L)
    if k < L:
        masked_d = torch.where(adj, distances, torch.full_like(distances, float("inf")))
        _, topk_idx = masked_d.topk(k, dim=-1, largest=False)  # [B,L,k]
        # topk may include +inf slots when degree < k -> keep finite picks only.
        picked_d = masked_d.gather(-1, topk_idx)
        keep = torch.zeros_like(adj)
        keep.scatter_(-1, topk_idx, picked_d < float("inf"))
        keep = keep | (eye & adj)  # self unconditionally kept
        adj = keep

    smooth = torch.where(adj, quintic_cutoff(distances, r_cut),
                         torch.zeros_like(distances))
    smooth = torch.where(eye, torch.where(adj, torch.ones_like(smooth), smooth), smooth)
    return distances, unit_dirs, adj, smooth
