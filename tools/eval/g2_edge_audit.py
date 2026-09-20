#!/usr/bin/env python3
"""G2a Q1 edge-count audit (no labels, CPU).

Traverses the existing Q1 train/valid/test splits using only CIF-derived
inputs (elements + positions) and the frozen G2a builder ``build_g2_edges``
at the fixed ``R = 5.5 A``. Writes per-structure records to
``results/g2_edge_audit_q1.csv`` and prints per-split quantiles for
per-structure edge counts, per-atom indegrees and enumeration half-width K,
plus max-sample identifiers.

Any non-finite or degenerate cell (s_min <= 1e-6 A) aborts with an explicit
error instead of silently skipping records.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.g2_periodic_edges import build_g2_edges


def _load_split(data_dir, split):
    el = np.load(os.path.join(data_dir, split, f"elements_{split}.npy"))
    po = np.load(os.path.join(data_dir, split, f"positions_{split}.npy"))
    assert el.shape[0] == po.shape[0], (el.shape, po.shape)
    assert el.shape[1] == 82, el.shape
    pos = po.reshape(-1, 82, 3)
    return el, pos


def _quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(x.size),
        "p50": float(np.quantile(x, 0.50)) if x.size else float("nan"),
        "p95": float(np.quantile(x, 0.95)) if x.size else float("nan"),
        "p99": float(np.quantile(x, 0.99)) if x.size else float("nan"),
        "max": float(x.max()) if x.size else float("nan"),
        "mean": float(x.mean()) if x.size else float("nan"),
    }


def audit_split(data_dir, split, batch_size, r_cut):
    el, pos = _load_split(data_dir, split)
    N = el.shape[0]
    rows = []
    all_indegrees = []
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        el_b = torch.from_numpy(el[start:end]).long()
        pos_b = torch.from_numpy(pos[start:end]).float()
        mask_b = el_b[:, 2:].eq(0)  # [b, 80] True = padding
        try:
            out = build_g2_edges(pos_b, mask_b, r_cut=r_cut)
        except ValueError as e:
            print(f"[audit] ABORT split={split} rows [{start},{end}): {e}",
                  flush=True)
            raise
        eb = out["batch"].numpy()
        ed = out["dst"].numpy()
        K = out["k"].numpy()
        L = int(out["L"])
        for j in range(end - start):
            gid = start + j
            n_atom = int((~mask_b[j]).sum().item())
            m = (eb == j)
            n_edges = int(m.sum())
            if n_atom > 0 and n_edges > 0:
                cnt = np.bincount(ed[m], minlength=L)[:L]
                # restrict to valid atoms for max/mean (padding stays 0)
                valid = (~mask_b[j].numpy())
                cnt_v = cnt[valid]
                max_in = int(cnt_v.max())
                mean_in = float(cnt_v.mean())
                all_indegrees.extend(cnt_v.tolist())
            elif n_atom > 0:
                max_in, mean_in = 0, 0.0
                all_indegrees.extend([0] * n_atom)
            else:
                max_in, mean_in = 0, 0.0
            rows.append((split, gid, n_atom, int(K[j]), n_edges, mean_in, max_in))
        if (start // batch_size + 1) % 20 == 0 or end == N:
            print(f"[audit] {split} {end}/{N}", flush=True)
    return rows, all_indegrees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data/train4ARPAT")
    ap.add_argument("--out", default="./results/g2_edge_audit_q1.csv")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--r_cut", type=float, default=5.5)
    args = ap.parse_args()
    assert args.r_cut == 5.5, "G2a audit uses the fixed R=5.5 A"

    import csv
    all_rows = []
    for split in ("train", "valid", "test"):
        rows, indegrees = audit_split(args.data_dir, split, args.batch, args.r_cut)
        all_rows.extend(rows)
        ec = np.array([r[4] for r in rows], dtype=np.float64)
        Ks = np.array([r[3] for r in rows], dtype=np.float64)
        ideg = np.array(indegrees, dtype=np.float64)
        qe, qk, qi = _quantiles(ec), _quantiles(Ks), _quantiles(ideg)
        amax = int(ec.argmax()) if len(ec) else -1
        kmax = int(Ks.argmax()) if len(Ks) else -1
        print(f"--- {split} (n={len(rows)}) ---")
        print(f"  per-structure edges: {qe} argmax_row={amax} "
              f"(n_atom={rows[amax][2] if amax>=0 else '-'}, "
              f"edges={rows[amax][4] if amax>=0 else '-'})")
        print(f"  per-atom indegree (pooled {len(indegrees)} atoms): {qi}")
        print(f"  K: {qk} argmax_row={kmax}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "idx", "n_atom", "K", "n_edges",
                    "mean_indegree", "max_indegree"])
        w.writerows(all_rows)
    print(f"[audit] wrote {len(all_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
