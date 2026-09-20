#!/usr/bin/env python3
"""Q1 同约化化学式的结构信号诊断（只读）。

本工具不训练、不改缓存。它在 Q1 test 划分中：

1. 从原子序数数组导出约化化学式，寻找同组成样本；
2. 仅保留同公式且缓存结构表示不同的组，计算其真实 eDOS/phDOS
   总和归一化谱形的组内 total-variation (TV) 与归一化 W1 距离；
3. 以等数量、不同约化化学式的随机样本对作描述性参照；
4. 将 B7 `_e9ctl` 的逐样本 oracle R²/失败率与组内谱形差异对齐。

这不是 composition-only 预测基线：Q1 使用 composition-hard-isolation 划分，
测试集化学式本来不在训练集出现。结果只回答“同组成样本仍有多少谱形变化，
以及 B7 在这些样本上的误差是否与变化程度有关”。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from functools import reduce
from math import gcd

import numpy as np
import pandas as pd


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _symbol_table() -> dict[int, str]:
    table = pd.read_csv(os.path.join(REPO, "utils", "periodic_table_v2.csv"))
    return {int(row.AtomicNumber): str(row.Symbol) for row in table.itertuples(index=False)}


def reduced_formula(row: np.ndarray, symbols: dict[int, str]) -> str:
    counts = Counter(int(z) for z in row[2:] if int(z) > 0)
    if not counts:
        return "<empty>"
    divisor = reduce(gcd, counts.values())
    return "".join(
        f"{symbols[z]}{'' if n // divisor == 1 else n // divisor}"
        for z, n in sorted(counts.items())
    )


def structure_signature(elements: np.ndarray, positions: np.ndarray) -> str:
    """Hash the cached structural representation after stable float rounding.

    It intentionally does not assert crystallographic inequivalence: equivalent
    cells written with a different origin or atom order may remain separate.
    The diagnostic calls this a *cache-structure variant*, not a polymorph
    certificate.
    """
    h = hashlib.sha1()
    h.update(np.asarray(elements, dtype=np.int64).tobytes())
    h.update(np.round(np.asarray(positions, dtype=np.float64), 6).tobytes())
    return h.hexdigest()


def normalized_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sums = x.sum(axis=1, keepdims=True)
    if not np.isfinite(x).all() or (sums <= 0).any():
        raise ValueError("non-finite or non-positive spectrum sum in Q1 test cache")
    return x / sums


def pair_distances(x: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Return shape TV and bin-normalized W1 for sample-index pairs."""
    left = x[[i for i, _ in pairs]]
    right = x[[j for _, j in pairs]]
    tv = 0.5 * np.abs(left - right).sum(axis=1)
    w1 = np.abs(np.cumsum(left - right, axis=1)).sum(axis=1) / (x.shape[1] - 1)
    return tv, w1


def all_pairs(indices: list[int]) -> list[tuple[int, int]]:
    return [(indices[a], indices[b]) for a in range(len(indices)) for b in range(a + 1, len(indices))]


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _summary(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=os.path.join(REPO, "data", "train4ARPAT"))
    ap.add_argument("--tag", default="_e9ctl", help="B7 result tag with samples_m1_<tag>_test.csv")
    ap.add_argument("--seed", type=int, default=42, help="between-formula reference-pair seed")
    ap.add_argument("--out_prefix", default=os.path.join(REPO, "results", "structure_signal_q1_test"))
    args = ap.parse_args()

    test_dir = os.path.join(args.data_dir, "test")
    elements = np.load(os.path.join(test_dir, "elements_test.npy"), mmap_mode="r")
    positions = np.load(os.path.join(test_dir, "positions_test.npy"), mmap_mode="r").reshape(-1, 82, 3)
    edos = normalized_rows(np.load(os.path.join(test_dir, "edos_tgtdos_test.npy"), mmap_mode="r"))
    phdos = normalized_rows(np.load(os.path.join(test_dir, "phdos_tgtdos_test.npy"), mmap_mode="r"))
    mpids = np.load(os.path.join(test_dir, "test_index.npy"))
    if not (len(elements) == len(positions) == len(edos) == len(phdos) == len(mpids)):
        raise ValueError("Q1 test arrays do not have the same length")

    tag = args.tag.lstrip("_")
    sample_path = os.path.join(REPO, "results", f"samples_m1_{tag}_test.csv")
    samples = pd.read_csv(sample_path)
    if len(samples) != len(elements):
        raise ValueError(f"B7 sample rows {len(samples)} != Q1 test rows {len(elements)}")
    required = {"r2_edos", "r2_phdos"}
    if not required.issubset(samples.columns):
        raise ValueError(f"{sample_path} missing {sorted(required - set(samples.columns))}")

    symbols = _symbol_table()
    formulas = [reduced_formula(row, symbols) for row in elements]
    signatures = [structure_signature(elements[i], positions[i]) for i in range(len(elements))]
    by_formula: dict[str, list[int]] = defaultdict(list)
    for i, formula in enumerate(formulas):
        by_formula[formula].append(i)

    group_rows: list[dict[str, object]] = []
    within_pairs: list[tuple[int, int]] = []
    diagnostic_indices: list[int] = []
    for formula, ids in sorted(by_formula.items()):
        variants = len({signatures[i] for i in ids})
        if len(ids) < 2 or variants < 2:
            continue
        pairs = all_pairs(ids)
        etv, ew1 = pair_distances(edos, pairs)
        ptv, pw1 = pair_distances(phdos, pairs)
        within_pairs.extend(pairs)
        diagnostic_indices.extend(ids)
        r2e = samples.iloc[ids]["r2_edos"].to_numpy(dtype=float)
        r2p = samples.iloc[ids]["r2_phdos"].to_numpy(dtype=float)
        group_rows.append({
            "formula": formula,
            "n_samples": len(ids),
            "cache_structure_variants": variants,
            "pair_count": len(pairs),
            "mpids": ";".join(map(str, mpids[ids])),
            "edos_tv_mean": float(etv.mean()),
            "edos_tv_median": float(np.median(etv)),
            "edos_w1_mean": float(ew1.mean()),
            "phdos_tv_mean": float(ptv.mean()),
            "phdos_tv_median": float(np.median(ptv)),
            "phdos_w1_mean": float(pw1.mean()),
            "b7_edos_r2_median": float(np.median(r2e)),
            "b7_edos_fail_pct": float((r2e < 0).mean() * 100),
            "b7_phdos_r2_median": float(np.median(r2p)),
            "b7_phdos_fail_pct": float((r2p < 0).mean() * 100),
        })

    if not group_rows:
        raise RuntimeError("no same-formula, distinct-cache-structure test groups found")
    diagnostic_indices = sorted(set(diagnostic_indices))

    # Match the number of within-formula pairs with random pairs from different
    # formulas.  This is a descriptive reference, not a train/test baseline.
    rng = np.random.default_rng(args.seed)
    between_pairs: list[tuple[int, int]] = []
    while len(between_pairs) < len(within_pairs):
        i, j = rng.integers(0, len(formulas), size=2).tolist()
        if i != j and formulas[i] != formulas[j]:
            between_pairs.append((i, j))
    etv_in, ew1_in = pair_distances(edos, within_pairs)
    ptv_in, pw1_in = pair_distances(phdos, within_pairs)
    etv_out, ew1_out = pair_distances(edos, between_pairs)
    ptv_out, pw1_out = pair_distances(phdos, between_pairs)

    groups = pd.DataFrame(group_rows).sort_values("phdos_tv_mean", ascending=False)
    groups.to_csv(f"{args.out_prefix}_groups.csv", index=False)

    group_spread_e = groups["edos_tv_mean"].to_list()
    group_spread_p = groups["phdos_tv_mean"].to_list()
    group_r2e = groups["b7_edos_r2_median"].to_list()
    group_r2p = groups["b7_phdos_r2_median"].to_list()
    all_r2e = samples["r2_edos"].to_numpy(dtype=float)
    all_r2p = samples["r2_phdos"].to_numpy(dtype=float)
    diag_r2e = samples.iloc[diagnostic_indices]["r2_edos"].to_numpy(dtype=float)
    diag_r2p = samples.iloc[diagnostic_indices]["r2_phdos"].to_numpy(dtype=float)

    summary = {
        "scope": "Q1 test; same reduced formula with distinct cached structure representations",
        "b7_tag": args.tag,
        "n_test": int(len(elements)),
        "n_reduced_formulas": int(len(by_formula)),
        "n_formula_groups_size_ge_2": int(sum(len(v) >= 2 for v in by_formula.values())),
        "n_structure_variant_groups": int(len(groups)),
        "n_samples_in_structure_variant_groups": int(len(diagnostic_indices)),
        "n_within_formula_pairs": int(len(within_pairs)),
        "within_formula_shape_distance": {
            "edos_tv": _summary(etv_in), "edos_w1": _summary(ew1_in),
            "phdos_tv": _summary(ptv_in), "phdos_w1": _summary(pw1_in),
        },
        "different_formula_reference": {
            "pair_count": int(len(between_pairs)),
            "edos_tv": _summary(etv_out), "edos_w1": _summary(ew1_out),
            "phdos_tv": _summary(ptv_out), "phdos_w1": _summary(pw1_out),
        },
        "b7_oracle_r2": {
            "all_test": {
                "edos_median": float(np.median(all_r2e)), "edos_fail_pct": float((all_r2e < 0).mean() * 100),
                "phdos_median": float(np.median(all_r2p)), "phdos_fail_pct": float((all_r2p < 0).mean() * 100),
            },
            "same_formula_structure_variant_samples": {
                "edos_median": float(np.median(diag_r2e)), "edos_fail_pct": float((diag_r2e < 0).mean() * 100),
                "phdos_median": float(np.median(diag_r2p)), "phdos_fail_pct": float((diag_r2p < 0).mean() * 100),
            },
            "group_spread_vs_group_median_r2_spearman": {
                "edos": spearman(group_spread_e, group_r2e),
                "phdos": spearman(group_spread_p, group_r2p),
            },
        },
        "interpretation_limits": [
            "组内谱差异证明同约化化学式并不唯一决定谱形；它不估计结构信息的全部因果贡献。",
            "不同化学式随机对仅作描述性参照，不能代替 composition-only 训练基线。",
            "缓存结构表示不同不等同于经空间群去重确认的不同多型。",
        ],
    }
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_prefix}_groups.csv and {args.out_prefix}_summary.json")


if __name__ == "__main__":
    main()
