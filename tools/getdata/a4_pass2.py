#!/usr/bin/env python3
"""A4 pass2: winsorize + 终盘 + 验收.

输入: v2_intermediate.parquet
输出: v2_processed.parquet + stats_v2.json + outlier_attribution.csv + audit_roundtrip.json
规则(DataSpec§6): per-bin p99.9(仅覆盖bin) + 单bin尖峰判据(邻<0.5v)->clip到阈值;
  f电子真峰保留策略: 宽峰(连续>=3bin超阈)不clip, 只记归因表(训练降权是C1的事).
验收: CIF round-trip 1000(seed42, 物种+体积1e-6); 分布抽查(v1中位max对照).
"""
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

RAW = Path("/root/home/newstudy/getdata/raw")
PCTL = 99.9


def main():
    t = pq.read_table(RAW / "v2_intermediate.parquet").to_pylist()
    print(f"[p2] n={len(t)}", flush=True)
    E = np.array([r["edos"] for r in t], dtype=np.float64)
    P = np.array([r["phdos"] for r in t], dtype=np.float64)
    EM = np.array([r["edos_mask"] for r in t], dtype=bool)
    PM = np.array([r["phdos_mask"] for r in t], dtype=bool)
    # per-bin p99.9(仅覆盖bin; 全掩膜bin阈值=inf即不处理)
    eth = np.array([np.quantile(E[EM[:, j], j], PCTL / 100) if EM[:, j].sum() else np.inf
                    for j in range(E.shape[1])])
    pth = np.array([np.quantile(P[PM[:, j], j], PCTL / 100) if PM[:, j].sum() else np.inf
                    for j in range(P.shape[1])])
    attr = []
    Ew, Pw = E.copy(), P.copy()
    for arr, W, th, name, M in ((E, Ew, eth, "edos", EM), (P, Pw, pth, "phdos", PM)):
        over = (arr > th) & M
        print(f"[p2] {name} over-threshold cells={int(over.sum())}", flush=True)
        left = np.zeros_like(arr)
        left[:, 1:] = arr[:, :-1]
        right = np.zeros_like(arr)
        right[:, :-1] = arr[:, 1:]
        neigh = np.maximum(left, right)
        wide = np.zeros_like(arr, dtype=bool)
        wide[:, 1:-1] = over[:, 1:-1] & over[:, :-2] & over[:, 2:]
        wide[:, 0] = over[:, 0] & over[:, 1]
        wide[:, -1] = over[:, -1] & over[:, -2]
        clip = over & ~wide & (neigh < 0.5 * arr)
        keep_sh = over & ~wide & ~clip
        ii, jj = np.where(clip)
        for i, j in zip(ii.tolist(), jj.tolist()):
            attr.append((t[i]["mpid"], name, int(j), round(float(arr[i][j]), 3),
                         f"clip->{round(float(th[j]), 3)}"))
        W[clip] = np.broadcast_to(th, arr.shape)[clip]
        ii, jj = np.where(over & wide)
        for i, j in zip(ii.tolist(), jj.tolist()):
            attr.append((t[i]["mpid"], name, int(j), round(float(arr[i][j]), 3),
                         "keep-wide"))
        ii, jj = np.where(keep_sh)
        for i, j in zip(ii.tolist(), jj.tolist()):
            attr.append((t[i]["mpid"], name, int(j), round(float(arr[i][j]), 3),
                         "keep-shoulder"))
    n_clip = sum(1 for a in attr if a[4].startswith("clip"))
    print(f"[p2] over-threshold events={len(attr)} clipped={n_clip}", flush=True)
    with open(RAW / "v2_outlier_attribution.csv", "w") as f:
        f.write("mpid,spec,bin,value,decision\n")
        for a in attr:
            f.write(f"{a[0]},{a[1]},{a[2]},{a[3]},{a[4]}\n")
    # 写回终盘
    import pyarrow as pa
    Ew32, Pw32 = Ew.astype(np.float32), Pw.astype(np.float32)
    cols = pq.read_table(RAW / "v2_intermediate.parquet")
    cols = cols.set_column(cols.schema.get_field_index("edos"),
                           "edos", pa.array(Ew32.tolist(), type=pa.list_(pa.float32())))
    cols = cols.set_column(cols.schema.get_field_index("phdos"),
                           "phdos", pa.array(Pw32.tolist(), type=pa.list_(pa.float32())))
    pq.write_table(cols, RAW / "v2_processed.parquet")
    # 分布抽查 vs v1中位max(edos 19.51 / phdos 0.2346)
    stats = {"n": len(t),
             "edos_thr_p999_med": round(float(np.median(eth[np.isfinite(eth)])), 3),
             "phdos_thr_p999_med": round(float(np.median(pth[np.isfinite(pth)])), 5),
             "edos_max_med": round(float(np.median(Ew.max(axis=1))), 3),
             "phdos_max_med": round(float(np.median(Pw.max(axis=1))), 5),
             "clipped_events": n_clip, "wide_kept": sum(1 for a in attr if a[4] == "keep-wide")}
    # CIF round-trip 1000
    rng = np.random.RandomState(42)
    sample = rng.choice(len(t), min(1000, len(t)), replace=False)
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter, CifParser
    ok = vol_err = 0
    for q, k in enumerate(sample):
        r = t[k]
        try:
            lat = r["lattice"]
            L = Lattice.from_parameters(lat[0], lat[1], lat[2], lat[3], lat[4], lat[5])
            Z = [z for z in r["elements"][2:] if z > 0]
            pos = np.array(r["positions"], dtype=float).reshape(82, 3)[2:2 + len(Z)]
            s0 = Structure(L, Z, pos)
            cif = str(CifWriter(s0))
            s1 = CifParser.from_str(cif).parse_structures(primitive=False)[0]
            if (sorted([s.specie.Z for s in s1]) == sorted(Z)
                    and abs(s1.volume - s0.volume) / s0.volume < 1e-6):
                ok += 1
            else:
                vol_err += 1
        except Exception:
            vol_err += 1
        if (q + 1) % 250 == 0:
            print(f"[p2] roundtrip {q + 1}/{len(sample)}", flush=True)
    stats["roundtrip_n"] = int(len(sample))
    stats["roundtrip_ok"] = int(ok)
    stats["roundtrip_fail"] = int(vol_err)
    json.dump(stats, open(RAW / "stats_v2.json", "w"), indent=1)
    json.dump({"edos_thr": [round(float(x), 4) if np.isfinite(x) else None for x in eth],
               "phdos_thr": [round(float(x), 6) if np.isfinite(x) else None for x in pth]},
              open(RAW / "v2_winsor_thresholds.json", "w"))
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
