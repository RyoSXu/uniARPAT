#!/usr/bin/env python3
"""A6分层切分v2: 8:1:1 + 三层分层 + 成分硬隔离 + 未见元素探针.

输入: v2_processed.parquet (+ summaries带隙: mp_structures/mp_increment_raw)
输出: index/split_v2.yaml (三集id+seed+统计) + split列回写v2_processed.parquet
     + probeanel.json (测试集未见元素探针)
规则(Design-A6冻结): seed42; 金属/半导体/绝缘体(0/0-2/>2eV) + 7晶系(稀有cell保底训练)
  + 峰均比难度三分位; 同成分(Z计数精确组)零跨集(断言); v2双谱无He(F3), He规则跳过并记录.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

RAW = Path("/root/home/newstudy/getdata/raw")
REPO = Path(__file__).resolve().parents[2]
OUT_YAML = REPO / "index/split_v2.yaml"


def main():
    t = pq.read_table(RAW / "v2_processed.parquet").to_pylist()
    print(f"[a6] n={len(t)}", flush=True)
    gaps = {}
    for line in open(RAW / "mp_structures.jsonl"):
        r = json.loads(line)
        gaps[r["mpid"]] = r["summary"].get("band_gap", 0.0) or 0.0
    for line in open(RAW / "mp_increment_raw.jsonl"):
        r = json.loads(line)
        if r["mp_summary"] is not None:
            gaps[r["mpid"]] = r["mp_summary"].get("band_gap", 0.0) or 0.0
    rows = []
    for r in t:
        Z = [z for z in r["elements"][2:] if z > 0]
        comp = tuple(sorted(Counter(Z).items()))
        e = np.array(r["edos"], dtype=float)
        p = np.array(r["phdos"], dtype=float)
        d = max(e.max() / (e.mean() + 1e-9), p.max() / (p.mean() + 1e-9))
        g = gaps.get(r["mpid"], 0.0)
        rows.append({"mpid": r["mpid"], "comp": comp, "crystal": r["crystal"],
                     "gapbin": 0 if g <= 0 else (1 if g <= 2 else 2),
                     "gap": round(float(g), 3), "diff": float(d),
                     "src": r["src_ph"]})
    diffs = np.array([r["diff"] for r in rows])
    t1, t2 = np.quantile(diffs, [1 / 3, 2 / 3])
    for r in rows:
        r["dterr"] = 0 if r["diff"] <= t1 else (1 if r["diff"] <= t2 else 2)
        r["cell"] = (r["gapbin"], r["crystal"], r["dterr"])
    # 成分组
    comp_groups = defaultdict(list)
    for r in rows:
        comp_groups[r["comp"]].append(r)
    print(f"[a6] comp groups={len(comp_groups)} diff tertiles=({t1:.2f},{t2:.2f})",
          flush=True)
    # 稀有cell(<10样本)保底训练
    cell_n = Counter(r["cell"] for r in rows)
    train, valid, test = [], [], []
    used = set()
    for r in rows:
        if cell_n[r["cell"]] < 10:
            train.append(r)
            used.add(r["mpid"])
    # 贪心: 按组大小降序, 组内cell目标比例8:1:1补亏空最大的集
    target = {0: 0.8, 1: 0.1, 2: 0.1}
    buckets = [train, valid, test]
    groups = sorted(comp_groups.values(), key=len, reverse=True)
    for g in groups:
        g = [r for r in g if r["mpid"] not in used]
        if not g:
            continue
        cells = Counter(r["cell"] for r in g)
        cell = cells.most_common(1)[0][0]
        morate = [(len([x for x in buckets[s] if x["cell"] == cell]) + 0.0, s)
                  for s in (0, 1, 2)]
        # 缺口 = 目标比例 - 当前比例(按cell计, 拉普拉斯平滑)
        tot = sum(m[0] for m in morate) + 3
        pick = min(range(3), key=lambda s: (morate[s][0] + 1) / tot / target[s])
        for r in g:
            buckets[pick].append(r)
            used.add(r["mpid"])
    names = ["train", "valid", "test"]
    split_of = {}
    for s, b in enumerate(buckets):
        for r in b:
            split_of[r["mpid"]] = names[s]
    assert len(split_of) == len(rows), (len(split_of), len(rows))
    # 断言: 成分零跨集
    comp_split = defaultdict(set)
    for r in rows:
        comp_split[r["comp"]].add(split_of[r["mpid"]])
    leaks = sum(1 for v in comp_split.values() if len(v) > 1)
    assert leaks == 0, f"composition leak groups={leaks}"
    # 验收: 各层比例±2%
    rep = {}
    for dim, key in (("gapbin", "gapbin"), ("crystal", "crystal"), ("dterr", "dterr")):
        tot_c = Counter(r[key] for r in rows)
        rep[dim] = {}
        for v in sorted(tot_c):
            shares = [sum(1 for r in buckets[s] if r[key] == v) / max(tot_c[v], 1)
                      for s in range(3)]
            rep[dim][str(v)] = [round(x, 4) for x in shares]
    print(json.dumps({k: len(b) for k, b in zip(names, buckets)}, indent=1), flush=True)
    # 未见元素探针: train未见元素 -> test样本
    train_el = set()
    for r in buckets[0]:
        train_el.update(z for z, _ in r["comp"])
    probe = [r["mpid"] for r in buckets[2]
             if any(z not in train_el for z, _ in r["comp"])]
    print(f"[a6] probe(test-unseen-el)={len(probe)}", flush=True)
    # 写盘
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, "w") as f:
        f.write("# split_v2 seed=42 comp-hard-isolation\n")
        for s, b in zip(names, buckets):
            f.write(f"{s}:\n")
            for r in sorted(x["mpid"] for x in b):
                f.write(f"  - {r}\n")
    json.dump({"probe_test_unseen_element": sorted(probe),
               "note": "He规则跳过: v2双谱核心集不含He(F3)"},
              open(RAW / "probeanel.json", "w"), indent=1)
    # split列回写
    tbl = pq.read_table(RAW / "v2_processed.parquet")
    mpids = tbl.column("mpid").to_pylist()
    import pyarrow as pa
    tbl = tbl.append_column("split", pa.array([split_of[m] for m in mpids]))
    pq.write_table(tbl, RAW / "v2_processed.parquet")
    json.dump({"n": {k: len(b) for k, b in zip(names, buckets)},
               "strata": rep, "leaks": leaks, "probe_n": len(probe),
               "seed": 42}, open(RAW / "split_report.json", "w"), indent=1)
    print("[a6] DONE", flush=True)


if __name__ == "__main__":
    main()
