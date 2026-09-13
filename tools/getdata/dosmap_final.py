#!/usr/bin/env python3
"""归属终判: vol规则落盘 + 模糊项StructureMatcher复核 + 池子结算.

输入: mp_increment_dosmap.json (unique/multi/none), mp_increment_raw.jsonl,
      Delta eDOS分区(按identifier取结构).
输出: mp_increment_dosmap.json (终判 mode: es_task/delta_unique/delta_vol/matched_multi/verified_vol5/unresolved),
      并池结算打印.
规则: es_task(REST直给, 5195) > delta_unique/vol<0.2% > matcher复核 > unresolved(241).
"""
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path("/root/home/newstudy/getdata/raw")


def main():
    res = json.load(open(RAW / "mp_increment_dosmap.json"))
    # 续跑兼容: 上轮Lattice(dict)bug把待复核写成unresolved, 按残留字段恢复
    for m, d in res.items():
        if d["mode"] == "unresolved" and "candidates" in d:
            d["mode"] = "multi"
        elif d["mode"] == "unresolved" and "gap" in d:
            d["mode"] = "unique_vol5"
    info = {}
    for line in open(RAW / "mp_increment_raw.jsonl"):
        r = json.loads(line)
        if r["mp_summary"]:
            info[r["mpid"]] = r["mp_summary"]["structure"]
    # (a) multi best<0.2% 接受
    tgts = {m: info[m]["lattice"]["volume"] for m, d in res.items()
            if d["mode"] == "multi"}
    for m, d in res.items():
        if d["mode"] == "multi":
            V = tgts[m]
            cands = sorted(d["candidates"], key=lambda c: abs(c[2] - V) / V)
            if abs(cands[0][2] - V) / V < 0.002:
                d["mode"] = "delta_vol"
                d["identifier"] = cands[0][0]
                d["run_type"] = cands[0][1]
    # (b) 待复核: multi剩余 + unique_vol5
    todo = [m for m, d in res.items() if d["mode"] in ("multi", "unique_vol5")]
    print(f"[final] verify queue={len(todo)}", flush=True)
    want = set()
    for m in todo:
        d = res[m]
        ids = ([c[0] for c in d["candidates"][:6]] if d["mode"] == "multi"
               else [d["identifier"]])
        want.update(ids)
    # Delta按identifier取结构
    rows = {}
    import glob
    import urllib.parse
    for f in sorted(glob.glob(str(RAW / "delta_mp/core/electronic-structure/total-dos/run_type=*/*.parquet"))):
        rt = urllib.parse.unquote(f.split("run_type=")[1].split("/")[0])
        t = pq.read_table(f, columns=["identifier", "structure"],
                          filters=[("identifier", "in", sorted(want))])
        for row in t.to_pylist():
            rows[row["identifier"]] = (row["structure"], rt)
        want -= set(rows)
        if not want:
            break
    print(f"[final] fetched structures={len(rows)} missing={len(want)}", flush=True)
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    ok = 0
    for m in todo:
        d = res[m]
        try:
            uc = info[m]
            # MP summary结构lattice是dict{matrix,...}(注意PhononDB侧是裸matrix, 别混)
            lat = Lattice(uc["lattice"]["matrix"] if isinstance(uc["lattice"], dict)
                          else uc["lattice"])
            ref = Structure(lat, [s["label"] for s in uc["sites"]],
                            [s["abc"] for s in uc["sites"]])
        except Exception:
            d["mode"] = "unresolved"
            continue
        cands = ([c[0] for c in d["candidates"][:6]] if d["mode"] == "multi"
                 else [d["identifier"]])
        hit = None
        for cid in cands:
            if cid not in rows:
                continue
            try:
                ds = rows[cid][0]
                cand = Structure(Lattice(ds["lattice"]["matrix"]),
                                 [s["label"] for s in ds["sites"]],
                                 [s["abc"] for s in ds["sites"]])
                if sm.fit(ref, cand):
                    hit = cid
                    break
            except Exception:
                continue
        if hit:
            d["mode"] = "matcher_ok"
            d["identifier"] = hit
            d["run_type"] = rows[hit][1]
            ok += 1
        else:
            d["mode"] = "unresolved"
    print(f"[final] matcher_ok={ok}/{len(todo)}", flush=True)
    # unresolved: none(241, Delta无) + 复核失败
    for m, d in res.items():
        if d["mode"] in ("none", "multi", "unique_vol5"):
            d["mode"] = "unresolved"
    json.dump(res, open(RAW / "mp_increment_dosmap.json", "w"), indent=1)
    print(Counter(v["mode"] for v in res.values()), flush=True)

    # ---- 池子结算 ----
    es_ok = set()
    for line in open(RAW / "mp_increment_raw.jsonl"):
        r = json.loads(line)
        if r["dos_task_id"] is not None:
            es_ok.add(r["mpid"])
    delta_ok = set(m for m, d in res.items()
                   if d["mode"] in ("unique", "delta_vol", "matcher_ok"))
    eff = set(json.load(open(RAW / "census_effective_ids.json")))
    dual = set(json.load(open(RAW / "dual_spectra_ids.json")))
    ph = set(json.loads(l)["canonical_mpid"]
             for l in open(RAW / "phonondb_a3c_map.jsonl"))
    jv = set(json.load(open(RAW / "jarvis_gap_hit_canon.json")))
    have_edos = es_ok | delta_ok
    for name, s in [("ph_inc", (ph & eff) - dual), ("jv_inc", jv)]:
        s = sorted(s)
        hit = sum(1 for m in s if m in have_edos)
        print(f"{name}: n={len(s)} MP-eDOS可得={hit} 缺={len(s) - hit}", flush=True)
    union_dual = set(m for m in ((ph & eff) - dual) | jv if m in have_edos)
    print(f"UNION new duals(MP-eDOS侧落实)={len(union_dual)} "
          f"true_double={18644 + len(union_dual)}", flush=True)


if __name__ == "__main__":
    main()
