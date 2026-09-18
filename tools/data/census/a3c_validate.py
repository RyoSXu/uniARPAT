#!/usr/bin/env python3
"""A3c分布验证(全本地): increment-4391 vs overlap-4660/885.

输入: phonondb_a3c_map.jsonl (+formula_pretty), phonondb_a3c_summary.json,
      phonondb_recomputed.jsonl (freq/dos/structure), dual/edos_absent (分组).
输出: phonondb_a3c_validation.json
  1. 元素直方图(formula_pretty正则解析, 不调API)
  2. maxfreq分布(中位/p90/p99, THz)
  3. 晶系分布(pymatgen SpacegroupAnalyzer, symprec=0.1, 每组抽样800, seed42)
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

RAW = Path("/root/home/newstudy/getdata/raw")

EL_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)?")


def parse_formula(f):
    out = Counter()
    for el, n in EL_RE.findall(f):
        if not el:
            continue
        out[el] += float(n) if n else 1.0
    return out


def main():
    summary = json.load(open(RAW / "phonondb_a3c_summary.json"))
    amap = {}
    with open(RAW / "phonondb_a3c_map.jsonl") as f:
        for line in f:
            r = json.loads(line)
            amap[r["canonical_mpid"]] = r
    dual = set(json.load(open(RAW / "dual_spectra_ids.json")))
    absent = set(json.load(open(RAW / "edos_absent.json")))
    true_double = dual - absent

    groups = {"increment_4391": [], "overlap_true_double": [], "overlap_absent": [],
              "outside_effective": []}
    outside = json.load(open(RAW / "phonondb_a3c_outside_effective.json"))
    outside_set = set(outside)
    for canon, r in amap.items():
        if canon in outside_set:
            groups["outside_effective"].append(r)
        elif canon in true_double:
            groups["overlap_true_double"].append(r)
        elif canon in absent:
            groups["overlap_absent"].append(r)
        elif canon not in dual:
            groups["increment_4391"].append(r)
    print({k: len(v) for k, v in groups.items()}, flush=True)

    # 1. 元素直方图
    el_stat = {}
    for g, rows in groups.items():
        if g == "outside_effective":
            continue
        c = Counter()
        n_oxide = 0
        for r in rows:
            f = r.get("formula_pretty", "")
            els = parse_formula(f)
            for el in els:
                c[el] += 1
            if "O" in els:
                n_oxide += 1
        el_stat[g] = {"n": len(rows), "n_elements_covered": len(c),
                      "top15": c.most_common(15),
                      "oxide_frac": round(n_oxide / max(len(rows), 1), 4)}

    # 2+3. maxfreq + 晶系抽样
    rec = {}
    with open(RAW / "phonondb_recomputed.jsonl") as f:
        for line in f:
            d = json.loads(line)
            rec[d["id"]] = d
    serial_of = {r["canonical_mpid"]: r["serial"]
                 for rows in groups.values() for r in rows if "serial" in r}

    rng = random.Random(42)
    val = {}
    for g, rows in groups.items():
        if g == "outside_effective":
            continue
        sers = [r["serial"] for r in rows if "serial" in r and r["serial"] in rec]
        maxf = []
        for s in sers:
            fr = rec[s].get("freq_THz", [])
            if fr:
                maxf.append(max(fr))
        maxf.sort()
        def pct(q):
            return round(maxf[min(int(q * len(maxf)), len(maxf) - 1)], 2) if maxf else None
        # 晶系全量(pymatgen ~3ms/条, symprec=0.1按DataSpec)
        cryst = Counter()
        n_fail = 0
        from pymatgen.core import Lattice, Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        for s in sers:
            try:
                uc = rec[s]["structure"]["unitcell"]
                lat = Lattice(uc["lattice"])
                sp = [p["symbol"] for p in uc["points"]]
                co = [p["coordinates"] for p in uc["points"]]
                st = Structure(lat, sp, co)
                sg = SpacegroupAnalyzer(st, symprec=0.1)
                cryst[str(sg.get_crystal_system())] += 1
            except Exception:
                n_fail += 1
        val[g] = {"n_dos": len(maxf),
                  "maxfreq_THz": {"median": pct(0.5), "p90": pct(0.9), "p99": pct(0.99)},
                  "crystal_system": dict(cryst), "spg_fail": n_fail}

    out = {"groups_n": {k: len(v) for k, v in groups.items()},
           "elements": el_stat, "spectra_structure": val,
           "conclusion": ""}
    # 简单一致性判决: 氧化物占比差<0.15, maxfreq中位差<20%, 主晶系相同
    try:
        inc, ov = el_stat["increment_4391"], el_stat["overlap_true_double"]
        d_ox = abs(inc["oxide_frac"] - ov["oxide_frac"])
        m_inc = val["increment_4391"]["maxfreq_THz"]["median"]
        m_ov = val["overlap_true_double"]["maxfreq_THz"]["median"]
        d_f = abs(m_inc - m_ov) / m_ov if m_ov else 1
        top_inc = max(val["increment_4391"]["crystal_system"],
                      key=val["increment_4391"]["crystal_system"].get)
        top_ov = max(val["overlap_true_double"]["crystal_system"],
                     key=val["overlap_true_double"]["crystal_system"].get)
        ok = d_ox < 0.15 and d_f < 0.20
        out["conclusion"] = (f"oxide_frac差{d_ox:.3f}, maxfreq中位差{d_f:.3f}, "
                             f"主晶系{top_inc}vs{top_ov} -> {'通过' if ok else '需人工复核'}")
    except Exception as e:
        out["conclusion"] = f"判决失败: {e}"
    json.dump(out, open(RAW / "phonondb_a3c_validation.json", "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False)[:3000], flush=True)


if __name__ == "__main__":
    main()
