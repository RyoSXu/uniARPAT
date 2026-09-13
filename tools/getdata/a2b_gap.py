#!/usr/bin/env python3
"""A2b三源gap分析: MP <-> JARVIS <-> PhononDB分歧矩阵(全本地).

输入:
  mp_phdos_raw.jsonl / mp_edos_raw.jsonl / mp_structures.jsonl (MP, canonical key)
  phonondb_recomputed.jsonl + phonondb_a3c_map.jsonl (PBEsol复算, serial->canon)
  jarvis_threesrc_raw.jsonl (JVASP eDOS+phDOS, canonical富化)
输出:
  a2b_pairs_phdos.csv / a2b_pairs_edos.csv (逐样本: r/位移/幅值比)
  a2b_summary.json (分臂median/p10/p90 + 权重建议 + 结构delta)
规则:
  - 单位: THz*33.35641 -> cm-1; JARVIS cm-1直用(DataSpec单位铁律).
  - 禁外推: 公共网格取三方交集, 网格外记mask不补.
  - 声子公共分辨率2cm-1; eDOS以MP Fermi对齐网格为锚, JARVIS滑移±15eV配准.
  - 形状相关用面积归一后Pearson r; 另记面积比与峰位移(lag).
  - MP声子优先pheasy, 无则dfpt(记method列).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path("/root/home/newstudy/getdata/raw")
THZ2CM = 33.35641


def fparse(s):
    q = (chr(39), chr(34))
    out = []
    for x in s.split(","):
        x = x.strip().strip(q[0]).strip(q[1])
        if x:
            try:
                out.append(float(x))
            except ValueError:
                pass
    return np.array(out, dtype=float)


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def interp_common(grids_vals, lo, hi, step):
    """grids_vals: [(x, y)] -> 公共等距网格上的插值矩阵(交集内, 外记nan)."""
    xs = np.arange(lo, hi + 1e-9, step)
    M = np.full((len(grids_vals), len(xs)), np.nan)
    for i, (x, y) in enumerate(grids_vals):
        m = (xs >= x.min()) & (xs <= x.max())
        if m.sum() > 3:
            M[i, m] = np.interp(xs[m], x, y)
    keep = ~np.isnan(M).any(axis=0)
    return xs[keep], M[:, keep]


def agg(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)])
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median": round(float(np.median(v)), 4),
            "p10": round(float(np.quantile(v, 0.1)), 4),
            "p90": round(float(np.quantile(v, 0.9)), 4)}


def main():
    # ---- load ----
    mp_ph = {}
    with open(RAW / "mp_phdos_raw.jsonl") as f:
        for line in f:
            r = json.loads(line)
            mp_ph[r["mpid"]] = r["phdos_raw"]
    mp_ed = {}
    with open(RAW / "mp_edos_raw.jsonl") as f:
        for line in f:
            r = json.loads(line)
            mp_ed[r["mpid"]] = r["edos_raw"][0]
    mp_st = {}
    with open(RAW / "mp_structures.jsonl") as f:
        for line in f:
            r = json.loads(line)
            mp_st[r["mpid"]] = r["summary"]
    ph_rec = {}
    with open(RAW / "phonondb_recomputed.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ph_rec[d["id"]] = d
    ph_map = {}  # canon -> serial
    with open(RAW / "phonondb_a3c_map.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ph_map[r["canonical_mpid"]] = r["serial"]
    jv = {}  # canon -> list of JVASP recs
    with open(RAW / "jarvis_threesrc_raw.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("canonical_mpid"):
                jv.setdefault(r["canonical_mpid"], []).append(r)
    print(f"[a2b] mp_ph={len(mp_ph)} mp_ed={len(mp_ed)} ph={len(ph_rec)} "
          f"jv_canon={len(jv)}", flush=True)

    # ---- phDOS pairs ----
    arms = defaultdict(list)  # arm -> list of r
    shifts, aratios = defaultdict(list), defaultdict(list)
    rows = []
    n3 = n2 = 0
    for canon, jrecs in sorted(jv.items()):
        if canon not in mp_ph or canon not in ph_map:
            continue
        # MP侧
        ment = sorted(mp_ph[canon], key=lambda e: 0 if e["method"] == "pheasy" else 1)[0]
        mx = np.array(ment["frequencies_THz"]) * THZ2CM
        my = np.array(ment["densities"])
        # PhononDB侧
        prec = ph_rec.get(ph_map[canon])
        if prec is None:
            continue
        px = np.array(prec["freq_THz"]) * THZ2CM
        py = np.array(prec["dos"])
        for jr in jrecs:
            if not jr.get("jarvis_phdos_raw"):
                continue
            if ("phonon_dos_frequencies" not in jr["jarvis_phdos_raw"]
                    or "phonon_dos_intensity" not in jr["jarvis_phdos_raw"]):
                continue  # 纯弹性记录, 无DOS
            jx = fparse(jr["jarvis_phdos_raw"]["phonon_dos_frequencies"])
            jy = fparse(jr["jarvis_phdos_raw"]["phonon_dos_intensity"])
            if len(jx) < 10 or len(jy) != len(jx):
                continue
            lo = max(mx.min(), px.min(), jx.min(), -300)
            hi = min(mx.max(), px.max(), jx.max())
            if hi - lo < 100:
                continue
            xs, M = interp_common([(mx, my), (px, py), (jx, jy)], lo, hi, 2.0)
            if M.shape[1] < 50:
                continue
            area = np.trapz(M, xs, axis=1)
            if (area <= 0).any():
                continue
            S = M / area[:, None]
            r_mp_ph = pearson(S[0], S[1])
            r_mp_jv = pearson(S[0], S[2])
            r_ph_jv = pearson(S[1], S[2])
            # 峰位移: MP为锚的互相关lag
            def lag(a, b):
                a0 = a - a.mean()
                b0 = b - b.mean()
                cc = np.correlate(a0, b0, mode="full")
                return float((cc.argmax() - (len(a) - 1)) * 2.0)
            lg_ph = lag(S[0], S[1])
            lg_jv = lag(S[0], S[2])
            arms["MP-PhononDB"].append(r_mp_ph)
            arms["MP-JARVIS"].append(r_mp_jv)
            arms["PhononDB-JARVIS"].append(r_ph_jv)
            shifts["MP-PhononDB"].append(lg_ph)
            shifts["MP-JARVIS"].append(lg_jv)
            aratios["MP-PhononDB"].append(float(area[1] / area[0]))
            aratios["MP-JARVIS"].append(float(area[2] / area[0]))
            rows.append({"canon": canon, "jvasp": jr["jvasp"],
                         "mp_method": ment["method"],
                         "r_mp_phdb": round(r_mp_ph, 4), "r_mp_jv": round(r_mp_jv, 4),
                         "r_phdb_jv": round(r_ph_jv, 4),
                         "lag_phdb_cm": lg_ph, "lag_jv_cm": lg_jv,
                         "area_phdb_mp": round(float(area[1] / area[0]), 4),
                         "area_jv_mp": round(float(area[2] / area[0]), 4)})
            n3 += 1
    print(f"[a2b] phdos 3-way pairs={n3}", flush=True)
    with open(RAW / "a2b_pairs_phdos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- eDOS MP-JARVIS (配准) ----
    erows, er_max, eshift = [], [], []
    for canon, jrecs in sorted(jv.items()):
        if canon not in mp_ed:
            continue
        me = mp_ed[canon]
        mx = np.array(me["energies"]) - me["efermi"]
        up = np.array(me["spin_up_densities"], dtype=float)
        dn = np.array(me["spin_down_densities"]
                      if me["spin_down_densities"] is not None
                      else np.zeros_like(up), dtype=float)
        my = np.abs(up) + np.abs(dn)  # 单自旋占83%, down=null即总量
        for jr in jrecs:
            if jr["jarvis_edos_raw"] is None:
                continue
            je = jr["jarvis_edos_raw"]
            jx0 = fparse(je["edos_energies"])
            ju = fparse(je["total_edos_up"])
            jd = fparse(je["total_edos_down"])
            if not (len(jx0) == len(ju) == len(jd)) or len(jx0) < 100:
                continue
            jy = np.abs(ju) + np.abs(jd)
            best, bshift = -2, 0.0
            for sh in np.arange(-15, 15.1, 0.5):
                jx = jx0 + sh
                lo = max(mx.min(), jx.min(), -6)
                hi = min(mx.max(), jx.max(), 6)
                if hi - lo < 8:
                    continue
                xs, M = interp_common([(mx, my), (jx, jy)], lo, hi, 0.1)
                if M.shape[1] < 80:
                    continue
                area = np.trapz(M, xs, axis=1)
                if (area <= 0).any():
                    continue
                r = pearson(M[0] / area[0], M[1] / area[1])
                if np.isfinite(r) and r > best:
                    best, bshift = r, float(sh)
            if best > -2:
                er_max.append(best)
                eshift.append(bshift)
                erows.append({"canon": canon, "jvasp": jr["jvasp"],
                              "r_max": round(float(best), 4),
                              "shift_eV": bshift})
    print(f"[a2b] edos pairs={len(erows)}", flush=True)
    with open(RAW / "a2b_pairs_edos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["canon", "jvasp", "r_max", "shift_eV"])
        w.writeheader()
        w.writerows(erows)

    # ---- 结构delta MP vs PhononDB (三源子集) ----
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    dvol, sg_match, sg_n = [], 0, 0
    for canon in sorted(set(r["canon"] for r in rows)):
        if canon not in mp_st or canon not in ph_map:
            continue
        try:
            ms = mp_st[canon]
            mvol = ms["structure"]["lattice"]["volume"] / ms["nsites"]
            mpg = ms["symmetry"]["number"]
            uc = ph_rec[ph_map[canon]]["structure"]["unitcell"]
            st = Structure(Lattice(uc["lattice"]),
                           [p["symbol"] for p in uc["points"]],
                           [p["coordinates"] for p in uc["points"]])
            pvol = st.volume / len(uc["points"])
            dvol.append(abs(pvol - mvol) / mvol)
            psg = SpacegroupAnalyzer(st, symprec=0.1).get_space_group_number()
            sg_n += 1
            sg_match += (psg == mpg)
        except Exception:
            pass
    struct = {"n": len(dvol), "dV_per_atom": agg(dvol),
              "sg_match_rate": round(sg_match / max(sg_n, 1), 4), "sg_n": sg_n}

    summary = {
        "phdos_r": {k: agg(v) for k, v in arms.items()},
        "phdos_lag_cm": {k: agg(v) for k, v in shifts.items()},
        "phdos_area_ratio": {k: agg(v) for k, v in aratios.items()},
        "edos_r_max": agg(er_max), "edos_shift_eV": agg(eshift),
        "struct_mp_vs_phonondb": struct,
    }
    # 预注册判决: median r>=0.8互换+校准; 0.5-0.8源标签加权; <0.5先审计
    rec = []
    for arm, st in summary["phdos_r"].items():
        m = st.get("median", 0)
        rec.append(f"{arm}: median_r={m} -> "
                   + ("互换+校准" if m >= 0.8 else ("源标签加权" if m >= 0.5 else "先审计再定")))
    summary["weight_advice"] = rec
    json.dump(summary, open(RAW / "a2b_summary.json", "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
