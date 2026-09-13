#!/usr/bin/env python3
"""A4 v2加工执行: raw三源 -> Parquet全集(谱落实口径26,002).

输入(全本地, 零API):
  mp_edos_raw.jsonl (18,644) / mp_phdos_raw.jsonl (26,609) / mp_structures.jsonl
  mp_increment_raw.jsonl (7,724 summary) + mp_increment_dosmap.json (eDOS归属)
  Delta eDOS分区 (增量dos task行提取)
  phonondb_recomputed.jsonl + phonondb_a3c_map.jsonl
  jarvis_gap_raw.jsonl + jarvis_threesrc_raw.jsonl (+ hit_canon)
  census_effective_ids.json / dual_spectra_ids.json / edos_absent.json
输出: v2_intermediate.parquet -> v2_processed.parquet + stats_v2.json + outlier_attribution.csv
冻结口径:
  GRID eDOS [-6,6]/128盒平均(trapz均值, Fermi对齐用dos-task行efermi);
  phDOS [-280,980]/64盒平均; THz源 x*33.35641且y/33.35641(密度守恒, Si标尺0.0602vs0.0605);
  JARVIS强度任意单位->按3N求和规则重归一(N=原胞原子数); PhononDB只除33.356(面积已1.0).
  结构: 原胞symprec=0.1 + 82格式(哨兵126/127 + 1/c, cif2dos契约); CAP=80, 超限截断+warn+prov.
  标签优先级: eDOS一律MP(血缘task); 声子MP pheasy>MP dfpt>PhononDB>JARVIS;
  外源声子视图进audit sidecar, 主表只存primary + 源标签/权重列.
  自旋: 总量=up+down(MP down>=0; JARVIS down为负用|down|).
  缺失记mask(存parquet+sidecar, 训练消费是C2的事); 永不填0(掩膜 bin值0+mask=1).
  winsorize: 两遍式, per-bin p99.9 + 单bin尖峰判据(邻<0.5v), 归因表留痕.
"""
import glob
import io
import json
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path("/root/home/newstudy/getdata/raw")
THZ2CM = 33.35641
E_EDGES = np.linspace(-6.0, 6.0, 129)
P_EDGES = np.linspace(-280.0, 980.0, 65)
CAP_ATOMS = 80
WINSOR_PCTL = 99.9
SYMPREC = 0.1


def box_average(x, y, edges):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)  # 缺失记mask: 非有限源点直接排除
    x, y = x[keep], y[keep]
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    cov = np.zeros(len(edges) - 1, dtype=np.int8)
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() > 1:
            out[i] = np.trapz(y[m], x[m]) / (edges[i + 1] - edges[i])
            cov[i] = 1
    return out.astype(np.float32), cov


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


def load_jsonl_map(path, key):
    d = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                d[r[key]] = r
            except Exception:
                pass
    return d


def main():
    t00 = time.time()
    print("[a4] loading raw...", flush=True)
    mp_ed = load_jsonl_map(RAW / "mp_edos_raw.jsonl", "mpid")
    mp_ph = load_jsonl_map(RAW / "mp_phdos_raw.jsonl", "mpid")
    mp_st = {json.loads(l)["mpid"]: json.loads(l)["summary"]
             for l in open(RAW / "mp_structures.jsonl")}
    mp_inc = load_jsonl_map(RAW / "mp_increment_raw.jsonl", "mpid")
    dosmap = json.load(open(RAW / "mp_increment_dosmap.json"))
    dual = set(json.load(open(RAW / "dual_spectra_ids.json")))
    absent = set(json.load(open(RAW / "edos_absent.json")))
    true_double = dual - absent
    ph_map = {}
    with open(RAW / "phonondb_a3c_map.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ph_map[r["canonical_mpid"]] = r["serial"]
    ph_rec = {}
    with open(RAW / "phonondb_recomputed.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ph_rec[d["id"]] = d
    jv_gap = {}
    with open(RAW / "jarvis_gap_raw.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["jarvis_phdos_raw"] is not None and r.get("canonical_mpid"):
                jv_gap.setdefault(r["canonical_mpid"], []).append(r)
    jv_three = {}
    with open(RAW / "jarvis_threesrc_raw.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["jarvis_phdos_raw"] is not None and r.get("canonical_mpid"):
                jv_three.setdefault(r["canonical_mpid"], []).append(r)

    # 增量eDOS Delta行: es直给 + dosmap归属
    need_tasks = {}
    for line in open(RAW / "mp_increment_raw.jsonl"):
        r = json.loads(line)
        if r["dos_task_id"] is not None:
            need_tasks[r["mpid"]] = r["dos_task_id"]
    for m, d in dosmap.items():
        if d["mode"] in ("unique", "delta_vol", "matcher_ok") and m not in need_tasks:
            need_tasks[m] = d["identifier"]
    print(f"[a4] increment edos tasks={len(need_tasks)}", flush=True)
    from deltalake import DeltaTable
    edt = DeltaTable(str(RAW / "delta_mp/core/electronic-structure/total-dos/"))
    inc_edos = {}
    ids = sorted(set(need_tasks.values()))
    for i in range(0, len(ids), 200):
        t = edt.to_pyarrow_table(filters=[("identifier", "in", ids[i:i + 200])])
        for row in t.to_pylist():
            inc_edos[row["identifier"]] = row
        if (i // 200 + 1) % 10 == 0:
            print(f"[a4] delta edos {i + 200}/{len(ids)}", flush=True)
    inv = {}
    for m, t in need_tasks.items():
        inv.setdefault(t, []).append(m)
    print(f"[a4] delta edos rows hit={len(inc_edos)}/{len(ids)}", flush=True)

    # ---- 主池: true_double(18644) + 谱落实增量 ----
    pool = sorted(true_double | set(
        m for m in list(mp_inc.keys())
        if (mp_inc[m]["dos_task_id"] is not None or m in dosmap and
            dosmap[m]["mode"] in ("unique", "delta_vol", "matcher_ok"))))
    # 注: dosmap unresolved(366)已排除
    print(f"[a4] pool={len(pool)}", flush=True)

    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        from utils.atom_feature import PeriodicTable
        pt = PeriodicTable()
        fmap = pt.atom_feature_map()
        FEAT = np.asarray(fmap[:119], dtype=np.float32)  # Z=0..118, 27维
        FDIM = FEAT.shape[1]
    except Exception as e:
        print(f"[a4] atom_feature fallback onehot: {e}", flush=True)
        FEAT, FDIM = None, 27

    def mp_summary_of(m):
        if m in mp_st:
            return mp_st[m], "mp_struct_dual"
        r = mp_inc.get(m)
        return (r["mp_summary"], "mp_struct_inc") if r and r["mp_summary"] else (None, None)

    def edos_of(m):
        if m in mp_ed:
            e = mp_ed[m]["edos_raw"][0]
            up = np.array(e["spin_up_densities"], dtype=float)
            dn = e["spin_down_densities"]
            dn = np.array(dn, dtype=float) if dn is not None else np.zeros_like(up)
            return (np.array(e["energies"], dtype=float) - e["efermi"], up + dn,
                    f"mp:{mp_ed[m]['dos_task_id']}", "mp")
        t = need_tasks.get(m)
        row = inc_edos.get(t) if t else None
        if row is None:
            return None
        up = np.array(row["spin_up_densities"], dtype=float)
        dn = row["spin_down_densities"]
        if dn is None or np.ndim(dn) == 0 or len(dn) != len(up):
            dn = np.zeros_like(up)  # 单自旋/标量: 即总量(与MP原生分支同规则)
        else:
            dn = np.array(dn, dtype=float)
        ef = row["efermi"]
        if ef is None or not np.isfinite(ef):
            return None  # 无Ef无法对齐, 记no_edos
        return (np.array(row["energies"], dtype=float) - ef, up + dn,
                f"mp:{t}", "mp")

    def phdos_of(m):
        cands = []
        if m in mp_ph:
            order = sorted(mp_ph[m]["phdos_raw"],
                           key=lambda e: 0 if e["method"] == "pheasy" else 1)
            e0 = order[0]
            cands.append((np.array(e0["frequencies_THz"], dtype=float) * THZ2CM,
                          np.array(e0["densities"], dtype=float) / THZ2CM,
                          f"mp:{e0['method']}", "mp", 1.0))
        if m in ph_map and ph_map[m] in ph_rec:
            pr = ph_rec[ph_map[m]]
            cands.append((np.array(pr["freq_THz"], dtype=float) * THZ2CM,
                          np.array(pr["dos"], dtype=float) / THZ2CM,
                          f"phdb:{ph_map[m]}", "phonondb", 1.0))
        for src, store in (("jarvis_gap", jv_gap), ("jarvis_3src", jv_three)):
            for jr in store.get(m, []):  # 同一材料多JVASP全扫, 首个可用者中标
                p = jr["jarvis_phdos_raw"]
                if "phonon_dos_frequencies" not in p:
                    continue
                cands.append((fparse(p["phonon_dos_frequencies"]),
                              fparse(p["phonon_dos_intensity"]),
                              f"jv:{jr['jvasp']}", "jarvis", np.nan))
        return cands

    recs, audit, stats = [], [], Counter()
    import warnings
    warnings.filterwarnings("ignore")
    for i, m in enumerate(pool):
        summ, ssrc = mp_summary_of(m)
        if summ is None:
            stats["no_struct"] += 1
            continue
        try:
            st0 = Structure(Lattice(summ["structure"]["lattice"]["matrix"]),
                            [s["label"] for s in summ["structure"]["sites"]],
                            [s["abc"] for s in summ["structure"]["sites"]])
            prim = SpacegroupAnalyzer(st0, symprec=SYMPREC).get_primitive_standard_structure()
        except Exception:
            stats["prim_fail"] += 1
            continue
        n = len(prim)
        trunc = n > CAP_ATOMS
        # CIF6: 82格式
        Z = [s.specie.Z for s in prim.sites][:CAP_ATOMS]
        src = [126, 127] + Z + [0] * (CAP_ATOMS - len(Z))
        a, b, c = prim.lattice.a, prim.lattice.b, prim.lattice.c
        al, be, ga = prim.lattice.angles
        pos = [[a, b, 1.0 / c], [al, be, ga]]
        pos += [list(prim.frac_coords[k]) for k in range(min(n, CAP_ATOMS))]
        pos += [[0, 0, 0]] * (CAP_ATOMS - min(n, CAP_ATOMS))
        occ = [1] * min(n, CAP_ATOMS) + [0] * (CAP_ATOMS - min(n, CAP_ATOMS))
        sg = SpacegroupAnalyzer(prim, symprec=SYMPREC)
        try:
            ds = sg.get_symmetry_dataset()
            wy = ds.wyckoffs if hasattr(ds, "wyckoffs") else ds["wyckoffs"]
            wy = "_".join(sorted(set(wy)))
        except Exception:
            wy = ""
        if FEAT is not None:
            af = np.zeros((CAP_ATOMS, FDIM), dtype=np.float32)
            for k, z in enumerate(Z):
                if 0 <= z < len(FEAT):
                    af[k] = FEAT[z]
        else:
            af = np.zeros((CAP_ATOMS, FDIM), dtype=np.float32)
        # eDOS
        eo = edos_of(m)
        if eo is None:
            stats["no_edos"] += 1
            continue
        ex, ey, eref, esrc = eo
        ev, emask = box_average(ex, ey, E_EDGES)
        # phDOS primary + audit
        cands = [c for c in phdos_of(m) if len(c[0]) >= 10 and len(c[0]) == len(c[1])]
        if not cands:
            stats["no_phdos"] += 1
            continue
        px, py, pref, psrc, pw = cands[0]
        sel = (px >= px.min()) & (px <= px.max())
        area = float(np.trapz(py[sel], px[sel])) if sel.sum() > 1 else 0.0
        if psrc == "jarvis":
            nmodes = 3 * n
            py = py * (nmodes / area) if area > 0 else py  # 3N求和规则重归一
            pw = 0.5  # A2b降权, 训练侧sampler消费
        elif psrc == "phonondb":
            pw = 1.0  # A2b互换档, 轻校准=权重1
        pv, pmask = box_average(px, py, P_EDGES)
        for ax, ay, aref, asrc, aw in cands[1:]:
            audit.append({"mpid": m, "primary": pref, "alt": aref,
                          "alt_src": asrc})
        recs.append({
            "mpid": m, "src_edos": esrc, "src_ph": psrc, "ph_weight": pw,
            "edos_ref": eref, "ph_ref": pref,
            "struct_ref": (f"mp_structures.jsonl:{m}" if ssrc == "mp_struct_dual"
                           else f"mp_increment_raw.jsonl:{m}"),
            "elements": src, "positions": np.array(pos, dtype=np.float32).flatten(),
            "lattice": [a, b, c, al, be, ga],
            "sg": sg.get_space_group_number(), "crystal": str(sg.get_crystal_system()),
            "wyckoff": wy, "occupancy": occ, "atom_feat": af,
            "edos": ev, "edos_mask": emask, "phdos": pv, "phdos_mask": pmask,
            "nsites_prim": n, "truncated": trunc,
            "prov": json.dumps({"struct": ssrc, "symprec": SYMPREC}),
        })
        stats[f"src_{psrc}"] += 1
        if trunc:
            stats["truncated"] += 1
        if (i + 1) % 2000 == 0:
            print(f"[a4] {i + 1}/{len(pool)} recs={len(recs)} {dict(stats)}",
                  flush=True)
    print(f"[a4] pass1 done recs={len(recs)} {dict(stats)} "
          f"elapsed={round(time.time() - t00, 1)}s", flush=True)

    tbl = pa.table({
        "mpid": [r["mpid"] for r in recs],
        "src_edos": [r["src_edos"] for r in recs],
        "src_ph": [r["src_ph"] for r in recs],
        "ph_weight": pa.array([r["ph_weight"] for r in recs], type=pa.float32()),
        "edos_ref": [r["edos_ref"] for r in recs],
        "ph_ref": [r["ph_ref"] for r in recs],
        "struct_ref": [r["struct_ref"] for r in recs],
        "elements": [r["elements"] for r in recs],
        "positions": pa.array([r["positions"] for r in recs], type=pa.list_(pa.float32())),
        "lattice": [r["lattice"] for r in recs],
        "sg": [r["sg"] for r in recs],
        "crystal": [r["crystal"] for r in recs],
        "wyckoff": [r["wyckoff"] for r in recs],
        "occupancy": [r["occupancy"] for r in recs],
        "atom_feat": pa.array([r["atom_feat"].flatten() for r in recs],
                              type=pa.list_(pa.float32())),
        "feat_dim": [int(r["atom_feat"].shape[1]) for r in recs],
        "edos": pa.array([r["edos"] for r in recs], type=pa.list_(pa.float32())),
        "edos_mask": [r["edos_mask"].tolist() for r in recs],
        "phdos": pa.array([r["phdos"] for r in recs], type=pa.list_(pa.float32())),
        "phdos_mask": [r["phdos_mask"].tolist() for r in recs],
        "nsites_prim": [r["nsites_prim"] for r in recs],
        "truncated": [r["truncated"] for r in recs],
        "prov": [r["prov"] for r in recs],
    })
    pq.write_table(tbl, RAW / "v2_intermediate.parquet")
    json.dump({"stats": dict(stats), "n": len(recs),
               "audit_alt_views": len(audit)},
              open(RAW / "v2_pass1.json", "w"), indent=1)
    with open(RAW / "v2_audit_alt.csv", "w") as f:
        f.write("mpid,primary,alt,alt_src\n")
        for a in audit:
            f.write(f"{a['mpid']},{a['primary']},{a['alt']},{a['alt_src']}\n")
    print("[a4] intermediate written", flush=True)


if __name__ == "__main__":
    main()
