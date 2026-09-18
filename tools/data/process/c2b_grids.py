#!/usr/bin/env python3
"""C2b网格标签全量制备: 原生谱 -> 七臂网格(Stage1 eDOS四臂+P1; Stage2 P0/P2).

臂定义(edges精确值, 见grids.json):
  E0 [-6,6]/128等距(anchor=现v2标签, 不重算) | E1 [-4,4]/160等距
  E2 非均匀160([-2,2]96 + 两尾各32) | E3 [-6,6]/256等距
  P0 [-280,980]/64(anchor=现v2标签) | P1 非均匀112(负频10 + 0~500按10 + 500~4000按67.3)
  P2 [-300,4000]/128等距
规则: 与A4完全同口径(自旋总量/单位/3N重归一/源优先级/掩膜/winsorize分臂重算);
  split复用v2(可比性); 输出data/grids_c2b/{arm}/{split}/标签npy,
  elements/positions/index软链v2缓存(同一样本同一切分).
用法: c2b_grids.py --arms E1,E2,E3,P1,P2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

RAW = Path("/root/home/newstudy/getdata/raw")
REPO = Path(__file__).resolve().parents[2]
OUTB = REPO / "data/grids_c2b"
THZ2CM = 33.35641

GRIDS = {
    "E0": np.linspace(-6, 6, 129),
    "E1": np.linspace(-4, 4, 161),
    "E2": np.concatenate([np.linspace(-6, -2, 33)[:-1],
                          np.linspace(-2, 2, 97)[:-1],
                          np.linspace(2, 6, 33)]),
    "E3": np.linspace(-6, 6, 257),
    "P0": np.linspace(-280, 980, 65),
    "P1": np.concatenate([np.linspace(-300, 0, 11)[:-1],
                          np.linspace(0, 500, 51)[:-1],
                          np.linspace(500, 4000, 53)]),
    "P2": np.linspace(-300, 4000, 129),
}
EDOS_ARMS = ["E0", "E1", "E2", "E3"]
PHDOS_ARMS = ["P0", "P1", "P2"]


def box_average(x, y, edges):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    cov = np.zeros(len(edges) - 1, dtype=np.int8)
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        n = int(m.sum())
        if n > 1:
            out[i] = np.trapz(y[m], x[m]) / (edges[i + 1] - edges[i])
            cov[i] = 1
        elif n == 1:
            out[i] = float(y[m][0])
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


def load_raw():
    mp_ed = {}
    with open(RAW / "mp_edos_raw.jsonl") as f:
        for line in f:
            r = json.loads(line)
            mp_ed[r["mpid"]] = r
    mp_ph = {}
    with open(RAW / "mp_phdos_raw.jsonl") as f:
        for line in f:
            r = json.loads(line)
            mp_ph[r["mpid"]] = r["phdos_raw"]
    need = set()
    for line in open(RAW / "mp_increment_raw.jsonl"):
        r = json.loads(line)
        if r["dos_task_id"]:
            need.add(r["dos_task_id"])
    dm = json.load(open(RAW / "mp_increment_dosmap.json"))
    for m, d in dm.items():
        if d["mode"] in ("unique", "delta_vol", "matcher_ok") and d.get("identifier"):
            need.add(d["identifier"])
    have = set()
    for line in open(RAW / "mp_edos_raw.jsonl"):
        have.add(json.loads(line)["dos_task_id"])
    need -= have
    print(f"[c2b] delta edos to read={len(need)}", flush=True)
    from deltalake import DeltaTable
    edt = DeltaTable(str(RAW / "delta_mp/core/electronic-structure/total-dos/"))
    inc = {}
    ids = sorted(need)
    for i in range(0, len(ids), 200):
        t = edt.to_pyarrow_table(filters=[("identifier", "in", ids[i:i + 200])])
        for row in t.to_pylist():
            inc[row["identifier"]] = row
    print(f"[c2b] delta edos hit={len(inc)}/{len(ids)}", flush=True)
    ph_rec = {}
    with open(RAW / "phonondb_recomputed.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ph_rec[d["id"]] = d
    ph_map = {}
    with open(RAW / "phonondb_a3c_map.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ph_map[r["canonical_mpid"]] = r["serial"]
    jv = {}
    for fn in ("jarvis_gap_raw.jsonl", "jarvis_threesrc_raw.jsonl"):
        with open(RAW / fn) as f:
            for line in f:
                r = json.loads(line)
                if r.get("jarvis_phdos_raw") and r.get("canonical_mpid"):
                    jv.setdefault(r["canonical_mpid"], []).append(r)
    return mp_ed, mp_ph, inc, ph_rec, ph_map, jv


def edos_raw_of(m, mp_ed, inc, need_map):
    if m in mp_ed:
        e = mp_ed[m]["edos_raw"][0]
        up = np.array(e["spin_up_densities"], dtype=float)
        dn = e["spin_down_densities"]
        dn = np.array(dn, dtype=float) if dn is not None else np.zeros_like(up)
        return np.array(e["energies"], dtype=float) - e["efermi"], up + dn
    t = need_map.get(m)
    row = inc.get(t) if t else None
    if row is None:
        return None
    up = np.array(row["spin_up_densities"], dtype=float)
    dn = row["spin_down_densities"]
    dn = (np.array(dn, dtype=float) if dn is not None and np.ndim(dn) != 0
          and len(dn) == len(up) else np.zeros_like(up))
    ef = row["efermi"]
    if ef is None or not np.isfinite(ef):
        return None
    return np.array(row["energies"], dtype=float) - ef, up + dn


def phdos_cands(m, mp_ph, ph_rec, ph_map, jv):
    out = []
    if m in mp_ph:
        order = sorted(mp_ph[m], key=lambda e: 0 if e["method"] == "pheasy" else 1)
        for e0 in order:
            x = np.array(e0["frequencies_THz"], dtype=float)
            y = np.array(e0["densities"], dtype=float)
            if len(x) >= 10 and len(x) == len(y):
                out.append((x * THZ2CM, y / THZ2CM, "mp"))
                break
    if m in ph_map and ph_map[m] in ph_rec:
        pr = ph_rec[ph_map[m]]
        out.append((np.array(pr["freq_THz"], dtype=float) * THZ2CM,
                    np.array(pr["dos"], dtype=float) / THZ2CM, "phonondb"))
    for jr in jv.get(m, []):
        p = jr["jarvis_phdos_raw"]
        if "phonon_dos_frequencies" not in p:
            continue
        x, y = fparse(p["phonon_dos_frequencies"]), fparse(p["phonon_dos_intensity"])
        if len(x) >= 10 and len(x) == len(y):
            out.append((x, y, "jarvis"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="E1,E2,E3,P1,P2")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"[c2b] arms={arms}", flush=True)
    pool = pq.read_table(RAW / "v2_processed.parquet",
                         columns=["mpid", "split", "nsites_prim", "src_ph"]).to_pylist()
    splits = {}
    for r in pool:
        splits.setdefault(r["split"], []).append(r["mpid"])
    print({k: len(v) for k, v in splits.items()}, flush=True)
    mp_ed, mp_ph, inc, ph_rec, ph_map, jv = load_raw()
    need_map = {}
    for line in open(RAW / "mp_increment_raw.jsonl"):
        r = json.loads(line)
        if r["dos_task_id"]:
            need_map[r["mpid"]] = r["dos_task_id"]
    dm = json.load(open(RAW / "mp_increment_dosmap.json"))
    for m, d in dm.items():
        if d["mode"] in ("unique", "delta_vol", "matcher_ok") and m not in need_map:
            need_map[m] = d.get("identifier")
    nsites = {r["mpid"]: r["nsites_prim"] for r in pool}
    report = {}
    for arm in arms:
        edges = GRIDS[arm]
        is_edos = arm in EDOS_ARMS
        nb = len(edges) - 1
        print(f"[c2b] {arm}: bins={nb} range=({edges[0]},{edges[-1]})", flush=True)
        armrep = {}
        for s, mpids in splits.items():
            V = np.zeros((len(mpids), nb), dtype=np.float32)
            M = np.zeros((len(mpids), nb), dtype=np.int8)
            for i, m in enumerate(mpids):
                if is_edos:
                    eo = edos_raw_of(m, mp_ed, inc, need_map)
                    if eo is None:
                        continue
                    v, c = box_average(eo[0], eo[1], edges)
                else:
                    cands = phdos_cands(m, mp_ph, ph_rec, ph_map, jv)
                    if not cands:
                        continue
                    x, y, src = cands[0]
                    if src == "jarvis":
                        area = float(np.trapz(y, x)) if len(x) > 1 else 0
                        if area > 0:
                            y = y * (3 * max(nsites.get(m, 1), 1) / area)
                    v, c = box_average(x, y, edges)
                V[i], M[i] = v, c
            # 分臂winsorize(同规则)
            th = np.array([np.quantile(V[M[:, j].astype(bool), j], 0.999)
                           if M[:, j].sum() else np.inf for j in range(nb)])
            left = np.zeros_like(V)
            left[:, 1:] = V[:, :-1]
            right = np.zeros_like(V)
            right[:, :-1] = V[:, 1:]
            over = (V > th) & M.astype(bool)
            wide = np.zeros_like(V, dtype=bool)
            wide[:, 1:-1] = over[:, 1:-1] & over[:, :-2] & over[:, 2:]
            clip = over & ~wide & (np.maximum(left, right) < 0.5 * V)
            V[clip] = np.broadcast_to(th, V.shape)[clip]
            d = OUTB / arm / s
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{'edos' if is_edos else 'phdos'}_tgtdos_{s}.npy", V)
            np.save(d / f"{'edos' if is_edos else 'phdos'}_mask_{s}.npy", M)
            np.save(d / f"{s}_index.npy", np.array(mpids))
            for base in ("elements", "positions"):
                src_f = REPO / "data/train4ARPAT-v2" / s / f"{base}_{s}.npy"
                lnk = d / f"{base}_{s}.npy"
                if not lnk.exists():
                    lnk.symlink_to(src_f)
            armrep[s] = {"n": len(mpids), "covered_frac": round(float(M.mean()), 4),
                         "max_med": round(float(np.median(V.max(axis=1))), 4),
                         "clipped": int(clip.sum())}
            print(f"[c2b] {arm}/{s}: {armrep[s]}", flush=True)
        report[arm] = armrep
    json.dump({k: v.tolist() for k, v in GRIDS.items()},
              open(OUTB / "grids.json", "w"))
    json.dump(report, open(OUTB / "c2b_grids_report.json", "w"), indent=1)
    print("[c2b] DONE", flush=True)


if __name__ == "__main__":
    main()
