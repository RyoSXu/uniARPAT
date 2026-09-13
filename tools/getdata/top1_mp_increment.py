#!/usr/bin/env python3
"""Top-up#1: MP增量集结构 + eDOS task映射补抓(A4必需, 无条件前置).

增量集 = PhononDB增量(4391) ∪ JARVIS增量(3600) = 7724唯一canonical.
抓: summary(structure/symmetry/标量/run_type) + electronic_structure(dos task_id).
出: mp_increment_raw.jsonl ({mpid, mp_summary, dos_task_id, provenance}).
断点续跑(mpid key); 整批100+数量校验(回声保序); 单体补查顽固分子.
"""
import json
import time
from pathlib import Path

import requests

KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
H = {"X-API-KEY": KEY}
RAW = Path("/root/home/newstudy/getdata/raw")
OUT = RAW / "mp_increment_raw.jsonl"
SUM_FIELDS = ("material_id,structure,symmetry,nsites,band_gap,efermi,"
              "formation_energy_per_atom,energy_above_hull,run_type,energy_type")


def build_increment():
    eff = set(json.load(open(RAW / "census_effective_ids.json")))
    dual = set(json.load(open(RAW / "dual_spectra_ids.json")))
    ph = set(json.loads(l)["canonical_mpid"]
             for l in open(RAW / "phonondb_a3c_map.jsonl"))
    jv = set(json.load(open(RAW / "jarvis_gap_hit_canon.json"))
             ) | set(json.loads(l)["canonical_mpid"]
                     for l in open(RAW / "jarvis_a7b_map.jsonl")
                     if json.loads(l)["canonical_mpid"] in dual)
    inc = ((ph | jv) & eff) - dual
    return sorted(inc)


def rest_batch(session, route, mpids, fields, chunk=100, tries=4):
    out, pending = {}, list(mpids)
    for p in range(tries):
        if not pending:
            break
        nxt = []
        for i in range(0, len(pending), chunk):
            ch = pending[i:i + chunk]
            try:
                r = session.get(BASE + route,
                                params={"material_ids": ",".join(ch),
                                        "_fields": fields},
                                headers=H, timeout=180)
                if r.status_code == 200 and len(r.json()["data"]) == len(ch):
                    for m, doc in zip(ch, r.json()["data"]):
                        out[m] = doc
                else:
                    nxt.extend(ch)
            except Exception:
                nxt.extend(ch)
            time.sleep(0.3)
        pending = [m for m in nxt if m not in out]
        if pending:
            print(f"[top1] {route} pass={p} resolved={len(out)}/{len(mpids)} "
                  f"retry={len(pending)}", flush=True)
    # 顽固单查
    still = []
    for m in pending:
        try:
            r = session.get(BASE + route,
                            params={"material_ids": m, "_fields": fields},
                            headers=H, timeout=60)
            if r.status_code == 200 and len(r.json()["data"]) == 1:
                out[m] = r.json()["data"][0]
            else:
                still.append(m)
        except Exception:
            still.append(m)
        time.sleep(0.2)
    return out, still


def main():
    inc = build_increment()
    print(f"[top1] increment={len(inc)}", flush=True)
    done = {}
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["mpid"]] = r
                except Exception:
                    pass
    todo = [m for m in inc if m not in done]
    print(f"[top1] todo={len(todo)}", flush=True)
    if not todo:
        return
    s = requests.Session()
    summ, s_un = rest_batch(s, "/materials/summary/", todo, SUM_FIELDS)
    print(f"[top1] summary {len(summ)}/{len(todo)} unresolved={len(s_un)}", flush=True)
    es, e_un = rest_batch(s, "/materials/electronic_structure/", todo,
                          "material_id,dos", chunk=50)
    print(f"[top1] es {len(es)}/{len(todo)} unresolved={len(e_un)}", flush=True)
    prov = {"mp_db": "2026.04.13",
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(OUT, "a") as f:
        for m in todo:
            d = (es.get(m) or {}).get("dos") or {}
            f.write(json.dumps({"mpid": m, "mp_summary": summ.get(m),
                                "dos_task_id": d.get("task_id"),
                                "summary_unresolved": m in s_un,
                                "es_unresolved": m in e_un,
                                "provenance": prov}) + "\n")
    print("[top1] DONE", flush=True)


if __name__ == "__main__":
    main()
