#!/usr/bin/env python3
"""A4 extraction step 1: dual-list spectra+structures -> raw JSONL.
In:  getdata/raw/dual_spectra_ids.json
Out: getdata/raw/mp_edos_raw.jsonl, mp_structures.jsonl (+ manifest)
Resume-safe (skip mpids already written). eDOS run_type pairing: structure's
functional first (GGA default; +U materials take +U), all variants referenced.
"""
import json
import time
from pathlib import Path

import requests

KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
H = {"X-API-KEY": KEY}
RAW = Path("/root/home/newstudy/getdata/raw")
SUMMARY_FIELDS = ("material_id,structure,band_gap,efermi,energy_above_hull,"
                  "nsites,nelements,symmetry,formation_energy_per_atom,"
                  "run_type,energy_type")


def rest_batched(session, mpids, fields, chunk=100, tries=6):
    out = {}
    for i in range(0, len(mpids), chunk):
        for _ in range(tries):
            try:
                r = session.get(
                    BASE + "/materials/summary/",
                    params={"material_ids": ",".join(mpids[i:i + chunk]),
                            "_fields": fields},
                    headers=H, timeout=180)
                if r.status_code == 200 and len(r.json()["data"]) == len(mpids[i:i + 100]):
                    for d in r.json()["data"]:
                        out[d["material_id"]] = d
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(5)
        time.sleep(0.3)
    return out


def main():
    dual = json.load(open(RAW / "dual_spectra_ids.json"))
    done_e, done_s = set(), set()
    fe = RAW / "mp_edos_raw.jsonl"
    fs = RAW / "mp_structures.jsonl"
    for path, acc in ((fe, done_e), (fs, done_s)):
        if path.exists():
            with open(path) as f:
                for line in f:
                    try:
                        acc.add(json.loads(line)["mpid"])
                    except Exception:  # noqa: BLE001
                        pass
    print(f"[extract] dual={len(dual)} edos_done={len(done_e)} struct_done={len(done_s)}",
          flush=True)
    session = requests.Session()
    # 1. structures (+ scalars for pairing/stratification)
    todo_s = [m for m in dual if m not in done_s]
    with open(fs, "a") as f:
        for i in range(0, len(todo_s), 100):
            docs = rest_batched(session, todo_s[i:i + 100], SUMMARY_FIELDS)
            for m in todo_s[i:i + 100]:
                if m in docs:
                    f.write(json.dumps({"mpid": m, "summary": docs[m],
                                        "provenance": {"mp_db": "2026.04.13"}}) + "\n")
            if (i // 100 + 1) % 20 == 0:
                print(f"[extract] structures {i + 100}/{len(todo_s)}", flush=True)
    # 2. eDOS task ids via es route, then local Delta read
    from deltalake import DeltaTable
    todo_e = [m for m in dual if m not in done_e]
    edt = DeltaTable(str(RAW / "delta_mp/core/electronic-structure/total-dos/"))
    with open(fe, "a") as f:
        for i in range(0, len(todo_e), 100):
            chunk = todo_e[i:i + 100]
            esmap = rest_batched(session, chunk, "material_id,dos")
            tids = {}
            for m in chunk:
                try:
                    tids[m] = esmap[m]["dos"]["task_id"]
                except Exception:  # noqa: BLE001
                    pass
            rows = {}
            if tids:
                for a in range(6):
                    try:
                        t = edt.to_pyarrow_table(
                            filters=[("identifier", "in", sorted(set(tids.values())))])
                        for r in t.to_pylist():
                            rows.setdefault(r["identifier"], []).append(r)
                        break
                    except Exception:  # noqa: BLE001
                        time.sleep(15)
            for m in chunk:
                recs = rows.get(tids[m], []) if m in tids else []
                if recs:
                    f.write(json.dumps({"mpid": m, "dos_task_id": tids[m],
                                        "edos_raw": recs}) + "\n")
            if (i // 100 + 1) % 10 == 0:
                print(f"[extract] edos {i + 100}/{len(todo_e)}", flush=True)
    print("[extract] DONE", flush=True)


if __name__ == "__main__":
    main()
