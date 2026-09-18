#!/usr/bin/env python3
"""v2 MP bulk raw fetcher: structures + eDOS + phDOS, all MP-native (same provenance).

Sources (verified 2026-09-09):
  - REST  /materials/summary/            : structure, scalars, phonon_IDs (batched)
  - REST  /materials/electronic_structure/: dos task_id per material (batched)
  - Delta s3a://materialsproject-parsed/core/electronic-structure/total-dos/
      schema: identifier, structure, spin_up/down_densities, energies, efermi, run_type
  - Delta s3a://materialsproject-parsed/phonon/electronic-structure/dos/
      schema: identifier, phonon_method, dos{frequencies, densities,
             projected_densities, run_type, structure}

Contract: FULL raw payloads + provenance per record. Resume-safe. Retried reads.
MP API key via env MP_API_KEY (REST only; Delta tables are unsigned public S3).

Usage:
  MP_API_KEY=... python3 fetch_mp_raw.py --mpids v1_test_ids.txt --out ../raw/v2_mp.jsonl
  (one legacy mp-id per line, e.g. mp-149)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

MP_BASE = "https://api.materialsproject.org"
OPT = {"AWS_SKIP_SIGNATURE": "true", "AWS_REGION": "us-east-1"}
EDOS_TABLE = "s3a://materialsproject-parsed/core/electronic-structure/total-dos/"
PHDOS_TABLE = "s3a://materialsproject-parsed/phonon/electronic-structure/dos/"


def rest_get(session, key, route, params, tries=5):
    for a in range(tries):
        try:
            r = session.get(MP_BASE + route, params=params,
                            headers={"X-API-KEY": key}, timeout=120)
            if r.status_code == 200:
                return r.json()["data"]
            time.sleep(6 * (a + 1))
        except Exception:  # noqa: BLE001
            time.sleep(6 * (a + 1))
    return None


def rest_batched(session, key, route, mpids, fields, chunk=50):
    out = {}
    for i in range(0, len(mpids), chunk):
        data = rest_get(session, key, route,
                        {"material_ids": ",".join(mpids[i:i + chunk]),
                         "_fields": fields, "id_format": "legacy"})
        if data:
            for d in data:
                out[d["material_id"]] = d
        time.sleep(0.3)
    return out


def delta_table(uri):
    from deltalake import DeltaTable
    last = None
    for a in range(8):
        try:
            return DeltaTable(uri, storage_options=OPT)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(12)
    raise RuntimeError(f"DeltaTable open failed for {uri}: {last}")


def delta_fetch(dt, col, ids, extra_filter=None, tries=6):
    """Fetch rows by identifier list (chunked IN-filter). Returns {id: row}."""
    import pyarrow.compute as pc
    out = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        filt = [(col, "in", chunk)]
        if extra_filter:
            filt.append(extra_filter)
        for a in range(tries):
            try:
                t = dt.to_pyarrow_table(filters=filt)
                for row in t.to_pylist():
                    out.setdefault(row[col], row)
                break
            except Exception:  # noqa: BLE001
                time.sleep(15)
        time.sleep(0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpids", required=True, help="text file, one legacy mp-id per line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit("MP_API_KEY not set")

    mpids = [l.strip() for l in open(args.mpids) if l.strip()]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["mpid"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [m for m in mpids if m not in done]
    print(f"[mp] total={len(mpids)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    session = requests.Session()
    print("[mp] REST summary (structure+phonon_IDs)...", flush=True)
    summ = rest_batched(session, key, "/materials/summary/", todo,
                        "material_id,structure,band_gap,efermi,energy_above_hull,"
                        "nsites,symmetry,formation_energy_per_atom,phonon_IDs,"
                        "energy_type,run_type")
    print(f"[mp] summary: {len(summ)}/{len(todo)}", flush=True)
    print("[mp] REST es (dos task_id)...", flush=True)
    es = rest_batched(session, key, "/materials/electronic_structure/", todo,
                      "material_id,dos")
    dos_task, es_missing = {}, []
    for m, d in es.items():
        try:
            t = (d.get("dos") or {}).get("task_id")
            if t:
                dos_task[m] = t
            else:
                es_missing.append(m)
        except Exception:  # noqa: BLE001
            es_missing.append(m)
    print(f"[mp] dos task_ids: {len(dos_task)}, missing: {len(es_missing)}", flush=True)

    print("[mp] Delta eDOS...", flush=True)
    edt = delta_table(EDOS_TABLE)
    edos = delta_fetch(edt, "identifier", sorted(set(dos_task.values())))
    print(f"[mp] edos rows: {len(edos)}", flush=True)
    inv_task = {}
    for m, t in dos_task.items():
        inv_task.setdefault(t, []).append(m)

    print("[mp] Delta phDOS...", flush=True)
    ph_ids = []
    for m in todo:
        pids = ((summ.get(m) or {}).get("phonon_IDs") or {})
        for _m, lst in pids.items():
            ph_ids.extend(lst or [])
    pdt = delta_table(PHDOS_TABLE)
    phdos = delta_fetch(pdt, "identifier", sorted(set(ph_ids)))
    print(f"[mp] phdos rows: {len(phdos)}", flush=True)
    inv_ph = {}
    for m in todo:
        pids = ((summ.get(m) or {}).get("phonon_IDs") or {})
        inv_ph[m] = {mm: [i for i in (lst or []) if i in phdos]
                     for mm, lst in pids.items()}

    prov = {"mp_db": "2026.04.13", "delta": "materialsproject-parsed/unsigned-s3",
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    n = 0
    with open(out_path, "a") as f:
        for m in todo:
            rec = {
                "mpid": m,
                "mp_summary": summ.get(m),
                "dos_task_id": dos_task.get(m),
                "edos_raw": [edos[t] for t in [dos_task[m]] if t in edos] or None,
                "phdos_raw": {mm: [phdos[i] for i in lst]
                              for mm, lst in inv_ph.get(m, {}).items()},
                "provenance": prov,
            }
            if rec["edos_raw"] is None:
                rec["edos_raw"] = None
            f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 25 == 0:
                f.flush()
                print(f"[mp] wrote {n}/{len(todo)}", flush=True)
    print(f"[mp] DONE wrote={n}", flush=True)


if __name__ == "__main__":
    main()
