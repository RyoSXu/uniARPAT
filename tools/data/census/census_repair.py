#!/usr/bin/env python3
"""Census repair: (1) normalize legacy pmap keys -> canonical new ids, dedupe;
(2) fetch phonon_IDs for ids never successfully queried. Resume-safe."""
import json
import time
from pathlib import Path

import requests

KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
H = {"X-API-KEY": KEY}
RAW = Path("/root/home/newstudy/getdata/raw")


def batched(session, ids, fields, chunk=200, tries=6):
    out = {}
    for i in range(0, len(ids), chunk):
        for a in range(tries):
            try:
                r = session.get(
                    BASE + "/materials/summary/",
                    params={"material_ids": ",".join(ids[i:i + chunk]),
                            "_fields": fields},
                    headers=H, timeout=180)
                if r.status_code == 200:
                    for d in r.json()["data"]:
                        out[d["material_id"]] = d
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(8)
        time.sleep(0.3)
    return out


def main():
    pmap = json.load(open(RAW / "census_phonon_map.json"))
    edos = set(l.strip() for l in open(RAW / "census_edos_materials.txt"))
    shorts = sorted({k for k in pmap if len(k) < 11})
    print(f"[repair] pmap={len(pmap)} shorts={len(shorts)}", flush=True)
    session = requests.Session()
    # 1. canonicalize shorts -> new ids (default format echoes canonical key)
    canon = {}
    if shorts:
        got = batched(session, shorts, "material_id,phonon_IDs")
        for k in shorts:
            pass
        # match by phonon ids (stable across formats)
        by_ph = {}
        for nk, doc in got.items():
            for lst in ((doc.get("phonon_IDs") or {}).values()):
                for pid in (lst or []):
                    by_ph.setdefault(pid, nk)
        moved = 0
        for k in shorts:
            pids = {p for lst in pmap[k].values() for p in (lst or [])}
            targets = {by_ph[p] for p in pids if p in by_ph}
            if len(targets) == 1:
                nk = targets.pop()
                if nk != k:
                    if nk in pmap:
                        for m, lst in pmap[k].items():
                            pmap[nk].setdefault(m, [])
                            pmap[nk][m] = sorted(set(pmap[nk][m]) | set(lst or []))
                    else:
                        pmap[nk] = pmap[k]
                    del pmap[k]
                    moved += 1
        print(f"[repair] canonicalized {moved}/{len(shorts)} shorts", flush=True)
        json.dump(pmap, open(RAW / "census_phonon_map.json", "w"))
    # 2. fetch never-queried ids
    missing = sorted(edos - set(pmap))
    print(f"[repair] missing to fetch: {len(missing)}", flush=True)
    n = 0
    for i in range(0, len(missing), 200):
        got = batched(session, missing[i:i + 200], "material_id,phonon_IDs")
        for nk, doc in got.items():
            pmap[nk] = doc.get("phonon_IDs") or {}
        n += 1
        if n % 25 == 0:
            json.dump(pmap, open(RAW / "census_phonon_map.json", "w"))
            print(f"[repair] batches {n}, with-phonon: "
                  f"{sum(1 for v in pmap.values() if v)}", flush=True)
    json.dump(pmap, open(RAW / "census_phonon_map.json", "w"))
    print(f"[repair] DONE entries={len(pmap)} "
          f"with-phonon={sum(1 for v in pmap.values() if v)}", flush=True)


if __name__ == "__main__":
    main()
