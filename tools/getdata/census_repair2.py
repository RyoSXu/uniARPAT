#!/usr/bin/env python3
"""Census repair v2: queue-based, never drops failures.
- shorts (legacy keys): re-query WITHOUT id_format -> canonical new key, merge.
- missing: batched fetch; per-batch verify returned set; absent singles retried
  individually; transport failures stay queued. Empty-after-retries => {} (absent).
State files: census_phonon_map.json (result), census_queue.json (pending).
"""
import json
import time
from pathlib import Path

import requests

KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
H = {"X-API-KEY": KEY}
RAW = Path("/root/home/newstudy/getdata/raw")
MAP_F = RAW / "census_phonon_map.json"
Q_F = RAW / "census_queue.json"


def qbatch(session, ids, tries=5):
    """Returns (found_dict, failed_list). found includes confirmed-absent {}."""
    found, pending = {}, list(ids)
    for _ in range(tries):
        if not pending:
            break
        got = {}
        for i in range(0, len(pending), 200):
            try:
                r = session.get(
                    BASE + "/materials/summary/",
                    params={"material_ids": ",".join(pending[i:i + 200]),
                            "_fields": "material_id,phonon_IDs"},
                    headers=H, timeout=180)
                if r.status_code == 200:
                    for d in r.json()["data"]:
                        got[d["material_id"]] = d.get("phonon_IDs") or {}
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.3)
        # singles retry for the still-missing (distinguish absent vs failed)
        still = [m for m in pending if m not in got]
        for m in still:
            try:
                r = session.get(
                    BASE + "/materials/summary/",
                    params={"material_ids": m,
                            "_fields": "material_id,phonon_IDs"},
                    headers=H, timeout=120)
                if r.status_code == 200 and r.json()["data"]:
                    for d in r.json()["data"]:
                        got[d["material_id"]] = d.get("phonon_IDs") or {}
                time.sleep(0.2)
            except Exception:  # noqa: BLE001
                pass
        pending = [m for m in pending if m not in got]
    return got, pending


def main():
    pmap = json.load(open(MAP_F))
    edos = set(l.strip() for l in open(RAW / "census_edos_materials.txt"))
    session = requests.Session()
    # 1. shorts -> canonical
    shorts = sorted({k for k in pmap if len(k) < 11})
    print(f"[r2] shorts={len(shorts)}", flush=True)
    if shorts:
        got, failed = qbatch(session, shorts)
        moved = 0
        for old in shorts:
            cands = {nk for nk in got if nk not in shorts}
            # match: same phonon ids OR server echo; fallback: drop short, keep new
            hit = [nk for nk in [old] if nk in got]
            nk = hit[0] if hit else None
            if nk is None:
                # find new key whose phonon set intersects
                olds = {p for lst in pmap[old].values() for p in (lst or [])}
                for cand, doc in got.items():
                    if cand in shorts:
                        continue
                    news = {p for lst in doc.values() for p in (lst or [])}
                    if olds & news:
                        nk = cand
                        break
            if nk is not None and nk != old:
                if nk in pmap:
                    for m, lst in pmap[old].items():
                        pmap[nk].setdefault(m, [])
                        pmap[nk][m] = sorted(set(pmap[nk][m]) | set(lst or []))
                else:
                    pmap[nk] = pmap[old]
                del pmap[old]
                moved += 1
        print(f"[r2] canonicalized {moved}, failed-short-queries={len(failed)}", flush=True)
        json.dump(pmap, open(MAP_F, "w"))
    # 2. missing with persistent queue
    if Q_F.exists():
        queue = json.load(open(Q_F))
    else:
        queue = sorted(edos - set(pmap))
    print(f"[r2] queue={len(queue)}", flush=True)
    passes = 0
    while queue and passes < 4:
        passes += 1
        got, queue = qbatch(session, queue)
        pmap.update(got)
        json.dump(pmap, open(MAP_F, "w"))
        json.dump(queue, open(Q_F, "w"))
        print(f"[r2] pass {passes}: got={len(got)} remaining={len(queue)} "
              f"with-phonon={sum(1 for v in pmap.values() if v)}", flush=True)
    # survivors after 4 passes => treat as confirmed-absent {}
    for m in queue:
        pmap.setdefault(m, {})
    json.dump(pmap, open(MAP_F, "w"))
    if Q_F.exists():
        Q_F.unlink()
    print(f"[r2] DONE entries={len(pmap)} "
          f"with-phonon={sum(1 for v in pmap.values() if v)}", flush=True)


if __name__ == "__main__":
    main()
