#!/usr/bin/env python3
"""A3 census: eDOS material list (dos-route pagination) + phonon_IDs via summary batches.

Outputs (incremental, resume-safe):
  getdata/raw/census_edos_materials.txt  (one legacy-or-new mp-id per line)
  getdata/raw/census_phonon_map.json     (mpid -> {method: [phonon ids]})
"""
import json
import time
from pathlib import Path

import requests

KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
H = {"X-API-KEY": KEY}
RAW = Path("/root/home/newstudy/getdata/raw")


def get(url, params, tries=6):
    for a in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=120)
            if r.status_code == 200:
                return r.json()
            time.sleep(6 * (a + 1))
        except Exception:  # noqa: BLE001
            time.sleep(6 * (a + 1))
    return None


def main():
    # 1. eDOS material census (id-only pages, cheap)
    edos_f = RAW / "census_edos_materials.txt"
    seen = set()
    if edos_f.exists():
        seen = set(l.strip() for l in open(edos_f) if l.strip())
    if not seen:
        skip, total = 0, None
        with open(edos_f, "a") as f:
            while True:
                j = get(BASE + "/materials/electronic_structure/dos/",
                        {"_fields": "material_id", "_skip": skip, "_limit": 1000,
                         "deprecated": False})
                if not j:
                    break
                total = j["meta"]["total_doc"]
                data = j["data"]
                if not data:
                    break
                for d in data:
                    m = d["material_id"]
                    if m not in seen:
                        seen.add(m)
                        f.write(m + "\n")
                skip += len(data)
                if skip % 20000 < 1000:
                    print(f"[census] edos ids: {len(seen)}/{total}", flush=True)
                if skip >= total:
                    break
    print(f"[census] eDOS materials: {len(seen)}", flush=True)

    # 2. phonon_IDs via summary batches of 200
    pmap_f = RAW / "census_phonon_map.json"
    pmap = {}
    if pmap_f.exists():
        pmap = json.load(open(pmap_f))
    ids = sorted(seen - set(pmap))
    print(f"[census] summary batches todo: {len(ids)}", flush=True)
    sess = requests.Session()
    n = 0
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        # NOTE: no id_format forcing — the id list mixes legacy (mp-149) and
        # new (mp-aaa...) ids; forcing legacy resolves new ids to empty.
        for a in range(6):
            try:
                r = sess.get(BASE + "/materials/summary/",
                             params={"material_ids": ",".join(chunk),
                                     "_fields": "material_id,phonon_IDs"},
                             headers=H, timeout=180)
                if r.status_code == 200:
                    for d in r.json()["data"]:
                        p = d.get("phonon_IDs") or {}
                        if any(p.values()):
                            pmap[d["material_id"]] = p
                        else:
                            pmap[d["material_id"]] = {}
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(8)
        n += 1
        if n % 25 == 0:
            json.dump(pmap, open(pmap_f, "w"))
            print(f"[census] batches {n}, with-phonon: {sum(1 for v in pmap.values() if v)}",
                  flush=True)
        time.sleep(0.3)
    json.dump(pmap, open(pmap_f, "w"))
    print(f"[census] DONE materials={len(seen)} with-phonon={sum(1 for v in pmap.values() if v)}",
          flush=True)


if __name__ == "__main__":
    main()
