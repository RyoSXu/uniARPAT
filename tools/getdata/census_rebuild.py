#!/usr/bin/env python3
"""Rebuild the MP phonon census from canonical eDOS material IDs.

The first census mixed legacy and current MP identifiers. This version starts
from the canonical IDs returned by the current DOS endpoint and only performs
batched summary queries. A failed batch remains in the queue for a later run;
no per-record retry storm is created.
"""
import json
import time
from pathlib import Path

import requests


KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
HEADERS = {"X-API-KEY": KEY}
RAW = Path("/root/home/newstudy/getdata/raw")
OUT = RAW / "census_phonon_map_canonical.json"
QUEUE = RAW / "census_phonon_queue.json"


def fetch_batch(session, ids, attempts=8):
    params = {
        "material_ids": ",".join(ids),
        "_fields": "material_id,phonon_IDs",
    }
    for attempt in range(attempts):
        try:
            response = session.get(
                f"{BASE}/materials/summary/",
                params=params,
                headers=HEADERS,
                timeout=180,
            )
            if response.status_code == 200:
                docs = response.json().get("data", [])
                return {
                    doc["material_id"]: doc.get("phonon_IDs") or {}
                    for doc in docs
                }
        except Exception:
            pass
        time.sleep(min(60, 5 * (attempt + 1)))
    return None


def main():
    edos_ids = sorted(
        {line.strip() for line in open(RAW / "census_edos_materials.txt") if line.strip()}
    )
    pmap = json.load(open(OUT)) if OUT.exists() else {}
    queue = json.load(open(QUEUE)) if QUEUE.exists() else [
        material_id for material_id in edos_ids if material_id not in pmap
    ]

    print(
        f"[rebuild] canonical eDOS={len(edos_ids)} existing={len(pmap)} "
        f"queue={len(queue)}",
        flush=True,
    )
    session = requests.Session()
    batch_count = 0
    failed_batches = []
    unverified = []  # requested but never echoed by server (see below)

    while queue:
        batch = queue[:200]
        result = fetch_batch(session, batch)
        if result is None:
            failed_batches.append(batch)
        else:
            pmap.update(result)
            for m in batch:
                if m not in result:
                    unverified.append(m)
        queue = queue[200:]
        batch_count += 1
        if batch_count % 25 == 0 or not queue:
            json.dump(pmap, open(OUT, "w"))
            json.dump(
                [item for group in failed_batches for item in group] + queue,
                open(QUEUE, "w"),
            )
            json.dump(unverified, open(RAW / "census_unverified.json", "w"))
            with_phonon = sum(1 for value in pmap.values() if value)
            print(
                f"[rebuild] batches={batch_count} remaining={len(queue)} "
                f"failed_ids={sum(map(len, failed_batches))} "
                f"unverified={len(unverified)} "
                f"with-phonon={with_phonon}",
                flush=True,
            )
        time.sleep(0.3)

    # Unresolved/unverified ids are NEVER marked {} (that would fake "no phonon").
    # Rule (2026-09-10): no summary doc => no canonical structure => excluded
    # from v2 training universe (kept in census_unresolved.json with reason).
    json.dump(pmap, open(OUT, "w"))
    json.dump([item for group in failed_batches for item in group], open(QUEUE, "w"))
    json.dump(unverified, open(RAW / "census_unverified.json", "w"))
    print(
        f"[rebuild] DONE entries={len(pmap)} "
        f"with-phonon={sum(1 for v in pmap.values() if v)} "
        f"failed_ids={sum(map(len, failed_batches))} "
        f"unverified={len(unverified)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
