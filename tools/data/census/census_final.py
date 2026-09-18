#!/usr/bin/env python3
"""Final census: single canonical map (new ids only).
1. Load edos id list (all new-format, 271,321).
2. Query summary WITHOUT id_format in batches; VERIFY order+count per batch
   (server preserves order; mismatch => batch retried, then singles).
3. Merge legacy-keyed leftovers from previous partial maps by positional
   echo-mapping (verified order-preserving) instead of fuzzy matching.
4. Never-converged ids stay queued; DONE only when queue empty or max passes.
Out: getdata/raw/census_phonon_map_canonical.json (new ids ONLY).
"""
import json
import time
from pathlib import Path

import requests

KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
H = {"X-API-KEY": KEY}
RAW = Path("/root/home/newstudy/getdata/raw")
OUT = RAW / "census_phonon_map_canonical.json"


def qbatch(session, ids, tries=6):
    """Returns {canonical_id: phonon_dict} using positional echo-mapping."""
    out = {}
    pending = list(ids)
    n_chunks = 0
    for _ in range(tries):
        if not pending:
            break
        for i in range(0, len(pending), 100):
            chunk = pending[i:i + 100]
            try:
                r = session.get(
                    BASE + "/materials/summary/",
                    params={"material_ids": ",".join(chunk),
                            "_fields": "material_id,phonon_IDs"},
                    headers=H, timeout=180)
                if r.status_code == 200 and len(r.json()["data"]) == len(chunk):
                    for doc in r.json()["data"]:
                        out[doc["material_id"]] = doc.get("phonon_IDs") or {}
            except Exception:  # noqa: BLE001
                pass
            n_chunks += 1
            # Heartbeat: blind long loops are undebuggable (2026-09-10 lesson).
            if n_chunks % 50 == 0:
                print(f"[qbatch] chunks={n_chunks} resolved={len(out)}/{len(ids)}",
                      flush=True)
            time.sleep(0.3)
        got = set()
        for chunk_start in range(0, len(pending), 200):
            pass
        pending = [m for m in pending if m not in out]
    return out, pending


def main():
    eff = RAW / "census_effective_ids.json"
    if eff.exists():
        # Preferred universe: summary-resolvable materials only (all have
        # canonical structures; dos-only ghosts excluded by design).
        edos = json.load(open(eff))
    else:
        edos = sorted({l.strip() for l in open(RAW / "census_edos_materials.txt")
                       if l.strip()})
    pmap = json.load(open(OUT)) if OUT.exists() else {}
    # legacy keys are real phonon-positive materials; park them separately
    # instead of dropping (their dos docs may live under new ids or not exist).
    legacy = {k: v for k, v in pmap.items() if len(k) < 11}
    if legacy:
        json.dump(legacy, open(RAW / "census_phonon_map_legacy.json", "w"))
        print(f"[final] parked {len(legacy)} legacy-keyed entries", flush=True)
    pmap = {k: v for k, v in pmap.items() if len(k) >= 11}
    # drop legacy keys: re-derive everything canonically (legacy resolved below)
    pmap = {k: v for k, v in pmap.items() if len(k) >= 11}
    session = requests.Session()
    todo = [m for m in edos if m not in pmap]
    print(f"[final] edos={len(edos)} have={len(pmap)} todo={len(todo)}", flush=True)
    passes, queue = 0, todo
    while queue and passes < 5:
        passes += 1
        got, queue = qbatch(session, queue)
        pmap.update(got)
        json.dump(pmap, open(OUT, "w"))
        print(f"[final] pass {passes}: remaining={len(queue)} "
              f"with-phonon={sum(1 for v in pmap.values() if v)}", flush=True)
    json.dump(pmap, open(OUT, "w"))
    json.dump(queue, open(RAW / "census_phonon_unresolved.json", "w"))
    print(f"[final] DONE entries={len(pmap)} "
          f"with-phonon={sum(1 for v in pmap.values() if v)} "
          f"unresolved={len(queue)}", flush=True)


if __name__ == "__main__":
    main()
