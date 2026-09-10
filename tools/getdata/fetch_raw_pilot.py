#!/usr/bin/env python3
"""v2 raw-data pilot fetcher: MP structures + JARVIS raw eDOS/phDOS, full raw archive.

Contract (per user decision 2026-09-09):
  - Store FULL raw responses (no rebinning, no clipping) + provenance per record.
  - Downstream binning is a *view*; raw archive is never rewritten.
  - Resume-safe (skip mpids already in output), throttled, retried.
  - MP API key via env MP_API_KEY (never committed).

Usage:
  MP_API_KEY=... python3 fetch_raw_pilot.py --n 100 --out ../raw/v2_pilot.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

MP_BASE = "https://api.materialsproject.org"
REF_DIR = Path("/root/home/newstudy/getdata/ref")  # outside repo; see tools/getdata/README


def load_mapping():
    with open(REF_DIR / "jarvis_to_mp_mapping.json") as f:
        j2m = json.load(f)
    m2j = {}
    for j, m in j2m.items():
        m2j.setdefault(str(m), j)
    return m2j


def mp_summary_batch(session, key, mpids, fields):
    out = {}
    for i in range(0, len(mpids), 50):
        chunk = mpids[i:i + 50]
        for attempt in range(4):
            try:
                r = session.get(
                    MP_BASE + "/materials/summary/",
                    params={"material_ids": ",".join(chunk),
                            "_fields": fields, "id_format": "legacy"},
                    headers={"X-API-KEY": key}, timeout=120)
                if r.status_code == 200:
                    for d in r.json()["data"]:
                        out[d["material_id"]] = d
                    break
                time.sleep(5 * (attempt + 1))
            except Exception:
                time.sleep(5 * (attempt + 1))
        time.sleep(0.3)
    return out


def jarvis_raw(jid, kind):
    """kind: 'electron' | 'phonon'. Returns raw dict or {'_error': ...}."""
    from jarvis.db.webpages import Webpage
    for attempt in range(3):
        try:
            w = Webpage(jid=jid)
            fn = w.get_dft_electron_dos if kind == "electron" else w.get_dft_phonon_dos
            d = fn()
            time.sleep(0.5)
            if not d:
                return {"_error": f"empty-{kind}-{jid}"}
            return d
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    return {"_error": f"failed-{kind}-{jid}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mp_index", type=str,
                    default=str(Path(__file__).resolve().parents[2]
                                / "data/train4ARPAT/test/test_index.npy"))
    args = ap.parse_args()
    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit("MP_API_KEY not set")

    import numpy as np
    mpids = [str(x) for x in np.load(args.mp_index, allow_pickle=True)][:args.n]
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
    print(f"[pilot] total={len(mpids)} done={len(done)} todo={len(todo)}", flush=True)

    m2j = load_mapping()
    session = requests.Session()
    structs = mp_summary_batch(
        session, key, todo,
        "material_id,structure,band_gap,efermi,energy_above_hull,nsites,"
        "symmetry,formation_energy_per_atom")
    print(f"[pilot] MP structures fetched: {len(structs)}/{len(todo)}", flush=True)

    n_ok = n_skip_nomap = n_err = 0
    with open(out_path, "a") as f:
        for mpid in todo:
            jid = m2j.get(mpid)
            if jid is None:
                n_skip_nomap += 1
                continue
            edos = jarvis_raw(jid, "electron")
            phdos = jarvis_raw(jid, "phonon")
            errs = [e.get("_error") for e in (edos, phdos)
                    if isinstance(e, dict) and e.get("_error")]
            rec = {
                "mpid": mpid, "jid": jid,
                "mp_summary": structs.get(mpid),
                "jarvis_edos_raw": edos if not edos.get("_error") else None,
                "jarvis_phdos_raw": phdos if not phdos.get("_error") else None,
                "fetch_errors": errs,
                "provenance": {
                    "mp_db": "2026.04.13", "jarvis": "webpages-live",
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            }
            if rec["jarvis_edos_raw"] is None and rec["jarvis_phdos_raw"] is None:
                n_err += 1
            else:
                n_ok += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (n_ok + n_err) % 20 == 0:
                print(f"[pilot] progress ok={n_ok} err={n_err} nomap={n_skip_nomap}",
                      flush=True)
    print(f"[pilot] DONE ok={n_ok} err={n_err} nomap={n_skip_nomap}", flush=True)


if __name__ == "__main__":
    main()
