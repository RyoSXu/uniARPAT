#!/usr/bin/env python3
"""Bulk MP structures for pretraining corpus (no spectra needed).
Paginates /materials/summary/ with structure field; raw JSONL sharded;
resume-safe (skip count persisted). Run: nohup ... & (hours).
"""
import argparse
import json
import time
from pathlib import Path

import requests

KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
H = {"X-API-KEY": KEY}
RAW = Path("/root/home/newstudy/getdata/raw/pretrain_structures")
FIELDS = ("material_id,structure,nsites,nelements,composition_reduced,"
          "symmetry,formation_energy_per_atom,band_gap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max", type=int, default=0,
                    help="max docs, 0 = all")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    state_f = RAW / "_state.json"
    skip = 0
    if state_f.exists():
        skip = json.load(open(state_f)).get("skip", 0)
    print(f"[struct] resume skip={skip}", flush=True)
    session = requests.Session()
    shard, n_in_shard, total = None, 0, 0
    while True:
        if args.max and skip >= args.max:
            break
        for a in range(8):
            try:
                r = session.get(
                    BASE + "/materials/summary/",
                    params={"_fields": FIELDS, "_skip": skip,
                            "_limit": args.limit},
                    headers=H, timeout=240)
                if r.status_code == 200:
                    data = r.json()["data"]
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(10)
        else:
            print(f"[struct] page failed skip={skip}, retry later", flush=True)
            json.dump({"skip": skip}, open(state_f, "w"))
            time.sleep(120)
            continue
        if not data:
            break
        if shard is None or n_in_shard >= 5000:
            if shard:
                shard.close()
            shard = open(RAW / f"structures_{skip:08d}.jsonl", "a")
            n_in_shard = 0
        for d in data:
            shard.write(json.dumps(d) + "\n")
        n_in_shard += len(data)
        total += len(data)
        skip += len(data)
        if total % 5000 < args.limit:
            shard.flush()
            json.dump({"skip": skip}, open(state_f, "w"))
            print(f"[struct] total={total} skip={skip}", flush=True)
        time.sleep(0.3)
    if shard:
        shard.close()
    json.dump({"skip": skip}, open(state_f, "w"))
    print(f"[struct] DONE total={total}", flush=True)


if __name__ == "__main__":
    main()
