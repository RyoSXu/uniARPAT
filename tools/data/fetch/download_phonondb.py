#!/usr/bin/env python3
"""Bulk PhononDB (NIMS MDR) downloader: 10k zips, threaded, resume-safe.
Manifest: getdata/raw/phonondb/manifest.jsonl ({id, url, file, bytes, status}).
Skip rule: final file exists => skip (integrity checked at processing stage).
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REF = Path("/root/home/newstudy/getdata/ref/download_list.md")
DEST = Path("/root/home/newstudy/getdata/raw/phonondb")
WORKERS = 6


def links():
    out = []
    for line in open(REF, errors="ignore"):
        m = re.search(r"https://mdr\.nims\.go\.jp/download_all/([0-9a-z]+\.zip)", line)
        if m:
            out.append((m.group(1)[:-4], m.group(0)))
    # dedupe, keep order
    seen, res = set(), []
    for k, u in out:
        if k not in seen:
            seen.add(k)
            res.append((k, u))
    return res


def fetch(session, key, url):
    dest = DEST / f"{key}.zip"
    if dest.exists():
        return "skip-exists", dest.stat().st_size
    part = DEST / f"{key}.zip.part"
    for _ in range(8):
        try:
            have = part.stat().st_size if part.exists() else 0
            h = {"Range": f"bytes={have}-"} if have else {}
            with session.get(url, headers=h, stream=True, timeout=180) as r:
                if have and r.status_code != 206:
                    have = 0
                mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    for ch in r.iter_content(chunk_size=1 << 20):
                        f.write(ch)
            part.rename(dest)
            return "ok", dest.stat().st_size
        except Exception:  # noqa: BLE001
            time.sleep(10)
    return "failed", part.stat().st_size if part.exists() else 0


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    all_links = links()
    print(f"[mdr] total={len(all_links)}", flush=True)
    done = {}
    man_f = DEST / "manifest.jsonl"
    if man_f.exists():
        with open(man_f) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except Exception:  # noqa: BLE001
                    pass
    todo = [(k, u) for k, u in all_links
            if k not in done or done[k].get("status") != "ok"]
    print(f"[mdr] todo={len(todo)} (have={len(all_links) - len(todo)})", flush=True)
    session = requests.Session()
    n_bytes = [0]

    def one(pair):
        k, u = pair
        st, sz = fetch(session, k, u)
        n_bytes[0] += sz if st == "ok" else 0
        return {"id": k, "url": u, "bytes": sz, "status": st}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        with open(man_f, "a") as mf:
            for i, rec in enumerate(ex.map(lambda p: one(p), todo)):
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                if (i + 1) % 200 == 0:
                    print(f"[mdr] {i + 1}/{len(todo)} "
                          f"MB={n_bytes[0] / 1e6:.0f}", flush=True)
    print(f"[mdr] DONE new-MB={n_bytes[0] / 1e6:.0f}", flush=True)


if __name__ == "__main__":
    main()
