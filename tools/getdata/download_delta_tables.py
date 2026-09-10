#!/usr/bin/env python3
"""Resume-capable bulk download of MP Delta tables (unsigned public S3).

Downloads entire table prefixes (data + _delta_log) so tables can be
opened locally with deltalake forever. Safe to re-run: skips complete files
(size match), resumes partial files via Range.

Usage:
  python3 download_delta_tables.py --dest ../raw/delta_mp
"""
import argparse
import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://s3.us-east-1.amazonaws.com/materialsproject-parsed"
TABLES = ["core/electronic-structure/total-dos/",
          "phonon/electronic-structure/dos/"]
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
from urllib.parse import quote


def s3_list(prefix):
    files, tok = [], None
    while True:
        u = BASE + "/?list-type=2&prefix=" + quote(prefix, safe="/=") + "&max-keys=1000"
        if tok:
            u += "&continuation-token=" + quote(tok, safe="")
        for a in range(6):
            try:
                x = urllib.request.urlopen(u, timeout=90).read()
                break
            except Exception:
                time.sleep(10)
        else:
            raise RuntimeError("LIST failed for " + prefix)
        r = ET.fromstring(x)
        for c in r.findall("s:Contents", NS):
            files.append((c.find("s:Key", NS).text,
                          int(c.find("s:Size", NS).text),
                          (c.find("s:ETag", NS).text or "").strip('"')))
        if r.findtext("s:IsTruncated", namespaces=NS) != "true":
            break
        tok = r.findtext("s:NextContinuationToken", namespaces=NS)
    return files


def fetch(url, dest, size):
    import requests
    for a in range(15):
        try:
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            if have == size:
                return True
            h = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=h, stream=True, timeout=120) as r:
                if have and r.status_code != 206:
                    have = 0
                    r = requests.get(url, stream=True, timeout=120)
                    r.raise_for_status()
                mode = "ab" if have else "wb"
                with open(dest, mode) as f:
                    for ch in r.iter_content(chunk_size=1 << 20):
                        f.write(ch)
            if os.path.getsize(dest) == size:
                return True
            print(f"  size mismatch {os.path.getsize(dest)}/{size}, retry", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  retry {a}: {str(e)[:90]}", flush=True)
            time.sleep(12)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    dest = Path(args.dest)
    manifest = {}
    for prefix in TABLES:
        print(f"[dl] LIST {prefix}", flush=True)
        files = s3_list(prefix)
        tot = sum(s for _, s, _ in files)
        print(f"[dl] {len(files)} files, {tot/1e9:.2f} GB", flush=True)
        for key, size, etag in files:
            # keys may contain literal %2B etc.; use exact key for URL path
            url = BASE + "/" + quote(key, safe="/")
            local = dest / key
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists() and local.stat().st_size == size:
                print(f"  skip {key.split('/')[-1]}", flush=True)
                ok = True
            else:
                print(f"  get {key.split('/')[-1]} ({size/1e6:.0f} MB)...", flush=True)
                ok = fetch(url, str(local), size)
            manifest[key] = {"size": size, "etag": etag, "ok": ok,
                             "done_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()) if ok else None}
            if not ok:
                print(f"  FAILED {key} (will retry next run)", flush=True)
    import json
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    bad = [k for k, v in manifest.items() if not v["ok"]]
    print(f"[dl] DONE ok={len(manifest)-len(bad)} failed={len(bad)}", flush=True)


if __name__ == "__main__":
    main()
