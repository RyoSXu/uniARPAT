#!/usr/bin/env python3
"""断点续传下载器(通用): Range续传 + 重试, 给figshare大文件用."""
import sys
import time

import requests


def fetch(url, dest, tries=20):
    have = dest.stat().st_size if dest.exists() else 0
    s = requests.Session()
    for a in range(tries):
        try:
            h = {"Range": f"bytes={have}-"} if have else {}
            with s.get(url, headers=h, stream=True, timeout=180) as r:
                if have and r.status_code != 206:
                    print("[dl] server ignores Range, restart", flush=True)
                    have = 0
                total = r.headers.get("Content-Length")
                mode = "ab" if have and r.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(dest, mode) as f:
                    for ch in r.iter_content(chunk_size=1 << 20):
                        if ch:
                            f.write(ch)
                            have += len(ch)
                print(f"[dl] done bytes={have} total_hint={total}", flush=True)
                return True
        except Exception as e:
            print(f"[dl] attempt {a}: {type(e).__name__} have={have}", flush=True)
            time.sleep(10)
    return False


if __name__ == "__main__":
    from pathlib import Path
    ok = fetch(sys.argv[1], Path(sys.argv[2]))
    sys.exit(0 if ok else 1)
