#!/usr/bin/env python3
"""A7c JARVIS谱拉取: JVASP清单 -> eDOS/phDOS raw落盘.

输入: --list (jvasp id的json数组, 如jarvis_threesrc_jvasp.json)
      --kinds electron|phonon|both
输出: --out jsonl ({jvasp, legacy_mpid, canonical_mpid, formula_pretty,
      jarvis_edos_raw, jarvis_phdos_raw, fetch_errors, provenance})
复用pilot路径(jarvis.db.webpages.Webpage, 一页 serving 两种谱).
断点续跑(jvasp key) + 多线程 + 心跳. 谱缺失记None+errors, 永不丢行.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW = Path("/root/home/newstudy/getdata/raw")


def fetch_one(jid, kinds):
    from jarvis.db.webpages import Webpage
    rec = {"jvasp": jid, "jarvis_edos_raw": None, "jarvis_phdos_raw": None,
           "fetch_errors": []}
    try:
        w = Webpage(jid=jid)
    except Exception as e:
        rec["fetch_errors"].append(f"page-{type(e).__name__}:{str(e)[:120]}")
        return rec
    if kinds in ("electron", "both"):
        try:
            d = w.get_dft_electron_dos()
            if d:
                rec["jarvis_edos_raw"] = d
            else:
                rec["fetch_errors"].append(f"empty-electron-{jid}")
        except Exception as e:
            rec["fetch_errors"].append(f"failed-electron-{type(e).__name__}")
    if kinds in ("phonon", "both"):
        try:
            d = w.get_dft_phonon_dos()
            if d:
                rec["jarvis_phdos_raw"] = d
            else:
                rec["fetch_errors"].append(f"empty-phonon-{jid}")
        except Exception as e:
            rec["fetch_errors"].append(f"failed-phonon-{type(e).__name__}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--kinds", default="both", choices=["electron", "phonon", "both"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    jids = json.load(open(args.list))
    if args.limit:
        jids = jids[:args.limit]
    # 映射富化
    meta = {}
    with open(RAW / "jarvis_a7b_map.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta[r["jvasp"]] = r
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["jvasp"])
                except Exception:
                    pass
    todo = [j for j in jids if j not in done]
    print(f"[a7c] total={len(jids)} done={len(done)} todo={len(todo)} "
          f"kinds={args.kinds} workers={args.workers}", flush=True)
    prov = {"jarvis": "webpages-live",
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    n = [0]

    def _one(jid):
        r = fetch_one(jid, args.kinds)
        m = meta.get(jid, {})
        r["legacy_mpid"] = m.get("legacy_mpid")
        r["canonical_mpid"] = m.get("canonical_mpid")
        r["formula_pretty"] = m.get("formula_pretty")
        r["provenance"] = prov
        return r

    f = open(out_path, "a")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_one, todo):
                f.write(json.dumps(r) + "\n")
                n[0] += 1
                if n[0] % 25 == 0:
                    f.flush()
                if n[0] % 100 == 0:
                    print(f"[a7c] {n[0]}/{len(todo)}", flush=True)
    finally:
        f.flush()
        f.close()
    print(f"[a7c] DONE wrote={n[0]} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
