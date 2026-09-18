#!/usr/bin/env python3
"""A5落盘: v2_processed.parquet(+split) -> data/train4ARPAT-v2/ npy训练缓存.

布局与v1同构: {train,valid,test}/{elements,positions,edos_tgtdos,phdos_tgtdos}_{s}.npy
  + {s}_index.npy(mpid) + edos/phdos_mask_{s}.npy(sidecar, C2消费) + bucket_idx.json(E6)
  + manifest.json(行数/sha/stats_v2/代码hash).
用法: build_v2_cache.py --out data/train4ARPAT-v2 [--check]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
RAW = Path("/root/home/newstudy/getdata/raw")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/train4ARPAT-v2"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    t = pq.read_table(RAW / "v2_processed.parquet").to_pylist()
    by = {"train": [], "valid": [], "test": []}
    for r in t:
        by[r["split"]].append(r)
    manifest = {"splits": {}, "files": {}}
    for s, rows in by.items():
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        el = np.array([r["elements"] for r in rows], dtype=np.int64)
        pos = np.array([r["positions"] for r in rows], dtype=np.float32)
        ed = np.array([r["edos"] for r in rows], dtype=np.float32)
        ph = np.array([r["phdos"] for r in rows], dtype=np.float32)
        em = np.array([r["edos_mask"] for r in rows], dtype=np.int8)
        pm = np.array([r["phdos_mask"] for r in rows], dtype=np.int8)
        idx = np.array([r["mpid"] for r in rows])
        assert el.shape[1] == 82 and pos.shape[1] == 246
        assert ed.shape[1] == 128 and ph.shape[1] == 64
        files = {f"elements_{s}.npy": el, f"positions_{s}.npy": pos,
                 f"edos_tgtdos_{s}.npy": ed, f"phdos_tgtdos_{s}.npy": ph,
                 f"edos_mask_{s}.npy": em, f"phdos_mask_{s}.npy": pm,
                 f"{s}_index.npy": idx}
        for fn, arr in files.items():
            np.save(d / fn, arr)
            manifest["files"][f"{s}/{fn}"] = {"shape": list(arr.shape),
                                              "sha": sha(d / fn)}
        manifest["splits"][s] = {"n": len(rows)}
        # 长度分桶(E6用)
        nats = (el[:, 2:] > 0).sum(axis=1)
        buckets, edges = np.histogram(nats, bins=[0, 8, 16, 32, 80, 10**9])
        manifest["splits"][s]["buckets"] = buckets.tolist()
        print(f"[a5] {s}: n={len(rows)} atoms_med={float(np.median(nats)):.0f}",
              flush=True)
    manifest["stats_v2"] = json.load(open(RAW / "stats_v2.json"))
    manifest["split"] = json.load(open(RAW / "split_report.json"))["n"]
    manifest["code_sha"] = sha(Path(__file__))
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    import shutil
    shutil.copy(RAW / "stats_v2.json", out / "stats_v2.json")
    print("[a5] cache written", flush=True)
    if args.check:
        import sys
        sys.path.insert(0, str(REPO))
        from datasets.dataset import Dos_Dataset as DD
        for s in ("train", "valid", "test"):
            ds = DD(data_dir=str(out), split=s)
            assert len(ds) == manifest["splits"][s]["n"]
            g = ds[0]
            assert g[0].shape[0] == 82 and g[2].shape[0] == 128 and g[3].shape[0] == 64
            print(f"[a5-check] {s}: len={len(ds)} ok", flush=True)
        # parquet对照抽查
        ed = np.load(out / "train/edos_tgtdos_train.npy")[:5]
        assert np.isfinite(ed).all()
        print("[a5-check] finite ok", flush=True)


if __name__ == "__main__":
    main()
