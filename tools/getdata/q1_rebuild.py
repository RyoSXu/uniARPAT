"""Q1 rebuild: v2_processed.parquet minus quarantine -> data/train4ARPAT (A5 mirror).

Usage: q1_rebuild.py --out data/train4ARPAT [--check]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path("/root/home/newstudy/uniARPAT")
RAW = Path("/root/home/newstudy/getdata/raw")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/train4ARPAT"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    q = json.load(open(REPO / "data/quarantine_q1.json"))
    qset = set(q["mpids"])
    print(f"[q1] quarantine {q['version']}: n={len(qset)}", flush=True)
    t = pq.read_table(RAW / "v2_processed.parquet").to_pylist()
    kept = [r for r in t if r["mpid"] not in qset]
    print(f"[q1] rows {len(t)} -> {len(kept)} (removed {len(t)-len(kept)})", flush=True)
    assert len(t) - len(kept) == len(qset), "quarantine coverage mismatch"
    by = {"train": [], "valid": [], "test": []}
    for r in kept:
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
        nats = (el[:, 2:] > 0).sum(axis=1)
        buckets, edges = np.histogram(nats, bins=[0, 8, 16, 32, 80, 10**9])
        manifest["splits"][s]["buckets"] = buckets.tolist()
        print(f"[q1] {s}: n={len(rows)} atoms_med={float(np.median(nats)):.0f}",
              flush=True)
    manifest["stats_v2"] = json.load(open(RAW / "stats_v2.json"))
    manifest["split"] = json.load(open(RAW / "split_report.json"))["n"]
    manifest["quarantine"] = {"version": q["version"], "criterion": q["criterion"],
                              "date": q["date"], "n_removed": len(qset),
                              "per_split_removed": q["per_split"],
                              "list": "../quarantine_q1.json"}
    manifest["code_sha"] = sha(Path(__file__))
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    shutil.copy(RAW / "stats_v2.json", out / "stats_v2.json")
    print("[q1] cache written", flush=True)
    if args.check:
        import sys
        sys.path.insert(0, str(REPO))
        from datasets.dataset import Dos_Dataset as DD
        for s in ("train", "valid", "test"):
            ds = DD(data_dir=str(out), split=s)
            assert len(ds) == manifest["splits"][s]["n"]
            g = ds[0]
            assert g[0].shape[0] == 82 and g[2].shape[0] == 128 and g[3].shape[0] == 64
            assert len(g) == 15 and g[14] is not None, "nvalence sidecar missing?"
            print(f"[q1-check] {s}: len={len(ds)} ok", flush=True)
        ed = np.load(out / "train/edos_tgtdos_train.npy")[:5]
        assert np.isfinite(ed).all()
        print("[q1-check] finite ok", flush=True)


if __name__ == "__main__":
    main()
