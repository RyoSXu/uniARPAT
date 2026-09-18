#!/usr/bin/env python3
"""A8: 逐样本canonical CIF生成(原胞symprec=0.1标准原胞).

输入: v2_processed.parquet(pool mpids) + mp_structures.jsonl + mp_increment_raw.jsonl
输出: getdata/raw/v2_cifs/{shard}/{mpid}.cif + v2_cifs.zip + a8_cif_report.json
断点续跑(文件存在即跳过). 与A4同一结构路径, round-trip已1000/1000.
"""
import json
import time
import zipfile
from pathlib import Path

RAW = Path("/root/home/newstudy/getdata/raw")
CIFD = RAW / "v2_cifs"


def main():
    import pyarrow.parquet as pq
    pool = pq.read_table(RAW / "v2_processed.parquet", columns=["mpid"]).column("mpid").to_pylist()
    print(f"[a8] pool={len(pool)}", flush=True)
    summ = {}
    with open(RAW / "mp_structures.jsonl") as f:
        for line in f:
            r = json.loads(line)
            summ[r["mpid"]] = r["summary"]["structure"]
    with open(RAW / "mp_increment_raw.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["mp_summary"] and r["mpid"] not in summ:
                summ[r["mpid"]] = r["mp_summary"]["structure"]
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    import warnings
    warnings.filterwarnings("ignore")
    n_ok = n_skip = n_fail = 0
    fails = []
    t0 = time.time()
    for i, m in enumerate(pool):
        shard = CIFD / m[3:5]
        dest = shard / f"{m}.cif"
        if dest.exists():
            n_skip += 1
            continue
        try:
            s = summ[m]
            st0 = Structure(Lattice(s["lattice"]["matrix"]),
                            [x["label"] for x in s["sites"]],
                            [x["abc"] for x in s["sites"]])
            prim = SpacegroupAnalyzer(st0, symprec=0.1).get_primitive_standard_structure()
            shard.mkdir(parents=True, exist_ok=True)
            CifWriter(prim, symprec=0.1).write_file(str(dest))
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if len(fails) < 20:
                fails.append((m, f"{type(e).__name__}:{str(e)[:100]}"))
        if (i + 1) % 2000 == 0:
            print(f"[a8] {i + 1}/{len(pool)} ok={n_ok} skip={n_skip} fail={n_fail} "
                  f"{round(time.time() - t0, 0)}s", flush=True)
    print(f"[a8] DONE ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
    # 打包
    zpath = RAW / "v2_cifs.zip"
    if n_fail == 0:
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for c in sorted(CIFD.glob("*/*.cif")):
                z.write(c, f"v2_cifs/{c.parent.name}/{c.name}")
        import os
        print(f"[a8] zip {round(os.path.getsize(zpath) / 1e6, 1)}MB", flush=True)
    json.dump({"pool": len(pool), "ok": n_ok, "skip": n_skip, "fail": n_fail,
               "fails": fails}, open(RAW / "a8_cif_report.json", "w"), indent=1)
    print("[a8] report written", flush=True)


if __name__ == "__main__":
    main()
