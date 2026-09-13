#!/usr/bin/env python3
"""Delta eDOS归属索引: task identifier -> (nsites, volume, species) 全本地.

动机: REST不再暴露material->dos task映射(es dos:null 33%), Delta行只有task键.
做法: 单遍扫描Delta eDOS各run_type分区, 抽(identifier, nsites, volume, 物种集),
      落盘复用; 再用增量集summary结构精确归属.
输出: delta_edos_index.parquet + top1归属结果并入mp_increment_dosmap.jsonl
"""
import glob
import json
import time
import urllib.parse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RAW = Path("/root/home/newstudy/getdata/raw")
DP = RAW / "delta_mp/core/electronic-structure/total-dos"


def build_index():
    import numpy as np
    ids, nss, vols, sps, rts = [], [], [], [], []
    for f in sorted(glob.glob(str(DP / "run_type=*/*.parquet"))):
        rt = urllib.parse.unquote(f.split("run_type=")[1].split("/")[0])
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=131072, columns=["identifier", "structure"]):
            t = pa.Table.from_batches([b])
            col_id = t.column("identifier").to_pylist()
            s = t.column("structure")
            lat = pc.struct_field(s, "lattice")
            vol = pc.struct_field(lat, "volume").to_pylist()
            sites = pc.struct_field(s, "sites").combine_chunks()
            off = sites.offsets.to_pylist()
            labels = pc.struct_field(sites.flatten(), "label").to_pylist()
            for k in range(len(col_id)):
                seg = labels[off[k]:off[k + 1]]
                ids.append(col_id[k])
                nss.append(off[k + 1] - off[k])
                vols.append(vol[k])
                sps.append(",".join(sorted(set(seg))))
                rts.append(rt)
        print(f"[idx] {rt}: total={len(ids)}", flush=True)
    out = pa.table({"identifier": ids, "nsites": nss, "volume": vols,
                    "species": sps, "run_type": rts})
    pq.write_table(out, RAW / "delta_edos_index.parquet")
    print(f"[idx] wrote {len(out)} rows", flush=True)


def match_targets():
    import numpy as np
    idx = pq.read_table(RAW / "delta_edos_index.parquet").to_pylist()
    # block: (nsites, species) -> rows
    from collections import defaultdict
    blocks = defaultdict(list)
    for r in idx:
        blocks[(r["nsites"], r["species"])].append(r)
    print(f"[match] blocks={len(blocks)}", flush=True)
    # 目标: dos_task缺失的增量材料
    tgts = []
    for line in open(RAW / "mp_increment_raw.jsonl"):
        r = json.loads(line)
        if r["dos_task_id"] is None and r["mp_summary"]:
            st = r["mp_summary"]["structure"]
            sp = ",".join(sorted(set(s["label"] for s in st["sites"])))
            tgts.append((r["mpid"], len(st["sites"]), sp, st["lattice"]["volume"]))
    print(f"[match] targets={len(tgts)}", flush=True)
    res = {}
    stat = {"unique": 0, "multi": 0, "none": 0}
    for mpid, ns, sp, V in tgts:
        cands = [c for c in blocks.get((ns, sp), [])
                 if abs(c["volume"] - V) / V < 0.01]
        if len(cands) == 1:
            res[mpid] = {"identifier": cands[0]["identifier"],
                         "run_type": cands[0]["run_type"], "mode": "unique"}
            stat["unique"] += 1
        elif len(cands) > 1:
            res[mpid] = {"candidates": [(c["identifier"], c["run_type"],
                                         round(c["volume"], 1)) for c in cands],
                         "mode": "multi"}
            stat["multi"] += 1
        else:
            res[mpid] = {"mode": "none"}
            stat["none"] += 1
    json.dump(res, open(RAW / "mp_increment_dosmap.json", "w"), indent=1)
    print(f"[match] {stat}", flush=True)


if __name__ == "__main__":
    import sys
    t0 = time.time()
    if "index" in sys.argv:
        build_index()
    if "match" in sys.argv:
        match_targets()
    print("elapsed", round(time.time() - t0, 1), flush=True)
