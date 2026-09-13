#!/usr/bin/env python3
"""JARVIS覆盖映射(A7后续, 对标A3c): JVASP -> legacy mp-N -> canonical.

输入:
  /root/home/newstudy/getdata/raw/jarvis_mp_mapping_v2.json (49,942 JVASP->MP)
  /root/home/newstudy/getdata/raw/phonondb_a3c_map.jsonl (免费翻译 9,938)
输出:
  /root/home/newstudy/getdata/raw/jarvis_a7b_map.jsonl
    ({jvasp, legacy_mpid, canonical_mpid, formula_pretty, src})
    src=free(A3c已翻) | api(本次翻译)
  /root/home/newstudy/getdata/raw/jarvis_a7b_nickname.json (37昵称ID, 直送L2/丢弃)
  /root/home/newstudy/getdata/raw/jarvis_a7b_summary.json (交集统计, 全量跑完才写)
策略: 整批先行(sleep0.3) + 顽固singles(--singles-only --workers4);
  --limit N 分段; 断点续跑(以jvasp为key).
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
H = {"X-API-KEY": KEY}
RAW = Path("/root/home/newstudy/getdata/raw")
MAP_IN = RAW / "jarvis_mp_mapping_v2.json"
FREE_IN = RAW / "phonondb_a3c_map.jsonl"
OUT_MAP = RAW / "jarvis_a7b_map.jsonl"
LEGACY_RE = re.compile(r"^mp-\d+$")


def qbatch(session, legs, tries=5, flush=None):
    out, pending = {}, list(legs)
    for p in range(tries):
        if not pending:
            break
        last = (p == tries - 1)
        nxt = []
        for i in range(0, len(pending), 100):
            ch = pending[i:i + 100]
            try:
                r = session.get(BASE + "/materials/summary/",
                                params={"material_ids": ",".join(ch),
                                        "_fields": "material_id,formula_pretty"},
                                headers=H, timeout=180)
                if r.status_code == 200 and len(r.json()["data"]) == len(ch):
                    for leg, doc in zip(ch, r.json()["data"]):
                        out[leg] = (doc["material_id"], doc.get("formula_pretty"))
                    if flush:
                        flush(ch, out)
                else:
                    nxt.extend(ch)
            except Exception:
                nxt.extend(ch)
            time.sleep(0.3)
        pending = [m for m in nxt if m not in out]
        if pending and not last:
            print(f"[a7b] pass={p} resolved={len(out)}/{len(legs)} "
                  f"retry={len(pending)}", flush=True)
    return out, pending


def singles(legs, workers, flush):
    from collections import Counter
    out, lock, done_n = {}, __import__("threading").Lock(), [0]
    fail_kind = Counter()
    import threading  # noqa: F811 (显式, 保持单文件)

    def _one(leg):
        # 200+0条 = definitive dead(删库, 不重试); 异常/非200才重试一次
        try:
            with requests.Session() as s:
                r = s.get(BASE + "/materials/summary/",
                          params={"material_ids": leg,
                                  "_fields": "material_id,formula_pretty"},
                          headers=H, timeout=30)
            if r.status_code == 200:
                docs = r.json()["data"]
                if len(docs) == 1:
                    doc = docs[0]
                    with lock:
                        out[leg] = (doc["material_id"], doc.get("formula_pretty"))
                        done_n[0] += 1
                        if flush:
                            flush([leg], out)
                        if done_n[0] % 200 == 0:
                            print(f"[a7b] singles {done_n[0]}/{len(legs)} "
                                  f"resolved={len(out)}", flush=True)
                    return leg, True
                with lock:
                    done_n[0] += 1
                    fail_kind["empty"] += 1
                return leg, False
            code = r.status_code
        except Exception:
            code = "exc"
        time.sleep(0.5)
        try:
            with requests.Session() as s:
                r = s.get(BASE + "/materials/summary/",
                          params={"material_ids": leg,
                                  "_fields": "material_id,formula_pretty"},
                          headers=H, timeout=30)
            if r.status_code == 200 and len(r.json()["data"]) == 1:
                doc = r.json()["data"][0]
                with lock:
                    out[leg] = (doc["material_id"], doc.get("formula_pretty"))
                    done_n[0] += 1
                    if flush:
                        flush([leg], out)
                return leg, True
            kind = "empty2" if r.status_code == 200 else f"http{r.status_code}"
        except Exception:
            kind = "exc2"
        with lock:
            done_n[0] += 1
            fail_kind[kind] += 1
            if done_n[0] % 200 == 0:
                print(f"[a7b] singles {done_n[0]}/{len(legs)} resolved={len(out)} "
                      f"fail={dict(fail_kind)}", flush=True)
        return leg, False

    if workers <= 1:
        res = [_one(l) for l in legs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_one, legs))
    return out, [l for l, ok in res if not ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mapping = json.load(open(MAP_IN))
    print(f"[a7b] pairs={len(mapping)}", flush=True)
    # legacy值去重; 昵称ID分流
    leg2jv = {}
    nick = {}
    for jv, mp in mapping.items():
        mp = str(mp)
        if LEGACY_RE.match(mp):
            leg2jv.setdefault(mp, []).append(jv)
        else:
            nick[jv] = mp
    json.dump(nick, open(RAW / "jarvis_a7b_nickname.json", "w"), indent=1)
    print(f"[a7b] unique_legacy={len(leg2jv)} nickname={len(nick)}", flush=True)

    # 免费翻译(A3c)
    free = {}
    with open(FREE_IN) as f:
        for line in f:
            r = json.loads(line)
            free[r["legacy_mpid"]] = (r["canonical_mpid"], r.get("formula_pretty"))
    # resume
    done_jv = set()
    if OUT_MAP.exists():
        with open(OUT_MAP) as f:
            for line in f:
                try:
                    done_jv.add(json.loads(line)["jvasp"])
                except Exception:
                    pass
    print(f"[a7b] resume have={len(done_jv)} free_hit_total="
          f"{sum(1 for v in leg2jv.values() for j in v if j not in done_jv)}",
          flush=True)

    # 先写免费行
    fmap = open(OUT_MAP, "a")
    n_free = 0
    for leg, jvs in leg2jv.items():
        if leg not in free:
            continue
        canon, form = free[leg]
        for jv in jvs:
            if jv not in done_jv:
                fmap.write(json.dumps({"jvasp": jv, "legacy_mpid": leg,
                                       "canonical_mpid": canon,
                                       "formula_pretty": form,
                                       "src": "free"}) + "\n")
                done_jv.add(jv)
                n_free += 1
    fmap.flush()
    print(f"[a7b] free_written={n_free}", flush=True)

    todo_legs = [l for l, jvs in leg2jv.items()
                 if l not in free and any(j not in done_jv for j in jvs)]
    if args.limit:
        todo_legs = todo_legs[:args.limit]
    print(f"[a7b] todo_legs={len(todo_legs)} singles_only={args.singles_only} "
          f"workers={args.workers}", flush=True)

    def _flush(chunk, out):
        for leg in chunk:
            if leg in out:
                canon, form = out[leg]
                for jv in leg2jv[leg]:
                    if jv not in done_jv:
                        fmap.write(json.dumps({"jvasp": jv, "legacy_mpid": leg,
                                               "canonical_mpid": canon,
                                               "formula_pretty": form,
                                               "src": "api"}) + "\n")
                        done_jv.add(jv)
        fmap.flush()

    session = requests.Session()
    if args.singles_only:
        trans, unres = singles(todo_legs, args.workers, _flush)
    else:
        trans, unres = qbatch(session, todo_legs, flush=_flush)
        if unres:
            print(f"[a7b] batch残留{len(unres)}, 转singles", flush=True)
            t2, unres = singles(unres, max(args.workers, 4), _flush)
            trans.update(t2)
    fmap.close()
    json.dump(unres, open(RAW / "jarvis_a7b_unresolved.part.json", "w"))
    print(f"[a7b] this_run api={len(trans)} unresolved={len(unres)} "
          f"total_mapped={len(done_jv)}", flush=True)
    if args.limit:
        return
    # 全量交集
    full_canon = set()
    with open(OUT_MAP) as f:
        for line in f:
            full_canon.add(json.loads(line)["canonical_mpid"])
    effective = set(json.load(open(RAW / "census_effective_ids.json")))
    dual = set(json.load(open(RAW / "dual_spectra_ids.json")))
    absent = set(json.load(open(RAW / "edos_absent.json")))
    true_double = dual - absent
    s = {"pairs": len(mapping), "mapped_jvasp": len(done_jv),
         "unique_canonical": len(full_canon),
         "unresolved_legs": unres, "nickname_n": len(nick),
         "in_effective": len(full_canon & effective),
         "in_mp_phonon": len(full_canon & dual),
         "in_true_double": len(full_canon & true_double),
         "in_edos_absent": len(full_canon & absent),
         "mp_no_phonon_but_jv_has_map": len((full_canon & effective) - dual),
         "outside_effective": len(full_canon - effective)}
    json.dump(s, open(RAW / "jarvis_a7b_summary.json", "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(s, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
