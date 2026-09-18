#!/usr/bin/env python3
"""A3c PhononDB覆盖映射: MDR serial -> legacy mp-N -> canonical mp-xxxx.

输入:
  /root/home/newstudy/getdata/ref/download_list.md  (serial<->legacy数字+元素名)
  /root/home/newstudy/getdata/raw/phonondb_recomputed.jsonl (10,034复算, id=serial)
  /root/home/newstudy/getdata/raw/census_effective_ids.json (154,373)
  /root/home/newstudy/getdata/raw/dual_spectra_ids.json (26,609 = MP有声子)
  /root/home/newstudy/getdata/raw/edos_absent.json (7,965 有声子无eDOS)
输出:
  /root/home/newstudy/getdata/raw/phonondb_a3c_map.jsonl
    ({serial, legacy_mpid, canonical_mpid, formula_pretty})
  /root/home/newstudy/getdata/raw/phonondb_a3c_summary.json
逻辑:
  MP summary支持legacy mp-N查询, 服务端保序返回canonical
  (Design-A3回声保序结论, 已用mp-25/149/39 + 990448等验证).
  逐批核对返回数, 缺的进unresolved, 永不静默丢数.
"""
import json
import re
import time
from pathlib import Path

import requests

KEY = "n7FHOn35u18KAUKwXt3zbCwYMUYi6you"
BASE = "https://api.materialsproject.org"
H = {"X-API-KEY": KEY}
REF = Path("/root/home/newstudy/getdata/ref/download_list.md")
RAW = Path("/root/home/newstudy/getdata/raw")
OUT_MAP = RAW / "phonondb_a3c_map.jsonl"
OUT_SUM = RAW / "phonondb_a3c_summary.json"


def parse_ref():
    rows = []  # (serial, legacy_numeric, name)
    for line in open(REF, errors="ignore"):
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) >= 5 and re.match(r"^\d+$", parts[0]):
            m = re.search(r"\[([0-9a-z]+\.zip)\]", parts[4])
            if m:
                rows.append((m.group(1)[:-4], int(parts[0]), parts[1]))
    # 去重保序
    seen, res = set(), []
    for s, n, nm in rows:
        if s not in seen:
            seen.add(s)
            res.append((s, n, nm))
    return res


def qbatch_translate(session, legacy_ids, tries=6, flush_cb=None,
                     singles_only=False):
    """legacy mp-N list -> {legacy: (canonical, formula)} + unresolved list.

    前tries-1轮只做整批重试(瞬时失败可恢复, 便宜);
    最后一轮才对顽固分子做singles补查(贵).
    singles_only=True: 跳过整批(整批已被1%死ID毒死, 见2026-09-11实测99/100),
    直接逐个singles, 命中即写, 无命中进unresolved.
    """
    out, pending = {}, list(legacy_ids)
    n_chunks = 0
    if singles_only:
        workers = getattr(qbatch_translate, "_workers", 1)
        if workers <= 1:
            still = []
            for j, leg in enumerate(pending):
                try:
                    r = session.get(
                        BASE + "/materials/summary/",
                        params={"material_ids": leg,
                                "_fields": "material_id,formula_pretty"},
                        headers=H, timeout=60)
                    if r.status_code == 200 and len(r.json()["data"]) == 1:
                        doc = r.json()["data"][0]
                        out[leg] = (doc["material_id"], doc.get("formula_pretty"))
                        if flush_cb:
                            flush_cb([leg], out)
                    else:
                        still.append(leg)
                except Exception:
                    still.append(leg)
                if (j + 1) % 50 == 0:
                    print(f"[a3c-q] singles {j + 1}/{len(pending)} "
                          f"resolved={len(out)}/{len(legacy_ids)}", flush=True)
                time.sleep(0.2)
            return out, still
        # 并发singles: 每条独立, 失败进still
        from concurrent.futures import ThreadPoolExecutor
        import threading
        lock = threading.Lock()
        done_n = [0]

        def _one(leg):
            for _ in range(2):
                try:
                    with requests.Session() as s:
                        r = s.get(
                            BASE + "/materials/summary/",
                            params={"material_ids": leg,
                                    "_fields": "material_id,formula_pretty"},
                            headers=H, timeout=60)
                    if r.status_code == 200 and len(r.json()["data"]) == 1:
                        doc = r.json()["data"][0]
                        with lock:
                            out[leg] = (doc["material_id"],
                                        doc.get("formula_pretty"))
                            done_n[0] += 1
                            if flush_cb:
                                flush_cb([leg], out)
                            if done_n[0] % 50 == 0:
                                print(f"[a3c-q] singles {done_n[0]}/{len(pending)} "
                                      f"resolved={len(out)}/{len(legacy_ids)}",
                                      flush=True)
                        return leg, True
                except Exception:
                    pass
                time.sleep(0.5)
            with lock:
                done_n[0] += 1
            return leg, False

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_one, pending))
        still = [leg for leg, ok in results if not ok]
        print(f"[a3c-q] singles done: resolved={len(out)}/{len(legacy_ids)} "
              f"dead={len(still)}", flush=True)
        return out, still
    for pass_no in range(tries):
        if not pending:
            break
        last_pass = (pass_no == tries - 1)
        # 分100一批, 要求返回数==请求数才采纳(回声保序前提)
        new_pending = []
        for i in range(0, len(pending), 100):
            chunk = pending[i:i + 100]
            try:
                r = session.get(
                    BASE + "/materials/summary/",
                    params={"material_ids": ",".join(chunk),
                            "_fields": "material_id,formula_pretty"},
                    headers=H, timeout=180)
                if r.status_code == 200 and len(r.json()["data"]) == len(chunk):
                    for leg, doc in zip(chunk, r.json()["data"]):
                        out[leg] = (doc["material_id"], doc.get("formula_pretty"))
                    if flush_cb:
                        flush_cb(chunk, out)
                else:
                    # 数量不对 -> 下一轮整批重试(最后才singles)
                    try:
                        code = r.status_code
                        got = len(r.json().get("data", [])) if code == 200 else -1
                    except Exception:
                        code, got = -99, -99
                    if n_chunks < 3:
                        print(f"[a3c-q] batch-miss chunk0={chunk[0]} "
                              f"want={len(chunk)} got={got} http={code}", flush=True)
                    new_pending.extend(chunk)
            except Exception as e:
                if n_chunks < 3:
                    print(f"[a3c-q] batch-exc chunk0={chunk[0]} {type(e).__name__}",
                          flush=True)
                new_pending.extend(chunk)
            n_chunks += 1
            if n_chunks % 10 == 0:
                print(f"[a3c-q] pass={pass_no} chunks={n_chunks} "
                      f"resolved={len(out)}/{len(legacy_ids)} "
                      f"pending={len(pending)}", flush=True)
            time.sleep(0.3)
        pending = [m for m in new_pending if m not in out]
        # 只有最后一轮才做singles补查
        if pending and last_pass:
            still = []
            for j, leg in enumerate(pending):
                try:
                    r = session.get(
                        BASE + "/materials/summary/",
                        params={"material_ids": leg,
                                "_fields": "material_id,formula_pretty"},
                        headers=H, timeout=60)
                    if r.status_code == 200 and len(r.json()["data"]) == 1:
                        doc = r.json()["data"][0]
                        out[leg] = (doc["material_id"], doc.get("formula_pretty"))
                        if flush_cb:
                            flush_cb([leg], out)
                    else:
                        still.append(leg)
                except Exception:
                    still.append(leg)
                if (j + 1) % 50 == 0:
                    print(f"[a3c-q] singles {j + 1}/{len(pending)} "
                          f"resolved={len(out)}/{len(legacy_ids)}", flush=True)
                time.sleep(0.2)
            pending = still
        elif pending:
            print(f"[a3c-q] pass={pass_no} end: resolved={len(out)}/{len(legacy_ids)} "
                  f"retry_batches={len(pending)} -> next pass", flush=True)
    return out, pending


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles-only", action="store_true",
                    help="跳过整批, 直接singles(整批被死ID毒死时用)")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前N个todo(分段跑, 默认全量)")
    ap.add_argument("--workers", type=int, default=1,
                    help="singles并发数(默认1串行; 建议4)")
    args = ap.parse_args()
    qbatch_translate._workers = args.workers
    rows = parse_ref()
    print(f"[a3c] ref rows={len(rows)}", flush=True)
    rec_ids = set()
    with open(RAW / "phonondb_recomputed.jsonl") as f:
        for line in f:
            try:
                rec_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    ref_serials = set(s for s, _, _ in rows)
    print(f"[a3c] recomputed={len(rec_ids)} ref={len(ref_serials)} "
          f"both={len(rec_ids & ref_serials)} "
          f"ref-rec={len(ref_serials - rec_ids)} rec-ref={len(rec_ids - ref_serials)}",
          flush=True)
    assert rec_ids == ref_serials, "serial对不上, 先停"

    # resume
    done = {}
    if OUT_MAP.exists():
        with open(OUT_MAP) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["serial"]] = r
                except Exception:
                    pass
    print(f"[a3c] resume have={len(done)}", flush=True)

    serial_to_leg = {s: f"mp-{n}" for s, n, _ in rows}
    todo_serials = [s for s, _, _ in rows if s not in done]
    todo_legs = [serial_to_leg[s] for s in todo_serials]
    # legacy去重(理论上1:1, 防御性)
    uniq_legs, seen = [], set()
    for leg in todo_legs:
        if leg not in seen:
            seen.add(leg)
            uniq_legs.append(leg)
    if args.limit:
        uniq_legs = uniq_legs[:args.limit]
    print(f"[a3c] todo_serials={len(todo_serials)} todo_uniq_legacy={len(uniq_legs)} "
          f"singles_only={args.singles_only}", flush=True)

    session = requests.Session()
    leg_to_serial = {}
    for s, n, _ in rows:
        leg_to_serial.setdefault(f"mp-{n}", s)
    fmap = open(OUT_MAP, "a")

    def _flush(chunk, out):
        for leg in chunk:
            if leg in out:
                s = leg_to_serial[leg]
                if s not in done:
                    canon, form = out[leg]
                    fmap.write(json.dumps({"serial": s, "legacy_mpid": leg,
                                           "canonical_mpid": canon,
                                           "formula_pretty": form}) + "\n")
                    done[s] = True
        fmap.flush()

    trans, unresolved = qbatch_translate(session, uniq_legs, flush_cb=_flush,
                                         singles_only=args.singles_only)
    print(f"[a3c] translated={len(trans)} unresolved={len(unresolved)}",
          flush=True)

    # singles补查的尾巴(flush_cb只覆盖整批命中, 这里补写剩余)
    for leg, (canon, form) in trans.items():
        s = leg_to_serial[leg]
        if s not in done:
            fmap.write(json.dumps({"serial": s, "legacy_mpid": leg,
                                   "canonical_mpid": canon,
                                   "formula_pretty": form}) + "\n")
            done[s] = True
    fmap.flush()
    fmap.close()
    if args.limit:
        print(f"[a3c] chunk done: mapped_this_run={len(trans)} "
              f"unresolved_this_run={len(unresolved)} (summary跳过, 全量跑完再算)",
              flush=True)
        json.dump(unresolved, open(RAW / "phonondb_a3c_unresolved.part.json", "w"))
        return
    json.dump(unresolved, open(RAW / "phonondb_a3c_unresolved.json", "w"), indent=1)

    # --- 交集统计 ---
    full = {}
    with open(OUT_MAP) as f:
        for line in f:
            r = json.loads(line)
            full[r["serial"]] = r
    canon_list = [r["canonical_mpid"] for r in full.values()]
    print(f"[a3c] mapped_total={len(full)} unique_canon={len(set(canon_list))}",
          flush=True)

    effective = set(json.load(open(RAW / "census_effective_ids.json")))
    dual = set(json.load(open(RAW / "dual_spectra_ids.json")))  # MP有声子26,609
    absent = set(json.load(open(RAW / "edos_absent.json")))  # 有声子无eDOS 7,965
    true_double = dual - absent  # 18,644
    canon_set = set(canon_list)

    summary = {
        "phonondb_total": len(rows),
        "mapped": len(full),
        "unique_canonical": len(canon_set),
        "unresolved": unresolved,
        "in_effective": len(canon_set & effective),
        "in_mp_phonon_dual26609": len(canon_set & dual),
        "in_true_double18644": len(canon_set & true_double),
        "in_edos_absent7965": len(canon_set & absent),
        "mp_no_phonon_but_phdb_has": len((canon_set & effective) - dual),
        "outside_effective": len(canon_set - effective),
        "note": "dual26609=MP有声子; true_double=dual-absent=18644; "
                "mp_no_phonon_but_phdb_has=PhononDB增量核心(在有效全集内但MP无声子)",
    }
    json.dump(summary, open(OUT_SUM, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)

    # 未命中清单(进L2 matcher候选)
    outside = sorted(canon_set - effective)
    json.dump(outside, open(RAW / "phonondb_a3c_outside_effective.json", "w"))
    print(f"[a3c] outside_effective_sample={outside[:10]}", flush=True)


if __name__ == "__main__":
    main()
