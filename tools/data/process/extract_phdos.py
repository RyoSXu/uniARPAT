#!/usr/bin/env python3
"""A4 extraction step 2: phDOS from local Delta -> raw JSONL.
In:  getdata/raw/dual_spectra_ids.json
Out: getdata/raw/mp_phdos_raw.jsonl  ({mpid, method, frequencies_THz,
      densities, projected, run_type, has_structure})
Resume-safe. Units converted downstream (THz->cm-1); raw keeps native.
"""
import json
import time
from pathlib import Path

RAW = Path("/root/home/newstudy/getdata/raw")
THZ = 33.35641


def main():
    dual = json.load(open(RAW / "dual_spectra_ids.json"))
    pmap = json.load(open(RAW / "census_phonon_map_canonical.json"))
    from deltalake import DeltaTable
    pdt = DeltaTable(str(RAW / "delta_mp/phonon/electronic-structure/dos/"))
    done = set()
    out_f = RAW / "mp_phdos_raw.jsonl"
    if out_f.exists():
        with open(out_f) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["mpid"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [m for m in dual if m not in done]
    print(f"[phdos] dual={len(dual)} done={len(done)} todo={len(todo)}", flush=True)
    with open(out_f, "a") as f:
        for i in range(0, len(todo), 200):
            chunk = todo[i:i + 200]
            pids, owner = [], {}
            for m in chunk:
                for meth, lst in (pmap.get(m) or {}).items():
                    for pid in (lst or []):
                        pids.append(pid)
                        owner[pid] = (m, meth)
            rows = {}
            if pids:
                for a in range(6):
                    try:
                        t = pdt.to_pyarrow_table(
                            filters=[("identifier", "in", sorted(set(pids)))])
                        for r in t.to_pylist():
                            rows[r["identifier"]] = r
                        break
                    except Exception:  # noqa: BLE001
                        time.sleep(15)
            by_m = {}
            for pid, r in rows.items():
                by_m.setdefault(owner[pid][0], []).append((owner[pid][1], r))
            for m in chunk:
                if m in by_m:
                    recs = [{"method": meth,
                             "frequencies_THz": r["dos"]["frequencies"],
                             "densities": r["dos"]["densities"],
                             "projected": r["dos"].get("projected_densities"),
                             "run_type": r["dos"].get("run_type"),
                             "has_structure": r["dos"].get("structure") is not None}
                            for meth, r in by_m[m]]
                    f.write(json.dumps({"mpid": m, "phdos_raw": recs}) + "\n")
            if (i // 200 + 1) % 10 == 0:
                print(f"[phdos] {i + 200}/{len(todo)}", flush=True)
    print("[phdos] DONE", flush=True)


if __name__ == "__main__":
    main()
