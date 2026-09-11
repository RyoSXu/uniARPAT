#!/usr/bin/env python3
"""PhononDB bulk recompute: zip -> phonopy mesh DOS (+structures/Born/dielectric).
Out: getdata/raw/phonondb_recomputed.jsonl  ({id, freq_THz, dos, pdos?,
      structure, born, dielectric, mesh, provenance})
Mesh rule: --mesh-grid 'Nx Nx Nz' (default 20; density calibration in A4).
Resume-safe (skip ids present). Pure CPU.
"""
import argparse
import json
import lzma
import subprocess
import tempfile
import zipfile
from pathlib import Path

import yaml

SRC = Path("/root/home/newstudy/getdata/raw/phonondb")
OUT = Path("/root/home/newstudy/getdata/raw/phonondb_recomputed.jsonl")


def recon_one(zip_path, mesh):
    """Returns record dict or {'_error': ...}."""
    try:
        z = zipfile.ZipFile(zip_path)
        raw = lzma.decompress(z.read("phonopy_params.yaml.xz")).decode()
        doc = yaml.safe_load(raw)
        with tempfile.TemporaryDirectory() as td:
            conf = Path(td) / "phonopy.conf"
            conf.write_text(raw)
            mesh_conf = Path(td) / "mesh.conf"
            mesh_conf.write_text(f"MP = {mesh} {mesh} {mesh}\nTETRAHEDRON = .TRUE.\n")
            r = subprocess.run(
                ["phonopy", "-c", str(conf), "--dos", str(mesh_conf)],
                cwd=td, capture_output=True, text=True, timeout=1200)
            dos_f = Path(td) / "total_dos.dat"
            if not dos_f.exists():
                return {"_error": f"no-dos {r.returncode} {r.stderr[-200:]}"}
            freq, dos = [], []
            for line in open(dos_f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()[:2]
                freq.append(float(a))
                dos.append(float(b))
        unitcell = doc.get("unit_cell", {})
        return {
            "freq_THz": freq,
            "dos": dos,
            "structure": {"unitcell": unitcell,
                          "primitive": doc.get("primitive_cell"),
                          "supercell_matrix": doc.get("supercell_matrix")},
            "born": doc.get("born_effective_charge"),
            "dielectric": doc.get("dielectric_constant"),
            "mesh": [mesh] * 3,
            "tetrahedron": True,
        }
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {str(e)[:150]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT),
                    help="output jsonl (per-worker part file for parallel runs)")
    ap.add_argument("--serials", type=str, default="",
                    help="comma-separated zip serials for pilot (default: all)")
    args = ap.parse_args()
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    pass
    zips = sorted(SRC.glob("*.zip"))
    if args.serials:
        want = set(args.serials.split(","))
        zips = [z for z in zips if z.stem in want]
    if args.limit:
        zips = zips[:args.limit]
    todo = [z for z in zips if z.stem not in done]
    print(f"[phdb] total={len(zips)} done={len(done)} todo={len(todo)} mesh={args.mesh}",
          flush=True)
    n_ok = n_err = 0
    with open(out_path, "a") as f:
        for i, zp in enumerate(todo):
            rec = recon_one(zp, args.mesh)
            if "_error" in rec:
                n_err += 1
                if n_err <= 5:
                    print(f"  ERR {zp.stem}: {rec['_error']}", flush=True)
            else:
                rec["id"] = zp.stem
                f.write(json.dumps(rec) + "\n")
                n_ok += 1
            if (i + 1) % 25 == 0:
                f.flush()
                print(f"[phdb] {i + 1}/{len(todo)} ok={n_ok} err={n_err}", flush=True)
    print(f"[phdb] DONE ok={n_ok} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
