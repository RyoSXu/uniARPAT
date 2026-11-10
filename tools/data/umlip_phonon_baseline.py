"""Phonon DOS from a universal interatomic potential (MACE) with phonopy: a non-learned-DOS baseline.

Phase A of the 2026-11-09 roadmap. For each selected material of a split, the
stored crystal (same cell and atom order as the model input) is optionally
relaxed with the potential, finite-displacement force constants are computed
with phonopy, and the total phonon DOS (tetrahedron method) is converted to the
dataset convention: frequency in cm^-1 (THz x 33.35641, density divided by the
same factor) and binned on the P0 window like the data release (left-closed
bins; trapezoid over the points inside a bin divided by the bin width; one
point -> its value; none -> 0). The DOS is per input cell (integral 3N).

Runs in the MACE environment (with phonopy) and imports nothing from the
repository. Writes <out>/phonons.npz (material ids, positions, binned DOS) and
<out>/records.jsonl (one line per material, written as it finishes, so a run
can resume) and <out>/manifest.json.

Usage (MACE environment):
    python tools/data/umlip_phonon_baseline.py --model <mace.model> --data data/crystal_q1_v1 \
        --split valid --sample 400 --seed 20261109 --out results/runs/umlip_phonon_20261109/mace_relaxed
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

THZ_TO_CM = 33.35641
WINDOW = (-280.0, 980.0)
BINS = 64


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(root, split):
    directory = Path(root) / split
    names = ("cell", "frac_coords", "atomic_numbers", "atom_offsets", "n_atoms", f"{split}_index")
    arrays = {name: np.load(directory / f"{name}_{split}.npy" if not name.endswith("_index") else directory / f"{name}.npy",
                            allow_pickle=False) for name in names}
    arrays["ids"] = arrays.pop(f"{split}_index")
    return arrays


def structure(arrays, index):
    from ase import Atoms

    start, end = arrays["atom_offsets"][index:index + 2]
    return Atoms(numbers=arrays["atomic_numbers"][start:end].astype(int),
                 scaled_positions=arrays["frac_coords"][start:end].astype(np.float64),
                 cell=arrays["cell"][index].astype(np.float64), pbc=True)


def supercell_matrix(cell, min_height):
    """Diagonal multiples so every perpendicular cell height reaches min_height (Angstrom)."""
    volume = abs(np.linalg.det(cell))
    heights = [volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    return np.diag([max(1, int(np.ceil(min_height / h))) for h in heights])


def bin_dos(frequencies_cm, density_cm):
    """Dataset binning: left-closed bins, trapezoid of in-bin points / width, single point -> value."""
    edges = np.linspace(WINDOW[0], WINDOW[1], BINS + 1)
    width = edges[1] - edges[0]
    out = np.zeros(BINS)
    for b in range(BINS):
        inside = (frequencies_cm >= edges[b]) & (frequencies_cm < edges[b + 1])
        x, y = frequencies_cm[inside], density_cm[inside]
        if len(x) >= 2:
            out[b] = np.trapezoid(y, x) / width if hasattr(np, "trapezoid") else np.trapz(y, x) / width
        elif len(x) == 1:
            out[b] = y[0]
    return out


def phonon_dos(atoms, calc, args):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell[:], scaled_positions=atoms.get_scaled_positions())
    matrix = supercell_matrix(np.asarray(atoms.cell[:]), args.min_height)
    phonon = Phonopy(unit, supercell_matrix=matrix, primitive_matrix=None)
    phonon.generate_displacements(distance=args.displacement)
    forces = []
    for cell in phonon.supercells_with_displacements:
        from ase import Atoms

        displaced = Atoms(symbols=cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True)
        displaced.calc = calc
        forces.append(displaced.get_forces())
    phonon.forces = np.asarray(forces)
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    phonon.run_mesh(args.mesh, with_eigenvectors=False, is_mesh_symmetry=True)
    step = args.pitch_cm / THZ_TO_CM
    phonon.run_total_dos(use_tetrahedron_method=True, freq_min=WINDOW[0] / THZ_TO_CM - 2 * step,
                         freq_max=WINDOW[1] / THZ_TO_CM + 2 * step, freq_pitch=step)
    dos = phonon.get_total_dos_dict()
    frequencies = np.asarray(dos["frequency_points"]) * THZ_TO_CM
    density = np.asarray(dos["total_dos"]) / THZ_TO_CM
    mesh = phonon.get_mesh_dict()["frequencies"] * THZ_TO_CM
    return bin_dos(frequencies, density), {
        "supercell_atoms": int(len(atoms) * round(abs(np.linalg.det(matrix)))),
        "supercell_matrix": np.diag(matrix).astype(int).tolist(), "displacements": len(forces),
        "min_mesh_frequency_cm": float(mesh.min()), "max_mesh_frequency_cm": float(mesh.max()),
        "window_integral_over_3n": float(bin_dos(frequencies, density).sum() * (WINDOW[1] - WINDOW[0]) / BINS / (3 * len(atoms)))}


def relax(atoms, calc, args):
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = atoms.copy()
    atoms.calc = calc
    optimizer = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = bool(optimizer.run(fmax=args.fmax, steps=args.max_steps))
    return atoms, {"relax_converged": converged, "relax_steps": int(optimizer.nsteps),
                   "volume_ratio": float(atoms.get_volume() / abs(np.linalg.det(structure_cell(atoms))))}


def structure_cell(atoms):
    return np.asarray(atoms.info["input_cell"])


def selection(arrays, args):
    count = len(arrays["ids"])
    if args.sample and args.sample < count:
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(count, size=args.sample, replace=False))
    else:
        chosen = np.arange(count)
    if args.max_atoms:
        chosen = np.asarray([i for i in chosen if arrays["n_atoms"][i] <= args.max_atoms])
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/crystal_q1_v1")
    parser.add_argument("--split", default="valid", choices=("train", "valid"))
    parser.add_argument("--sample", type=int, default=0, help="random materials (0 = all)")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--max-atoms", type=int, default=0, help="optional cap on input-cell atoms (0 = none)")
    parser.add_argument("--no-relax", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.005)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--min-height", type=float, default=10.0)
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--mesh", type=float, default=100.0, help="phonopy mesh length (or use 20 20 20 style via int)")
    parser.add_argument("--pitch-cm", type=float, default=1.0)
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new materials (pilot)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from mace.calculators import MACECalculator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = load_split(args.data, args.split)
    chosen = selection(arrays, args)
    manifest = {"schema": "umlip-phonon-baseline-v1", "model": {"path": str(Path(args.model).resolve()), "sha256": sha256(args.model)},
                "data": {"root": str(Path(args.data).resolve()), "manifest_sha256": sha256(Path(args.data) / "manifest.json"),
                         "split": args.split},
                "settings": {k: v for k, v in vars(args).items() if k not in ("out", "model", "data", "limit")},
                "convention": {"axis": "cm^-1 (THz x 33.35641)", "density": "per input cell, divided by 33.35641",
                               "window": WINDOW, "bins": BINS, "binning": "dataset release rule"},
                "software": {"torch": torch.__version__}, "selected": [str(arrays["ids"][i]) for i in chosen]}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["settings"] != manifest["settings"] or saved["selected"] != manifest["selected"] or saved["model"] != manifest["model"]:
            raise ValueError("existing output directory was produced with different settings")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    records_path = out / "records.jsonl"
    done = set()
    if records_path.exists():
        done = {json.loads(line)["material_id"] for line in records_path.read_text().splitlines() if line.strip()}
    calc = MACECalculator(model_paths=args.model, device=args.device, default_dtype=args.dtype)
    new = 0
    for index in chosen:
        material = str(arrays["ids"][index])
        if material in done:
            continue
        started = time.time()
        atoms = structure(arrays, index)
        atoms.info["input_cell"] = np.asarray(atoms.cell[:]).tolist()
        record = {"material_id": material, "position": int(index), "atoms": len(atoms)}
        try:
            if not args.no_relax:
                atoms, info = relax(atoms, calc, args)
                record.update(info)
            binned, info = phonon_dos(atoms, calc, args)
            record.update(info, dos=binned.tolist(), ok=True)
        except Exception as error:  # recorded per material; the baseline reports failures
            record.update(ok=False, error=f"{type(error).__name__}: {error}"[:300])
        record["seconds"] = round(time.time() - started, 2)
        with records_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(material, record.get("ok"), record["seconds"], record.get("supercell_atoms"), flush=True)
        new += 1
        if args.limit and new >= args.limit:
            break
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    ok = [r for r in rows if r["ok"]]
    np.savez(out / "phonons.npz", material_ids=np.asarray([r["material_id"] for r in ok]),
             positions=np.asarray([r["position"] for r in ok], dtype=np.int64),
             pred_phdos=np.asarray([r["dos"] for r in ok], dtype=np.float64).reshape(len(ok), BINS))
    print(json.dumps({"records": len(rows), "ok": len(ok), "selected": len(chosen)}))


if __name__ == "__main__":
    main()
