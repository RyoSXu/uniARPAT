#!/usr/bin/env python3
"""Export the frozen B7 M1 model's CIF-only blind eDOS/phDOS prediction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from utils.b7_cif_inference import (
    B7_EPOCH,
    B7_SEED,
    load_b7_grids,
    load_b7_model,
    predict_b7_blind,
    resolve_device,
    structure_to_b7_inputs,
)


def collect_cif_files(cif: str | None, cif_dir: str | None) -> list[Path]:
    files: list[Path] = []
    if cif is not None:
        files.append(Path(cif))
    if cif_dir is not None:
        directory = Path(cif_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"CIF directory not found: {directory}")
        files.extend(sorted(directory.glob("*.cif")))
        files.extend(sorted(directory.glob("*.CIF")))
    unique = list(dict.fromkeys(files))
    if not unique:
        raise ValueError("provide --cif or --cif-dir")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CIF file not found: {missing[0]}")
    stems = [path.stem for path in unique]
    if len(set(stems)) != len(stems):
        raise ValueError("CIF basenames must be unique because they name output files")
    return unique


def process_cif(cif_path: Path, model, device, checkpoint_metadata: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    structure = Structure.from_file(str(cif_path))
    src, pos, metadata = structure_to_b7_inputs(structure)
    prediction = predict_b7_blind(model, src, pos, metadata["n_valence"], device)
    edos_x, phdos_x = load_b7_grids()
    stem = cif_path.stem
    npz_path = output_dir / f"{stem}_b7_blind.npz"
    np.savez_compressed(npz_path, edos_x=edos_x, edos=prediction["edos"], phdos_x=phdos_x, phdos=prediction["phdos"])
    metadata = {
        **metadata,
        **{key: value for key, value in prediction.items() if key not in {"edos", "phdos"}},
        "cif": str(cif_path),
        "checkpoint": checkpoint_metadata,
        "model_contract": "B7 _e9ctl M1 SumNorm + H1 blind; no label-derived scale",
        "grid": {"edos": "E0 centers [-6,6] eV, 128 bins", "phdos": "P0 centers [-280,980] cm^-1, 64 bins"},
        "spectra_npz": str(npz_path),
    }
    json_path = output_dir / f"{stem}_b7_blind.json"
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "cif": str(cif_path),
        "formula": metadata["formula"],
        "n_atoms": metadata["n_atoms"],
        "n_valence": metadata["n_valence"],
        "eta_ph": metadata["eta_ph"],
        "gamma_e": metadata["gamma_e"],
        "edos_sum": metadata["edos_sum"],
        "phdos_sum": metadata["phdos_sum"],
        "spectra_npz": str(npz_path),
        "metadata_json": str(json_path),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen B7 M1 CIF-only blind eDOS/phDOS inference.")
    parser.add_argument("--cif", help="single CIF path")
    parser.add_argument("--cif-dir", help="directory containing .cif/.CIF files")
    parser.add_argument("--checkpoint", required=True, help="B7 _e9ctl epoch-33 checkpoint_best.pth")
    parser.add_argument("--output", default="results/b7_cif_blind", help="output directory")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    files = collect_cif_files(args.cif, args.cif_dir)
    device = resolve_device(args.device)
    model, checkpoint_metadata = load_b7_model(args.checkpoint, device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [process_cif(path, model, device, checkpoint_metadata, output_dir) for path in files]
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"B7 _e9ctl epoch {B7_EPOCH} (seed {B7_SEED}): {len(results)} CIF prediction(s) -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
