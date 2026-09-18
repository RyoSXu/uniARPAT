#!/usr/bin/env python3
"""v2 processing skeleton (interfaces frozen, parameters filled after census).

Stage contract (see docs/04_前沿探索/2026-09-09_v2数据加工规范.md):
  raw JSONL / local Delta  ->  process.py  ->  Parquet full set (+split col)
                             -> build_v2_cache.py -> npy/memmap train cache
"""
from pathlib import Path

# Filled by A3 census before execution:
CAP_ATOMS = None        # NOT fixed at 80 (v1 legacy); set from atom-count p99.9
                        # after extraction stats (candidates 96/128); outliers
                        # truncated + warned + logged in prov.
GRID_EDOS = None        # e.g. ("linspace", -6.0, 6.0, 128)
GRID_PHDOS = None       # e.g. ("linspace", -280.0, 980.0, 64)
WINSOR_PCTL = 99.9      # per-bin winsorize percentile
SYMPREC = 0.1           # pymatgen canonical-cell symprec


def canonicalize_structure(mp_structure: dict) -> dict:
    """MP structure JSON -> primitive cell @ SYMPREC + symmetry + CIF text."""
    raise NotImplementedError


def extract_cif6(mp_structure: dict) -> dict:
    """Return the CIF-6 bundle: lattice9, sg(id+system+wyckoff),
    occupancy[80], atom_feat[80,24] placeholder, mask[80], cif_text."""
    raise NotImplementedError


def resample_spectrum(energies, densities, grid, efermi) -> tuple:
    """Fermi-align + trapz box-average onto grid. No extrapolation; gaps->mask."""
    raise NotImplementedError


def process_record(raw: dict) -> dict:
    """One raw JSONL record -> one processed dict (pre-Parquet)."""
    raise NotImplementedError


def build_cache(parquet_path: str, split_yaml: str, out_dir: str,
                batch_buckets=(8, 16, 32, 80)):
    """Parquet + split -> npy/memmap train cache (+ length-bucket index)."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("skeleton only; fill CAP_ATOMS/GRID_* after A3 census")
