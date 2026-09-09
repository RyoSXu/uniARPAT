#!/usr/bin/env python3
"""uniARPAT-v2 pure end-to-end production tool: CIF -> DOS (Step 4).

Implements docs/01_开发文档/2026-09-09_uniARPAT_v2生产级集成与P0修复开发方案.md §3:

  1. CLI: --cif / --cif_dir / --weights / --output / --device (+ --smoke)
  2. Pure pymatgen parsing: Structure -> elements (82) + positions (82,3 / flat 246),
     row-0 col-2 strictly stores 1/c (reciprocal contract, FIX-P0-02).
  3. Zero-oracle forward with uniARPAT-v2 M4 backbone -> 128-d eDOS + 64-d phDOS.
  4. Macro-property linkage via thermo_props.ThermodynamicCalculator:
     kappa_L(300K, Julian-Slack) + Theta_D + Cv(T) over 50K~1000K.
  5. Publication multi-panel figure (eDOS Fermi@0eV + phDOS + Cv(T)) saved as PNG (+PDF).
  6. --smoke fast verification; `python -m unittest discover tests -v` stays 15/15 green.

Feature contract (must stay in sync with training data):
  elements: [82] long, [126, 127] sentinels + up to 80 atomic numbers + 0 padding.
  positions: [82, 3] float, row0=[a, b, 1/c], row1=[alpha, beta, gamma] (deg),
             rows2..=[frac coords] + 0 padding. Flat form is 246 (=82*3).
  energy axes: eDOS E-E_F in [-10, 10] eV (128 bins);
               phDOS omega in [-280, 980] cm^-1 (64 bins).
"""

import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Robust import path: work both when CWD is uniARPAT/ and when invoked as
# `python uniARPAT/cif2dos.py` from the parent workspace.
# ---------------------------------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
_CANDIDATE_ROOTS = [FILE_DIR, FILE_DIR / "uniARPAT", Path.cwd(), Path.cwd() / "uniARPAT"]
for _p in _CANDIDATE_ROOTS:
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from model.transformer import Transformer  # noqa: E402
except ImportError:  # fallback when imported from parent dir
    from uniARPAT.model.transformer import Transformer  # noqa: E402

try:
    from thermo_props import ThermodynamicCalculator  # noqa: E402
except ImportError:
    from uniARPAT.thermo_props import ThermodynamicCalculator  # noqa: E402

try:
    from pymatgen.core import Element, Lattice, Structure  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise ImportError("cif2dos.py requires pymatgen (`pip install pymatgen`)") from exc


# ---------------------------------------------------------------------------
# Constants (frozen contracts)
# ---------------------------------------------------------------------------
SRC_LEN = 82
MAX_ATOMS = 80
POS_ROWS = 82
POS_FLAT = 246
SENTINELS = (126, 127)

EDOS_DIM = 128
PHDOS_DIM = 64
E_EDOS = np.linspace(-10.0, 10.0, EDOS_DIM)
FREQ_PHDOS = np.linspace(-280.0, 980.0, PHDOS_DIM)
T_RANGE_DEFAULT = np.linspace(50.0, 1000.0, 100)

# M4 robust backbone (Week-2 verdict): decoupled + gated cross-attn + symmetric
# heads, direct regression, no oracle scale branch.
M4_PARAMS = dict(
    token_num=118,
    d_model=512,
    nhead=8,
    edos_num=EDOS_DIM,
    phdos_num=PHDOS_DIM,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation="gelu",
    normalize_before=False,
    decoupled_decoder=True,
    use_gated_cross_attn=True,
    head_type="symmetric",
    predict_scale=False,
)


# ---------------------------------------------------------------------------
# Structure -> model features (strict 1/c contract)
# ---------------------------------------------------------------------------

def structure_to_tensors(structure):
    """Convert a pymatgen Structure to (src [82], pos [82,3], meta).

    Strict contract: pos[0] = [a, b, 1/c] (NOT c), pos[1] = angles in degrees.
    """
    a, b, c = float(structure.lattice.a), float(structure.lattice.b), float(structure.lattice.c)
    if not np.isfinite([a, b, c]).all() or min(a, b, c) <= 1e-4:
        raise ValueError(f"Illegal lattice constants a={a} b={b} c={c}")
    inv_c = 1.0 / max(c, 1e-4)
    angles = [float(x) for x in structure.lattice.angles]
    atomic_numbers = [int(z) for z in structure.atomic_numbers]
    frac_coords = np.asarray(structure.frac_coords, dtype=np.float64)

    n = len(atomic_numbers)
    if n == 0:
        raise ValueError("Structure contains no atoms")
    if n > MAX_ATOMS:
        warnings.warn(
            f"Structure has {n} atoms > {MAX_ATOMS} slots; truncating to first {MAX_ATOMS}."
        )
        atomic_numbers = atomic_numbers[:MAX_ATOMS]
        frac_coords = frac_coords[:MAX_ATOMS]
        n = MAX_ATOMS

    src = torch.zeros(SRC_LEN, dtype=torch.long)
    src[0], src[1] = SENTINELS[0], SENTINELS[1]
    src[2: 2 + n] = torch.tensor(atomic_numbers, dtype=torch.long)

    pos = torch.zeros(POS_ROWS, 3, dtype=torch.float32)
    pos[0, :] = torch.tensor([a, b, inv_c], dtype=torch.float32)
    pos[1, :] = torch.tensor(angles, dtype=torch.float32)
    if n > 0:
        pos[2: 2 + n, :] = torch.tensor(frac_coords, dtype=torch.float32)

    # Material metadata for Slack kappa_L
    try:
        masses = [float(Element.from_Z(int(z)).atomic_mass) for z in atomic_numbers]
        m_avg = float(np.mean(masses))
    except Exception:
        m_avg = 50.0
    try:
        vol_cell = float(structure.lattice.volume)
    except Exception:
        vol_cell = 20.0 * max(n, 1)
    v_atom = float(vol_cell / max(n, 1))

    meta = {
        "formula": structure.composition.reduced_formula,
        "n_atoms": int(n),
        "a": a,
        "b": b,
        "c": c,
        "inv_c": float(inv_c),
        "angles": angles,
        "m_avg": m_avg,
        "vol_cell": vol_cell,
        "vol_per_atom": v_atom,
    }
    return src, pos, meta


def parse_cif_to_features(cif_path):
    """Parse a .cif file via pymatgen into model-ready tensors.

    Returns (src [82], pos [82,3], meta). Raises FileNotFoundError/ValueError.
    """
    cif_path = Path(cif_path)
    if not cif_path.is_file():
        raise FileNotFoundError(f"CIF not found: {cif_path}")
    structure = Structure.from_file(str(cif_path))
    src, pos, meta = structure_to_tensors(structure)
    meta["cif"] = str(cif_path)
    meta["stem"] = cif_path.stem
    return src, pos, meta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_m4_model(device, weights_path=None):
    """Build uniARPAT-v2 M4 Transformer and optionally load weights."""
    torch.manual_seed(0)
    model = Transformer(**M4_PARAMS)
    if weights_path:
        load_weights_into_model(model, weights_path, device)
    model.to(device)
    model.eval()
    return model


def load_weights_into_model(model, weights_path, device=None):
    """Load checkpoint into an M4 model.

    Accepts: raw Transformer state_dict (ablation checkpoints), or full
    basemodel dicts {'model': {'transformer': ...}} / {'state_dict': ...}.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    ckpt = torch.load(str(weights_path), map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                inner = ckpt[key]
                # basemodel format: {'model': {'transformer': state}}
                if key == "model" and "transformer" in inner:
                    state = inner["transformer"]
                else:
                    state = inner
                break
        if state is None:
            # Heuristic: raw state_dict contains transformer parameter names
            if any("tok_emb" in k or "edos_query" in k for k in ckpt.keys()):
                state = ckpt
    if state is None:
        raise ValueError(
            f"Unrecognised checkpoint format: {weights_path} "
            f"(keys={list(ckpt.keys())[:8] if isinstance(ckpt, dict) else type(ckpt)})"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Strictness: query/head/decoder shapes must match; missing norm buffers are fatal.
    critical_missing = [k for k in missing if "query_embed" in k or "out_head" in k]
    if critical_missing:
        raise RuntimeError(f"Checkpoint incompatible, missing keys: {critical_missing}")
    if missing:
        warnings.warn(f"Checkpoint loaded with missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        warnings.warn(f"Checkpoint loaded with unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if device is not None:
        model.to(device)
    return model


@torch.no_grad()
def predict_dos(model, src, pos, device):
    """Single-sample forward -> (edos [128], phdos [64]) numpy, clamped >= 0."""
    model.eval()
    inp = src.unsqueeze(0).to(device)
    p = pos.unsqueeze(0).to(device)
    mask = (inp == 0)
    outputs = model(inp, mask, p)
    edos = outputs["edos"]
    phdos = outputs["phdos"]
    # M5-style checkpoints carry phys_* blind outputs; prefer them if present.
    if "phys_edos" in outputs and "phys_phdos" in outputs:
        edos, phdos = outputs["phys_edos"], outputs["phys_phdos"]
    if edos.dim() == 3:
        edos = edos.squeeze(1)
    if phdos.dim() == 3:
        phdos = phdos.squeeze(1)
    edos = torch.clamp(edos[0].detach().cpu(), min=0.0)
    phdos = torch.clamp(phdos[0].detach().cpu(), min=0.0)
    for name, t in (("eDOS", edos), ("phDOS", phdos)):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"Inference produced non-finite {name}")
    return edos.numpy(), phdos.numpy()


# ---------------------------------------------------------------------------
# Thermodynamics linkage
# ---------------------------------------------------------------------------

def compute_thermo_props(phdos, meta, t_range=None):
    """Julian-Slack kappa_L(300K) + Debye Theta_D + continuous Cv(T)."""
    calc = ThermodynamicCalculator()
    if t_range is None:
        t_range = T_RANGE_DEFAULT
    t_range = np.asarray(t_range, dtype=np.float64)
    ph = np.clip(np.asarray(phdos, dtype=np.float64), 0.0, None)
    cv = np.asarray(calc.compute_Cv(ph, T_range=t_range), dtype=np.float64)
    theta_d = float(calc.compute_Debye_T(ph))
    kappa = float(
        calc.compute_Slack_kappaL(
            ph,
            M_avg=float(meta.get("m_avg", 50.0)),
            volume_per_atom=float(meta.get("vol_per_atom", 20.0)),
            n_atoms=int(meta.get("n_atoms", 1)),
            T=300.0,
        )
    )
    # Cv at 300 K (nearest grid point)
    cv_300 = float(cv[int(np.argmin(np.abs(t_range - 300.0)))]) if len(cv) else 0.0
    return {
        "T_range": t_range,
        "Cv": cv,
        "Cv_300K": cv_300,
        "Theta_D": theta_d,
        "kappa_L_300K": kappa,
    }


# ---------------------------------------------------------------------------
# Publication figure
# ---------------------------------------------------------------------------

def plot_publication_figure(edos, phdos, thermo, meta, save_png, save_pdf=None):
    """Save eDOS (Fermi@0eV) + phDOS + Cv(T) multi-panel figure."""
    save_png = Path(save_png)
    save_png.parent.mkdir(parents=True, exist_ok=True)
    t_range = np.asarray(thermo["T_range"])
    cv = np.asarray(thermo["Cv"])
    formula = meta.get("formula", meta.get("stem", "material"))
    title = (
        f"{formula}  uniARPAT-v2 (M4)  "
        f"$\\kappa_L(300K)$={thermo['kappa_L_300K']:.2f} W/mK  "
        f"$\\Theta_D$={thermo['Theta_D']:.0f} K  "
        f"$C_v(300K)$={thermo['Cv_300K']:.2f} J/molK"
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)

    # Panel 1: eDOS with Fermi level at 0 eV
    axs[0].plot(E_EDOS, edos, color="#d32f2f", linewidth=2.0, label="uniARPAT-v2 eDOS")
    axs[0].axvline(0.0, color="gray", linestyle=":", linewidth=1.5, label="Fermi level ($E_F$=0 eV)")
    axs[0].set_title("Electronic DOS", fontweight="bold")
    axs[0].set_xlabel("Energy ($E - E_F$, eV)")
    axs[0].set_ylabel("eDOS (states/eV)")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    # Panel 2: phDOS
    axs[1].plot(FREQ_PHDOS, phdos, color="#2b5c8f", linewidth=2.0, label="uniARPAT-v2 phDOS")
    axs[1].axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
    axs[1].set_title("Phonon DOS", fontweight="bold")
    axs[1].set_xlabel(r"Frequency ($\mathrm{cm}^{-1}$)")
    axs[1].set_ylabel("phDOS")
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].grid(alpha=0.3)

    # Panel 3: Cv(T) 50K~1000K
    axs[2].plot(t_range, cv, color="#388e3c", linewidth=2.0, label="$C_v(T)$")
    axs[2].axvline(300.0, color="gray", linestyle=":", linewidth=1.0, label="300 K")
    axs[2].set_title(r"Lattice Heat Capacity $C_v(T)$", fontweight="bold")
    axs[2].set_xlabel("Temperature (K)")
    axs[2].set_ylabel(r"$C_v$ (J / mol K)")
    axs[2].set_xlim(float(t_range.min()), float(t_range.max()))
    axs[2].set_ylim(0.0, max(float(cv.max()) * 1.1, 1e-6))
    axs[2].legend(loc="lower right", fontsize=8)
    axs[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save_png), dpi=300, bbox_inches="tight")
    if save_pdf is not None:
        save_pdf = Path(save_pdf)
        save_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_pdf), bbox_inches="tight")
    plt.close(fig)
    return str(save_png)


# ---------------------------------------------------------------------------
# End-to-end per-file pipeline
# ---------------------------------------------------------------------------

def process_single_cif(cif_path, model, device, output_dir, t_range=None):
    """Full CIF->DOS->thermo->figure pipeline for one file. Returns result dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src, pos, meta = parse_cif_to_features(cif_path)
    edos, phdos = predict_dos(model, src, pos, device)
    assert edos.shape == (EDOS_DIM,) and phdos.shape == (PHDOS_DIM,)
    assert np.all(np.isfinite(edos)) and np.all(np.isfinite(phdos))
    thermo = compute_thermo_props(phdos, meta, t_range=t_range)
    stem = meta["stem"]

    np.save(str(output_dir / f"{stem}_edos.npy"), edos)
    np.save(str(output_dir / f"{stem}_phdos.npy"), phdos)
    np.save(str(output_dir / f"{stem}_Cv.npy"), np.stack([thermo["T_range"], thermo["Cv"]], axis=1))
    png = str(output_dir / f"{stem}.png")
    pdf = str(output_dir / f"{stem}.pdf")
    plot_publication_figure(edos, phdos, thermo, meta, png, pdf)

    props = {
        "file": meta["cif"],
        "formula": meta["formula"],
        "n_atoms": meta["n_atoms"],
        "a": meta["a"],
        "b": meta["b"],
        "c": meta["c"],
        "kappa_L_300K": thermo["kappa_L_300K"],
        "Theta_D": thermo["Theta_D"],
        "Cv_300K": thermo["Cv_300K"],
        "edos_npy": str(output_dir / f"{stem}_edos.npy"),
        "phdos_npy": str(output_dir / f"{stem}_phdos.npy"),
        "figure_png": png,
        "figure_pdf": pdf,
    }
    with open(str(output_dir / f"{stem}_props.json"), "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in props.items()}, f, indent=2)
    return props


def collect_cif_files(cif=None, cif_dir=None):
    files = []
    if cif:
        files.append(Path(cif))
    if cif_dir:
        files.extend(sorted(Path(cif_dir).glob("*.cif")))
        files.extend(sorted(Path(cif_dir).glob("*.CIF")))
    # Deduplicate preserving order
    seen, uniq = set(), []
    for f in files:
        sf = str(f)
        if sf not in seen:
            seen.add(sf)
            uniq.append(f)
    return uniq


def resolve_device(device_str):
    device_str = (device_str or "cpu").lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device_str.startswith("cuda"):
        return torch.device(device_str if ":" in device_str else "cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# --smoke: fast end-to-end verification (no weights / no files needed)
# ---------------------------------------------------------------------------

def run_smoke(device=None, save_dir=None):
    """Synthetic diamond-Si end-to-end check: contract + forward + thermo.

    Mirrors tests/test_p0_fixes.py::test_cif2dos_smoke but routes through the
    production helpers (structure_to_tensors / predict_dos / compute_thermo).
    Returns True on success, raises AssertionError otherwise.
    """
    import math as _math

    device = device or torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    lattice = Lattice.cubic(5.43)
    struct = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    src, pos, meta = structure_to_tensors(struct)

    # Contract assertions
    assert tuple(src.shape) == (SRC_LEN,), f"src shape {tuple(src.shape)} != (82,)"
    assert tuple(pos.shape) == (POS_ROWS, 3), f"pos shape {tuple(pos.shape)} != (82,3)"
    assert pos.flatten().numel() == POS_FLAT
    assert src[0].item() == 126 and src[1].item() == 127
    expected_inv_c = 1.0 / 5.43
    assert abs(float(pos[0, 2]) - expected_inv_c) < 1e-6, (
        f"1/c contract violated: got {float(pos[0, 2])}, expect {expected_inv_c}"
    )
    assert meta["n_atoms"] == 2

    # Geometry contract: real-space volume ~= a*b*c after 1/c recovery
    try:
        from utils.relative_features import build_cell_from_lattice  # local import
    except ImportError:  # fallback when invoked from workspace parent
        from uniARPAT.utils.relative_features import build_cell_from_lattice
    cell, _ = build_cell_from_lattice(pos.unsqueeze(0))
    vol = torch.linalg.det(cell[0]).item()
    assert _math.isfinite(vol) and vol > 50.0, f"smoke geometry failed, vol={vol}"

    # Forward with fresh M4 (no checkpoint needed -> fast, CPU-friendly)
    model = build_m4_model(device, weights_path=None)
    edos, phdos = predict_dos(model, src, pos, device)
    assert edos.shape == (EDOS_DIM,) and phdos.shape == (PHDOS_DIM,)
    assert np.all(np.isfinite(edos)) and np.all(np.isfinite(phdos))

    # Thermo linkage
    thermo = compute_thermo_props(phdos, meta, t_range=np.linspace(50, 1000, 10))
    assert np.all(np.isfinite(thermo["Cv"])) and np.all(thermo["Cv"] >= 0.0)
    assert _math.isfinite(thermo["Theta_D"]) and thermo["Theta_D"] >= 0.0
    assert _math.isfinite(thermo["kappa_L_300K"]) and thermo["kappa_L_300K"] >= 0.0

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_publication_figure(
            edos, phdos, thermo, {**meta, "stem": "Si_smoke"},
            str(save_dir / "Si_smoke.png"),
        )

    print(
        "[cif2dos smoke] PASS  "
        f"src={tuple(src.shape)} pos={tuple(pos.shape)} inv_c={float(pos[0,2]):.5f}  "
        f"eDOS{edos.shape} phDOS{phdos.shape} no-NaN  "
        f"Theta_D={thermo['Theta_D']:.1f}K kappa_L={thermo['kappa_L_300K']:.3f}W/mK"
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="uniARPAT-v2 production tool: pure end-to-end CIF -> eDOS/phDOS + thermo + figure."
    )
    p.add_argument("--cif", type=str, default=None, help="Single CIF file path")
    p.add_argument("--cif_dir", type=str, default=None, help="Batch directory of *.cif")
    p.add_argument("--weights", type=str, default=None, help="M4 checkpoint (.pth)")
    p.add_argument("--output", type=str, default="./results/cif2dos_output", help="Output directory")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda[:id]")
    p.add_argument("--smoke", action="store_true", help="Fast end-to-end smoke verification")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    device = resolve_device(args.device)

    if args.smoke:
        ok = run_smoke(device)
        return 0 if ok else 1

    cif_files = collect_cif_files(args.cif, args.cif_dir)
    if not cif_files:
        print("ERROR: provide --cif <file> or --cif_dir <dir> (or use --smoke).", file=sys.stderr)
        return 2
    for f in cif_files:
        if not Path(f).is_file():
            print(f"ERROR: CIF not found: {f}", file=sys.stderr)
            return 2

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cif2dos] device={device}  files={len(cif_files)}  output={output_dir}")
    if args.weights:
        print(f"[cif2dos] weights={args.weights}")
    else:
        print("[cif2dos] WARNING: --weights not given; using randomly initialised M4 (demo mode).")

    model = build_m4_model(device, weights_path=args.weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cif2dos] M4 loaded: {n_params/1e6:.2f}M params")

    results = []
    for cif_path in cif_files:
        try:
            props = process_single_cif(cif_path, model, device, output_dir)
            results.append(props)
            print(
                f"[cif2dos] {Path(cif_path).name}: {props['formula']} "
                f"kappa_L={props['kappa_L_300K']:.3f} W/mK  "
                f"Theta_D={props['Theta_D']:.1f} K  Cv(300K)={props['Cv_300K']:.2f} J/molK  "
                f"-> {props['figure_png']}"
            )
        except Exception as exc:
            print(f"[cif2dos] FAILED {cif_path}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()
    if not results:
        print("[cif2dos] ERROR: all files failed.", file=sys.stderr)
        return 1

    # Batch summary
    import csv

    summary_csv = output_dir / "summary.csv"
    with open(str(summary_csv), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[cif2dos] Done: {len(results)}/{len(cif_files)} succeeded. Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
