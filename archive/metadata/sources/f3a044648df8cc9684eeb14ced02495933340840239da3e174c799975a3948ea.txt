"""Frozen B7 M1 CIF-only blind-inference contract.

This module intentionally does not reuse the legacy M4 ``cif2dos.py`` path:
B7 predicts a SumNorm shape plus H1 eta/gamma scale, whereas M4's output
semantics and eDOS grid are different.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pymatgen.core import Structure

from model.transformer import Transformer


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_ATOMS = 80
SRC_LEN = MAX_ATOMS + 2
SENTINELS = (126, 127)
B7_EPOCH = 33
B7_SEED = 42
DELTA_EDOS = 0.09375
DELTA_PHDOS = 19.6875

# This is the effective B7 `_e9ctl` transformer configuration, not the
# editable template in configs/default.yaml.
B7_TRANSFORMER_PARAMS = {
    "token_num": 118,
    "d_model": 512,
    "nhead": 8,
    "edos_num": 128,
    "phdos_num": 64,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "dim_feedforward": 2048,
    "dropout": 0.05,
    "activation": "gelu",
    "normalize_before": False,
    "decoupled_decoder": False,
    "use_gated_cross_attn": False,
    "head_type": "legacy",
    "predict_scale": False,
    "atom_feat_mode": "legacy3",
    "energy_code": "none",
    "scale_mode": "eta",
    "scalar_mode": "none",
}


def resolve_device(device_name: str) -> torch.device:
    """Resolve a requested device without silently accepting an invalid CUDA run."""
    name = str(device_name).lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" or name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(name)
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")


def load_b7_grids() -> tuple[np.ndarray, np.ndarray]:
    """Return B7's E0/P0 bin centers from the versioned edge contract."""
    path = REPO_ROOT / "data" / "grids_c2b" / "grids.json"
    with path.open(encoding="utf-8") as handle:
        grids = json.load(handle)
    e_edges = np.asarray(grids["E0"], dtype=np.float64)
    p_edges = np.asarray(grids["P0"], dtype=np.float64)
    if e_edges.shape != (129,) or p_edges.shape != (65,):
        raise ValueError("B7 E0/P0 grid shape drift")
    if not np.allclose(np.diff(e_edges), DELTA_EDOS) or not np.allclose(np.diff(p_edges), DELTA_PHDOS):
        raise ValueError("B7 E0/P0 grid spacing drift")
    return (e_edges[:-1] + e_edges[1:]) / 2.0, (p_edges[:-1] + p_edges[1:]) / 2.0


def _zval_table() -> dict[str, dict[str, Any]]:
    with (REPO_ROOT / "index" / "z0_zval.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def structure_to_b7_inputs(structure: Structure) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Make B7's fixed 82-row input and CIF-derived H1 scale metadata.

    The CIF cell is used exactly as supplied.  A user who changes a primitive
    cell into a supercell changes ``N_atoms`` and ``N_val`` and must treat it as
    a different B7 input rather than an interchangeable serialization.
    """
    if not structure.is_ordered:
        raise ValueError("B7 CIF inference requires an ordered structure")
    n_atoms = len(structure)
    if n_atoms == 0:
        raise ValueError("CIF structure contains no atoms")
    if n_atoms > MAX_ATOMS:
        raise ValueError(f"B7 supports at most {MAX_ATOMS} atoms per CIF cell; got {n_atoms}")

    lattice = structure.lattice
    lengths = np.asarray(lattice.abc, dtype=np.float64)
    angles = np.asarray(lattice.angles, dtype=np.float64)
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    if not np.isfinite(lengths).all() or not np.isfinite(angles).all() or not np.isfinite(frac).all():
        raise ValueError("CIF lattice or fractional coordinates contain non-finite values")
    if np.any(lengths <= 1e-4):
        raise ValueError(f"invalid CIF lattice lengths: {lengths.tolist()}")

    atomic_numbers = [int(site.specie.Z) for site in structure]
    if any(z <= 0 or z >= B7_TRANSFORMER_PARAMS["token_num"] for z in atomic_numbers):
        raise ValueError("CIF contains an atomic number outside B7 token embedding support")

    zval = _zval_table()
    symbols = [site.specie.symbol for site in structure]
    missing = sorted({symbol for symbol in symbols if symbol not in zval})
    if missing:
        raise ValueError(f"CIF elements absent from frozen Z0 valence table: {', '.join(missing)}")
    n_valence = float(sum(float(zval[symbol]["zval"]) for symbol in symbols))
    confidence_min = int(min(int(zval[symbol]["conf"]) for symbol in symbols))

    src = torch.zeros(SRC_LEN, dtype=torch.long)
    src[0], src[1] = SENTINELS
    src[2:2 + n_atoms] = torch.as_tensor(atomic_numbers, dtype=torch.long)
    pos = torch.zeros(SRC_LEN, 3, dtype=torch.float32)
    pos[0] = torch.as_tensor([lengths[0], lengths[1], 1.0 / lengths[2]], dtype=torch.float32)
    pos[1] = torch.as_tensor(angles, dtype=torch.float32)
    pos[2:2 + n_atoms] = torch.as_tensor(frac, dtype=torch.float32)
    metadata = {
        "formula": structure.composition.reduced_formula,
        "n_atoms": n_atoms,
        "n_valence": n_valence,
        "zval_confidence_min": confidence_min,
        "cell_lengths_angstrom": [float(x) for x in lengths],
        "cell_angles_deg": [float(x) for x in angles],
        "input_contract": "CIF cell as supplied; pos[0]=[a,b,1/c], pos[1]=[alpha,beta,gamma]",
    }
    return src, pos, metadata


def checkpoint_state(checkpoint_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read and authenticate the specific B7 checkpoint payload before loading."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"B7 checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("B7 checkpoint must be an ablation payload with a 'model' state dict")
    expected = {"model_name": "M1", "epoch": B7_EPOCH, "seed": B7_SEED}
    mismatched = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatched:
        raise ValueError(f"checkpoint is not frozen B7 _e9ctl: {mismatched}")
    metadata = {key: checkpoint.get(key) for key in ("model_name", "epoch", "seed", "use_amp", "best_val_score")}
    return checkpoint["model"], metadata


def load_b7_model(checkpoint_path: str | Path, device: torch.device) -> tuple[Transformer, dict[str, Any]]:
    """Strictly load B7's state dict into its frozen M1 architecture."""
    state, metadata = checkpoint_state(checkpoint_path)
    model = Transformer(**B7_TRANSFORMER_PARAMS)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, metadata


@torch.no_grad()
def predict_b7_blind(
    model: torch.nn.Module,
    src: torch.Tensor,
    pos: torch.Tensor,
    n_valence: float,
    device: torch.device,
) -> dict[str, Any]:
    """Apply the frozen SumNorm + H1 blind scale reconstruction."""
    inp = src.unsqueeze(0).to(device)
    positions = pos.unsqueeze(0).to(device)
    mask = inp.eq(0)
    outputs = model(inp, mask, positions)
    if not {"edos", "phdos", "eta"}.issubset(outputs):
        raise RuntimeError("B7 model output is missing eDOS, phDOS, or H1 eta/gamma")
    logits_e, logits_p, eta_gamma = outputs["edos"], outputs["phdos"], outputs["eta"]
    if logits_e.shape != (1, 128) or logits_p.shape != (1, 64) or eta_gamma.shape != (1, 2):
        raise RuntimeError("B7 model output shapes do not match the frozen E0/P0 + H1 contract")
    if not all(torch.isfinite(value).all() for value in (logits_e, logits_p, eta_gamma)):
        raise RuntimeError("B7 inference produced non-finite logits or H1 scales")
    if (eta_gamma < -1e-6).any() or (eta_gamma > 1.0 + 1e-6).any():
        raise RuntimeError("B7 H1 eta/gamma must be bounded in [0, 1]")
    eta_gamma = eta_gamma.clamp(0.0, 1.0)

    n_atoms = int((src[2:] != 0).sum().item())
    if n_atoms < 1 or not np.isfinite(n_valence) or n_valence <= 0:
        raise ValueError("CIF-derived atom count and N_valence must be positive and finite")
    shape_e = F.softmax(logits_e, dim=-1)
    shape_p = F.softmax(logits_p, dim=-1)
    eta_ph, gamma_e = eta_gamma[0, 0], eta_gamma[0, 1]
    total_e = torch.as_tensor(n_valence, device=device) * gamma_e / DELTA_EDOS
    total_p = 3.0 * n_atoms * eta_ph / DELTA_PHDOS
    edos = (shape_e[0] * total_e).detach().cpu().numpy()
    phdos = (shape_p[0] * total_p).detach().cpu().numpy()
    if not np.isfinite(edos).all() or not np.isfinite(phdos).all() or (edos < 0).any() or (phdos < 0).any():
        raise RuntimeError("B7 blind reconstruction produced invalid spectra")
    return {
        "edos": edos,
        "phdos": phdos,
        "eta_ph": float(eta_ph.item()),
        "gamma_e": float(gamma_e.item()),
        "edos_sum": float(total_e.item()),
        "phdos_sum": float(total_p.item()),
        "n_atoms": n_atoms,
        "n_valence": float(n_valence),
    }
