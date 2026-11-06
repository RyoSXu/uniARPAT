"""Training-only structure regularizers for the LR2 E+S generalization trial (design 2026-11-05).

G perturbs the training input geometry with a small random strain and Cartesian
atom displacement; labels stay fixed. D drops the summed local residual per
sample (DropPath convention). Neither changes evaluation inputs or the eval path.
"""
from __future__ import annotations

import torch

STRAIN_ISOTROPIC_HALF_WIDTH = 0.01   # u ~ U(-0.01, 0.01): about +/-3 % volume
STRAIN_SHEAR_SD = 0.003               # symmetric off-diagonal strain components
DISPLACEMENT_SD_ANGSTROM = 0.02
DISPLACEMENT_CLIP_ANGSTROM = 0.06
LOCAL_DROP_PROB = 0.25

STRAIN_AUGMENT_CONTRACT = {
    "schema": "lr2-es-strain-augment-v1",
    "applies_to": "training inputs only; labels, valid and test inputs unchanged",
    "strain": "cell' = cell (I + eps)^T, eps = u I + S, u ~ U(-0.01, 0.01), "
              "S symmetric off-diagonal with entries N(0, 0.003^2)",
    "displacement": "Cartesian N(0, (0.02 A)^2) per atom component, clipped to +/-0.06 A, "
                    "then wrapped into the strained cell",
    "random_source": "torch generator of the DataLoader worker (seeded from the training seed)",
}


def local_drop_contract(prob):
    return {"schema": "lr2-es-local-drop-v1", "prob": prob,
            "rule": "training only: per sample, the summed residual of both local calls is zeroed with "
                    "probability p and scaled by 1/(1-p) otherwise; eval uses the full path unscaled",
            "dropped_sample": "identical to the structure-blind S0 path for that step"}


def strain_perturb(cell: torch.Tensor, frac: torch.Tensor, generator: torch.Generator | None = None):
    """Return a strained cell and displaced fractional coordinates in [0, 1)."""
    dtype = cell.dtype
    cell64, frac64 = cell.to(torch.float64), frac.to(torch.float64)
    u = (torch.rand((), generator=generator, dtype=torch.float64) * 2 - 1) * STRAIN_ISOTROPIC_HALF_WIDTH
    shear = torch.randn(3, 3, generator=generator, dtype=torch.float64) * STRAIN_SHEAR_SD
    shear = torch.triu(shear, diagonal=1)
    eps = u * torch.eye(3, dtype=torch.float64) + shear + shear.T
    strained = cell64 @ (torch.eye(3, dtype=torch.float64) + eps).T
    displacement = torch.randn(frac64.shape, generator=generator, dtype=torch.float64) * DISPLACEMENT_SD_ANGSTROM
    displacement = displacement.clamp(-DISPLACEMENT_CLIP_ANGSTROM, DISPLACEMENT_CLIP_ANGSTROM)
    moved = (frac64 + displacement @ torch.linalg.inv(strained)) % 1.0
    return strained.to(dtype), moved.to(frac.dtype)


def drop_local_residual(initial: torch.Tensor, updated: torch.Tensor, prob: float,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """DropPath over the whole local stack, one Bernoulli draw per sample."""
    if not 0. < prob < 1.:
        raise ValueError("local drop probability must lie in (0, 1)")
    keep = (torch.rand((updated.shape[0],) + (1,) * (updated.dim() - 1), device=updated.device,
                       generator=generator) >= prob).to(updated.dtype)
    return initial + (updated - initial) * keep / (1. - prob)
