"""Batched SumNorm/H1 reconstruction using only structure-derived scale inputs."""
import math

import torch


def reconstruct_h1_spectra(outputs, n_atoms, n_valence, *, delta_edos=0.09375,
                           delta_phdos=19.6875):
    """Generalize the established B7 H1 formula to arbitrary batch/bin counts.

    No target statistics enter blind spectra. Empty structures have zero totals;
    valid structures require positive finite atom and valence counts.
    """
    if not all(math.isfinite(x) and x > 0 for x in (delta_edos, delta_phdos)):
        raise ValueError("positive finite spectral bin widths required")
    e, p, scales = (outputs[key] for key in ("edos", "phdos", "eta"))
    if e.ndim != 2 or p.ndim != 2 or p.shape[0] != e.shape[0] or scales.shape != (e.shape[0], 2):
        raise ValueError("H1 logits/scales must have aligned batch shapes")
    if not all(torch.isfinite(value).all() for value in (e, p, scales)):
        raise ValueError("finite H1 logits and scales required")
    if (scales < 0).any() or (scales > 1).any():
        raise ValueError("H1 scales must lie in [0,1]")
    atoms = torch.as_tensor(n_atoms, device=e.device, dtype=e.dtype).reshape(-1, 1)
    valence = torch.as_tensor(n_valence, device=e.device, dtype=e.dtype).reshape(-1, 1)
    if atoms.shape != (e.shape[0], 1) or valence.shape != atoms.shape:
        raise ValueError("H1 structure counts must align with the batch")
    if not torch.isfinite(atoms).all() or not torch.isfinite(valence).all() \
            or (atoms < 0).any() or (valence < 0).any() \
            or ((atoms > 0) != (valence > 0)).any():
        raise ValueError("finite positive counts required for valid structures")
    shape_e, shape_p = e.softmax(-1), p.softmax(-1)
    return {"shape_edos": shape_e, "shape_phdos": shape_p,
            "blind_edos": shape_e * valence * scales[:, 1:2] / delta_edos,
            "blind_phdos": shape_p * 3 * atoms * scales[:, 0:1] / delta_phdos}
