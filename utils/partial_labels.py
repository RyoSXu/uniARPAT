"""Partial-label rows of v2 views (design 2026-11-11 splits-v2, W1 task weighting).

A row may carry only one of the two spectra. Its missing target is stored as zeros with an
all-false coverage mask; ``has_<task>`` says which targets exist. The flags ride in a fixed
batch field after the optional external atom features, as a [batch, 2] bool tensor (eDOS, phDOS).
"""
from __future__ import annotations

import torch

from utils.external_atom_features import FIELD as EXTERNAL_ATOM_FIELD

FIELD = EXTERNAL_ATOM_FIELD + 1
TASKS = ("edos", "phdos")


def batch_labels(batch):
    """[batch, 2] bool label flags, or None for a fully labelled (v1 or dual-only) batch."""
    if len(batch) <= FIELD or batch[FIELD] is None:
        return None
    labels = batch[FIELD]
    if not torch.is_tensor(labels) or labels.dtype != torch.bool or labels.shape != (len(batch[0]), 2):
        raise ValueError("partial-label flags must be a [batch, 2] bool tensor")
    return labels


def labelled_mean(values, labels, denominator=None):
    """Sum over labelled rows divided by their count (or a caller-fixed batch count); 0 with no label.

    ``values`` is per-row and may be non-zero on unlabelled rows; those rows contribute nothing.
    """
    if values.shape != labels.shape:
        raise ValueError("per-row values and label flags must align")
    count = labels.sum() if denominator is None else torch.as_tensor(denominator, device=values.device)
    total = torch.where(labels, values, torch.zeros_like(values)).sum()
    return total / count.clamp_min(1).to(values.dtype)
