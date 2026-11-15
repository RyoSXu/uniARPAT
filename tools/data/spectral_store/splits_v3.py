"""Full re-split of spectral-store-v1 (user decision 2026-11-12: old valid/test membership not kept).

Every unit -- legacy material, new material or independent entry -- is placed only by its reduced
formula:
- formula containing a rare element (fixed list) -> train (D7);
- otherwise b = SHA-256(SALT : formula)[:12 hex] / 16^12: b < 0.10 -> test, b < 0.20 -> valid,
  else train (D3).
The rule does not depend on labels, models, earlier splits or the order in which data arrive.
"""
from __future__ import annotations

import hashlib

SALT = 'spectral-store-v1/splits-v3/2026-11-12'
RARE_ELEMENTS = frozenset({'He', 'Ne', 'Ar', 'Kr'})
TEST_MAX, VALID_MAX = 0.10, 0.20


def bucket(formula: str) -> float:
    return int(hashlib.sha256(f'{SALT}:{formula}'.encode()).hexdigest()[:12], 16) / 16 ** 12


def assign(formula: str, elements) -> str:
    if RARE_ELEMENTS.intersection(elements):
        return 'train'
    b = bucket(formula)
    return 'test' if b < TEST_MAX else 'valid' if b < VALID_MAX else 'train'
