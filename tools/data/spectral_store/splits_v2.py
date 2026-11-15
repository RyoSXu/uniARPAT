"""Fixed split rule of spectral-store-v1 (design 2026-11-11 splits-v2, approved 2026-11-11/12).

A material or independent entry is placed only by its reduced formula:
- formula group of a legacy valid_q1 / test_q1 material -> that side (legacy members keep
  their split; other members of the group become the evaluation supplement of that side,
  never training) -- decision C3/D2; legacy train members of such groups are held out
  (eval_group_holdout);
- formula group of a legacy train_q1 material (not on an evaluation side) -> train (design 3.2);
- formula containing a rare element (fixed list) -> train (D7);
- otherwise b = SHA-256(SALT : formula)[:12 hex] / 16^12: b < 0.10 -> test, b < 0.20 -> dev,
  else train (D3).
The rule does not depend on labels, models or the order in which data arrive.
"""
from __future__ import annotations

import hashlib

SALT = 'spectral-store-v1/splits-v2/2026-11-12'
RARE_ELEMENTS = frozenset({'He', 'Ne', 'Ar', 'Kr'})  # < 20 labelled formula groups each (Xe: 53, not rare)
TEST_MAX, DEV_MAX = 0.10, 0.20


def bucket(formula: str) -> float:
    return int(hashlib.sha256(f'{SALT}:{formula}'.encode()).hexdigest()[:12], 16) / 16 ** 12


def assign(formula: str, elements, legacy_valid_formulas: set, legacy_test_formulas: set,
           legacy_train_formulas: set) -> str:
    """'test_side' / 'valid_side' (legacy groups), or 'train' / 'dev' / 'test' (new groups)."""
    if formula in legacy_test_formulas:
        return 'test_side'
    if formula in legacy_valid_formulas:
        return 'valid_side'
    if formula in legacy_train_formulas or RARE_ELEMENTS.intersection(elements):
        return 'train'
    b = bucket(formula)
    return 'test' if b < TEST_MAX else 'dev' if b < DEV_MAX else 'train'
