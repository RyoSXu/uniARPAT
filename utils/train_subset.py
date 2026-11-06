"""Nested composition-group train subsets for the LR2 E+S learning curve (design 2026-11-06).

A subset keeps whole absolute-composition groups of the default train view, so
materials of one composition never straddle the subset boundary. Groups are
shuffled once with a fixed seed and taken as prefixes, so every smaller subset
is contained in every larger one. Each training epoch repeats the subset
``repeat`` times (``repeat * fraction = 1``): one epoch keeps about the number
of optimizer steps of a full-train epoch, so the 35-epoch warmup-cosine schedule,
the per-epoch valid evaluation and the checkpoint rule stay unchanged.
The frozen balance protocol (train spectral energies and reference scales) is
always the full-train one; a subset never refits it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "lr2-es-learning-curve-subset-v1"
SAMPLING_SEED = 20261106
# Fraction -> repeat; repeat * fraction == 1 keeps one full-train epoch of steps.
FRACTIONS = {0.25: 4, 0.5: 2}
GROUP_KEY = "absolute composition: sorted (atomic number, count) pairs of the complete structure"


def composition_key(numbers) -> tuple:
    z, counts = np.unique(np.asarray(numbers, dtype=int), return_counts=True)
    return tuple((int(a), int(c)) for a, c in zip(z, counts))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_view(data_dir):
    """Material ids and composition keys of the default train view, in row order."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    excluded = set(manifest.get("training_exclusions", []))
    directory = data_dir / "train"
    ids = np.load(directory / "train_index.npy", allow_pickle=False).astype(str)
    offsets = np.load(directory / "atom_offsets_train.npy")
    numbers = np.load(directory / "atomic_numbers_train.npy")
    keep = [i for i, material in enumerate(ids) if material not in excluded]
    return ([str(ids[i]) for i in keep],
            [composition_key(numbers[offsets[i]:offsets[i + 1]]) for i in keep])


def build_subsets(data_dir, fractions=tuple(FRACTIONS), seed=SAMPLING_SEED):
    """Nested group-prefix subsets; each is the smallest prefix reaching its fraction."""
    data_dir = Path(data_dir)
    if any(f not in FRACTIONS for f in fractions):
        raise ValueError(f"learning-curve fractions are fixed at {sorted(FRACTIONS)}")
    ids, keys = train_view(data_dir)
    groups = {}
    for material, key in zip(ids, keys):
        groups.setdefault(key, []).append(material)
    ordered = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(ordered))
    position = {material: row for row, material in enumerate(ids)}
    manifest_sha = _sha256(data_dir / "manifest.json")
    subsets, chosen, taken = {}, [], 0
    for fraction in sorted(fractions):
        target = fraction * len(ids)
        while taken < target:
            key = ordered[order[len(chosen)]]
            chosen.append(key)
            taken += len(groups[key])
        members = sorted((m for key in chosen for m in groups[key]), key=position.__getitem__)
        subsets[fraction] = {
            "schema": SCHEMA, "fraction": fraction, "repeat": FRACTIONS[fraction],
            "group_key": GROUP_KEY, "sampling_seed": seed, "nested": "group-prefix of one fixed permutation",
            "source_manifest_sha256": manifest_sha, "train_view_materials": len(ids),
            "train_view_groups": len(groups), "groups": len(chosen), "materials": len(members),
            "realized_fraction": len(members) / len(ids),
            "materials_per_epoch": len(members) * FRACTIONS[fraction],
            "balance_protocol": "full-train frozen protocol; not refitted on the subset",
            "material_ids": members,
        }
    return subsets


def write_subset(subset: dict, path) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite a frozen subset: {path}")
    path.write_text(json.dumps(subset, ensure_ascii=False, indent=1) + "\n")
    return _sha256(path)


def load_train_subset(path, data_dir) -> dict:
    """Validated subset record plus its file hash; rejects a subset of another data version."""
    path = Path(path)
    subset = json.loads(path.read_text())
    if subset.get("schema") != SCHEMA:
        raise ValueError("not a learning-curve train subset")
    if FRACTIONS.get(subset.get("fraction")) != subset.get("repeat"):
        raise ValueError("subset fraction/repeat pair is not one of the fixed learning-curve points")
    if subset["source_manifest_sha256"] != _sha256(Path(data_dir) / "manifest.json"):
        raise ValueError("subset was drawn from another data manifest")
    ids = subset["material_ids"]
    if len(ids) != subset["materials"] or len(set(ids)) != len(ids):
        raise ValueError("subset material list is inconsistent")
    return {**subset, "sha256": _sha256(path), "path": str(path.resolve())}


def subset_contract(subset: dict) -> dict:
    """What a checkpoint records about its training data; ids are bound by the file hash."""
    return {"schema": SCHEMA, "fraction": subset["fraction"], "repeat": subset["repeat"],
            "materials": subset["materials"], "groups": subset["groups"],
            "sampling_seed": subset["sampling_seed"], "file_sha256": subset["sha256"],
            "design": "docs/design/design-2026-11-06-learning-curve.md",
            "scope": "train rows restricted to whole composition groups and repeated per epoch; model, loss, "
                     "frozen full-train protocol, schedule, valid evaluation and checkpoint guards unchanged"}
