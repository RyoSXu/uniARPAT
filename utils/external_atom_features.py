"""Frozen external per-atom features (design 2026-11-08, MACE frozen-feature reference).

`tools/data/extract_mace_descriptors.py` writes, in a separate environment,
one float32 array per split aligned with the packed crystal atom order plus a
manifest. This module verifies that manifest against the crystal cache,
standardizes with the frozen train statistics and serves per-material blocks.
The batch carries the padded block as field 17; `Model.data_preprocess`
attaches it to the geometry so every model call site receives it unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from utils.paths import file_sha256

SCHEMA = "external-atom-features-v1"
FIELD = 17  # after the 15 base fields and the two bin-center fields


def load_manifest(root, data_dir=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"not an external atom-feature manifest: {root}")
    if data_dir is not None:
        if file_sha256(Path(data_dir) / "manifest.json") != manifest["data"]["manifest_sha256"]:
            raise ValueError("external atom features were extracted from a different crystal cache")
    std = np.asarray(manifest["standardization"]["std"], dtype=np.float64)
    if std.shape != (manifest["layout"]["dim"],):
        raise ValueError("standardization width differs from the feature width")
    return manifest


def feature_contract(root, data_dir):
    """Identity of the frozen features for training records, contracts and export checks."""
    root = Path(root)
    manifest = load_manifest(root, data_dir)
    return {"schema": SCHEMA, "root": str(root), "manifest_sha256": file_sha256(root / "manifest.json"),
            "model_sha256": manifest["model"]["sha256"], "model_name": manifest["model"]["name"],
            "dim": int(manifest["layout"]["dim"]), "layers": int(manifest["layout"]["layers"]),
            "split_sha256": {split: entry["sha256"] for split, entry in manifest["splits"].items()},
            "standardization": "frozen train mean/std from the manifest (std floor applied)"}


class ExternalAtomFeatures:
    """Standardized float32 features of one split, sliced by packed atom offsets."""

    def __init__(self, root, split, data_dir, offsets):
        root = Path(root)
        manifest = load_manifest(root, data_dir)
        entry = manifest["splits"][split]
        path = root / f"{split}.npy"
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"external atom features changed since extraction: {path}")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (entry["atoms"], manifest["layout"]["dim"]):
            raise ValueError(f"unexpected external feature array: {array.dtype} {array.shape}")
        if array.shape[0] != int(offsets[-1]):
            raise ValueError("external features and packed crystal atoms disagree")
        stats = manifest["standardization"]
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.maximum(np.asarray(stats["std"], dtype=np.float64), float(stats["std_floor"]))
        standardized = ((array.astype(np.float64) - mean) / std).astype(np.float32)
        if not np.isfinite(standardized).all():
            raise ValueError("non-finite standardized external features")
        self.values = torch.from_numpy(standardized)
        self.offsets = offsets
        self.dim = int(manifest["layout"]["dim"])

    def __getitem__(self, structure_index):
        start, end = self.offsets[structure_index:structure_index + 2]
        return self.values[start:end]
