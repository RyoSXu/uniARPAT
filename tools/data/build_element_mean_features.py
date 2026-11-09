"""Build element-only external atom features from frozen per-atom descriptors (M0, design 2026-11-09).

Every atom receives the mean descriptor of its element over the train atoms, so
the features keep the pretrained model's per-element chemistry but carry no
information about the atom's environment or the material. Elements absent from
train receive the train mean over all atoms (zero after standardization). The
output uses the same `external-atom-features-v1` layout as the source, with
train-only standardization statistics of the new arrays.

Usage (training environment, numpy only):
    python tools/data/build_element_mean_features.py \
        --source data/external/mace_mp0b3_medium_v1 --out data/external/mace_mp0b3_medium_elemmean_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from utils.external_atom_features import SCHEMA, load_manifest
from utils.paths import file_sha256, input_path

SPLITS = ("train", "valid", "test")


def split_arrays(source, data_dir, split, manifest):
    path = source / f"{split}.npy"
    if file_sha256(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"source features changed since extraction: {path}")
    features = np.load(path, allow_pickle=False)
    numbers = np.load(data_dir / split / f"atomic_numbers_{split}.npy", allow_pickle=False).astype(np.int64)
    if features.shape[0] != numbers.shape[0]:
        raise ValueError(f"{split}: features and atomic numbers disagree")
    return features, numbers


def element_table(features, numbers):
    """Per-element mean of the train descriptors (float64) and the global fallback."""
    elements = np.unique(numbers)
    table = {int(z): features[numbers == z].astype(np.float64).mean(0) for z in elements}
    counts = {int(z): int((numbers == z).sum()) for z in elements}
    return table, counts, features.astype(np.float64).mean(0)


def build(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f"use a new output directory: {out}")
    manifest = load_manifest(source)
    data_dir = Path(manifest["data"]["root"])
    if file_sha256(data_dir / "manifest.json") != manifest["data"]["manifest_sha256"]:
        raise ValueError("source features were extracted from a different crystal cache")
    train_features, train_numbers = split_arrays(source, data_dir, "train", manifest)
    table, counts, fallback = element_table(train_features, train_numbers)
    out.mkdir(parents=True)
    splits, unseen, train_stats = {}, {}, None
    for split in SPLITS:
        features, numbers = (train_features, train_numbers) if split == "train" else \
            split_arrays(source, data_dir, split, manifest)
        lookup = np.stack([table.get(int(z), fallback) for z in numbers]).astype(np.float32)
        unseen[split] = int(sum(int(z) not in table for z in numbers))
        if not np.isfinite(lookup).all():
            raise ValueError(f"{split}: non-finite element features")
        path = out / f"{split}.npy"
        np.save(path, lookup)
        if split == "train":
            values = lookup.astype(np.float64)
            std = values.std(0)
            train_stats = {"mean": values.mean(0).tolist(), "std": std.tolist(), "std_floor": 1e-6,
                           "near_constant_channels": int((std < 1e-6).sum())}
        splits[split] = {"materials": manifest["splits"][split]["materials"], "atoms": int(lookup.shape[0]),
                         "dim": int(lookup.shape[1]), "sha256": file_sha256(path)}
    result = {"schema": SCHEMA,
              "source": "element-only: per-element train mean of the source descriptors (no environment information)",
              "derived_from": {"root": str(source.resolve()), "manifest_sha256": file_sha256(source / "manifest.json"),
                               "source": manifest["source"]},
              "model": manifest["model"], "data": manifest["data"], "splits": splits,
              "layout": manifest["layout"],
              "elements": {"train_counts": counts, "unseen_atoms": unseen,
                           "unseen_fallback": "train mean over all atoms"},
              "standardization": {"fit": "train split atoms only", **train_stats}}
    (out / "manifest.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/external/mace_mp0b3_medium_v1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = build(input_path(args.source), input_path(args.out))
    print(json.dumps({k: v for k, v in result.items() if k not in ("standardization",)}, indent=1)[:2000])


if __name__ == "__main__":
    main()
