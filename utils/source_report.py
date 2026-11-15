"""Per-source and per-flag evaluation of a v2 partial-label view (design 2026-11-11 splits-v2, T5).

Groups are formed only from view metadata (label source, link kind, disputed and volume flags,
independent entries) and, for legacy rows, from the structure check of the record their label
came from. Each group reports count, mean and median blind energy, median R2 and failure rate,
computed on the rows that carry that task's label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ("edos", "phdos")
VOLUME_FLAG = 0.10
RECORD_PREFIX = {"edos": {"mp": "mp_edos"}, "phdos": {"mp": "mp_phonon", "phonondb": "phonondb", "jarvis": "jarvis"}}
# Origin ids only (no spectra) of the step-2 legacy labels; a re-split may place any legacy material anywhere.
LEGACY_RAW = ("exports/q1_v2_raw/train.parquet", "exports/q1_v2_raw/valid.parquet",
              "sealed/exports/q1_v2_raw/test.parquet")


def view_metadata(root, split):
    """One row per material/entry of a public split, with decoded label sources and flags."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    source_names = {code: name for name, code in manifest["source_codes"].items()}
    link_names = {code: name for name, code in manifest["link_codes"].items()}

    def load(key):
        name = f"{split}_index.npy" if key == "index" else f"{key}_{split}.npy"
        return np.load(root / split / name, allow_pickle=False)

    frame = pd.DataFrame({"material_id": load("index").astype(str), "is_entry": load("is_entry")})
    for task in TASKS:
        frame[f"has_{task}"] = load(f"has_{task}")
        frame[f"source_{task}"] = [source_names[int(c)] for c in load(f"source_{task}")]
        frame[f"link_{task}"] = [link_names[int(c)] for c in load(f"link_{task}")]
        frame[f"disputed_{task}"] = load(f"disputed_{task}")
        frame[f"volume_flag_{task}"] = load(f"volume_flag_{task}")
        frame[f"structure_mismatch_{task}"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    return frame.set_index("material_id")


def legacy_structure_flags(store, material_ids):
    """Structure check of the record behind each legacy label (step-2 origin); NA when not checkable.

    structure_mismatch: the record's calculation structure fails the default matcher (or the crystal
    system) against the legacy material; volume_flag: per-atom volume differs by more than 10%.
    """
    store = Path(store)
    raw = pd.concat([pd.read_parquet(store / path, columns=["material_id", "task", "origin"]) for path in LEGACY_RAW])
    raw = raw[raw.material_id.isin(set(material_ids))]
    if raw.duplicated(["material_id", "task"]).any():
        raise ValueError("a legacy label has more than one origin")
    features = pd.read_parquet(store / "staging/identity/record_features.parquet",
                               columns=["record_id", "hint", "cs_record", "cs_hint", "default", "volume_ratio"])
    features = features.set_index("record_id")
    out = {}
    for row in raw.itertuples():
        prefix, _, ident = row.origin.partition(":")
        record = f"{RECORD_PREFIX[row.task].get(prefix, prefix)}:{ident}"
        mismatch = volume = pd.NA
        if record in features.index:
            f = features.loc[record]
            if f.hint == row.material_id and not pd.isna(f.default):
                mismatch = not (bool(f.default) and f.cs_record == f.cs_hint)
                volume = bool(abs(float(f.volume_ratio) - 1) > VOLUME_FLAG)
        out[(row.material_id, row.task)] = (mismatch, volume)
    return out


def attach_legacy_flags(meta, store, split):
    legacy = meta.index[(meta.source_edos == "legacy") | (meta.source_phdos == "legacy")]
    if not len(legacy):
        return meta
    flags = legacy_structure_flags(store, legacy)
    meta = meta.copy()
    for task in TASKS:
        rows = meta.index[meta[f"source_{task}"] == "legacy"]
        values = [flags.get((m, task), (pd.NA, pd.NA)) for m in rows]
        meta.loc[rows, f"structure_mismatch_{task}"] = pd.array([v[0] for v in values], dtype="boolean")
        meta.loc[rows, f"volume_flag_{task}"] = [bool(v[1]) if not pd.isna(v[1]) else False for v in values]
        meta.loc[rows, f"volume_unknown_{task}"] = [pd.isna(v[1]) for v in values]
    return meta


def _summary(values, r2):
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "energy_mean": float(values.mean()), "energy_median": float(values.median()),
            "r2_median": float(r2.median()), "fail_rate": float((r2 < 0).mean() * 100)}


def grouped_report(samples, root, split, mode="blind"):
    """samples: per-material frame with material_id, {mode}_energy_<task> and {mode}_r2_<task>."""
    root = Path(root)
    meta = attach_legacy_flags(view_metadata(root, split), root.parent.parent, split)
    frame = samples.set_index("material_id")
    if set(frame.index) != set(meta.index) or frame.index.duplicated().any():
        raise ValueError("samples must cover the view split exactly once")
    frame = frame.join(meta[[c for c in meta.columns if c not in frame.columns]])
    report = {"split": split, "mode": mode, "tasks": {}}
    for task in TASKS:
        rows = frame[frame[f"has_{task}"].astype(bool)]
        energy, r2 = rows[f"{mode}_energy_{task}"], rows[f"{mode}_r2_{task}"]
        if energy.isna().any() or r2.isna().any():
            raise ValueError(f"labelled {task} rows lack scores")
        mismatch = rows[f"structure_mismatch_{task}"]
        clean = ~rows[f"disputed_{task}"].astype(bool) & ~rows[f"volume_flag_{task}"].astype(bool) \
            & ~mismatch.fillna(False).astype(bool)
        groups = {"all": rows.index, "clean": rows.index[clean]}
        for column in (f"source_{task}", f"link_{task}"):
            for value in sorted(rows[column].unique()):
                groups[f"{column.rsplit('_', 1)[0]}={value}"] = rows.index[rows[column] == value]
        for column in (f"disputed_{task}", f"volume_flag_{task}", "is_entry"):
            groups[f"{column.removesuffix('_' + task)}"] = rows.index[rows[column].astype(bool)]
        groups["structure_mismatch"] = rows.index[mismatch.fillna(False).astype(bool)]
        groups["structure_unchecked"] = rows.index[mismatch.isna() & (rows[f"source_{task}"] == "legacy")]
        report["tasks"][task] = {name: _summary(energy.loc[index], r2.loc[index]) for name, index in groups.items()}
    return report
