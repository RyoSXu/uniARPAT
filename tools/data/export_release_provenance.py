#!/usr/bin/env python3
"""Export source metadata for an existing crystal cache; never rebuild labels.

Requires the historical v2 processed parquet only when producing the release.
Consumers of the resulting release do not need that external source file.
"""

import argparse
from collections import Counter
import csv
from functools import reduce
import gzip
import hashlib
import io
import json
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_provenance(cache, processed, out):
    """Check every frozen spectrum/mask and export metadata into a fresh folder."""
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output: {out}")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "crystal-q1-v1":
        raise ValueError("expected crystal-q1-v1")
    columns = ["mpid", "split", "src_edos", "src_ph", "ph_weight",
               "edos_ref", "ph_ref", "struct_ref", "prov", "nsites_prim",
               "edos", "phdos", "edos_mask", "phdos_mask"]
    parent = pq.read_table(processed, columns=columns).to_pylist()
    lookup = {row["mpid"]: row for row in parent}
    if len(lookup) != len(parent):
        raise ValueError("duplicate parent material IDs")
    excluded = set(manifest.get("training_exclusions", []))
    report = {
        "schema": "crystal-q1-v1-provenance-v1",
        "cache_manifest_sha256": sha256(cache / "manifest.json"),
        "parent_processed_sha256": sha256(processed),
        "parent_rows": len(parent),
        "verification_scope": "stored arrays and source metadata; no model evaluation",
        "splits": {}, "overlap": {},
    }
    records, ids, exact, reduced = [], {}, {}, {}
    for split in ("train", "valid", "test"):
        directory = cache / split
        material_ids = np.load(directory / f"{split}_index.npy", allow_pickle=False).astype(str)
        ids[split] = set(material_ids)
        if len(ids[split]) != len(material_ids):
            raise ValueError(f"duplicate IDs: {split}")
        rows = [lookup[material_id] for material_id in material_ids]
        if any(row["split"] != split for row in rows):
            raise ValueError(f"parent split mismatch: {split}")
        zero_spectra = {}
        for task in ("edos", "phdos"):
            for suffix, column in (("tgtdos", task), ("mask", task + "_mask")):
                values = np.load(directory / f"{task}_{suffix}_{split}.npy", allow_pickle=False)
                original = np.asarray([row[column] for row in rows], dtype=values.dtype)
                if not np.array_equal(values, original):
                    raise ValueError(f"frozen values differ: {split}/{task}/{suffix}")
                if suffix == "tgtdos":
                    zero_spectra[task] = np.all(values == 0, axis=1)
        numbers = np.load(directory / f"atomic_numbers_{split}.npy", allow_pickle=False)
        offsets = np.load(directory / f"atom_offsets_{split}.npy", allow_pickle=False)
        counts = np.load(directory / f"n_atoms_{split}.npy", allow_pickle=False)
        if not np.array_equal(np.diff(offsets), counts) or offsets[0] != 0 or offsets[-1] != len(numbers):
            raise ValueError(f"invalid atom offsets: {split}")
        exact[split], reduced[split] = set(), set()
        for position, row in enumerate(rows):
            composition = Counter(map(int, numbers[offsets[position]:offsets[position + 1]]))
            divisor = reduce(gcd, composition.values())
            exact[split].add(tuple(sorted(composition.items())))
            reduced[split].add(tuple(sorted((z, n // divisor) for z, n in composition.items())))
            if int(counts[position]) != row["nsites_prim"]:
                raise ValueError(f"parent complete atom count differs: {row['mpid']}")
            source = row["src_ph"]
            reference = row["ph_ref"]
            if source == "phonondb":
                phonon_url = "https://mdr.nims.go.jp/concern/datasets/" + reference.removeprefix("phdb:")
            elif source == "jarvis":
                phonon_url = "https://www.ctcms.nist.gov/~knc6/static/JARVIS/" + reference.removeprefix("jv:") + ".xml"
            elif source == "mp":
                phonon_url = "https://materialsproject.org/materials/" + row["mpid"]
            else:
                raise ValueError(f"unknown phonon source: {source}")
            active = split != "train" or row["mpid"] not in excluded
            records.append({
                "material_id": row["mpid"], "split": split, "row_index": position,
                "active_in_default_view": int(active), "n_atoms": int(counts[position]),
                "edos_zero_window": int(zero_spectra["edos"][position]),
                "phdos_zero_window": int(zero_spectra["phdos"][position]),
                "edos_source": row["src_edos"], "edos_ref": row["edos_ref"],
                "phdos_source": source, "phdos_ref": reference,
                "phdos_source_url": phonon_url,
                "historical_ph_weight": row["ph_weight"],
                "structure_ref": row["struct_ref"], "structure_provenance": row["prov"],
            })
        active_rows = [row for row in rows if split != "train" or row["mpid"] not in excluded]
        active_mask = np.asarray([split != "train" or row["mpid"] not in excluded for row in rows])
        report["splits"][split] = {
            "stored": len(rows), "default_view": len(active_rows),
            "complete_atoms_stored": int(counts.sum()), "max_atoms_stored": int(counts.max()),
            "edos_sources": dict(sorted(Counter(row["src_edos"] for row in rows).items())),
            "phdos_sources": dict(sorted(Counter(row["src_ph"] for row in rows).items())),
            "default_view_phdos_sources": dict(sorted(Counter(row["src_ph"] for row in active_rows).items())),
            "spectra_and_masks_equal_parent": True,
            "zero_window_spectra_stored": {task: int(mask.sum()) for task, mask in zero_spectra.items()},
            "zero_window_spectra_default_view": {
                task: int((mask & active_mask).sum()) for task, mask in zero_spectra.items()},
        }
    if excluded != {r["material_id"] for r in records if not r["active_in_default_view"]}:
        raise ValueError("training exclusion IDs not found exactly once")
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        report["overlap"][first + "_" + second] = {
            "material_ids": len(ids[first] & ids[second]),
            "complete_atom_count_compositions": len(exact[first] & exact[second]),
            "reduced_compositions": len(reduced[first] & reduced[second]),
        }
        if ids[first] & ids[second]:
            raise ValueError("material ID overlap between splits")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    out.mkdir(parents=True, exist_ok=False)
    csv_path = out / "material-provenance.csv.gz"
    with csv_path.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
            compressed.write(buffer.getvalue().encode("utf-8"))
    report["provenance"] = {"file": csv_path.name, "rows": len(records), "sha256": sha256(csv_path)}
    (out / "provenance-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="fresh metadata-only output directory")
    args = parser.parse_args()
    report = export_provenance(args.cache.resolve(), args.processed.resolve(), args.out.resolve())
    print(json.dumps({"rows": report["provenance"]["rows"], "splits": report["splits"]}, indent=2))


if __name__ == "__main__":
    main()
