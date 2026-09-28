#!/usr/bin/env python3
"""D3a local-only adjudication of MP eDOS-only support candidates.

The sample is frozen without validation/test labels. Candidate attribution and
quality checks use only local MP structures and local Delta tables. Q1 valid
labels are opened only after all candidate usability decisions are complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

import numpy as np

SEED = "d3a-edos-support-v1"
RANDOM_N = 500
TARGETED_N = 500
TRAIN_Q3_TV = 0.253176
ALLOWED_RUN_TYPES = {"GGA", "GGA+U"}
E0_EDGES = np.linspace(-6.0, 6.0, 129)


def stable_hash(mpid: str, seed: str = SEED) -> str:
    return hashlib.sha256(f"{seed}\0{mpid}".encode()).hexdigest()


def structure_labels(structure: dict) -> list[str]:
    labels = []
    for site in structure.get("sites", []):
        species = site.get("species") or []
        if species:
            labels.append(str(species[0].get("element")))
        else:
            labels.append(str(site.get("label")))
    return labels


def composition_signature(labels: Iterable[str]) -> str:
    counts = Counter(str(x) for x in labels)
    return "|".join(f"{k}:{counts[k]}" for k in sorted(counts))


def nsites_bin(nsites: int) -> str:
    if nsites <= 4:
        return "01-04"
    if nsites <= 8:
        return "05-08"
    if nsites <= 16:
        return "09-16"
    if nsites <= 32:
        return "17-32"
    if nsites <= 80:
        return "33-80"
    return "81+"


def nelements_bin(nelements: int) -> str:
    return str(nelements) if nelements <= 3 else "4+"


def gap_bin(gap: float) -> str:
    if gap <= 0:
        return "metal"
    if gap <= 2:
        return "gap_0_2"
    return "gap_gt_2"


def base_stratum(record: dict) -> tuple[str, str, str, str]:
    return (
        nelements_bin(int(record["nelements"])),
        nsites_bin(int(record["nsites"])),
        str(record["crystal_system"]).lower(),
        gap_bin(float(record["band_gap"])),
    )


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def candidate_ids(raw_root: Path, processed_path: Path) -> list[str]:
    import pyarrow.parquet as pq

    effective = set(load_json(raw_root / "census_effective_ids.json"))
    phonon = set(load_json(raw_root / "dual_spectra_ids.json"))
    current = set(
        pq.read_table(processed_path, columns=["mpid"]).column("mpid").to_pylist()
    )
    candidates = sorted(effective - (phonon | current))
    if len(candidates) != 121420:
        raise ValueError(f"candidate contract mismatch: {len(candidates)} != 121420")
    return candidates


def load_metadata(shard_dir: Path, wanted: set[str]) -> dict[str, dict]:
    result = {}
    for path in sorted(shard_dir.glob("structures_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                mpid = row.get("material_id")
                if mpid not in wanted:
                    continue
                structure = row["structure"]
                labels = structure_labels(structure)
                lattice = structure.get("lattice") or {}
                result[mpid] = {
                    "mpid": mpid,
                    "structure": structure,
                    "composition": composition_signature(labels),
                    "species": ",".join(sorted(set(labels))),
                    "nsites": int(row.get("nsites") or len(labels)),
                    "nelements": int(row.get("nelements") or len(set(labels))),
                    "crystal_system": (row.get("symmetry") or {}).get(
                        "crystal_system", "unknown"
                    ),
                    "band_gap": float(row.get("band_gap") or 0.0),
                    "volume": float(lattice.get("volume")),
                }
    missing = wanted - set(result)
    if missing:
        raise ValueError(f"metadata missing for {len(missing)} ids; first={sorted(missing)[:3]}")
    return result


def select_sample(
    candidates: list[str], metadata: dict[str, dict], train_ids: list[str]
) -> list[dict]:
    ranked = sorted(candidates, key=lambda m: (stable_hash(m), m))
    random_ids = ranked[:RANDOM_N]
    random_set = set(random_ids)

    train_compositions = {metadata[m]["composition"] for m in train_ids}
    train_cells = Counter(base_stratum(metadata[m]) for m in train_ids)
    groups = defaultdict(list)
    for mpid in candidates:
        if mpid in random_set:
            continue
        rec = metadata[mpid]
        unseen = rec["composition"] not in train_compositions
        key = (unseen,) + base_stratum(rec)
        groups[key].append(mpid)
    for ids in groups.values():
        ids.sort(key=lambda m: (stable_hash(m), m))
    group_order = sorted(
        groups,
        key=lambda key: (
            0 if key[0] else 1,
            train_cells[key[1:]],
            tuple(str(x) for x in key[1:]),
        ),
    )
    targeted = []
    offset = 0
    while len(targeted) < TARGETED_N:
        progressed = False
        for key in group_order:
            ids = groups[key]
            if offset < len(ids):
                targeted.append(ids[offset])
                progressed = True
                if len(targeted) == TARGETED_N:
                    break
        if not progressed:
            raise ValueError("not enough candidates for targeted sample")
        offset += 1

    rows = []
    for arm, ids in (("probability", random_ids), ("targeted", targeted)):
        for rank, mpid in enumerate(ids, 1):
            rec = metadata[mpid]
            b = base_stratum(rec)
            rows.append(
                {
                    "mpid": mpid,
                    "arm": arm,
                    "arm_rank": rank,
                    "sample_hash": stable_hash(mpid),
                    "composition": rec["composition"],
                    "composition_seen_train": int(
                        rec["composition"] in train_compositions
                    ),
                    "nelements": rec["nelements"],
                    "nsites": rec["nsites"],
                    "crystal_system": rec["crystal_system"],
                    "band_gap": rec["band_gap"],
                    "nelements_bin": b[0],
                    "nsites_bin": b[1],
                    "gap_bin": b[3],
                    "train_stratum_count": train_cells[b],
                }
            )
    if len(rows) != 1000 or len({r["mpid"] for r in rows}) != 1000:
        raise AssertionError("sample must contain 1000 unique materials")
    return rows


def atomic_write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("refusing to write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    except Exception:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp, path)
    except Exception:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_direct_task_map(raw_root: Path, sample_ids: set[str]) -> dict[str, str]:
    result = {}
    path = raw_root / "mp_increment_raw.jsonl"
    if path.exists():
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("mpid") in sample_ids and row.get("dos_task_id"):
                    result[row["mpid"]] = row["dos_task_id"]
    path = raw_root / "mp_increment_dosmap.json"
    if path.exists():
        for mpid, row in load_json(path).items():
            if (
                mpid in sample_ids
                and mpid not in result
                and row.get("mode") in {"unique", "delta_vol", "matcher_ok"}
                and row.get("identifier")
            ):
                result[mpid] = row["identifier"]
    return result


def index_candidates(
    index_path: Path, metadata: dict[str, dict], direct: dict[str, str]
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    import pyarrow.parquet as pq

    blocks = {(r["nsites"], r["species"]) for r in metadata.values()}
    table = pq.read_table(index_path, columns=["identifier", "nsites", "volume", "species", "run_type"])
    by_mpid = {m: [] for m in metadata}
    task_run_type = {}
    meta_by_block = defaultdict(list)
    for mpid, row in metadata.items():
        meta_by_block[(row["nsites"], row["species"])].append((mpid, row["volume"]))
    for row in table.to_pylist():
        block = (int(row["nsites"]), row["species"])
        if block not in blocks:
            continue
        task_run_type[row["identifier"]] = row["run_type"]
        for mpid, volume in meta_by_block[block]:
            rel = abs(float(row["volume"]) - volume) / max(volume, 1e-12)
            if rel < 0.01:
                by_mpid[mpid].append(
                    {
                        "identifier": row["identifier"],
                        "run_type": row["run_type"],
                        "volume": float(row["volume"]),
                        "volume_rel_diff": rel,
                    }
                )
    for mpid, identifier in direct.items():
        if identifier not in {r["identifier"] for r in by_mpid[mpid]}:
            run_type = task_run_type.get(identifier)
            if run_type:
                by_mpid[mpid].append(
                    {
                        "identifier": identifier,
                        "run_type": run_type,
                        "volume": math.nan,
                        "volume_rel_diff": math.nan,
                    }
                )
    return by_mpid, task_run_type


def load_delta_rows(
    delta_root: Path, identifiers: set[str], task_run_type: dict[str, str]
) -> dict[str, dict]:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    rows = {}
    columns = [
        "identifier",
        "structure",
        "spin_up_densities",
        "spin_down_densities",
        "energies",
        "efermi",
    ]
    identifiers_by_run_type = defaultdict(list)
    for identifier in sorted(identifiers):
        run_type = task_run_type.get(identifier)
        if run_type is not None:
            identifiers_by_run_type[run_type].append(identifier)
    for directory in sorted(delta_root.glob("run_type=*")):
        paths = sorted(directory.glob("*.parquet"))
        if not paths:
            continue
        run_type = unquote(directory.name.split("=", 1)[1])
        wanted = identifiers_by_run_type.get(run_type, [])
        if not wanted:
            continue
        dataset = ds.dataset([str(p) for p in paths], format="parquet")
        for start in range(0, len(wanted), 4000):
            values = pa.array(wanted[start : start + 4000])
            table = dataset.to_table(
                columns=columns, filter=pc.field("identifier").isin(values)
            )
            for row in table.to_pylist():
                row["run_type"] = run_type
                rows[row["identifier"]] = row
    return rows


def to_structure(payload: dict):
    from pymatgen.core import Lattice, Structure

    lattice = payload["lattice"]
    matrix = lattice["matrix"] if isinstance(lattice, dict) else lattice
    labels = structure_labels(payload)
    coords = [site["abc"] for site in payload["sites"]]
    return Structure(Lattice(matrix), labels, coords)


def match_sample_tasks(
    metadata: dict[str, dict], candidates: dict[str, list[dict]], delta_rows: dict[str, dict]
) -> tuple[list[dict], dict[str, dict]]:
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
    )
    audit = []
    selected = {}
    for mpid in sorted(metadata):
        base = {
            "mpid": mpid,
            "candidate_task_count": len(candidates[mpid]),
            "structure_match_count": 0,
            "task_id": "",
            "run_type": "",
            "volume_rel_diff": "",
            "status": "",
        }
        if not candidates[mpid]:
            base["status"] = "unresolved_no_task"
            audit.append(base)
            continue
        try:
            reference = to_structure(metadata[mpid]["structure"])
        except Exception:
            base["status"] = "unresolved_bad_candidate_structure"
            audit.append(base)
            continue
        matches = []
        for candidate in candidates[mpid]:
            row = delta_rows.get(candidate["identifier"])
            if row is None:
                continue
            try:
                if matcher.fit(reference, to_structure(row["structure"])):
                    matches.append((candidate, row))
            except Exception:
                continue
        base["structure_match_count"] = len(matches)
        if not matches:
            base["status"] = "unresolved_no_match"
            audit.append(base)
            continue
        matches.sort(
            key=lambda pair: (
                pair[0]["volume_rel_diff"]
                if math.isfinite(pair[0]["volume_rel_diff"])
                else float("inf"),
                pair[0]["identifier"],
            )
        )
        choice, row = matches[0]
        base.update(
            {
                "task_id": choice["identifier"],
                "run_type": row["run_type"],
                "volume_rel_diff": choice["volume_rel_diff"],
            }
        )
        if row["run_type"] not in ALLOWED_RUN_TYPES:
            base["status"] = "incompatible_run_type"
        else:
            base["status"] = "matched"
            selected[mpid] = row
        audit.append(base)
    return audit, selected


def box_average(x: np.ndarray, y: np.ndarray, edges: np.ndarray = E0_EDGES):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    coverage = np.zeros(len(edges) - 1, dtype=np.int8)
    for index in range(len(edges) - 1):
        mask = (x >= edges[index]) & (x < edges[index + 1])
        count = int(mask.sum())
        if count > 1:
            order = np.argsort(x[mask])
            xx, yy = x[mask][order], y[mask][order]
            out[index] = np.trapz(yy, xx) / (edges[index + 1] - edges[index])
            coverage[index] = 1
        elif count == 1:
            out[index] = float(y[mask][0])
            coverage[index] = 1
    return out, coverage


def winsorize_isolated(values: np.ndarray, mask: np.ndarray, thresholds: np.ndarray):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    thresholds = np.asarray(thresholds, dtype=float)
    output = values.copy()
    over = (values > thresholds) & mask
    left = np.zeros_like(values)
    left[1:] = values[:-1]
    right = np.zeros_like(values)
    right[:-1] = values[1:]
    neighbor = np.maximum(left, right)
    wide = np.zeros_like(over)
    wide[1:-1] = over[1:-1] & over[:-2] & over[2:]
    wide[0] = over[0] & over[1]
    wide[-1] = over[-1] & over[-2]
    clip = over & ~wide & (neighbor < 0.5 * values)
    output[clip] = thresholds[clip]
    return output, int(clip.sum())


def valence_count(structure: dict, zval: dict) -> float:
    total = 0.0
    for site in structure.get("sites", []):
        species = site.get("species") or []
        if not species:
            raise ValueError("site lacks species")
        for item in species:
            symbol = item["element"]
            if symbol not in zval:
                raise ValueError(f"missing zval for {symbol}")
            total += float(item.get("occu", 1.0)) * float(zval[symbol]["zval"])
    return total


def normalize_spectrum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("spectrum has no positive finite mass")
    return values / total


def process_spectrum(row: dict, zval: dict, thresholds: np.ndarray) -> dict:
    result = {
        "quality_status": "",
        "usable": 0,
        "coverage_fraction": math.nan,
        "trunc_ratio": math.nan,
        "task_nval": math.nan,
        "primitive_nval": math.nan,
        "e0_mass": math.nan,
        "winsor_clipped_bins": 0,
        "shape": None,
    }
    try:
        energies = np.asarray(row["energies"], dtype=float)
        up = np.asarray(row["spin_up_densities"], dtype=float)
        down = row.get("spin_down_densities")
        if down is None or np.ndim(down) == 0 or len(down) != len(up):
            down_array = np.zeros_like(up)
        else:
            down_array = np.asarray(down, dtype=float)
        efermi = float(row["efermi"])
        if not np.isfinite(efermi) or len(energies) != len(up):
            raise ValueError("invalid energy/density arrays")
        finite = np.isfinite(energies) & np.isfinite(up) & np.isfinite(down_array)
        if finite.sum() < 2:
            raise ValueError("fewer than two finite points")
        x = energies[finite]
        y = up[finite] + down_array[finite]
        order = np.argsort(x)
        x, y = x[order], y[order]
        unique_x, inverse = np.unique(x, return_inverse=True)
        if len(unique_x) != len(x):
            sums = np.zeros(len(unique_x))
            counts = np.zeros(len(unique_x))
            np.add.at(sums, inverse, y)
            np.add.at(counts, inverse, 1)
            x, y = unique_x, sums / counts
        task_nval = valence_count(row["structure"], zval)
        if task_nval <= 0:
            raise ValueError("nonpositive task N_val")
        full_mass = float(np.trapz(y, x))
        trunc_ratio = full_mass / task_nval
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        primitive = SpacegroupAnalyzer(to_structure(row["structure"]), symprec=0.1).get_primitive_standard_structure()
        primitive_nval = sum(
            float(zval[str(site.specie.symbol)]["zval"]) for site in primitive.sites
        )
        binned, coverage = box_average(x - efermi, y)
        binned, clipped = winsorize_isolated(binned, coverage, thresholds)
        e0_mass = float(binned.sum())
        result.update(
            {
                "coverage_fraction": float(coverage.mean()),
                "trunc_ratio": trunc_ratio,
                "task_nval": task_nval,
                "primitive_nval": primitive_nval,
                "e0_mass": e0_mass,
                "winsor_clipped_bins": clipped,
            }
        )
        if not coverage.any():
            result["quality_status"] = "empty_coverage"
        elif not np.isfinite(e0_mass) or e0_mass <= 0:
            result["quality_status"] = "invalid_e0_mass"
        elif not np.isfinite(primitive_nval) or primitive_nval <= 0:
            result["quality_status"] = "invalid_primitive_nval"
        else:
            result["shape"] = normalize_spectrum(binned)
            if trunc_ratio < 0.5:
                result["quality_status"] = "truncated"
            else:
                result["quality_status"] = "usable"
                result["usable"] = 1
    except Exception as exc:
        result["quality_status"] = f"unreadable:{type(exc).__name__}"
    return result


def total_variation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum(axis=-1)


def nearest_tv(query: np.ndarray, reference: np.ndarray, chunk: int = 256) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference shape mismatch")
    answer = np.full(query.shape[0], np.inf, dtype=np.float64)
    for start in range(0, len(reference), chunk):
        block = reference[start : start + chunk]
        distances = 0.5 * np.abs(query[:, None, :] - block[None, :, :]).sum(axis=2)
        answer = np.minimum(answer, distances.min(axis=1))
    return answer


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def bootstrap_median_ci(values: np.ndarray, draws: int = 2000, seed: int = 20260928):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    medians = np.empty(draws)
    for index in range(draws):
        medians[index] = np.median(rng.choice(values, size=len(values), replace=True))
    return tuple(np.quantile(medians, [0.025, 0.975]).tolist())


def support_metrics(
    sample_rows: list[dict], usable_shapes: dict[str, np.ndarray], q1_root: Path, support_csv: Path
):
    train = np.asarray(np.load(q1_root / "train/edos_tgtdos_train.npy"), dtype=float)
    train = train / np.maximum(train.sum(axis=1, keepdims=True), 1e-12)
    candidate_ids_ordered = [r["mpid"] for r in sample_rows if r["mpid"] in usable_shapes]
    candidate = np.stack([usable_shapes[m] for m in candidate_ids_ordered])
    candidate_novelty = nearest_tv(candidate, train)
    novelty = dict(zip(candidate_ids_ordered, candidate_novelty))

    valid = np.asarray(np.load(q1_root / "valid/edos_tgtdos_valid.npy"), dtype=float)
    valid = valid / np.maximum(valid.sum(axis=1, keepdims=True), 1e-12)
    valid_ids = np.load(q1_root / "valid/valid_index.npy").astype(str)
    support_rows = [r for r in read_csv(support_csv) if r["split"] == "valid" and int(r["train_support_quartile"]) == 4]
    indices = np.array([int(r["sample_index"]) for r in support_rows], dtype=int)
    if any(valid_ids[i] != r["mpid"] for i, r in zip(indices, support_rows)):
        raise ValueError("frozen support CSV does not match Q1 valid order")
    baseline = np.array([float(r["nearest_train_target_tv"]) for r in support_rows])
    candidate_nearest = nearest_tv(valid[indices], candidate)
    new = np.minimum(baseline, candidate_nearest)
    reduction = baseline - new
    ci = bootstrap_median_ci(reduction)
    q4_rows = [
        {
            "sample_index": int(i),
            "mpid": str(valid_ids[i]),
            "baseline_nearest_tv": float(b),
            "candidate_nearest_tv": float(c),
            "new_nearest_tv": float(n),
            "reduction": float(d),
        }
        for i, b, c, n, d in zip(indices, baseline, candidate_nearest, new, reduction)
    ]
    median_baseline = float(np.median(baseline))
    median_reduction = float(np.median(reduction))
    metrics = {
        "q4_n": len(reduction),
        "baseline_median_nearest_tv": median_baseline,
        "new_median_nearest_tv": float(np.median(new)),
        "median_absolute_reduction": median_reduction,
        "median_relative_reduction": median_reduction / median_baseline,
        "strict_improvement_fraction": float(np.mean(reduction > 1e-12)),
        "median_reduction_bootstrap_ci95": list(ci),
    }
    return novelty, metrics, q4_rows


def ensure_outputs_available(paths: list[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")


def parse_args():
    repo = Path(__file__).resolve().parents[2]
    raw = Path("/root/home/newstudy/getdata/raw")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=raw)
    parser.add_argument("--processed", type=Path, default=Path("/root/home/newstudy/getdata/v2_release/v2_processed.parquet"))
    parser.add_argument("--q1-root", type=Path, default=repo / "data/train4ARPAT")
    parser.add_argument("--support-csv", type=Path, default=repo / "results/edos_spectral_support_q1_train_valid_samples.csv")
    parser.add_argument("--zval", type=Path, default=repo / "index/z0_zval.json")
    parser.add_argument("--winsor", type=Path, default=raw / "v2_winsor_thresholds.json")
    parser.add_argument("--delta-index", type=Path, default=raw / "delta_edos_index.parquet")
    parser.add_argument("--delta-root", type=Path, default=raw / "delta_mp/core/electronic-structure/total-dos")
    parser.add_argument("--sample-manifest", type=Path, default=repo / "output/d3a_edos_support_sample_manifest.csv")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--sample-out", type=Path, default=repo / "results/d3a_edos_support_sample.csv")
    parser.add_argument("--summary-out", type=Path, default=repo / "results/d3a_edos_support_adjudication.json")
    parser.add_argument("--q4-out", type=Path, default=repo / "results/d3a_edos_support_q1_valid_q4.csv")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = candidate_ids(args.raw_root, args.processed)
    train_ids = np.load(args.q1_root / "train/train_index.npy").astype(str).tolist()
    wanted = set(candidates) | set(train_ids)
    metadata_all = load_metadata(args.raw_root / "pretrain_structures", wanted)
    sample_rows = select_sample(candidates, metadata_all, train_ids)
    candidate_set = set(candidates)
    if any(r["mpid"] not in candidate_set for r in sample_rows):
        raise AssertionError("sample contains a non-candidate")

    if args.sample_only:
        ensure_outputs_available([args.sample_manifest], args.force)
        atomic_write_csv(sample_rows, args.sample_manifest)
        print(json.dumps({"sample_n": len(sample_rows), "sha256": sha256_file(args.sample_manifest)}))
        return

    if not args.sample_manifest.exists():
        raise FileNotFoundError("run --sample-only first to freeze the sample manifest")
    frozen = read_csv(args.sample_manifest)
    if [r["mpid"] for r in frozen] != [r["mpid"] for r in sample_rows]:
        raise ValueError("sample manifest does not match deterministic reconstruction")
    outputs = [args.sample_out, args.summary_out, args.q4_out]
    ensure_outputs_available(outputs, args.force)

    sample_ids = {r["mpid"] for r in sample_rows}
    sample_meta = {m: metadata_all[m] for m in sample_ids}
    direct = load_direct_task_map(args.raw_root, sample_ids)
    candidates_by_mpid, task_run_type = index_candidates(
        args.delta_index, sample_meta, direct
    )
    wanted_tasks = {c["identifier"] for rows in candidates_by_mpid.values() for c in rows}
    delta_rows = load_delta_rows(args.delta_root, wanted_tasks, task_run_type)
    attribution, selected = match_sample_tasks(sample_meta, candidates_by_mpid, delta_rows)
    attribution_by_id = {r["mpid"]: r for r in attribution}
    zval = load_json(args.zval)
    thresholds = np.asarray(load_json(args.winsor)["edos_thr"], dtype=float)
    if thresholds.shape != (128,):
        raise ValueError("frozen eDOS winsor threshold must have 128 bins")

    usable_shapes = {}
    readable_shapes = {}
    final_rows = []
    for sample in sample_rows:
        mpid = sample["mpid"]
        row = dict(sample)
        row.update(attribution_by_id[mpid])
        if row["status"] == "matched":
            quality = process_spectrum(selected[mpid], zval, thresholds)
        else:
            quality = {
                "quality_status": row["status"],
                "usable": 0,
                "coverage_fraction": math.nan,
                "trunc_ratio": math.nan,
                "task_nval": math.nan,
                "primitive_nval": math.nan,
                "e0_mass": math.nan,
                "winsor_clipped_bins": 0,
                "shape": None,
            }
        shape = quality.pop("shape")
        row.update(quality)
        row["nearest_train_target_tv"] = ""
        if shape is not None:
            readable_shapes[mpid] = shape
            if int(row["usable"]):
                usable_shapes[mpid] = shape
        final_rows.append(row)

    if not usable_shapes:
        raise RuntimeError("no usable candidate spectra; cannot evaluate support")
    novelty, support, q4_rows = support_metrics(
        sample_rows, usable_shapes, args.q1_root, args.support_csv
    )
    _, readable_support, readable_q4_rows = support_metrics(
        sample_rows, readable_shapes, args.q1_root, args.support_csv
    )
    for strict_row, readable_row in zip(q4_rows, readable_q4_rows):
        if strict_row["mpid"] != readable_row["mpid"]:
            raise AssertionError("strict and sensitivity Q4 rows are misaligned")
        strict_row["all_readable_candidate_nearest_tv"] = readable_row[
            "candidate_nearest_tv"
        ]
        strict_row["all_readable_new_nearest_tv"] = readable_row["new_nearest_tv"]
        strict_row["all_readable_reduction"] = readable_row["reduction"]
    for row in final_rows:
        if row["mpid"] in novelty:
            row["nearest_train_target_tv"] = float(novelty[row["mpid"]])

    probability = [r for r in final_rows if r["arm"] == "probability"]
    prob_usable = sum(int(r["usable"]) for r in probability)
    wilson = wilson_interval(prob_usable, len(probability))
    all_novelty = np.array(list(novelty.values()))
    gates = {
        "probability_wilson_lower_ge_0_50": wilson[0] >= 0.50,
        "q4_median_relative_reduction_ge_0_05": support["median_relative_reduction"] >= 0.05,
        "q4_bootstrap_lower_gt_0": support["median_reduction_bootstrap_ci95"][0] > 0,
        "q4_strict_improvement_fraction_ge_0_25": support["strict_improvement_fraction"] >= 0.25,
        "sample_size_and_uniqueness": len(final_rows) == 1000 and len({r["mpid"] for r in final_rows}) == 1000,
        "usable_run_types_compatible": all(r["run_type"] in ALLOWED_RUN_TYPES for r in final_rows if int(r["usable"])),
    }
    summary = {
        "status": "pass" if all(gates.values()) else "stop",
        "decision": "design_10_20k_data_pilot" if all(gates.values()) else "close_data_route_return_to_model_mechanism",
        "candidate_count": len(candidates),
        "sample_manifest_sha256": sha256_file(args.sample_manifest),
        "sample_counts": Counter(r["arm"] for r in final_rows),
        "usable_counts": Counter(r["arm"] for r in final_rows if int(r["usable"])),
        "status_counts": Counter(r["quality_status"] for r in final_rows),
        "probability_usable_rate": prob_usable / len(probability),
        "probability_usable_wilson_ci95": list(wilson),
        "candidate_novelty": {
            "usable_n": len(all_novelty),
            "median_nearest_train_target_tv": float(np.median(all_novelty)),
            "fraction_above_train_q3_threshold": float(np.mean(all_novelty > TRAIN_Q3_TV)),
            "train_q3_threshold": TRAIN_Q3_TV,
        },
        "valid_q4_support": support,
        "all_readable_sensitivity": {
            "definition": (
                "matched readable GGA/GGA+U spectra with valid E0 shape, including "
                "records below the approximate truncation threshold"
            ),
            "readable_n": len(readable_shapes),
            "valid_q4_support": readable_support,
        },
        "truncation_ratio_note": (
            "full raw DOS integral divided by task-cell elemental-table N_val; "
            "classification is audited against frozen Z0 separately"
        ),
        "gates": gates,
        "test_labels_read": False,
    }
    # Counter is dict-like but normalize for stable JSON.
    summary["sample_counts"] = dict(summary["sample_counts"])
    summary["usable_counts"] = dict(summary["usable_counts"])
    summary["status_counts"] = dict(summary["status_counts"])

    staged = []
    try:
        for rows, path in ((final_rows, args.sample_out), (q4_rows, args.q4_out)):
            temp = path.with_name(path.name + ".staged")
            if temp.exists():
                temp.unlink()
            atomic_write_csv(rows, temp)
            staged.append((temp, path))
        temp_json = args.summary_out.with_name(args.summary_out.name + ".staged")
        if temp_json.exists():
            temp_json.unlink()
        atomic_write_json(summary, temp_json)
        staged.append((temp_json, args.summary_out))
        for temp, path in staged:
            os.replace(temp, path)
    except Exception:
        for temp, _ in staged:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
