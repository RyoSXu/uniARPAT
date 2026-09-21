#!/usr/bin/env python3
"""Read-only D4b provenance audit for D4's negative-coordinate phDOS stratum.

This audit intentionally distinguishes a source's frequency/DOS grid from a
per-material stability or convergence verdict.  It neither changes Q1 nor
turns a negative coordinate into an imaginary-mode label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ARCHIVES = {
    "mp": "mp_phdos_raw.jsonl",
    "phonondb": "phonondb_recomputed.jsonl",
    "jarvis": "jarvis_gap_raw.jsonl + jarvis_threesrc_raw.jsonl",
}


def parse_bool(value: object) -> bool:
    """Accept CSV booleans exactly; reject silent truthiness such as ``'False'``."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def parse_numeric_csv(value: object) -> np.ndarray:
    """Parse JARVIS's quoted comma-delimited numeric fields without coercion."""
    if not isinstance(value, str):
        raise ValueError("expected a comma-delimited string")
    tokens = [item.strip().strip("'\"") for item in value.split(",")]
    values = np.asarray([float(item) for item in tokens if item], dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("expected at least two finite numeric values")
    return values


def negative_coordinate_mass_fraction(frequency: object, dos: object) -> float:
    """Trapezoid-integrate non-negative-coordinate support, splitting at zero.

    The value describes the *source DOS grid*, not a mode count or stability
    label.  Exact zero is included in both partition endpoints but has zero
    measure, so the two integrals still sum to the whole spectrum.
    """
    x = np.asarray(frequency, dtype=np.float64)
    y = np.asarray(dos, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("frequency and DOS must be matching one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0.0).any():
        raise ValueError("frequency must be finite and DOS must be finite/non-negative")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("frequency coordinates must be strictly increasing")
    if x[0] < 0.0 < x[-1] and not np.any(x == 0.0):
        y0 = float(np.interp(0.0, x, y))
        insert = int(np.searchsorted(x, 0.0))
        x = np.insert(x, insert, 0.0)
        y = np.insert(y, insert, y0)
    total = float(np.trapz(y, x))
    if total <= 0.0:
        return 0.0
    negative = float(np.trapz(y[x <= 0.0], x[x <= 0.0]))
    return negative / total


def select_mp_record(records: list[dict[str, Any]], ph_ref: str) -> dict[str, Any] | None:
    """Select the C2b source preference, but require the frozen ``ph_ref`` match."""
    if not ph_ref.startswith("mp:"):
        raise ValueError(f"MP source received non-MP reference {ph_ref!r}")
    method = ph_ref.removeprefix("mp:")
    matching = [record for record in records if record.get("method") == method]
    if not matching:
        return None
    return matching[0]


def raw_evidence(frequency: object, dos: object, *, reference_matched: bool,
                 min_fd_phonon_mode: object = None, run_type: object = None,
                 has_structure: object = None) -> dict[str, object]:
    x = np.asarray(frequency, dtype=np.float64)
    y = np.asarray(dos, dtype=np.float64)
    return {
        "raw_record_found": True,
        "raw_reference_matched": reference_matched,
        "raw_frequency_min": float(x.min()),
        "raw_frequency_max": float(x.max()),
        "raw_negative_coordinate_dos_mass_fraction": negative_coordinate_mass_fraction(x, y),
        "source_min_fd_phonon_mode": min_fd_phonon_mode,
        "source_run_type": run_type,
        "source_has_structure": has_structure,
    }


def missing_evidence() -> dict[str, object]:
    return {
        "raw_record_found": False,
        "raw_reference_matched": False,
        "raw_frequency_min": np.nan,
        "raw_frequency_max": np.nan,
        "raw_negative_coordinate_dos_mass_fraction": np.nan,
        "source_min_fd_phonon_mode": None,
        "source_run_type": None,
        "source_has_structure": None,
    }


def _scan_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def collect_mp_evidence(raw_root: Path, needed: dict[str, str]) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for row in _scan_jsonl(raw_root / "mp_phdos_raw.jsonl"):
        mpid = row.get("mpid")
        if mpid not in needed:
            continue
        record = select_mp_record(row.get("phdos_raw", []), needed[mpid])
        if record is None:
            continue
        found[mpid] = raw_evidence(
            record["frequencies_THz"], record["densities"], reference_matched=True,
            run_type=record.get("run_type"), has_structure=record.get("has_structure"),
        )
        if len(found) == len(needed):
            break
    return found


def collect_phonondb_evidence(raw_root: Path, needed: dict[str, str]) -> dict[str, dict[str, object]]:
    serial_to_mpid = {ph_ref.removeprefix("phdb:"): mpid for mpid, ph_ref in needed.items()}
    found: dict[str, dict[str, object]] = {}
    for row in _scan_jsonl(raw_root / "phonondb_recomputed.jsonl"):
        mpid = serial_to_mpid.get(row.get("id"))
        if mpid is None:
            continue
        found[mpid] = raw_evidence(row["freq_THz"], row["dos"], reference_matched=True)
        if len(found) == len(needed):
            break
    return found


def collect_jarvis_evidence(raw_root: Path, needed: dict[str, str]) -> dict[str, dict[str, object]]:
    """Stream the two raw archives; only retain rows selected by frozen JARVIS ref."""
    wanted = {(mpid, ph_ref.removeprefix("jv:")) for mpid, ph_ref in needed.items()}
    found: dict[str, dict[str, object]] = {}
    for filename in ("jarvis_gap_raw.jsonl", "jarvis_threesrc_raw.jsonl"):
        for row in _scan_jsonl(raw_root / filename):
            key = (row.get("canonical_mpid"), row.get("jvasp"))
            if key not in wanted:
                continue
            payload = row.get("jarvis_phdos_raw")
            if not payload or "phonon_dos_frequencies" not in payload:
                continue
            try:
                frequency = parse_numeric_csv(payload["phonon_dos_frequencies"])
                dos = parse_numeric_csv(payload["phonon_dos_intensity"])
                found[key[0]] = raw_evidence(
                    frequency, dos, reference_matched=True,
                    min_fd_phonon_mode=payload.get("min_fd_phonon_mode"),
                )
            except (TypeError, ValueError):
                continue
        if len(found) == len(needed):
            break
    return found


def source_field_catalog() -> list[dict[str, object]]:
    """The source schemas are reported explicitly instead of inferred as labels."""
    return [
        {"src_ph": "mp", "archive": SOURCE_ARCHIVES["mp"],
         "frequency_field": "phdos_raw[].frequencies_THz", "dos_field": "phdos_raw[].densities",
         "direct_binary_stability_flag": False, "direct_convergence_flag": False,
         "auxiliary_fields": "method, run_type, has_structure"},
        {"src_ph": "phonondb", "archive": SOURCE_ARCHIVES["phonondb"],
         "frequency_field": "freq_THz", "dos_field": "dos",
         "direct_binary_stability_flag": False, "direct_convergence_flag": False,
         "auxiliary_fields": "mesh, tetrahedron, born, dielectric"},
        {"src_ph": "jarvis", "archive": SOURCE_ARCHIVES["jarvis"],
         "frequency_field": "jarvis_phdos_raw.phonon_dos_frequencies",
         "dos_field": "jarvis_phdos_raw.phonon_dos_intensity",
         "direct_binary_stability_flag": False, "direct_convergence_flag": False,
         "auxiliary_fields": "min_fd_phonon_mode, phonon_modes, elastic fields"},
    ]


def load_d4_test_rows(data_root: Path, results_root: Path, raw_root: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    d4 = pd.read_csv(results_root / "d4_phdos_b7_test_samples.csv")
    test_index = np.load(data_root / "test" / "test_index.npy", allow_pickle=True).astype(str)
    if len(d4) != len(test_index):
        raise ValueError("D4 rows and Q1 test index do not align")
    if d4["sample_index"].astype(str).tolist() != test_index.tolist():
        raise ValueError("D4 sample_index must exactly match Q1 test order")
    d4["d4_high_negative_coordinate"] = d4["high_negative_mass_fraction"].map(parse_bool)
    d4["mpid"] = test_index
    required = raw_root / "v2_processed.parquet"
    if not required.is_file():
        raise FileNotFoundError(f"missing provenance table: {required}")
    provenance = pq.read_table(required, columns=["mpid", "split", "src_ph", "ph_ref", "prov"]).to_pandas()
    test_provenance = provenance.loc[provenance["split"] == "test"]
    if test_provenance["mpid"].duplicated().any():
        raise ValueError("test provenance has duplicate mpid")
    rows = d4.merge(test_provenance, on="mpid", how="left", validate="one_to_one")
    if rows["src_ph"].isna().any():
        raise ValueError("one or more Q1 test samples lack phDOS provenance")
    unsupported = set(rows["src_ph"]) - set(SOURCE_ARCHIVES)
    if unsupported:
        raise ValueError(f"unknown phDOS provenance source(s): {sorted(unsupported)}")
    return rows


def attach_raw_evidence(rows: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    evidence: dict[str, dict[str, object]] = {}
    collectors = {
        "mp": collect_mp_evidence,
        "phonondb": collect_phonondb_evidence,
        "jarvis": collect_jarvis_evidence,
    }
    for source, group in rows.groupby("src_ph", sort=True):
        needed = dict(zip(group["mpid"], group["ph_ref"], strict=True))
        for mpid, ph_ref in needed.items():
            if not isinstance(ph_ref, str) or not ph_ref.startswith({"mp": "mp:", "phonondb": "phdb:", "jarvis": "jv:"}[source]):
                raise ValueError(f"{mpid} has inconsistent {source!r} reference {ph_ref!r}")
        evidence.update(collectors[source](raw_root, needed))
    table = pd.DataFrame([
        {"mpid": mpid, **evidence.get(mpid, missing_evidence())}
        for mpid in rows["mpid"]
    ])
    return rows.merge(table, on="mpid", how="left", validate="one_to_one")


def provenance_summary(rows: pd.DataFrame) -> pd.DataFrame:
    report_rows = []
    for (source, high), group in rows.groupby(["src_ph", "d4_high_negative_coordinate"], sort=True):
        r2 = group["r2_phdos"].to_numpy(dtype=float)
        raw_mass = group["raw_negative_coordinate_dos_mass_fraction"].to_numpy(dtype=float)
        minimum = pd.to_numeric(group["source_min_fd_phonon_mode"], errors="coerce").to_numpy(dtype=float)
        report_rows.append({
            "src_ph": source,
            "d4_stratum": "high" if high else "other",
            "n": len(group),
            "b7_phdos_r2_median": float(np.median(r2)),
            "b7_phdos_fail_n": int((r2 < 0.0).sum()),
            "b7_phdos_fail_rate": float((r2 < 0.0).mean() * 100.0),
            "raw_record_found_n": int(group["raw_record_found"].sum()),
            "raw_reference_matched_n": int(group["raw_reference_matched"].sum()),
            "raw_negative_coordinate_mass_q50": float(np.nanmedian(raw_mass)) if np.isfinite(raw_mass).any() else np.nan,
            "min_fd_phonon_mode_available_n": int(np.isfinite(minimum).sum()),
            "min_fd_phonon_mode_q50": float(np.nanmedian(minimum)) if np.isfinite(minimum).any() else np.nan,
        })
    return pd.DataFrame(report_rows)


def run_audit(data_root: Path, results_root: Path, raw_root: Path) -> dict[str, object]:
    rows = attach_raw_evidence(load_d4_test_rows(data_root, results_root, raw_root), raw_root)
    field_catalog = pd.DataFrame(source_field_catalog())
    summary = provenance_summary(rows)
    results_root.mkdir(parents=True, exist_ok=True)
    samples_path = results_root / "d4b_phdos_provenance_test_samples.csv"
    summary_path = results_root / "d4b_phdos_provenance_summary.csv"
    catalog_path = results_root / "d4b_phdos_provenance_field_catalog.csv"
    result_path = results_root / "d4b_phdos_provenance_audit_summary.json"
    rows.to_csv(samples_path, index=False)
    summary.to_csv(summary_path, index=False)
    field_catalog.to_csv(catalog_path, index=False)
    result = {
        "audit": "D4b negative-coordinate provenance and stability audit",
        "n_test": int(len(rows)),
        "n_d4_high": int(rows["d4_high_negative_coordinate"].sum()),
        "source_counts": dict(Counter(rows["src_ph"])),
        "source_counts_d4_high": dict(Counter(rows.loc[rows["d4_high_negative_coordinate"], "src_ph"])),
        "raw_record_found_n": int(rows["raw_record_found"].sum()),
        "raw_reference_matched_n": int(rows["raw_reference_matched"].sum()),
        "direct_binary_stability_flag_present": False,
        "direct_convergence_flag_present": False,
        "label_or_split_change_authorized": False,
        "interpretation": (
            "Raw frequency/DOS provenance is not a per-material stability or convergence verdict; "
            "the available archives contain no direct binary stability or convergence flag."
        ),
        "outputs": {
            "test_samples": str(samples_path), "source_summary": str(summary_path),
            "field_catalog": str(catalog_path),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "train4ARPAT")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Git-external raw archive containing v2_processed.parquet and source JSONL files")
    args = parser.parse_args()
    print(json.dumps(run_audit(args.data_root, args.results_root, args.raw_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
