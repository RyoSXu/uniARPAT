#!/usr/bin/env python3
"""Read-only semantic audit of JARVIS ``min_fd_phonon_mode``.

The JARVIS field is preserved as source metadata.  This script establishes its
stored representation and relation to the other frequency arrays; it does not
promote it to a dataset quality label.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_numeric_csv(value: object) -> np.ndarray:
    if not isinstance(value, str):
        raise ValueError("expected a quoted comma-delimited numeric string")
    tokens = [item.strip().strip("'\"") for item in value.split(",")]
    result = np.asarray([float(item) for item in tokens if item], dtype=np.float64)
    if len(result) < 1 or not np.isfinite(result).all():
        raise ValueError("expected one or more finite values")
    return result


def classify_min_fd(value: object) -> tuple[float, str]:
    """Keep a negative serialized zero distinct from a strictly negative value."""
    if not isinstance(value, str):
        raise ValueError("min_fd_phonon_mode must be its source string")
    text = value.strip().strip("'\"")
    numeric = float(text)
    if not np.isfinite(numeric):
        raise ValueError("min_fd_phonon_mode must be finite")
    if numeric < 0.0:
        return numeric, "negative"
    if numeric == 0.0 and text.startswith("-"):
        return numeric, "zero_serialized_negative"
    if numeric == 0.0:
        return numeric, "zero"
    return numeric, "positive"


def exact_numeric_match(left: object, right: object) -> bool:
    """No tolerance is permitted for a claim of stored-field equality."""
    return float(left) == float(right)


def jarvis_accessor_contract() -> dict[str, str]:
    """Record the installed parser route without making a network request."""
    from jarvis.db.webpages import Webpage

    source = inspect.getsource(Webpage.get_dft_phonon_dos)
    required = 'self.data["basic_info"]["main_elastic"]["main_elastic_info"]'
    if required not in source:
        raise RuntimeError("installed JARVIS accessor no longer exposes MAIN-ELAST phonon data")
    return {
        "accessor": "jarvis.db.webpages.Webpage.get_dft_phonon_dos",
        "source_file": str(Path(inspect.getfile(Webpage)).resolve()),
        "raw_path": "basic_info.main_elastic.main_elastic_info",
    }


def _scan_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def collect_jarvis_rows(raw_root: Path, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for filename in ("jarvis_gap_raw.jsonl", "jarvis_threesrc_raw.jsonl"):
        for row in _scan_jsonl(raw_root / filename):
            key = (row.get("canonical_mpid"), row.get("jvasp"))
            if key not in wanted:
                continue
            payload = row.get("jarvis_phdos_raw")
            if not payload:
                raise ValueError(f"{key} matches ph_ref but has no JARVIS phDOS payload")
            if key in found:
                raise ValueError(f"duplicate JARVIS raw payload for frozen reference {key}")
            found[key] = payload
    if set(found) != wanted:
        missing = sorted(wanted - set(found))
        raise ValueError(f"missing JARVIS raw payloads for {missing[:5]} (total {len(missing)})")
    return found


def load_rows(results_root: Path) -> pd.DataFrame:
    rows = pd.read_csv(results_root / "d4b_phdos_provenance_test_samples.csv")
    jarvis = rows.loc[rows["src_ph"] == "jarvis"].copy()
    if len(jarvis) != 209:
        raise ValueError(f"expected D4b's frozen 209 JARVIS test rows, found {len(jarvis)}")
    if jarvis["ph_ref"].isna().any() or not jarvis["ph_ref"].str.startswith("jv:").all():
        raise ValueError("all JARVIS rows require a frozen jv: ph_ref")
    if jarvis["mpid"].duplicated().any():
        raise ValueError("JARVIS test mpids must be unique")
    return jarvis


def enrich_rows(rows: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    wanted = set(zip(rows["mpid"], rows["ph_ref"].str.removeprefix("jv:"), strict=True))
    payloads = collect_jarvis_rows(raw_root, wanted)
    evidence = []
    for row in rows.itertuples(index=False):
        key = (row.mpid, row.ph_ref.removeprefix("jv:"))
        payload = payloads[key]
        min_fd, category = classify_min_fd(payload.get("min_fd_phonon_mode"))
        mode_min = float(parse_numeric_csv(payload["phonon_modes"]).min())
        dos_min = float(parse_numeric_csv(payload["phonon_dos_frequencies"]).min())
        evidence.append({
            "mpid": row.mpid,
            "jvasp": key[1],
            "min_fd_phonon_mode_raw": payload["min_fd_phonon_mode"],
            "min_fd_phonon_mode": min_fd,
            "min_fd_category": category,
            "phonon_modes_min": mode_min,
            "phdos_frequency_min": dos_min,
            "min_fd_equals_phonon_modes_min": exact_numeric_match(min_fd, mode_min),
            "min_fd_equals_phdos_frequency_min": exact_numeric_match(min_fd, dos_min),
        })
    return rows.merge(pd.DataFrame(evidence), on="mpid", how="left", validate="one_to_one")


def category_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for category, group in rows.groupby("min_fd_category", sort=True):
        r2 = group["r2_phdos"].to_numpy(dtype=float)
        output.append({
            "min_fd_category": category,
            "n": len(group),
            "d4_high_n": int(group["d4_high_negative_coordinate"].sum()),
            "b7_phdos_fail_n": int((r2 < 0.0).sum()),
            "b7_phdos_fail_rate": float((r2 < 0.0).mean() * 100.0),
            "b7_phdos_r2_median": float(np.median(r2)),
            "phonon_modes_min_q50": float(np.median(group["phonon_modes_min"])),
            "phdos_frequency_min_q50": float(np.median(group["phdos_frequency_min"])),
        })
    return pd.DataFrame(output)


def run_audit(results_root: Path, raw_root: Path) -> dict[str, object]:
    contract = jarvis_accessor_contract()
    rows = enrich_rows(load_rows(results_root), raw_root)
    summary = category_summary(rows)
    results_root.mkdir(parents=True, exist_ok=True)
    sample_path = results_root / "d4c_jarvis_min_fd_test_samples.csv"
    summary_path = results_root / "d4c_jarvis_min_fd_category_summary.csv"
    result_path = results_root / "d4c_jarvis_min_fd_audit_summary.json"
    rows.to_csv(sample_path, index=False)
    summary.to_csv(summary_path, index=False)
    result = {
        "audit": "D4c JARVIS min_fd_phonon_mode semantic audit",
        "n_jarvis_test": int(len(rows)),
        "category_counts": dict(Counter(rows["min_fd_category"])),
        "min_fd_equals_phonon_modes_min_n": int(rows["min_fd_equals_phonon_modes_min"].sum()),
        "min_fd_equals_phdos_frequency_min_n": int(rows["min_fd_equals_phdos_frequency_min"].sum()),
        "source_contract": contract,
        "data_policy_authorized": False,
        "interpretation": (
            "min_fd_phonon_mode is preserved as MAIN-ELAST source metadata; it is not a cross-source "
            "stability/convergence truth or an authorization for Q1 data changes."
        ),
        "outputs": {"test_samples": str(sample_path), "category_summary": str(summary_path)},
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.results_root, args.raw_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
