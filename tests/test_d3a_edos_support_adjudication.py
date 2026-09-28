import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.eval.d3a_edos_support_adjudication import (
    RANDOM_N,
    TARGETED_N,
    atomic_write_csv,
    atomic_write_json,
    base_stratum,
    bootstrap_median_ci,
    box_average,
    composition_signature,
    ensure_outputs_available,
    gap_bin,
    nearest_tv,
    nelements_bin,
    normalize_spectrum,
    nsites_bin,
    process_spectrum,
    select_sample,
    stable_hash,
    total_variation,
    valence_count,
    wilson_interval,
    winsorize_isolated,
)


class D3aSupportAdjudicationTests(unittest.TestCase):
    def test_stable_hash_is_deterministic_and_seeded(self):
        self.assertEqual(stable_hash("mp-1"), stable_hash("mp-1"))
        self.assertNotEqual(stable_hash("mp-1"), stable_hash("mp-2"))
        self.assertNotEqual(stable_hash("mp-1"), stable_hash("mp-1", "other"))

    def test_composition_signature_counts_and_sorts(self):
        self.assertEqual(composition_signature(["O", "Si", "O"]), "O:2|Si:1")

    def test_bins_have_frozen_boundaries(self):
        self.assertEqual(nsites_bin(4), "01-04")
        self.assertEqual(nsites_bin(5), "05-08")
        self.assertEqual(nsites_bin(80), "33-80")
        self.assertEqual(nsites_bin(81), "81+")
        self.assertEqual(nelements_bin(4), "4+")
        self.assertEqual(gap_bin(0), "metal")
        self.assertEqual(gap_bin(2), "gap_0_2")
        self.assertEqual(gap_bin(2.01), "gap_gt_2")

    def test_sample_has_two_disjoint_frozen_arms(self):
        train_ids = [f"train-{i}" for i in range(20)]
        candidates = [f"mp-{i:05d}" for i in range(1200)]
        metadata = {}
        for i, mpid in enumerate(train_ids + candidates):
            metadata[mpid] = {
                "composition": f"X:{i % 31 + 1}",
                "nelements": i % 4 + 1,
                "nsites": i % 90 + 1,
                "crystal_system": ["cubic", "hexagonal", "triclinic"][i % 3],
                "band_gap": [0.0, 1.0, 3.0][i % 3],
            }
        first = select_sample(candidates, metadata, train_ids)
        second = select_sample(list(reversed(candidates)), metadata, train_ids)
        self.assertEqual([r["mpid"] for r in first], [r["mpid"] for r in second])
        self.assertEqual(len(first), RANDOM_N + TARGETED_N)
        self.assertEqual(sum(r["arm"] == "probability" for r in first), RANDOM_N)
        self.assertEqual(sum(r["arm"] == "targeted" for r in first), TARGETED_N)
        self.assertEqual(len({r["mpid"] for r in first}), len(first))

    def test_base_stratum_uses_only_metadata(self):
        record = {
            "nelements": 2,
            "nsites": 9,
            "crystal_system": "Cubic",
            "band_gap": 1.5,
        }
        self.assertEqual(base_stratum(record), ("2", "09-16", "cubic", "gap_0_2"))

    def test_box_average_matches_single_and_multi_point_contract(self):
        edges = np.array([0.0, 1.0, 2.0])
        values, mask = box_average(
            np.array([0.25, 1.1, 1.9]), np.array([2.0, 2.0, 4.0]), edges
        )
        self.assertEqual(mask.tolist(), [1, 1])
        self.assertAlmostEqual(values[0], 2.0)
        self.assertAlmostEqual(values[1], 2.4)

    def test_box_average_drops_nonfinite_pairs(self):
        values, mask = box_average(
            np.array([-5.95, np.nan]), np.array([3.0, 2.0])
        )
        self.assertEqual(int(mask.sum()), 1)
        self.assertAlmostEqual(float(values[0]), 3.0)

    def test_winsor_clips_isolated_but_keeps_wide_peak(self):
        threshold = np.full(5, 5.0)
        clipped, count = winsorize_isolated(
            np.array([1.0, 10.0, 1.0, 1.0, 1.0]), np.ones(5), threshold
        )
        self.assertEqual(count, 1)
        self.assertEqual(clipped[1], 5.0)
        wide, count = winsorize_isolated(
            np.array([1.0, 10.0, 10.0, 10.0, 1.0]), np.ones(5), threshold
        )
        self.assertEqual(count, 0)
        np.testing.assert_allclose(wide[1:4], 10.0)

    def test_valence_count_honors_occupancy(self):
        structure = {
            "sites": [
                {"species": [{"element": "Si", "occu": 1.0}]},
                {"species": [{"element": "O", "occu": 0.5}]},
            ]
        }
        zval = {"Si": {"zval": 4}, "O": {"zval": 6}}
        self.assertEqual(valence_count(structure, zval), 7.0)

    def test_normalize_and_total_variation(self):
        a = normalize_spectrum(np.array([1.0, 1.0]))
        b = normalize_spectrum(np.array([2.0, 0.0]))
        self.assertAlmostEqual(float(total_variation(a, b)), 0.5)
        with self.assertRaises(ValueError):
            normalize_spectrum(np.zeros(2))

    def test_truncated_spectrum_keeps_shape_for_sensitivity(self):
        structure = {
            "lattice": {"matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]},
            "sites": [
                {
                    "species": [{"element": "H", "occu": 1.0}],
                    "abc": [0.0, 0.0, 0.0],
                }
            ],
        }
        row = {
            "structure": structure,
            "energies": [-7.0, -5.99, -5.95, 0.0, 0.05, 7.0],
            "spin_up_densities": [0.01] * 6,
            "spin_down_densities": None,
            "efermi": 0.0,
        }
        result = process_spectrum(
            row, {"H": {"zval": 1.0}}, np.full(128, 100.0)
        )
        self.assertEqual(result["quality_status"], "truncated")
        self.assertEqual(result["usable"], 0)
        self.assertIsNotNone(result["shape"])
        self.assertAlmostEqual(float(result["shape"].sum()), 1.0)

    def test_nearest_tv(self):
        query = np.array([[1.0, 0.0], [0.5, 0.5]])
        ref = np.array([[0.0, 1.0], [0.6, 0.4]])
        np.testing.assert_allclose(nearest_tv(query, ref, chunk=1), [0.4, 0.1])

    def test_wilson_interval_known_bounds(self):
        lo, hi = wilson_interval(250, 500)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertAlmostEqual(lo, 0.456, places=3)
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)

    def test_bootstrap_is_deterministic(self):
        values = np.arange(10, dtype=float)
        self.assertEqual(
            bootstrap_median_ci(values, draws=50, seed=7),
            bootstrap_median_ci(values, draws=50, seed=7),
        )

    def test_atomic_writers_leave_complete_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "x.csv"
            json_path = root / "x.json"
            atomic_write_csv([{"a": 1, "b": "x"}], csv_path)
            atomic_write_json({"ok": True}, json_path)
            self.assertNotIn(b"\r\n", csv_path.read_bytes())
            with csv_path.open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["a"], "1")
            self.assertTrue(json.loads(json_path.read_text())["ok"])
            self.assertFalse(list(root.glob("*.tmp")))

    def test_output_guard_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exists"
            path.write_text("x")
            with self.assertRaises(FileExistsError):
                ensure_outputs_available([path], force=False)
            ensure_outputs_available([path], force=True)


if __name__ == "__main__":
    unittest.main()
