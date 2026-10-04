"""Negative controls for CUDA acceptance metrics; these tests need only CPU."""

from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from tools.eval.periodic_message_gpu_check import (
    benchmark_samples, compare_gradients, compare_local, geometry_metrics,
    make_batch, summarize_benchmark,
)
from utils.periodic_geometry import PeriodicNeighborRecords, build_periodic_records

ROOT = Path(__file__).resolve().parents[1]


class GPUAcceptanceMetrics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_gradient_metric_detects_missing_zero_and_nonfinite_gradients(self):
        expected = {"input": torch.tensor([1., -2., 3.]), "weight": torch.ones(2, 4)}
        self.assertTrue(compare_gradients(expected, expected, torch.float32)["pass"])
        zero = {key: torch.zeros_like(value) for key, value in expected.items()}
        self.assertFalse(compare_gradients(expected, zero, torch.float32)["pass"])
        with self.assertRaises(ValueError):
            compare_gradients(expected, {"input": expected["input"]}, torch.float32)
        wrong = dict(expected, weight=torch.full((2, 4), float("nan")))
        with self.assertRaises(AssertionError):
            compare_gradients(expected, wrong, torch.float32)
        # Exact-zero symmetry channels are legitimate; small drift is explicitly
        # reported below the absolute floor, while the total norm is checked.
        expected = dict(expected, symmetric=torch.zeros(2))
        drift = dict(expected, symmetric=torch.full((2,), 1e-8))
        result = compare_gradients(expected, drift, torch.float32)
        self.assertTrue(result["pass"])
        self.assertIn("symmetric", result["tensors_below_absolute_floor"])

    def test_extra_angular_statistic_leak_is_not_hidden_by_scalar_match(self):
        zeros = torch.zeros(1, 2, 512)
        debug = {"states": [torch.zeros(1, 2, 108)] * 3,
                 "degree": torch.ones(1, 2, dtype=torch.long),
                 "pair_count": torch.zeros(1, 2, dtype=torch.long),
                 "angular_invariants": [torch.zeros(1, 2, 16)] * 2}
        reference = zeros, debug
        self.assertTrue(compare_local(reference, reference, torch.float32)["pass"])
        altered = dict(debug, angular_invariants=[torch.full((1, 2, 16), .01)] * 2)
        self.assertFalse(compare_local(reference, (zeros, altered), torch.float32)["pass"])
        missing = dict(debug, angular_invariants=[])
        with self.assertRaises(ValueError):
            compare_local(reference, (zeros, missing), torch.float32)
        wrong_count = dict(debug, degree=torch.zeros(1, 2, dtype=torch.long))
        self.assertFalse(compare_local(reference, (zeros, wrong_count), torch.float32)["pass"])

    def test_boundary_record_loss_is_reported_and_stable_loss_rejected(self):
        mask = torch.zeros(1, 2, dtype=torch.bool)
        edges = {"batch": torch.tensor([0]), "dst": torch.tensor([0]), "src": torch.tensor([1]),
                 "shifts": torch.zeros(1, 3, dtype=torch.long), "distances": torch.tensor([5.99995])}
        reference = PeriodicNeighborRecords.from_edges(edges, torch.tensor([[5.99995, 0., 0.]]), mask, 6.)
        empty = replace(reference, batch=reference.batch[:0], dst=reference.dst[:0], src=reference.src[:0],
                        shifts=reference.shifts[:0], vectors=reference.vectors[:0], distances=reference.distances[:0])
        metric = geometry_metrics(reference, empty, torch.float32)
        self.assertFalse(metric["pass"])
        self.assertEqual(metric["geometry_status"], "boundary_limited")
        self.assertEqual(metric["boundary_missing"], 1)
        stable = replace(reference, vectors=torch.tensor([[5., 0., 0.]]), distances=torch.tensor([5.]))
        with self.assertRaises(AssertionError):
            geometry_metrics(stable, empty, torch.float32)

    def test_fixed_real_batch_preserves_counts_and_timing_summary_uses_medians(self):
        prior = json.loads((ROOT / "results/periodic_message_precheck_q1_r6_20261003/manifest.json").read_text())
        definitions = benchmark_samples(prior["selected_samples"])
        self.assertEqual([len(v) for v in definitions.values()], [1, 4, 8, 1])
        pos, mask, _ = make_batch(ROOT / "data/train4ARPAT", definitions["typical_b4"])
        records = build_periodic_records(pos, mask, 6.)
        counts = torch.bincount(records.batch, minlength=4).tolist()
        self.assertEqual(counts, [s["n_edges"] for s in definitions["typical_b4"]])
        rows = []
        for device, times in (("cpu", (1., 10., 2.)), ("cuda", (.5, .1, .2))):
            for backward in (False, True):
                for duration in times:
                    rows.append({"category": "benchmark", "batch_label": "test", "route": "A", "device": device,
                                 "backward": backward, "seconds": duration, "peak_allocated_MiB": 20., "peak_reserved_MiB": 30.})
        summary = summarize_benchmark(rows)["test"]["A"]
        self.assertEqual(summary["cpu/forward_backward"]["median_seconds"], 2.)
        self.assertEqual(summary["cuda/forward_backward"]["median_seconds"], .2)
        self.assertEqual(summary["cpu_over_gpu_forward_backward"], 10.)


if __name__ == "__main__":
    unittest.main()
