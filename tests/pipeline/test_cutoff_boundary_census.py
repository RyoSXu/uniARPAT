"""Boundary exposure includes excluded edges and resolves per-atom count flips."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tools.eval.cutoff_boundary_census import (
    RADIUS, SCAN_RADIUS, band_key, degree_change_summary, geometry_batches, summarize_batch,
)
from utils.crystal import CrystalGeometry
from utils.periodic_geometry import build_periodic_records


class CutoffBoundaryCensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def cubic(self, length):
        return CrystalGeometry(torch.eye(3)[None]*length, torch.zeros(1, 1, 3))

    def test_expanded_scan_covers_neighbors_excluded_from_production(self):
        mask = torch.zeros(1, 1, dtype=torch.bool)
        geometry = self.cubic(6.00002)
        original = build_periodic_records(geometry, mask, RADIUS)
        expanded = build_periodic_records(geometry, mask, SCAN_RADIUS)
        rows, degree, _, _ = summarize_batch(expanded, mask)
        self.assertEqual(len(original.distances), 0)
        self.assertEqual(len(expanded.distances), 6)
        self.assertEqual(rows[0]['records_kept'], 0)
        self.assertEqual(degree.tolist(), [[0]])
        self.assertEqual(rows[0][band_key(1e-4)+'_outside'], 6)
        self.assertEqual(rows[0][band_key(1e-4)+'_centers'], 1)

    def test_exact_boundary_is_reported_separately_and_remains_excluded(self):
        mask = torch.zeros(1, 1, dtype=torch.bool)
        records = build_periodic_records(self.cubic(6.), mask, SCAN_RADIUS)
        rows, _, _, _ = summarize_batch(records, mask)
        self.assertEqual(rows[0]['records_kept'], 0)
        for band in (1e-6, 1e-4, 1e-2):
            key = band_key(band)
            self.assertEqual([rows[0][f'{key}_{side}'] for side in ('inside', 'exact', 'outside')], [0, 6, 0])

    def test_side_counts_and_receiver_counts_do_not_confuse_edges_with_atoms(self):
        records = SimpleNamespace(batch=torch.tensor([0, 0, 0, 0, 1, 1]),
            dst=torch.tensor([0, 1, 1, 2, 0, 0]),
            distances=torch.tensor([5., 5.9999995, 6., 6.0000005, 6.0099, 5.5]))
        mask = torch.tensor([[False, False, False], [False, True, True]])
        rows, degree, _, _ = summarize_batch(records, mask)
        self.assertEqual([r['n_atoms'] for r in rows], [3, 1])
        self.assertEqual(degree.tolist(), [[1, 1, 0], [1, 0, 0]])
        key = band_key(1e-6)
        self.assertEqual([rows[0][f'{key}_{side}'] for side in ('inside', 'exact', 'outside')], [1, 1, 1])
        self.assertEqual(rows[0][f'{key}_centers'], 2)
        self.assertEqual(rows[1][band_key(1e-2)+'_outside'], 1)

    def test_padding_records_are_rejected(self):
        records = SimpleNamespace(batch=torch.tensor([0]), dst=torch.tensor([1]), distances=torch.tensor([5.]))
        with self.assertRaisesRegex(ValueError, 'padding'):
            summarize_batch(records, torch.tensor([[False, True]]))

    def test_opposite_degree_changes_are_not_hidden_by_zero_total_change(self):
        result = degree_change_summary(np.array([3, 2, 0]), np.array([2, 3, 0]))
        self.assertEqual(result['net_degree_delta'], 0)
        self.assertEqual(result['changed_centers'], 2)
        self.assertEqual(result['max_abs_degree_delta'], 1)
        self.assertAlmostEqual(result['max_abs_log_degree_delta'], np.log(4)-np.log(3))

    def test_geometry_batching_preserves_complete_large_structures_without_labels(self):
        samples = [(torch.full((n,), 14), CrystalGeometry(torch.eye(3)*10, torch.zeros(n, 3)))
                   for n in (1, 2, 160, 3)]
        batches = list(geometry_batches(samples, np.arange(4), list(range(4))))
        self.assertEqual(np.concatenate([positions for positions, _, _ in batches]).tolist(), list(range(4)))
        self.assertEqual([int(count) for _, numbers, _ in batches for count in numbers.ne(0).sum(-1)], [1, 2, 160, 3])


if __name__ == '__main__':
    unittest.main()
