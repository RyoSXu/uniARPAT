"""spectral-store-v1 (design 2026-11-10 P0): binning, spikes, keys, splits and sealing."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/data/spectral_store'))

from common import (box_average, choose_edos_task, exact_bin, clip_isolated_spikes, composition_key, reduced_key,  # noqa: E402
                    spike_thresholds)
import make_splits  # noqa: E402
import make_view  # noqa: E402
from tools.data.c2b_grids import box_average as label_box_average  # noqa: E402


class Binning(unittest.TestCase):
    def test_identical_to_label_code(self):
        rng = np.random.default_rng(0)
        edges = np.linspace(-6, 6, 129)
        for _ in range(5):
            x = np.sort(rng.uniform(-8, 8, 3000))
            y = rng.gamma(2.0, 1.0, 3000)
            ours, cov = box_average(x, y, edges)
            ref, ref_cov = label_box_average(x, y, edges)
            np.testing.assert_array_equal(ours, ref)
            np.testing.assert_array_equal(cov, ref_cov)


class ExactBin(unittest.TestCase):
    def test_matches_dense_integration_and_keeps_area(self):
        rng = np.random.default_rng(3)
        x = np.sort(rng.uniform(-8, 8, 300))
        y = rng.gamma(2.0, 1.0, 300)
        edges = np.linspace(-6, 6, 33)
        values, coverage = exact_bin(x, y, edges)
        fine = np.linspace(-6, 6, 32 * 2000 + 1)
        dense = np.interp(fine, x, y)
        ref = np.array([np.trapz(dense[i * 2000:(i + 1) * 2000 + 1], fine[i * 2000:(i + 1) * 2000 + 1])
                        for i in range(32)]) / (12 / 32)
        np.testing.assert_allclose(values, ref, rtol=1e-4, atol=1e-4)  # dense grid error
        np.testing.assert_array_equal(coverage, 1)

    def test_linear_function_exact_and_outside_zero(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])          # y = x
        values, coverage = exact_bin(x, y, np.array([-1.0, 0.0, 0.5, 1.5, 3.0]))
        np.testing.assert_allclose(values, [0.0, 0.25, 1.0, 0.875 / 1.5], rtol=1e-6)  # int_1.5^2 x dx = 0.875
        np.testing.assert_allclose(coverage, [0.0, 1.0, 1.0, 0.5 / 1.5], rtol=1e-6)

    def test_box_average_drops_edge_segments(self):
        x = np.arange(0.0, 1.0001, 0.3)        # 4 points per unit bin, y = 1
        values_box, _ = box_average(x, np.ones_like(x), np.array([0.0, 1.0]))
        values_exact, _ = exact_bin(x, np.ones_like(x), np.array([0.0, 0.9]))
        self.assertAlmostEqual(float(values_box[0]), 0.9, places=6)   # 10% of the area lost
        self.assertAlmostEqual(float(values_exact[0]), 1.0, places=6)

    def test_nonnegative_input_gives_nonnegative_bins(self):
        rng = np.random.default_rng(5)
        x = np.sort(rng.uniform(0, 1000, 4000))
        y = np.where(rng.random(4000) < 0.5, 0.0, rng.gamma(0.5, 1e-3, 4000))  # long zero runs
        values, _ = exact_bin(x, y, np.linspace(-280, 980, 65))
        self.assertTrue((values >= 0).all())


class Spikes(unittest.TestCase):
    def test_isolated_clipped_wide_kept(self):
        values = np.ones((3, 8), dtype=np.float32)
        values[0, 3] = 10.0          # isolated spike
        values[1, 2:5] = 10.0        # wide peak
        mask = np.ones_like(values, dtype=bool)
        thresholds = np.full(8, 5.0)
        out, clip = clip_isolated_spikes(values, mask, thresholds)
        self.assertEqual(out[0, 3], 5.0)
        np.testing.assert_array_equal(out[1], values[1])
        self.assertEqual(int(clip.sum()), 1)

    def test_thresholds_use_covered_values_only(self):
        values = np.array([[1.0, 100.0], [2.0, 0.0]])
        mask = np.array([[True, False], [True, True]])
        th = spike_thresholds(values, mask)
        self.assertLessEqual(th[1], 0.0 + 1e-12)


class Keys(unittest.TestCase):
    def test_reduced(self):
        key = composition_key([8, 8, 8, 8, 22, 22])
        self.assertEqual(key, '8:4,22:2')
        self.assertEqual(reduced_key(key), '8:2,22:1')


class EdosRule(unittest.TestCase):
    @staticmethod
    def task(t, canonical=False, functional='GGA', volume=10.0, status='ok'):
        return {'source_task_id': t, 'is_canonical': canonical, 'functional': functional,
                'calc_volume_per_atom': volume, 'status': status}

    def test_canonical_first_then_type_then_volume(self):
        types = {'a': 'GGA Static', 'b': 'GGA NSCF Uniform', 'c': 'GGA NSCF Uniform', 'd': 'HSE06 Static'}
        tasks = [self.task('a', volume=10.0), self.task('b', volume=13.0), self.task('c', volume=11.0),
                 self.task('d', functional='HSE06', volume=10.0)]
        self.assertEqual(choose_edos_task(tasks + [self.task('z', canonical=True)], 10.0, types)[1], 'canonical')
        row, kind = choose_edos_task(tasks, 10.0, types, ('NSCF Uniform', 'Static'))
        self.assertEqual((row['source_task_id'], kind), ('c', 'fallback'))
        self.assertEqual(choose_edos_task(tasks, 10.0, types)[0]['source_task_id'], 'a')   # volume only
        self.assertEqual(choose_edos_task(tasks, 10.0, types, allow_fallback=False), (None, None))
        truncated_canonical = [self.task('z', canonical=True, status='truncated'), self.task('a')]
        self.assertEqual(choose_edos_task(truncated_canonical, 10.0, types)[1], 'fallback')


def material(i, reduced, legacy=None, n_atoms=4):
    return {'material_id': f'mp-{i}', 'legacy_split': legacy, 'reduced_composition': reduced,
            'composition': reduced, 'in_current_mp': True, 'deprecated': False, 'license': 'BY-C',
            'structure_status': 'ok', 'n_atoms': n_atoms}


class Splits(unittest.TestCase):
    def setUp(self):
        self.old = dict(make_splits.TARGETS), make_splits.DUAL_POOL_MIN
        make_splits.TARGETS.update({'test_dual': 2, 'test_dual_small': 2, 'test_edos': 2, 'dev_edos': 2})

    def tearDown(self):
        make_splits.TARGETS.clear()
        make_splits.TARGETS.update(self.old[0])
        make_splits.DUAL_POOL_MIN = self.old[1]

    def test_groups_disjoint_and_exclusions(self):
        mats = [material(0, 'A', 'train'), material(1, 'B', 'valid'), material(2, 'C', 'test')]
        mats += [material(10, 'B'), material(11, 'A'), material(12, 'A', n_atoms=200)]
        letters = 'DEFGHIJKLMNOPQRS'
        for j, letter in enumerate(letters):
            mats += [material(100 + 2 * j, letter), material(101 + 2 * j, letter)]
        edos = {m['material_id'] for m in mats}
        phdos = {m['material_id'] for m in mats if m['reduced_composition'] in 'DEFG'}
        splits, report = make_splits.assign(mats, edos, phdos)
        by_id = {m['material_id']: m for m in mats}
        make_splits.check_disjoint(splits, by_id)
        self.assertNotIn('mp-10', sum(splits.values(), []))         # valid composition excluded
        self.assertIn('mp-11', splits['train_new'])                 # shares only train composition
        self.assertEqual(report['eligibility_exclusions']['too_many_atoms'], 1)
        for name in ('test_dual', 'test_edos', 'dev_edos'):
            self.assertTrue(splits[name])
            self.assertFalse({by_id[i]['reduced_composition'] for i in splits[name]} & {'A', 'B', 'C'})
        self.assertTrue(all(i in phdos for i in splits['test_dual']))
        self.assertFalse(any(i in phdos for i in splits['test_edos'] + splits['dev_edos']))


class Sealing(unittest.TestCase):
    def test_view_refuses_sealed_splits(self):
        config = {'name': 'x', 'splits': {'test': ['test_dual']}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                make_view.build(config, Path(tmp) / 'view', unseal=False)

    def test_nested_subsets(self):
        ids = [f'mp-{i}' for i in range(500)]
        small, large = make_view.nested_subset(ids, 50), make_view.nested_subset(ids, 200)
        self.assertTrue(small <= large)
        self.assertEqual(len(large), 200)


if __name__ == '__main__':
    unittest.main()
