"""Learning-curve train subsets (design 2026-11-06): nesting, whole groups, repeat, contracts."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import training_record
from utils.train_subset import FRACTIONS, build_subsets, composition_key, load_train_subset, write_subset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/crystal_q1_v1'


def fake_data(root: Path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    (root / 'train').mkdir(parents=True)
    sizes = rng.integers(1, 5, n)
    numbers = np.concatenate([rng.integers(1, 6, s) for s in sizes]).astype(np.uint8)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    ids = np.asarray([f'mp-{i}' for i in range(n)])
    np.save(root / 'train/train_index.npy', ids)
    np.save(root / 'train/atom_offsets_train.npy', offsets)
    np.save(root / 'train/atomic_numbers_train.npy', numbers)
    (root / 'manifest.json').write_text(json.dumps({'training_exclusions': ['mp-0']}))
    return ids, numbers, offsets


class SubsetTest(unittest.TestCase):
    def test_nested_whole_groups_and_fractions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, numbers, offsets = fake_data(root)
            subsets = build_subsets(root)
            keys = {str(ids[i]): composition_key(numbers[offsets[i]:offsets[i + 1]]) for i in range(1, len(ids))}
            small, large = (set(subsets[f]['material_ids']) for f in (0.25, 0.5))
            self.assertTrue(small < large)
            self.assertNotIn('mp-0', large)  # training exclusions stay out
            for fraction, members in ((0.25, small), (0.5, large)):
                inside = {keys[m] for m in members}
                outside = {k for m, k in keys.items() if m not in members}
                self.assertFalse(inside & outside)
                self.assertGreaterEqual(len(members), fraction * len(keys))
                self.assertEqual(subsets[fraction]['repeat'] * fraction, 1.)
            self.assertEqual(build_subsets(root)[0.25]['material_ids'], subsets[0.25]['material_ids'])

    def test_frozen_file_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_data(root)
            subset = build_subsets(root)[0.5]
            path = root / 'subset.json'
            digest = write_subset(subset, path)
            self.assertEqual(load_train_subset(path, root)['sha256'], digest)
            with self.assertRaises(FileExistsError):
                write_subset(subset, path)
            other = root / 'other'
            fake_data(other, seed=1)
            (other / 'manifest.json').write_text(json.dumps({'training_exclusions': []}))
            with self.assertRaises(ValueError):
                load_train_subset(path, other)
            bad = root / 'bad.json'
            bad.write_text(json.dumps({**subset, 'repeat': 3}))
            with self.assertRaises(ValueError):
                load_train_subset(bad, root)
        with self.assertRaises(ValueError):
            build_subsets(ROOT, fractions=(0.1,))

    def test_default_contract_unchanged(self):
        self.assertNotIn('train_subset', training_record(ExperimentConfig()))
        self.assertEqual(training_record(replace(ExperimentConfig(), train_subset='x'))['train_subset'], 'x')
        self.assertEqual(FRACTIONS, {0.25: 4, 0.5: 2})

    def test_recipe_guard(self):
        from utils.spectral_balance_experiment import validate_learning_curve

        self.assertIsNone(validate_learning_curve(ExperimentConfig()))
        for change in ({'local_route': 'I3'}, {'strain_augment': True}, {'seed': 43}):
            recipe = {'train_subset': 'x', 'local_route': 'LR2', 'local_layout': 'staged',
                      'spectral_shape_aux': True, 'selection_metric': 'balanced_blind_energy'}
            cfg = replace(ExperimentConfig(), **{**recipe, **change})
            with self.assertRaises(ValueError):
                validate_learning_curve(cfg)


@unittest.skipUnless((DATA / 'manifest.json').is_file(), 'local crystal data required')
class DatasetRowsTest(unittest.TestCase):
    def test_rows_repeat_subset_and_valid_untouched(self):
        from utils.dataset import Dos_Dataset

        with tempfile.TemporaryDirectory() as tmp:
            subset = build_subsets(DATA)[0.25]
            path = Path(tmp) / 'f25.json'
            write_subset(subset, path)
            train = Dos_Dataset(data_dir=str(DATA), split='train', dos_sumnorm=True, train_subset=str(path))
            counts = Counter(str(m) for m in train.material_ids)
            self.assertEqual(set(counts), set(subset['material_ids']))
            self.assertEqual(set(counts.values()), {4})
            self.assertEqual(len(train), 4 * subset['materials'])
            with self.assertRaises(ValueError):
                Dos_Dataset(data_dir=str(DATA), split='valid', dos_sumnorm=True, train_subset=str(path))


if __name__ == '__main__':
    unittest.main()
