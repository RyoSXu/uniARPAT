"""Element-only MACE attribution arm M0 (design 2026-11-09): feature construction and verdict rules."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.data.build_element_mean_features import build
from tools.eval.mace_m0_analyze import edos_reading, verdict
from utils.external_atom_features import SCHEMA, ExternalAtomFeatures
from utils.paths import file_sha256


def source_store(tmp):
    data, source = Path(tmp) / 'data', Path(tmp) / 'source'
    numbers = {'train': np.array([8, 8, 26, 8, 26]), 'valid': np.array([26, 8, 3]), 'test': np.array([8])}
    features = {'train': np.array([[1, 0], [3, 0], [10, 2], [5, 0], [20, 4]], dtype=np.float32),
                'valid': np.array([[99, 9], [7, 7], [5, 5]], dtype=np.float32),
                'test': np.array([[0, 0]], dtype=np.float32)}
    source.mkdir()
    (data).mkdir()
    (data / 'manifest.json').write_text('{}')
    splits = {}
    for split, z in numbers.items():
        (data / split).mkdir()
        np.save(data / split / f'atomic_numbers_{split}.npy', z.astype(np.int16))
        np.save(source / f'{split}.npy', features[split])
        splits[split] = {'materials': 1, 'atoms': len(z), 'dim': 2, 'sha256': file_sha256(source / f'{split}.npy')}
    manifest = {'schema': SCHEMA, 'source': 'test', 'model': {'sha256': 'x', 'name': 'm'},
                'data': {'root': str(data), 'manifest_sha256': file_sha256(data / 'manifest.json')},
                'splits': splits, 'layout': {'dim': 2, 'layers': 1},
                'standardization': {'mean': [0, 0], 'std': [1, 1], 'std_floor': 1e-6}}
    (source / 'manifest.json').write_text(json.dumps(manifest))
    return data, source


class ElementMeanFeatures(unittest.TestCase):
    def test_element_lookup_drops_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, source = source_store(tmp)
            out = Path(tmp) / 'm0'
            manifest = build(source, out)
            valid = np.load(out / 'valid.npy')
            np.testing.assert_allclose(valid[0], [15, 3])            # Fe: train mean, not its own 99/9
            np.testing.assert_allclose(valid[1], [3, 0])             # O
            np.testing.assert_allclose(valid[2], [7.8, 1.2], rtol=1e-6)  # Li unseen: global train mean
            self.assertEqual(manifest['elements']['unseen_atoms'], {'train': 0, 'valid': 1, 'test': 0})
            self.assertEqual(manifest['derived_from']['manifest_sha256'], file_sha256(source / 'manifest.json'))
            store = ExternalAtomFeatures(out, 'valid', data, np.array([0, 3]))
            self.assertEqual(tuple(store[0].shape), (3, 2))
            with self.assertRaises(FileExistsError):
                build(source, out)


class Rules(unittest.TestCase):
    def test_verdict_thresholds(self):
        base = {'edos': 0., 'phdos': 0.}
        self.assertEqual(verdict({**base, 'joint': -0.005}, True), 'element_dominant')
        self.assertEqual(verdict({**base, 'joint': -0.002}, True), 'partial')
        self.assertEqual(verdict({**base, 'joint': 0.001}, True), 'environment_dominant')
        self.assertEqual(verdict({**base, 'joint': 0.005}, True), 'worse')
        self.assertEqual(verdict({'joint': -0.005, 'edos': -0.02, 'phdos': 0.01}, True), 'worse')
        self.assertEqual(verdict({**base, 'joint': -0.005}, False), 'worse')

    def test_edos_reading(self):
        yes, no = {'excludes_zero': True}, {'excludes_zero': False}
        self.assertEqual(edos_reading(yes, no), 'element')
        self.assertEqual(edos_reading(no, yes), 'environment')
        self.assertEqual(edos_reading(yes, yes), 'partial')
        self.assertEqual(edos_reading(no, no), 'unresolved')


if __name__ == '__main__':
    unittest.main()
