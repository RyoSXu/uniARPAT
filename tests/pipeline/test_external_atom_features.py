"""MACE frozen-feature reference (design 2026-11-08): batching, attachment, guards, contracts."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from utils.bucket_batch import crystal_collate, split_crystal_batch
from utils.crystal import CrystalGeometry, FeaturedCrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.external_atom_features import FIELD, SCHEMA, ExternalAtomFeatures, load_manifest
from utils.local_global_experiment import training_record
from utils.paths import file_sha256
from utils.spectral_balance_experiment import _backbone_params, validate_external_features

RECIPE = {'local_route': 'LR2', 'local_layout': 'staged', 'spectral_shape_aux': True, 'atom_feat': 'z_only_proj',
          'selection_metric': 'balanced_blind_energy'}


def sample(n, dim=4):
    fields = [torch.arange(1, n + 1), CrystalGeometry(torch.eye(3) * 4, torch.rand(n, 3))]
    fields += [torch.zeros(1)] * 15
    fields.append(torch.full((n, dim), float(n)))
    return fields


class FeatureStore(unittest.TestCase):
    def write(self, root, data_dir, atoms, dim=3):
        values = np.arange(atoms * dim, dtype=np.float32).reshape(atoms, dim)
        np.save(root / 'valid.npy', values)
        manifest = {'schema': SCHEMA, 'data': {'manifest_sha256': file_sha256(data_dir / 'manifest.json')},
                    'layout': {'dim': dim, 'layers': 1},
                    'model': {'sha256': 'x', 'name': 'm'},
                    'splits': {'valid': {'atoms': atoms, 'sha256': file_sha256(root / 'valid.npy')}},
                    'standardization': {'mean': [1.0] * dim, 'std': [2.0, 0.0, 4.0], 'std_floor': 1e-6}}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return values

    def test_standardized_blocks_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, data = Path(tmp) / 'f', Path(tmp) / 'd'
            root.mkdir(), data.mkdir()
            (data / 'manifest.json').write_text('{}')
            values = self.write(root, data, 5)
            store = ExternalAtomFeatures(root, 'valid', data, np.array([0, 2, 5]))
            expected = (values[2:5] - 1.0) / np.array([2.0, 1e-6, 4.0])
            np.testing.assert_allclose(store[1].numpy(), expected.astype(np.float32), rtol=1e-6)
            with self.assertRaises(ValueError):
                ExternalAtomFeatures(root, 'valid', data, np.array([0, 2, 4]))
            (data / 'manifest.json').write_text('{"changed": 1}')
            with self.assertRaises(ValueError):
                load_manifest(root, data)


class Batching(unittest.TestCase):
    def test_collate_pads_and_split_trims(self):
        batch = crystal_collate([sample(2), sample(5), sample(3)])
        self.assertEqual(tuple(batch[FIELD].shape), (3, 5, 4))
        self.assertTrue(torch.all(batch[FIELD][0, 2:] == 0))
        chunks = split_crystal_batch(batch, pair_budget=2 * 5 * 5)
        for chunk in chunks:
            self.assertEqual(chunk[FIELD].shape[1], chunk[1].frac_coords.shape[1])
            self.assertEqual(chunk[FIELD].shape[0], chunk[0].shape[0])

    def test_attach_survives_device_and_dtype_moves(self):
        geometry = CrystalGeometry(torch.eye(3)[None].repeat(2, 1, 1), torch.zeros(2, 3, 3))
        featured = FeaturedCrystalGeometry.attach(geometry, torch.ones(2, 3, 4))
        cell, frac = featured
        self.assertIsInstance(featured, CrystalGeometry)
        moved = featured.to(torch.float64)
        self.assertEqual(moved.atom_features.dtype, torch.float64)
        self.assertEqual(moved.cell.dtype, torch.float64)
        with self.assertRaises(ValueError):
            FeaturedCrystalGeometry.attach(geometry, torch.ones(2, 4, 4))


class Guards(unittest.TestCase):
    def test_default_contract_unchanged(self):
        self.assertNotIn('external_atom_features', training_record(ExperimentConfig()))
        self.assertIsNone(validate_external_features(ExperimentConfig()))
        self.assertNotIn('external_atom_dim', _backbone_params({'external_atom_dim': 8, 'local_route': 'LR2'}))

    def test_recipe_guard(self):
        for change in ({'local_route': 'I3'}, {'strain_augment': True}, {'seed': 43}, {'train_subset': 'x'},
                       {'atom_feat': 'legacy3'}, {'single_task': 'edos'}):
            cfg = replace(ExperimentConfig(), **{**RECIPE, 'external_atom_features': 'x', **change})
            with self.assertRaises(ValueError):
                validate_external_features(cfg)

    def test_transformer_guard_and_zero_projection(self):
        from model.transformer import Transformer

        with self.assertRaises(ValueError):
            Transformer(local_route='I3', local_layout='staged', local_depth=2, atom_feat_mode='z_only_proj',
                        external_atom_dim=8)
        small = dict(d_model=16, nhead=2, dim_feedforward=32, num_encoder_layers=1, num_decoder_layers=1,
                     local_route='LR2', local_layout='staged', local_depth=2, global_depth=1,
                     atom_feat_mode='z_only_proj')
        torch.manual_seed(0)
        Transformer(**small)
        plain = torch.random.get_rng_state()
        torch.manual_seed(0)
        model = Transformer(**small, external_atom_dim=8)
        self.assertTrue(torch.equal(plain, torch.random.get_rng_state()))  # projection draws nothing
        self.assertTrue(torch.all(model.external_atom_proj.weight == 0))
        self.assertTrue(torch.all(model.external_atom_proj.bias == 0))


if __name__ == '__main__':
    unittest.main()
