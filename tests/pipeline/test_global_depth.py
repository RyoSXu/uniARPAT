"""Decoupled global depth: ordering, matched init, contract and E(3) contracts."""
from dataclasses import asdict, replace
import random
import unittest

import torch

from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture, parameters
from tools.eval.periodic_geometry_acceptance import e3_operations
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import (
    DEEP_CONTRACT_SCHEMA, DEEP_INIT_PROTOCOL, INIT_PROTOCOL,
    capture_training_rng, comparison_contract, comparison_configs,
    depth_comparison_configs, initialize_comparison_model, restore_training_rng,
    untrained_reference_state, validate_local_global_config,
)
from utils.spectral_inference import reconstruct_h1_spectra
from utils.zproj_alignment import state_hash


def deep_parameters(layout='staged', route='B+', depth=2, global_depth=6):
    return {**parameters(route, layout, depth), 'global_depth': global_depth}


def configuration():
    return ExperimentConfig(epochs=2, batch_size=2, seed=42, tag='_unit',
        atom_feat='z_only_proj', skip_test_eval=True, local_route='B+',
        local_layout='staged', local_depth=2, global_depth=6,
        selection_metric='balanced_oracle_mae')


class TestGlobalDepth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, layout='staged', global_depth=6, dtype=torch.float32):
        before = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            params = deep_parameters(layout, global_depth=global_depth)
            model = Transformer(**params)
            info = initialize_comparison_model(model, params, 42)
        finally:
            torch.set_default_dtype(before)
        return model.eval(), info

    def test_default_depth_keeps_the_historical_order_counts_and_initialization(self):
        for layout, order in (
                ('staged', (('local', 0), ('local', 1), ('global', 0), ('global', 1))),
                ('alternating', (('local', 0), ('global', 0), ('local', 1), ('global', 1)))):
            params = parameters('B+', layout, 2)
            plain = Transformer(**params)
            plain_info = initialize_comparison_model(plain, params, 42)
            with_depth = Transformer(**{**params, 'global_depth': 2})
            depth_info = initialize_comparison_model(with_depth, {**params, 'global_depth': 2}, 42)
            self.assertEqual(plain.encoder.execution_order, order)
            self.assertEqual(plain.encoder.update_counts,
                             {'local_calls': 2, 'internal_local_blocks': 4,
                              'global_attention': 2, 'global_ffn': 2})
            self.assertEqual(depth_info['update_counts'], plain_info['update_counts'])
            self.assertEqual(depth_info['state_sha256'], plain_info['state_sha256'])
            self.assertEqual(depth_info['common_state_sha256'], plain_info['common_state_sha256'])
            self.assertEqual(plain_info['protocol'], INIT_PROTOCOL)

    def test_depth_orders_are_the_requested_sequences_and_keep_update_counts(self):
        staged = Transformer(**deep_parameters('staged')).encoder
        alternating = Transformer(**deep_parameters('alternating')).encoder
        self.assertEqual(staged.execution_order,
                         (('local', 0), ('local', 1)) + tuple(('global', i) for i in range(6)))
        self.assertEqual(alternating.execution_order,
                         (('local', 0), ('global', 0), ('local', 1), ('global', 1))
                         + tuple(('global', i) for i in range(2, 6)))
        for encoder in (staged, alternating):
            self.assertEqual(encoder.update_counts,
                             {'local_calls': 2, 'internal_local_blocks': 4,
                              'global_attention': 6, 'global_ffn': 6})

    def test_deep_initialization_copies_the_reference_and_is_layout_independent(self):
        reference = untrained_reference_state(42, parameters('B+', 'staged', 2))
        reference_hash = state_hash(reference)
        common_hash = state_hash({key: value for key, value in reference.items()
                                  if not (key.startswith('encoder.local_layers.')
                                          and '.message.' in key)})
        self.assertFalse(any(key.startswith('encoder.global_layers.2.') for key in reference))
        infos = {}
        for layout in ('staged', 'alternating'):
            params = deep_parameters(layout)
            infos[layout] = initialize_comparison_model(Transformer(**params), params, 42)
        for layout, info in infos.items():
            self.assertEqual(info['protocol'], DEEP_INIT_PROTOCOL, layout)
            self.assertEqual(info['shared_init_protocol'], INIT_PROTOCOL, layout)
            self.assertEqual(info['shared_state_sha256'], reference_hash, layout)
            self.assertEqual(info['reference_common_state_sha256'], common_hash, layout)
            self.assertEqual(info['local_depth'], 2, layout)
            self.assertEqual(info['global_depth'], 6, layout)
            self.assertEqual(info['n_extended_global_tensors'], 48, layout)
        for key in ('shared_state_sha256', 'extended_global_state_sha256', 'state_sha256',
                    'common_state_sha256', 'reference_common_state_sha256'):
            self.assertEqual(infos['staged'][key], infos['alternating'][key], key)
        self.assertNotEqual(infos['staged']['state_sha256'], infos['staged']['shared_state_sha256'])

    def test_deep_contract_is_distinct_and_cross_depth_checkpoints_are_rejected(self):
        source = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(source.parameters())
        deep_cfg = configuration()
        deep_contract = comparison_contract(deep_cfg, deep_parameters())
        self.assertEqual(deep_contract['schema'], DEEP_CONTRACT_SCHEMA)
        self.assertEqual(deep_contract['init_protocol'], DEEP_INIT_PROTOCOL)
        self.assertEqual(deep_contract['reference'],
                         {'protocol': INIT_PROTOCOL, 'layout': 'staged',
                          'local_depth': 2, 'global_depth': 2})
        shallow_contract = comparison_contract(
            replace(deep_cfg, global_depth=None), parameters('B+', 'staged', 2))
        self.assertEqual(shallow_contract['schema'], 'local-global-model-v1')
        self.assertNotIn('global_depth', shallow_contract['training'])
        self.assertIn('global_depth', deep_contract['training'])
        self.assertNotEqual(deep_contract, shallow_contract)
        payload = build_ablation_checkpoint(1, 'M1', 42, False, source, optimizer, .2,
            model_contract=deep_contract, training_rng=capture_training_rng())
        for rejected in (shallow_contract, None,
                         comparison_contract(replace(deep_cfg, local_layout='alternating'),
                                             deep_parameters('alternating'))):
            with self.assertRaisesRegex(ValueError, 'model contract'):
                restore_ablation_checkpoint(payload, source, optimizer, False,
                                            model_contract=rejected)

    def test_validation_rejects_unsupported_decouplings_before_any_access(self):
        base = configuration()
        bad = [replace(base, global_depth=2), replace(base, global_depth=1),
               replace(base, global_depth=0), replace(base, global_depth=True),
               replace(base, global_depth=6, local_route='none',
                       local_layout=None, local_depth=None),
               replace(base, local_route='none', local_layout=None, local_depth=None,
                       global_depth=None)]
        for cfg in bad:
            with self.assertRaises(ValueError, msg=asdict(cfg)):
                validate_local_global_config(cfg)
        for depth, global_depth in ((2, 6), (1, 3), (3, 4)):
            configs = depth_comparison_configs(base, depth=depth, global_depth=global_depth,
                                               seeds=[42], tag_prefix='unit')
            self.assertEqual(len(configs), 2)
            self.assertEqual(len({cfg.tag for cfg in configs}), 2)
            for cfg in configs:
                self.assertGreater(cfg.global_depth, cfg.local_depth)
                validate_local_global_config(cfg)
        for kwargs in ({'depth': 0, 'global_depth': 6}, {'depth': 2, 'global_depth': 2},
                       {'depth': 2, 'global_depth': 6, 'seeds': []},
                       {'depth': 2, 'global_depth': 6, 'seeds': [42, 42]},
                       {'depth': 2, 'global_depth': 6, 'tag_prefix': 'bad/prefix'}):
            arguments = {'depth': 2, 'global_depth': 6, 'seeds': [42], 'tag_prefix': 'unit'}
            arguments.update(kwargs)
            with self.assertRaises(ValueError, msg=arguments):
                depth_comparison_configs(base, **arguments)
        # The historical four-arm generator still produces its own tags and depths.
        arms = comparison_configs(replace(base, global_depth=None), depth=2, seeds=[42],
                                  tag_prefix='unit')
        self.assertEqual([cfg.tag for cfg in arms],
                         ['_lg_unit_k2_a_staged_s42', '_lg_unit_k2_a_alternating_s42',
                          '_lg_unit_k2_bp_staged_s42', '_lg_unit_k2_bp_alternating_s42'])
        self.assertTrue(all(cfg.global_depth is None for cfg in arms))

    def test_full_outputs_keep_e3_invariance_for_both_spectra(self):
        for dtype in (torch.float32, torch.float64):
            tolerance = 2e-5 if dtype == torch.float32 else 2e-8
            z, geometry = fixture(dtype)
            for layout in ('staged', 'alternating'):
                model, _ = self.create(layout=layout, dtype=dtype)
                with torch.no_grad():
                    expected = model(z, z.eq(0), geometry)
                    expected.update(reconstruct_h1_spectra(expected, [3, 2], [16., 8.]))
                    for name, matrix, translation in e3_operations():
                        q, t = (torch.as_tensor(matrix, dtype=dtype),
                                torch.as_tensor(translation, dtype=dtype))
                        cell = geometry.cell @ q
                        frac = geometry.frac_coords + torch.bmm(
                            t.expand(2, 1, 3), torch.linalg.inv(cell))
                        actual = model(z, z.eq(0), type(geometry)(cell, frac))
                        actual.update(reconstruct_h1_spectra(actual, [3, 2], [16., 8.]))
                        for key in expected:
                            with self.subTest(dtype=dtype, layout=layout, operation=name, key=key):
                                torch.testing.assert_close(actual[key], expected[key],
                                                           atol=tolerance, rtol=tolerance)

    def test_resume_restores_weights_optimizer_and_rng_for_the_deep_contract(self):
        params = deep_parameters('alternating')
        model = Transformer(**params)
        initialize_comparison_model(model, params, 42)
        optimizer = torch.optim.AdamW(model.parameters())
        total = sum(parameter.sum() for parameter in model.parameters())
        total.backward()
        optimizer.step()
        contract = comparison_contract(configuration(), params)
        payload = build_ablation_checkpoint(1, 'M1', 42, False, model, optimizer, .2,
            model_contract=contract, training_rng=capture_training_rng())
        fresh = Transformer(**params)
        restore_ablation_checkpoint(payload, fresh, optimizer, False, model_contract=contract)
        left, right = model.state_dict(), fresh.state_dict()
        self.assertEqual(sorted(key for key in left if not torch.equal(left[key], right[key])), [])
        stream = capture_training_rng()
        restore_training_rng(payload['training_rng'])
        expected = (float(random.random()), torch.rand(2))
        restore_training_rng(payload['training_rng'])
        actual = (float(random.random()), torch.rand(2))
        restore_training_rng(stream)
        self.assertEqual(expected[0], actual[0])
        self.assertTrue(torch.equal(expected[1], actual[1]))


if __name__ == '__main__':
    unittest.main()
