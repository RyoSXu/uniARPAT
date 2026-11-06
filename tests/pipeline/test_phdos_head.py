"""phDOS head depth study (design 2026-11-06): construction, guards, contracts, verdicts."""
from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from tools.eval.phdos_head_analyze import verdict
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import training_record
from utils.spectral_balance_experiment import _backbone_params, phdos_head_state, validate_phdos_head

RECIPE = {'local_route': 'LR2', 'local_layout': 'staged', 'spectral_shape_aux': True,
          'selection_metric': 'balanced_blind_energy'}


class HeadDepthTest(unittest.TestCase):
    def test_default_contract_unchanged(self):
        self.assertNotIn('phdos_head_layers', training_record(ExperimentConfig()))
        self.assertEqual(training_record(replace(ExperimentConfig(), phdos_head_layers=1))['phdos_head_layers'], 1)
        self.assertIsNone(validate_phdos_head(ExperimentConfig()))
        self.assertNotIn('phdos_head_layers', _backbone_params({'phdos_head_layers': 1, 'local_route': 'LR2'}))

    def test_recipe_guard(self):
        self.assertEqual(validate_phdos_head(replace(ExperimentConfig(), phdos_head_layers=2, **RECIPE)), 2)
        for change in ({'phdos_head_layers': 6}, {'phdos_head_layers': 3}, {'local_route': 'I3'},
                       {'strain_augment': True}, {'seed': 43}, {'train_subset': 'x'}, {'local_layout': 'midpoint'}):
            cfg = replace(ExperimentConfig(), **{**RECIPE, 'phdos_head_layers': 1, **change})
            with self.assertRaises(ValueError):
                validate_phdos_head(cfg)

    def test_head_state_is_seeded_and_sized(self):
        one, again = phdos_head_state(16, 1, 42), phdos_head_state(16, 1, 42)
        self.assertEqual(set(one), {'phdos_out_head.layers.0.weight', 'phdos_out_head.layers.0.bias'})
        self.assertTrue(all(torch.equal(one[k], again[k]) for k in one))
        two = phdos_head_state(16, 2, 42)
        self.assertEqual(tuple(two['phdos_out_head.layers.0.weight'].shape), (48, 16, 3))
        self.assertEqual(tuple(two['phdos_out_head.layers.2.weight'].shape), (1, 48, 3))
        before = torch.random.get_rng_state()
        phdos_head_state(16, 2, 7)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_transformer_guard(self):
        from model.transformer import Transformer

        with self.assertRaises(ValueError):
            Transformer(local_route='I3', local_layout='staged', local_depth=2, phdos_head_layers=1)

    def test_verdict_rule(self):
        self.assertEqual(verdict({'joint': -0.005, 'edos': 0., 'phdos': -0.01}, True), 'better')
        self.assertEqual(verdict({'joint': 0.002, 'edos': 0., 'phdos': 0.005}, True), 'not_distinguished')
        self.assertEqual(verdict({'joint': 0.002, 'edos': 0., 'phdos': 0.011}, True), 'worse')
        self.assertEqual(verdict({'joint': 0.0045, 'edos': 0., 'phdos': 0.}, True), 'worse')
        self.assertEqual(verdict({'joint': -0.01, 'edos': 0., 'phdos': -0.02}, False), 'worse')


if __name__ == '__main__':
    unittest.main()
