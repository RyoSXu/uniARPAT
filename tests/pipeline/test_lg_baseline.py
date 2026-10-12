"""Baseline comparison arm: matched init, contract isolation and shared evaluation."""
from copy import deepcopy
from dataclasses import replace
import logging
import random
import unittest

import numpy as np
import torch

from model.model import basemodel
from model.transformer import Transformer
from tests.data.test_crystal_input import sample
from tools.eval.local_global_compare import training_command
import tools.train as runner
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.bucket_batch import crystal_collate
from utils.experiment_config import ExperimentConfig
from utils.lg_baseline_experiment import (
    BASELINE_CONTRACT_SCHEMA, BASELINE_INIT_PROTOCOL, baseline_contract,
    baseline_reference_params, initialize_baseline_model, validate_baseline_config,
)
from utils.local_global_experiment import (
    comparison_contract, initialize_comparison_model, untrained_reference_state,
    validate_local_global_config, capture_training_rng, restore_training_rng,
)


def shared():
    return dict(d_model=16, nhead=4, edos_num=8, phdos_num=4,
                num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=32,
                dropout=0., atom_feat_mode='z_only_proj', scale_mode='eta')


def arm_parameters():
    return {**shared(), 'local_route': 'A', 'local_layout': 'staged', 'local_depth': 2}


def baseline_parameters():
    return {**shared(), 'local_route': 'none', 'lg_baseline': True}


def configuration():
    return ExperimentConfig(epochs=2, batch_size=2, seed=42, tag='_unit',
        atom_feat='z_only_proj', skip_test_eval=True, lg_baseline=True,
        selection_metric='balanced_oracle_mae')


class TestLgBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_validation_freezes_the_only_difference_to_the_encoder(self):
        validate_baseline_config(configuration())
        validate_local_global_config(configuration())
        with self.assertRaisesRegex(ValueError, 'old-encoder arm'):
            validate_baseline_config(replace(configuration(), local_route='A',
                                             local_layout='staged', local_depth=2))
        for option, value in (('use_amp', True), ('init_ckpt', 'x.pth'),
                              ('freeze_backbone', True), ('reset_rng_after_init', True),
                              ('use_g1', True), ('use_g2', True),
                              ('use_periodic_manybody', True), ('use_macro_lattice', True),
                              ('c5_moe', True), ('use_atom_additive_phdos', True),
                              ('r1a_point', True), ('r1b_coord', True)):
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                validate_baseline_config(replace(configuration(), **{option: value}))
        with self.assertRaisesRegex(ValueError, 'selection_metric'):
            validate_baseline_config(replace(configuration(), selection_metric=None))
        with self.assertRaisesRegex(ValueError, 'SumNorm/H1'):
            validate_baseline_config(replace(configuration(), norm='minmax'))
        with self.assertRaisesRegex(ValueError, 'M1 and ZP'):
            validate_baseline_config(replace(configuration(), atom_feat='legacy3'))
        with self.assertRaisesRegex(ValueError, 'isolated tag'):
            validate_baseline_config(replace(configuration(), tag=''))
        # A non-baseline arm still cannot select checkpoints without a route.
        with self.assertRaisesRegex(ValueError, 'explicit selection metrics'):
            validate_local_global_config(replace(configuration(), lg_baseline=False))
        # The runner refuses conflicting options before any data or model access.
        with self.assertRaises(ValueError):
            runner.train_and_eval(replace(configuration(), use_amp=True))
        parsed = runner.build_arg_parser().parse_args(training_command(configuration())[2:])
        self.assertTrue(parsed.lg_baseline)
        self.assertEqual(parsed.selection_metric, 'balanced_oracle_mae')

    def test_common_part_is_exactly_the_four_arm_reference(self):
        arm, target = Transformer(**arm_parameters()), Transformer(**baseline_parameters())
        arm_init = initialize_comparison_model(arm, arm_parameters(), 42)
        encoder_before = {key: value.clone() for key, value in target.state_dict().items()
                          if key.startswith('encoder.')}
        init = initialize_baseline_model(target, baseline_parameters(), 42)

        self.assertEqual(init['protocol'], BASELINE_INIT_PROTOCOL)
        arm_state, current = arm.state_dict(), target.state_dict()
        common = sorted(key for key in current if not key.startswith('encoder.'))
        self.assertEqual(common, sorted(key for key in arm_state if not key.startswith('encoder.')))
        self.assertTrue(common and encoder_before)
        for key in common:
            torch.testing.assert_close(current[key], arm_state[key], atol=0, rtol=0)
        for key, value in encoder_before.items():
            torch.testing.assert_close(current[key], value, atol=0, rtol=0)
        # Same subset definition as the four arms, so the recorded digest is
        # directly comparable with their runtime.local_global_init record.
        self.assertEqual(init['reference_common_state_sha256'], arm_init['common_state_sha256'])
        self.assertEqual(init['n_encoder_tensors'], len(encoder_before))
        self.assertEqual(init['n_common_tensors'], len(common))
        for key in ('common_state_sha256', 'encoder_state_sha256', 'state_sha256'):
            self.assertEqual(len(init[key]), 64, key)
        self.assertNotEqual(init['common_state_sha256'], init['state_sha256'])
        # The target never shares tensors with the discarded reference.
        reference = untrained_reference_state(42, baseline_reference_params(baseline_parameters()))
        self.assertTrue(all(current[key] is not reference[key] for key in common))

    def test_common_part_aligns_while_the_old_encoder_stays_independent(self):
        target = Transformer(**baseline_parameters())
        init = initialize_baseline_model(target, baseline_parameters(), 42)
        torch.manual_seed(20261008)
        other = Transformer(**baseline_parameters())
        other_init = initialize_baseline_model(other, baseline_parameters(), 42)
        # Same reference seed: the copied part matches exactly. The encoder is
        # whatever its own construction produced, so it differs.
        self.assertEqual(init['common_state_sha256'], other_init['common_state_sha256'])
        self.assertNotEqual(init['state_sha256'], other_init['state_sha256'])
        self.assertNotEqual(init['encoder_state_sha256'], other_init['encoder_state_sha256'])

    def test_contract_is_distinct_and_cross_arm_checkpoints_are_rejected(self):
        cfg, params = configuration(), baseline_parameters()
        contract = baseline_contract(cfg, params)
        self.assertEqual(contract['schema'], BASELINE_CONTRACT_SCHEMA)
        self.assertEqual(contract['reference'], {'route': 'A', 'layout': 'staged',
                                                 'depth': 2, 'protocol': 'local-global-init-v1'})
        arm_cfg = replace(cfg, lg_baseline=False, local_route='A',
                          local_layout='staged', local_depth=2)
        arm_contract = comparison_contract(arm_cfg, arm_parameters())
        self.assertNotEqual(contract, arm_contract)
        self.assertNotEqual(contract, baseline_contract(
            replace(cfg, selection_metric='balanced_blind_mae'), params))

        source = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(source.parameters())
        payload = build_ablation_checkpoint(1, 'M1', 42, False, source, optimizer, .2,
            model_contract=contract, training_rng=capture_training_rng())
        before = deepcopy(source.state_dict())
        restored = restore_ablation_checkpoint(payload, source, optimizer, False,
                                               model_contract=contract)
        self.assertEqual(restored['epoch'], 1)
        self.assertTrue(restored['optimizer_restored'])
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        for wrong in (None, arm_contract,
                      baseline_contract(replace(cfg, selection_metric='balanced_blind_mae'), params)):
            with self.assertRaisesRegex(ValueError, 'model contract'):
                restore_ablation_checkpoint(payload, source, optimizer, False,
                                            model_contract=wrong)

    def test_saved_rng_replays_and_baseline_shares_the_blind_oracle_evaluator(self):
        state = capture_training_rng()
        expected = (random.random(), np.random.random(), torch.rand(3))
        restore_training_rng(state)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)

        wrapper = basemodel(logging.getLogger('baseline-test'), metrics_list=[],
            loss_form='sumnorm_klw', sub_model={'transformer': baseline_parameters()},
            optimizer={}, lr_scheduler={})
        wrapper.to(torch.device('cpu'))
        batch = crystal_collate([sample(2), sample(3, 1)])
        result = runner.evaluate_split(wrapper, [batch], sumnorm=True, return_sample_level=True,
                                       sample_ids=['a', 'b'])
        self.assertEqual(list(result['sample_df']['material_id']), ['a', 'b'])
        for mode in ('blind', 'oracle'):
            for task in ('edos', 'phdos'):
                for key in (f'{mode}_r2_{task}_median', f'{mode}_fail_rate_{task}',
                            f'{mode}_mae_{task}_median'):
                    self.assertTrue(np.isfinite(result[key]), key)
        self.assertEqual(result['r2_edos_median'], result['oracle_r2_edos_median'])
        self.assertNotEqual(result['balanced_blind_mae'], result['balanced_oracle_mae'])
        with self.assertRaisesRegex(ValueError, 'label every evaluated sample'):
            runner.evaluate_split(wrapper, [batch], sumnorm=True, return_sample_level=True,
                                  sample_ids=['a'])
        legacy = basemodel(logging.getLogger('baseline-test'), metrics_list=[],
            loss_form='sumnorm_klw', sub_model={'transformer': shared()},
            optimizer={}, lr_scheduler={})
        legacy.to(torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'blind/oracle evaluator'):
            runner.evaluate_split(legacy, [batch], sumnorm=True, sample_ids=['a', 'b'])


if __name__ == '__main__':
    unittest.main()
