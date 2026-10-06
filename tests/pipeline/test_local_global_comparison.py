"""Comparison planning, initialization/restore identity and blind/oracle reporting."""
from copy import deepcopy
from dataclasses import asdict, replace
import logging
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import pandas as pd
import yaml

from model.model import basemodel
from model.transformer import Transformer
from tests.data.test_crystal_input import sample
from tests.model.test_local_global_encoder import parameters
from tests.pipeline import test_checkpoint_boundary as checkpoint_fixtures
from tools.eval.local_global_compare import training_command, summarize
import tools.train as runner
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.bucket_batch import crystal_collate
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import (
    comparison_configs, comparison_contract, validate_local_global_config,
    capture_training_rng, restore_training_rng,
)
from utils.spectral_inference import reconstruct_h1_spectra


def configuration():
    return ExperimentConfig(epochs=2, batch_size=2, seed=42, tag='_unit',
        atom_feat='z_only_proj', skip_test_eval=True, local_route='A',
        local_layout='staged', local_depth=2, selection_metric='balanced_oracle_mae')


class TestLocalGlobalComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_four_arm_commands_round_trip_and_early_rejections(self):
        configs = comparison_configs(configuration(), depth=2, seeds=[42, 43], tag_prefix='unit')
        self.assertEqual(len(configs), 8)
        self.assertEqual(len({cfg.tag for cfg in configs}), 8)
        for cfg in configs:
            parsed = runner.build_arg_parser().parse_args(training_command(cfg)[2:])
            self.assertEqual(asdict(cfg), asdict(ExperimentConfig.from_args(parsed)))
        for depth in (0, 1):
            with self.assertRaisesRegex(ValueError, 'depth >= 2'):
                comparison_configs(configuration(), depth=depth, seeds=[42], tag_prefix='unit')
        bad = [replace(configuration(), use_amp=True), replace(configuration(), use_macro_lattice=True),
               replace(configuration(), skip_test_eval=False), replace(configuration(), selection_metric=None),
               replace(configuration(), init_ckpt='old.pth'), replace(configuration(), atom_feat='legacy3')]
        with patch.object(runner, 'ConfigBuilder', side_effect=AssertionError('data/model access')) as builder:
            for cfg in bad:
                with self.assertRaises(ValueError):
                    runner.train_and_eval(cfg)
            builder.assert_not_called()
        # The new ZP path bypasses the historical fingerprint protocol only after its own validation.
        validate_local_global_config(configuration())
        runner.validate_zproj_config(configuration())
        # Common options remain explicit and identical across all four arms;
        # validation does not reclassify them as geometric scoring side paths.
        allowed = replace(configuration(), augment=True, disp_sigma=.02, use_mask=True,
                          q2_fourier=True, energy_code='edos', edos_grid='unit-grid.npy')
        for cfg in comparison_configs(allowed, depth=2, seeds=[42], tag_prefix='allowed'):
            parsed = runner.build_arg_parser().parse_args(training_command(cfg)[2:])
            self.assertEqual(asdict(cfg), asdict(ExperimentConfig.from_args(parsed)))
            self.assertNotEqual(comparison_contract(cfg, parameters()),
                                comparison_contract(configuration(), parameters()))

    def test_checkpoint_rejects_layout_and_protocol_mismatch_before_loading(self):
        source = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(source.parameters())
        cfg = configuration()
        contract = comparison_contract(cfg, parameters())
        payload = build_ablation_checkpoint(1, 'M1', 42, False, source, optimizer, .2,
            model_contract=contract, training_rng=capture_training_rng())
        before = deepcopy(source.state_dict())
        other = comparison_contract(replace(cfg, local_layout='alternating'), parameters(layout='alternating'))
        with self.assertRaisesRegex(ValueError, 'model contract'):
            restore_ablation_checkpoint(payload, source, optimizer, False, model_contract=other)
        with self.assertRaisesRegex(ValueError, 'model contract'):
            restore_ablation_checkpoint(payload, source, optimizer, False)
        with self.assertRaisesRegex(ValueError, 'model contract'):
            restore_ablation_checkpoint(payload, source, optimizer, False,
                model_contract=comparison_contract(replace(cfg, tv_w=.1), parameters()))
        damaged = {**payload, 'training_rng': {}}
        with self.assertRaisesRegex(ValueError, 'incomplete training RNG'):
            restore_ablation_checkpoint(damaged, source, optimizer, False, model_contract=contract)
        damaged = {**payload, 'optimizer': {'param_groups': []}}
        with self.assertRaisesRegex(ValueError, 'optimizer state'):
            restore_ablation_checkpoint(damaged, source, optimizer, False, model_contract=contract)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            torch.save(payload, path)
            loaded = torch.load(path, weights_only=True)
            meta = restore_ablation_checkpoint(loaded, source, optimizer, False, model_contract=contract)
            self.assertEqual(meta['epoch'], 1)
            self.assertTrue(meta['optimizer_restored'])

    def test_saved_rng_replays_all_training_streams(self):
        state = capture_training_rng()
        expected = (random.random(), np.random.random(), torch.rand(3))
        restore_training_rng(state)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[0:2], actual[0:2])
        torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)

    def test_execution_history_preserves_contract_metadata_and_rng(self):
        model = Transformer(**parameters())
        blocks = [block for layer in model.encoder.local_layers for block in layer.message.blocks]
        cfg = configuration()
        saved = {'cli': asdict(cfg), 'model_contract': comparison_contract(cfg, parameters()),
                 'runtime': {'local_global_init': {'protocol': 'unit'}, 'other_metadata': .125}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config_used.yaml'
            path.write_text(yaml.safe_dump(saved))
            for block in blocks:
                block.recompute_angles = True
            old = runner.write_local_global_execution(path, model, 0)
            for block in blocks:
                block.recompute_angles = False
            rng = capture_training_rng()
            expected = (random.random(), np.random.random(), torch.rand(3))
            restore_training_rng(rng)
            new = runner.write_local_global_execution(path, model, 21)
            actual = (random.random(), np.random.random(), torch.rand(3))
            self.assertEqual(expected[:2], actual[:2])
            torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)
            self.assertEqual(new['from_epoch'], 22)
            self.assertEqual(new['recompute_angles'], [False] * 4)
            self.assertEqual(new['pair_chunk'], [4096] * 4)
            result = yaml.safe_load(path.read_text())
            self.assertEqual(result['cli'], saved['cli'])
            self.assertEqual(result['model_contract'], saved['model_contract'])
            self.assertEqual(result['runtime']['local_global_init'], saved['runtime']['local_global_init'])
            self.assertEqual(result['runtime']['other_metadata'], .125)
            self.assertEqual(result['runtime']['local_global_execution'], [old, new])
            unchanged = path.read_bytes()
            runner.write_local_global_execution(path, model, 21)
            self.assertEqual(path.read_bytes(), unchanged)

    def test_h1_structure_only_reconstruction_and_full_evaluator_modes(self):
        outputs = {'edos': torch.zeros(2, 8), 'phdos': torch.zeros(2, 4),
                   'eta': torch.tensor([[.5, .25], [.3, .7]])}
        spectra = reconstruct_h1_spectra(outputs, [2, 3], [8, 12])
        torch.testing.assert_close(spectra['blind_edos'].sum(-1), torch.tensor([8*.25, 12*.7])/.09375)
        torch.testing.assert_close(spectra['blind_phdos'].sum(-1), torch.tensor([3*2*.5, 3*3*.3])/19.6875)
        wrapper = basemodel(logging.getLogger('comparison-test'), metrics_list=[],
            sub_model={'transformer': parameters()}, optimizer={}, lr_scheduler={})
        wrapper.to(torch.device('cpu'))
        batch = crystal_collate([sample(2), sample(3, 1)])
        result = runner.evaluate_split(wrapper, [batch], sumnorm=True, return_sample_level=True)
        for mode in ('blind', 'oracle'):
            for task in ('edos', 'phdos'):
                for key in (f'{mode}_r2_{task}_median', f'{mode}_fail_rate_{task}',
                            f'{mode}_mae_{task}_median'):
                    self.assertTrue(np.isfinite(result[key]), key)
        self.assertEqual(len(result['sample_df']), 2)
        self.assertEqual(result['r2_edos_median'], result['oracle_r2_edos_median'])
        self.assertNotEqual(result['balanced_blind_mae'], result['balanced_oracle_mae'])

    def test_summary_keeps_unfinished_arms_pending_and_uses_selected_valid_epoch(self):
        configs = comparison_configs(configuration(), depth=2, seeds=[42], tag_prefix='summary')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / f'results/reports/history_m1{configs[0].tag}.csv'
            history.parent.mkdir(parents=True)
            history.write_text('epoch,balanced_score,balanced_oracle_mae,oracle_r2_edos_median\n1,.2,.2,.3\n')
            saved = root / f'results/runs/ablation_m1{configs[0].tag}/config_used.yaml'
            saved.parent.mkdir(parents=True)
            params = parameters()
            saved.write_text(yaml.safe_dump({'config': {'model': {'params': {'sub_model': {'transformer': params}}}},
                'model_contract': comparison_contract(configs[0], params)}))
            with patch('tools.eval.local_global_compare.repository_path', side_effect=lambda path: root / path):
                rows = summarize(configs)
            self.assertEqual([row['status'] for row in rows], ['partial', 'pending', 'pending', 'pending'])
            self.assertEqual(rows[0]['best_epoch'], 1)
            with patch('tools.eval.local_global_compare.repository_path', side_effect=lambda path: root / path):
                with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                    summarize([replace(configs[0], selection_metric='balanced_blind_mae')])

    def test_actual_runner_fresh_and_resume_keep_training_streams_and_selection(self):
        # Reuse the isolated runner fixture; the step/evaluation are synthetic,
        # with no dataset access or model parameter update.
        fixture = checkpoint_fixtures.TestRunnerRecovery()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.cfg = replace(configuration(), tag='_recovery_contract')
        fixture.model.model['transformer'].local_route = 'A'
        fixture.model.model['transformer'].encoder = SimpleNamespace(local_layers=[
            SimpleNamespace(message=SimpleNamespace(blocks=[
                SimpleNamespace(recompute_angles=False, pair_chunk=4096)]))])
        fixture.evaluate.return_value.update({
            'balanced_oracle_mae': .25, 'balanced_blind_mae': .4,
            'blind_r2_edos_median': .3, 'blind_r2_phdos_median': .4})
        observed = []
        def synthetic_step(*args, **kwargs):
            observed.append((random.random(), np.random.random(), torch.rand(1).item()))
            return {'loss': 1., 'loss_edos': .5, 'loss_phdos': .5}
        fixture.model.train_one_step.side_effect = synthetic_step
        saved = []
        original_build = runner.build_ablation_checkpoint
        def record_checkpoint(*args, **kwargs):
            payload = original_build(*args, **kwargs)
            saved.append(deepcopy(payload))
            return payload
        with patch.object(runner, 'initialize_comparison_model', return_value={'protocol': 'unit'}), \
                patch.object(runner, 'build_ablation_checkpoint', side_effect=record_checkpoint):
            runner.train_and_eval(fixture.cfg)
            self.assertEqual(len(observed), 2)
            expected = observed[-1]
            first = next(payload for payload in saved if payload['epoch'] == 1)
            torch.save(first, fixture.latest_path)
            history = pd.read_csv(fixture.history_path)
            history.iloc[:1].to_csv(fixture.history_path, index=False)
            observed.clear()
            runner.train_and_eval(fixture.cfg)
        self.assertEqual(observed, [expected])
        history = pd.read_csv(fixture.history_path)
        self.assertEqual(list(history['balanced_score']), [.25, .25])
        self.assertIn('training_rng', saved[-1])
        self.assertEqual(saved[-1]['model_contract']['selection_metric'], 'balanced_oracle_mae')
        execution = yaml.safe_load(fixture.config_path.read_text())['runtime']['local_global_execution']
        self.assertEqual([item['from_epoch'] for item in execution], [1, 2])
        self.assertTrue(all(item['recompute_angles'] == [False] for item in execution))

    def test_existing_history_without_configuration_is_explicitly_unverifiable(self):
        configs = comparison_configs(configuration(), depth=2, seeds=[42], tag_prefix='missing_config')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / f'results/reports/history_m1{configs[0].tag}.csv'
            history.parent.mkdir(parents=True)
            history.write_text('epoch,balanced_score,balanced_oracle_mae\n1,.2,.2\n')
            original = history.read_bytes()
            with patch('tools.eval.local_global_compare.repository_path', side_effect=lambda path: root / path):
                with self.assertRaisesRegex(ValueError, 'missing configuration') as error:
                    summarize(configs)
            self.assertIn(configs[0].tag, str(error.exception))
            self.assertIn(str(history), str(error.exception))
            self.assertIn('config_used.yaml', str(error.exception))
            self.assertEqual(history.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
