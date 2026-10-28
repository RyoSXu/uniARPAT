"""Balanced blind objective through model updates, selection, resume and export."""
from copy import deepcopy
from dataclasses import replace
import json
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from model.model import basemodel
from model.transformer import Transformer
from tests.data.test_crystal_input import sample
from tests.data.test_spectral_balance import protocol, row
from tests.model.test_local_global_encoder import fixture
from tests.pipeline import test_checkpoint_boundary as checkpoint_fixtures
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_structure_controls import predict
from tools.eval.lg_valid_export import select_best
from tools.eval import lg_valid_export as exporter
from tools.eval.local_global_compare import training_command
import tools.train as trainer
from utils.bucket_batch import crystal_collate
from utils.checkpoint_selection import selected_history_row
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import initialize_comparison_model, training_record, validate_local_global_config
from utils.local_structure_experiment import FROZEN
from utils.spectral_balance import ENERGY_METRIC, reconstruct_balanced_spectra


def grid_sample(n, seed):
    return sample(n, seed) + [torch.linspace(-6, 6, 8), torch.linspace(0, 1200, 4)]


class BalancedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initialization_reuses_a64_weights_and_preserves_rng(self):
        params = model_params('I3')
        original = Transformer(**params)
        initialize_comparison_model(original, params, 42)
        selected_params = {**params, 'scale_mode': 'per_atom'}
        candidate = Transformer(**selected_params)
        rng = torch.get_rng_state().clone()
        info = initialize_comparison_model(candidate, selected_params, 42)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(info['backbone_initial_tensors_identical'])
        for key, value in candidate.state_dict().items():
            old_key = key.replace('mass_head.', 'eta_head.', 1)
            torch.testing.assert_close(value, original.state_dict()[old_key], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), sum(p.numel() for p in original.parameters()))

    def test_split_decoder_copies_one_initial_decoder_into_both_tasks(self):
        params = {**model_params('I3'), 'scale_mode': 'per_atom'}
        shared = Transformer(**params)
        initialize_comparison_model(shared, params, 42)
        split_params = {**params, 'decoupled_decoder': True}
        split = Transformer(**split_params)
        info = initialize_comparison_model(split, split_params, 42)
        self.assertTrue(info['split_decoder'])
        shared_state, split_state = shared.state_dict(), split.state_dict()
        for key, value in shared_state.items():
            if key.startswith('decoder.'):
                suffix = key[len('decoder.'):]
                torch.testing.assert_close(split_state['edos_decoder.' + suffix], value, rtol=0, atol=0)
                torch.testing.assert_close(split_state['phdos_decoder.' + suffix], value, rtol=0, atol=0)
            else:
                torch.testing.assert_close(split_state[key], value, rtol=0, atol=0)
        self.assertFalse(any(key.startswith('decoder.') for key in split_state))

    def test_phonon_head_keeps_every_a64_tensor_and_seeds_only_the_head(self):
        params = {**model_params('I3'), 'scale_mode': 'per_atom'}
        shared = Transformer(**params)
        initialize_comparison_model(shared, params, 42)
        phonon_params = {**params, 'phonon_head': True}
        first, second = Transformer(**phonon_params), Transformer(**phonon_params)
        info = initialize_comparison_model(first, phonon_params, 42)
        initialize_comparison_model(second, phonon_params, 42)
        self.assertIn('phonon_head', info)
        state, again = first.state_dict(), second.state_dict()
        for key, value in shared.state_dict().items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        head = [key for key in state if key.startswith('phonon_head.')]
        self.assertTrue(head and set(state) == set(shared.state_dict()) | set(head))
        for key in head:
            torch.testing.assert_close(state[key], again[key], rtol=0, atol=0)

    def test_phonon_head_trains_through_shared_objective_and_skips_old_phdos_readout(self):
        model = self.training_model(shape_aux=True, phonon_head=True)
        items = [grid_sample(3, 42), grid_sample(2, 43)]
        for item in items:
            item[14] = None
        batch = crystal_collate(items)
        stats = model.train_one_step(batch, 0)
        metrics = trainer.evaluate_local_global_split(model, [batch])
        auxiliary = .5 * (stats['loss_shape_e'] + stats['loss_shape_p'])
        self.assertAlmostEqual(stats['loss'] - auxiliary, metrics[ENERGY_METRIC], places=6)
        transformer = model.model['transformer']
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in transformer.phonon_head.parameters()))
        self.assertTrue(all(p.grad is None for p in transformer.phdos_out_head.parameters()))
        self.assertTrue(all(p.grad is not None for p in transformer.edos_out_head.parameters()))

    def test_phonon_correction_starts_exactly_at_the_physics_shape(self):
        pure = self.training_model(shape_aux=True, phonon_head=True)
        prior = self.training_model(shape_aux=True, phonon_head=True, phonon_correction=True)
        for model in (pure, prior):
            params = model.params['sub_model']['transformer']
            initialize_comparison_model(model.model['transformer'], params, 42)
        pure_state, prior_state = pure.model['transformer'].state_dict(), prior.model['transformer'].state_dict()
        final = [key for key in prior_state if key.startswith('phdos_out_head.layers.10.')]
        self.assertEqual(len(final), 2)
        for key, value in prior_state.items():
            if key in final:
                self.assertEqual(int(torch.count_nonzero(value)), 0)
            else:
                torch.testing.assert_close(value, pure_state[key], rtol=0, atol=0)
        items = [grid_sample(3, 42), grid_sample(2, 43)]
        for item in items:
            item[14] = None
        batch = crystal_collate(items)
        outputs = []
        for model in (pure, prior):
            model.model['transformer'].eval()
            inp, pos, mask, *rest = model.data_preprocess(batch)
            with torch.no_grad():
                outputs.append(model.model['transformer'](inp, mask, pos, rest[-2], rest[-1]))
        for key in ('edos', 'phdos', 'mass_logits'):
            torch.testing.assert_close(outputs[1][key], outputs[0][key], rtol=0, atol=0)

    def test_phonon_correction_learns_through_the_zeroed_final_convolution(self):
        model = self.training_model(shape_aux=True, phonon_head=True, phonon_correction=True)
        initialize_comparison_model(model.model['transformer'], model.params['sub_model']['transformer'], 42)
        items = [grid_sample(3, 42), grid_sample(2, 43)]
        for item in items:
            item[14] = None
        batch = crystal_collate(items)
        stats = model.train_one_step(batch, 0)
        metrics = trainer.evaluate_local_global_split(model, [batch])
        auxiliary = .5 * (stats['loss_shape_e'] + stats['loss_shape_p'])
        self.assertAlmostEqual(stats['loss'] - auxiliary, metrics[ENERGY_METRIC], places=6)
        transformer = model.model['transformer']
        final = transformer.phdos_out_head.layers[10]
        self.assertGreater(float(final.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in transformer.phonon_head.parameters()))

    def training_model(self, *, shape_aux=False, phonon_head=False, phonon_correction=False):
        transformer = {**model_params('I3'), 'scale_mode': 'per_atom'}
        if phonon_head:
            transformer['phonon_head'] = True
        if phonon_correction:
            transformer['phonon_correction'] = True
        model = basemodel(logging.getLogger('balanced-test'), loss_form=ENERGY_METRIC,
            spectral_balance=protocol((8, 4)), spectral_shape_aux=shape_aux, metrics_list=[],
            sub_model={'transformer': transformer},
            optimizer={'transformer': {'type': 'AdamW', 'params': {'lr': 0.}}}, lr_scheduler={})
        model.to(torch.device('cpu'))
        return model

    def test_loss_and_eval_share_reconstruction_and_do_not_require_valence(self):
        model = self.training_model()
        items = [grid_sample(3, 42), grid_sample(2, 43)]
        for item in items:
            item[14] = None
        batch = crystal_collate(items)
        stats = model.train_one_step(batch, 0)
        metrics = trainer.evaluate_local_global_split(model, [batch], return_sample_level=True)
        self.assertAlmostEqual(stats['loss'], .5*(stats['loss_edos'] + stats['loss_phdos']), places=6)
        self.assertAlmostEqual(stats['loss'], metrics[ENERGY_METRIC], places=6)
        self.assertEqual(metrics['selection_samples'], 2)
        self.assertIn('blind_r2_edos_p10', metrics)
        self.assertEqual(stats['loss_eta'], 0.)
        grads = [p.grad for p in model.model['transformer'].mass_head.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_accumulation_preserves_sample_weighting_and_one_optimizer_update(self):
        full, split = self.training_model(), self.training_model()
        split.model['transformer'].load_state_dict(deepcopy(full.model['transformer'].state_dict()))
        batch = crystal_collate([grid_sample(3, 42), grid_sample(2, 43), grid_sample(1, 44)])
        left = full.train_one_step(batch, 0, pair_budget=27)
        with patch.object(split.optimizer['transformer'], 'step', wraps=split.optimizer['transformer'].step) as step:
            right = split.train_one_step(batch, 0, pair_budget=9)
        self.assertEqual(step.call_count, 1)
        self.assertAlmostEqual(left['loss'], right['loss'], places=5)
        for (name, p), (_, q) in zip(full.model['transformer'].named_parameters(), split.model['transformer'].named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=name)

    def test_new_mass_reconstruction_preserves_fixture_e3_and_permutation(self):
        params = {**model_params('I3'), 'scale_mode': 'per_atom'}
        model = Transformer(**params).eval()
        initialize_comparison_model(model, params, 42)
        z, geometry = fixture()
        norm = protocol((8, 4))
        def spectra(numbers, geom):
            return reconstruct_balanced_spectra(predict(model, numbers, geom), numbers.ne(0).sum(-1), norm)
        with torch.no_grad():
            expected = spectra(z, geometry)
            # Orthogonal reflection plus translation in Cartesian coordinates.
            matrix = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
            cell = geometry.cell @ matrix
            shift = torch.bmm(torch.tensor([[[.2, -.3, .4]]]).expand(2, -1, -1), torch.linalg.inv(cell))
            actual = spectra(z, CrystalGeometry(cell, geometry.frac_coords + shift))
            order = torch.tensor([2, 0, 3, 1])
            permuted = spectra(z[:, order], CrystalGeometry(geometry.cell, geometry.frac_coords[:, order]))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(permuted[key], expected[key], atol=1e-4, rtol=1e-4)

    def test_new_selection_requires_opt_in_and_legacy_training_contract_is_unchanged(self):
        cfg = ExperimentConfig(**{**FROZEN, 'local_route': 'I3', 'selection_metric': ENERGY_METRIC})
        with self.assertRaisesRegex(ValueError, 'spectral_balance_config'):
            validate_local_global_config(cfg)
        self.assertNotIn('spectral_balance_config', training_record(replace(cfg, selection_metric='balanced_blind_r2')))
        parsed = trainer.build_arg_parser().parse_args(training_command(replace(cfg,
            spectral_balance_config='/tmp/normalizer.json', scale_mode='per_atom'))[2:])
        self.assertEqual(parsed.spectral_balance_config, '/tmp/normalizer.json')

    def runner_fixture(self):
        fixture = checkpoint_fixtures.TestRunnerRecovery()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        path = fixture.root / 'protocol.json'
        path.write_text(json.dumps(protocol()))
        fixture.cfg = ExperimentConfig(**{**FROZEN, 'local_route': 'I3',
            'tag': '_recovery_contract', 'scale_mode': 'per_atom', 'selection_metric': ENERGY_METRIC,
            'spectral_balance_config': str(path)})
        fixture.model.model['transformer'].local_route = 'I3'
        fixture.model.model['transformer'].encoder = SimpleNamespace(local_layers=[])
        self.enterContext(patch('utils.spectral_balance_experiment.validate_balance_config', return_value=protocol()))
        self.enterContext(patch.object(trainer, 'initialize_comparison_model', return_value={'protocol': 'unit'}))
        return fixture

    def metric(self, error=.3, failure=10.):
        metrics = row(e=error, p=error, blind_fail_rate_edos=failure)
        metrics.pop('epoch')
        return {**metrics,
                'blind_mae_edos_median': .2, 'blind_mae_phdos_median': .01,
                'mae_edos_median': .1, 'mae_phdos_median': .001}

    def test_runner_filters_saves_resumes_and_export_rederives_same_best(self):
        fixture = self.runner_fixture()
        metrics = [self.metric(.3), self.metric(.1, 11.), self.metric(.2), self.metric(.2)] + [self.metric(.6)]*31
        fixture.evaluate.side_effect = deepcopy(metrics)
        checkpoints = []
        original = trainer.build_ablation_checkpoint
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            checkpoints.append(deepcopy(result))
            return result
        with patch.object(trainer, 'build_ablation_checkpoint', side_effect=capture):
            result = trainer.train_and_eval(fixture.cfg)
        self.assertEqual(result['best_epoch'], 3)
        best = torch.load(fixture.best_path, weights_only=True)
        self.assertEqual(best['epoch'], 3)
        saved = {'cli': {'model': 'M1'}, 'model_contract': best['model_contract']}
        with patch('tools.eval.lg_valid_export.repository_path', side_effect=lambda p: fixture.root / p):
            self.assertEqual(select_best(fixture.run_dir, ENERGY_METRIC, saved)['row'].epoch, 3)
        fourth = next(c for c in checkpoints if c['epoch'] == 4)
        torch.save(fourth, fixture.latest_path)
        pd.read_csv(fixture.history_path).iloc[:4].to_csv(fixture.history_path, index=False)
        fixture.evaluate.side_effect = deepcopy(metrics[4:])
        trainer.train_and_eval(fixture.cfg)
        self.assertEqual(torch.load(fixture.best_path, weights_only=True)['epoch'], 3)
        self.assertEqual(torch.load(fixture.latest_path, weights_only=True)['epoch'], 35)
        self.assertEqual(selected_history_row(pd.read_csv(fixture.history_path), ENERGY_METRIC, protocol()).epoch, 3)

    def test_no_eligible_epoch_preserves_latest_and_never_exports_it_as_best(self):
        fixture = self.runner_fixture()
        fixture.evaluate.side_effect = [self.metric(.1, 11.)]*35
        result = trainer.train_and_eval(fixture.cfg)
        self.assertEqual(result['selection_status'], 'no_eligible_best')
        self.assertFalse(fixture.best_path.exists())
        self.assertTrue(fixture.latest_path.exists())
        latest = torch.load(fixture.latest_path, weights_only=True)
        saved = {'cli': {'model': 'M1'}, 'model_contract': latest['model_contract']}
        with patch('tools.eval.lg_valid_export.repository_path', side_effect=lambda p: fixture.root / p), \
                self.assertRaisesRegex(ValueError, 'no_eligible_best'):
            select_best(fixture.run_dir, ENERGY_METRIC, saved)

    def test_export_rejects_recomputed_guard_or_score_drift_before_writing_files(self):
        fixture = self.runner_fixture()
        fixture.evaluate.side_effect = [self.metric()]*35
        trainer.train_and_eval(fixture.cfg)
        checkpoint = torch.load(fixture.best_path, weights_only=True)
        identity = {'schema': 'unit-data'}
        checkpoint['data_contract'] = identity
        contract = checkpoint['model_contract']
        saved = {'cli': {'model': 'M1', 'batch_size': 32, 'norm': 'sumnorm', 'data_contract': identity},
                 'config': {'dataset': {'valid': {'data_dir': 'unit-data'}}}}
        ids = [str(i) for i in range(100)]
        loader = SimpleNamespace(sampler=list(range(100)), dataset=SimpleNamespace(material_ids=ids))
        fixture.builder.get_dataloader.return_value = loader
        picked = {'checkpoint': checkpoint, 'row': pd.Series({'balanced_score': .3})}
        for metrics, message in ((self.metric(failure=11.), 'task guards'),
                                 (self.metric(error=.2), 'energy score')):
            metrics.update(sample_df=pd.DataFrame({'material_id': ids, 'blind_r2_edos': [.5]*100}))
            out = fixture.root / 'export.csv'
            with patch.object(exporter, 'load_run', return_value=(saved, {}, contract)), \
                    patch.object(exporter, 'select_best', return_value=picked), \
                    patch.object(exporter, 'ConfigBuilder', return_value=fixture.builder), \
                    patch.object(exporter, 'evaluate_split', return_value=metrics), \
                    patch('utils.crystal_cache.crystal_data_contract', return_value=identity), \
                    self.assertRaisesRegex(ValueError, message):
                exporter.export(fixture.run_dir, out, fixture.root / 'summary.json')
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
