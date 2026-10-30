"""LR2 on E+S preserves initialization, objective and checkpoint boundaries."""
from copy import deepcopy
from dataclasses import replace
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from model.model import basemodel
from model.transformer import Transformer
from tests.data.test_spectral_balance import protocol
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_structure_controls import predict
from tests.pipeline.test_spectral_balance import grid_sample
from tools.checks.check_long_budget import captured_config
from tools.eval.lg_valid_export import load_run
from tools.eval.local_global_compare import training_command
from tools.train import build_arg_parser, evaluate_local_global_split
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.bucket_batch import crystal_collate
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import capture_training_rng, comparison_contract, initialize_comparison_model
from utils.local_residual_experiment import residual_ablation_contract, validate_residual_config, validate_residual_model
from utils.local_structure_experiment import FROZEN
from utils.spectral_balance import ENERGY_METRIC
from utils.spectral_balance_experiment import BALANCE_RESIDUAL_SCHEMA, validate_balance_config
from utils.zproj_alignment import state_hash


class SpectralResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def configuration(self, route='LR2'):
        return ExperimentConfig(**{**FROZEN, 'local_route': route, 'scale_mode': 'per_atom',
            'selection_metric': ENERGY_METRIC, 'spectral_shape_aux': True,
            'spectral_balance_config': '/tmp/unit-residual-protocol.json', 'tag': '_unit_residual_es'})

    def contract(self, route='LR2'):
        cfg = self.configuration(route)
        params = {**model_params(route), 'scale_mode': 'per_atom'}
        with patch('utils.spectral_balance_experiment.validate_balance_config', return_value=protocol()):
            with patch('utils.spectral_balance_experiment.file_sha256', return_value='unit'):
                return comparison_contract(cfg, params)

    def test_equal_initial_tensors_and_rng_with_only_four_additions_disabled(self):
        models, infos = {}, {}
        for route in ('I3', 'LR2'):
            params = {**model_params(route), 'scale_mode': 'per_atom'}
            models[route] = Transformer(**params).eval()
            rng = torch.get_rng_state().clone()
            infos[route] = initialize_comparison_model(models[route], params, 42)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(infos['I3']['state_sha256'], infos['LR2']['state_sha256'])
        self.assertEqual(infos['I3']['a64_initialization'], infos['LR2']['a64_initialization'])
        self.assertEqual(sum(p.numel() for p in models['I3'].parameters()),
                         sum(p.numel() for p in models['LR2'].parameters()))
        candidate = models['LR2']
        validate_residual_model(candidate, {**model_params('LR2'), 'scale_mode': 'per_atom'}, require_identity=True)
        blocks = [block for layer in candidate.encoder.local_layers for block in layer.message.blocks]
        self.assertEqual(len(blocks), 4)
        self.assertTrue(all(block.residual_add is False for block in blocks))
        z, geometry = fixture()
        expected = predict(models['I3'], z, geometry)
        altered = predict(candidate, z, geometry)
        self.assertTrue(any(not torch.equal(expected[key], altered[key]) for key in expected))
        for block in blocks:
            block.residual_add = True
        restored = predict(candidate, z, geometry)
        for key in expected:
            torch.testing.assert_close(restored[key], expected[key], rtol=0, atol=0)

    def test_config_rejects_unapproved_combinations_and_preserves_legacy_lr2(self):
        cfg = self.configuration()
        policy = protocol((cfg.edos_num, cfg.phdos_num))
        for task, delta in (('edos', cfg.delta_edos), ('phdos', cfg.delta_phdos)):
            policy['tasks'][task]['bin_width'] = delta
        with patch('utils.spectral_balance_experiment.load_balance_protocol', return_value=policy):
            validate_balance_config(cfg)
            for changes in ({'spectral_shape_aux': False}, {'split_decoder': True}, {'phonon_head': True},
                            {'phonon_correction': True}, {'local_route': 'LR1'}, {'seed': 43},
                            {'selection_metric': 'balanced_blind_r2'}, {'scale_mode': 'eta'}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_balance_config(replace(cfg, **changes))
        legacy = ExperimentConfig(**{**FROZEN, 'local_route': 'LR2', 'selection_metric': 'balanced_blind_r2',
                                     'tag': '_unit_legacy_residual'})
        validate_residual_config(legacy)
        self.assertEqual(comparison_contract(legacy, model_params('LR2'))['schema'], 'local-residual-model-v1')

    def test_production_assembly_and_contract_change_only_route_and_protocol_identity(self):
        cfg = self.configuration()
        with patch('utils.spectral_balance_experiment.validate_balance_config', return_value=protocol()):
            reference = captured_config(replace(cfg, local_route='I3'))
            candidate = captured_config(cfg)
        expected = deepcopy(reference)
        expected['model']['params']['sub_model']['transformer']['local_route'] = 'LR2'
        self.assertEqual(candidate, expected)
        control, altered = self.contract('I3'), self.contract()
        self.assertEqual(altered['schema'], BALANCE_RESIDUAL_SCHEMA)
        self.assertEqual(altered['ablation'], residual_ablation_contract())
        for key in ('shape_supervision', 'spectral_balance', 'checkpoint_selection', 'backbone_reference'):
            self.assertEqual(control[key], altered[key])
        parsed = build_arg_parser().parse_args(training_command(cfg)[2:])
        self.assertEqual(parsed.local_route, 'LR2')
        self.assertTrue(parsed.spectral_shape_aux and parsed.skip_test_eval)

    def training_model(self):
        params = {**model_params('LR2'), 'scale_mode': 'per_atom'}
        model = basemodel(logging.getLogger('residual-es-test'), loss_form=ENERGY_METRIC,
            spectral_balance=protocol((8, 4)), spectral_shape_aux=True, metrics_list=[],
            sub_model={'transformer': params},
            optimizer={'transformer': {'type': 'AdamW', 'params': {'lr': 0.}}}, lr_scheduler={})
        initialize_comparison_model(model.model['transformer'], params, 42)
        model.to(torch.device('cpu'))
        model.model['transformer'].eval()
        return model

    def test_joint_gradient_objective_and_selected_metric_agree_without_valence(self):
        model = self.training_model()
        items = [grid_sample(3, 42), grid_sample(2, 43)]
        z, geometry = fixture()
        for index, item in enumerate(items):
            item[14] = None
            n = len(item[0])
            item[0] = z[index, :n]
            item[1] = CrystalGeometry(geometry.cell[index], geometry.frac_coords[index, :n])
        batch = crystal_collate(items)
        losses = model.train_one_step(batch, 0)
        measured = evaluate_local_global_split(model, [batch])
        self.assertAlmostEqual(losses['loss'], .5 * sum(losses[key] for key in
            ('loss_energy_e', 'loss_energy_p', 'loss_shape_e', 'loss_shape_p')), places=6)
        self.assertAlmostEqual(losses['loss']-.5*(losses['loss_shape_e']+losses['loss_shape_p']),
                               measured[ENERGY_METRIC], places=6)
        for layer in model.model['transformer'].encoder.local_layers:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()))

    def test_resume_preserves_optimizer_and_rejects_same_shape_control_and_old_lr2(self):
        model = self.training_model()
        model.train_one_step(crystal_collate([grid_sample(3, 42)]), 0)
        transformer, optimizer = model.model['transformer'], model.optimizer['transformer']
        contract = self.contract()
        checkpoint = deepcopy(build_ablation_checkpoint(1, 'M1', 42, False, transformer, optimizer, .3,
            model_contract=contract, training_rng=capture_training_rng()))
        fresh = self.training_model()
        result = restore_ablation_checkpoint(checkpoint, fresh.model['transformer'],
            fresh.optimizer['transformer'], False, model_contract=contract)
        self.assertTrue(result['optimizer_restored'])
        self.assertEqual(state_hash(transformer.state_dict()), state_hash(fresh.model['transformer'].state_dict()))
        for index, slots in optimizer.state_dict()['state'].items():
            for key, value in slots.items():
                torch.testing.assert_close(fresh.optimizer['transformer'].state_dict()['state'][index][key],
                                           value, atol=0, rtol=0)
        for other in (self.contract('I3'), {**contract, 'schema': 'local-residual-model-v1'}):
            with self.assertRaisesRegex(ValueError, 'different model contract'):
                restore_ablation_checkpoint(checkpoint, transformer, optimizer, False, model_contract=other)

    def test_export_rejects_route_schema_or_formula_drift(self):
        contract = self.contract()
        params = {'loss_form': ENERGY_METRIC, 'spectral_balance': protocol(), 'spectral_shape_aux': True,
                  'sub_model': {'transformer': contract['transformer']}}
        saved = {'model_contract': contract, 'config': {'model': {'params': params}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config_used.yaml'
            path.write_text(yaml.safe_dump(saved))
            load_run(Path(directory))
            for mutation in ('schema', 'route', 'formula', 'auxiliary'):
                changed = deepcopy(saved)
                target = changed['model_contract']
                if mutation == 'schema':
                    target['schema'] = 'joint-spectral-balance-shape-model-v1'
                elif mutation == 'route':
                    target['training']['local_route'] = 'I3'
                elif mutation == 'formula':
                    target['ablation']['internal_residual_add'] = True
                else:
                    changed['config']['model']['params']['spectral_shape_aux'] = False
                path.write_text(yaml.safe_dump(changed))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_run(Path(directory))


if __name__ == '__main__':
    unittest.main()
