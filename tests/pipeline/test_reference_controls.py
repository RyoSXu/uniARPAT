"""LR2 E+S reference controls: seed replicates, structure-blind S0 and single-task supervision."""
from dataclasses import asdict, replace
import logging
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from model.model import basemodel
from model.transformer import Transformer
from tests.data.test_spectral_balance import protocol, row
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_structure_controls import predict
from tests.pipeline.test_spectral_balance import grid_sample
from tools.eval.local_global_compare import training_command
from tools.train import build_arg_parser, evaluate_local_global_split
from utils.bucket_batch import crystal_collate
from utils.builder import ConfigBuilder
from utils.checkpoint_selection import selected_history_row, selection_contract, validation_score
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import comparison_contract, initialize_comparison_model, training_record
from utils.local_structure_experiment import FROZEN
from utils.spectral_balance import ENERGY_METRIC
from utils.spectral_balance_experiment import (
    REFERENCE_CONTROL_SCHEMA, STRUCTURE_BLIND_SCHEMA, validate_balance_config)
from utils.zproj_alignment import state_hash


def params(route='LR2'):
    base = {**model_params('LR2'), 'scale_mode': 'per_atom'}
    return {**base, 'local_route': 'S0', 'local_depth': 0} if route == 'S0' else base


def batch_from(items):
    z, geometry = fixture()
    for index, item in enumerate(items):
        item[14] = None
        n = len(item[0])
        item[0] = z[index, :n]
        item[1] = CrystalGeometry(geometry.cell[index], geometry.frac_coords[index, :n])
    return crystal_collate(items)


class ReferenceControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def configuration(self, **changes):
        cfg = ExperimentConfig(**{**FROZEN, 'local_route': 'LR2', 'scale_mode': 'per_atom',
            'selection_metric': ENERGY_METRIC, 'spectral_shape_aux': True,
            'spectral_balance_config': '/tmp/unit-reference-protocol.json', 'tag': '_unit_refctl'})
        return replace(cfg, **changes)

    def policy(self, cfg):
        policy = protocol((cfg.edos_num, cfg.phdos_num))
        for task, delta in (('edos', cfg.delta_edos), ('phdos', cfg.delta_phdos)):
            policy['tasks'][task]['bin_width'] = delta
        return policy

    def test_validation_accepts_only_declared_controls(self):
        cfg = self.configuration()
        accepted = [replace(cfg, seed=43, data_order_seed=43),
                    replace(cfg, local_route='S0', local_depth=0, selection_metric='unguarded_blind_energy'),
                    replace(cfg, single_task='edos', selection_metric='edos_blind_energy'),
                    replace(cfg, single_task='phdos', selection_metric='phdos_blind_energy')]
        rejected = [replace(cfg, local_route='I3', seed=43),
                    replace(cfg, local_route='S0', local_depth=0),
                    replace(cfg, local_route='S0', local_depth=2, selection_metric='unguarded_blind_energy'),
                    replace(cfg, single_task='edos'),
                    replace(cfg, single_task='edos', selection_metric='phdos_blind_energy'),
                    replace(cfg, local_route='S0', local_depth=0, single_task='edos',
                            selection_metric='edos_blind_energy'),
                    replace(cfg, data_order_seed=-1),
                    replace(cfg, seed=43, local_layout='midpoint')]
        with patch('utils.spectral_balance_experiment.load_balance_protocol', return_value=self.policy(cfg)):
            for item in accepted:
                with self.subTest(accepted=item.local_route + str(item.single_task)):
                    validate_balance_config(item)
            for item in rejected:
                with self.subTest(rejected=(item.local_route, item.seed, item.single_task, item.selection_metric)):
                    with self.assertRaises(ValueError):
                        validate_balance_config(item)

    def contract(self, cfg, route='LR2'):
        with patch('utils.spectral_balance_experiment.validate_balance_config', return_value=protocol()):
            with patch('utils.spectral_balance_experiment.file_sha256', return_value='unit'):
                return comparison_contract(cfg, params(route))

    def test_contracts_isolate_controls_and_leave_default_records_unchanged(self):
        cfg = self.configuration()
        adopted = self.contract(cfg)
        self.assertNotIn('reference_control', adopted)
        self.assertFalse({'data_order_seed', 'single_task'} & set(training_record(cfg)))
        replicate = self.contract(replace(cfg, seed=43, data_order_seed=43))
        self.assertEqual(replicate['reference_control']['schema'], REFERENCE_CONTROL_SCHEMA)
        self.assertEqual(replicate['reference_control']['kind'], 'seed_replicate')
        self.assertEqual(replicate['training']['data_order_seed'], 43)
        self.assertEqual(replicate['backbone_reference'], adopted['backbone_reference'])
        blind = self.contract(replace(cfg, local_route='S0', local_depth=0,
                                      selection_metric='unguarded_blind_energy'), 'S0')
        self.assertEqual(blind['schema'], STRUCTURE_BLIND_SCHEMA)
        self.assertEqual(blind['checkpoint_selection']['schema'], 'reference-blind-energy-selection-v1')
        single = self.contract(replace(cfg, single_task='edos', selection_metric='edos_blind_energy'))
        self.assertEqual(single['reference_control']['supervised_tasks'], ['edos'])
        self.assertEqual(single['checkpoint_selection']['task_weights'], {'edos': 1., 'phdos': 0.})
        parsed = build_arg_parser().parse_args(training_command(
            replace(cfg, local_route='S0', local_depth=0, seed=44, data_order_seed=44,
                    selection_metric='unguarded_blind_energy'))[2:])
        self.assertEqual((parsed.local_route, parsed.local_depth, parsed.seed, parsed.data_order_seed),
                         ('S0', 0, 44, 44))

    def test_structure_blind_copies_same_seed_lr2_tensors_and_ignores_geometry(self):
        reference = Transformer(**params())
        initialize_comparison_model(reference, params(), 42)
        model = Transformer(**params('S0'))
        rng = torch.get_rng_state().clone()
        info = initialize_comparison_model(model, params('S0'), 42)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(model.encoder.local_layers), 0)
        shared = {k: v for k, v in reference.state_dict().items() if not k.startswith('encoder.local_layers.')}
        self.assertEqual(set(model.state_dict()), set(shared))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, shared[key], rtol=0, atol=0)
        self.assertLess(info['parameter_count'], info['reference_parameter_count'])
        model.eval()
        z, geometry = fixture()
        expected = predict(model, z, geometry)
        moved = CrystalGeometry(geometry.cell * 1.3, (geometry.frac_coords + .17) % 1.)
        for key, value in predict(model, z, moved).items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_seed_replicate_changes_initial_tensors_deterministically(self):
        hashes = {}
        for seed in (42, 43, 43):
            model = Transformer(**params())
            hashes.setdefault(seed, set()).add(initialize_comparison_model(model, params(), seed)['state_sha256'])
        self.assertEqual(len(hashes[43]), 1)
        self.assertNotEqual(hashes[42], hashes[43])

    def training_model(self, task=None, route='LR2'):
        model = basemodel(logging.getLogger('reference-control-test'), loss_form=ENERGY_METRIC,
            spectral_balance=protocol((8, 4)), spectral_shape_aux=True, metrics_list=[],
            sub_model={'transformer': params(route)}, single_task=task,
            optimizer={'transformer': {'type': 'AdamW', 'params': {'lr': 0.}}}, lr_scheduler={})
        initialize_comparison_model(model.model['transformer'], params(route), 42)
        model.to(torch.device('cpu'))
        model.model['transformer'].eval()
        return model

    def test_single_task_loss_supervises_only_its_task(self):
        for task, other in (('edos', 'phdos'), ('phdos', 'edos')):
            with self.subTest(task=task):
                model = self.training_model(task)
                batch = batch_from([grid_sample(3, 42), grid_sample(2, 43)])
                losses = model.train_one_step(batch, 0)
                suffix = 'e' if task == 'edos' else 'p'
                self.assertAlmostEqual(losses['loss'], losses[f'loss_energy_{suffix}'] + losses[f'loss_shape_{suffix}'],
                                       places=6)
                measured = evaluate_local_global_split(model, [batch])
                self.assertAlmostEqual(losses['loss'] - losses[f'loss_shape_{suffix}'],
                                       measured[f'blind_energy_{task}_mean'], places=6)
                grads = [p.grad for name, p in model.model['transformer'].named_parameters()
                         if name.startswith(f'{other}_')]
                self.assertTrue(grads)
                self.assertTrue(all(g is None or not torch.count_nonzero(g) for g in grads))
        with self.assertRaises(ValueError):
            basemodel(logging.getLogger('reference-control-test'), loss_form='sumnorm_klw', metrics_list=[],
                      sub_model={'transformer': params()}, single_task='edos',
                      optimizer={'transformer': {'type': 'AdamW', 'params': {'lr': 0.}}}, lr_scheduler={})

    def test_reference_selection_ignores_guards_but_verifies_records(self):
        policy = protocol()
        for metric, weights in (('unguarded_blind_energy', (.5, .5)), ('edos_blind_energy', (1., 0.)),
                                ('phdos_blind_energy', (0., 1.))):
            with self.subTest(metric=metric):
                rows = []
                for epoch, (e, p, failures) in enumerate(((.3, .3, 10.), (.1, .2, 30.), (.2, .1, 10.)), 1):
                    record = row(epoch, e, p, blind_fail_rate_edos=failures,
                                 checkpoint_eligible=failures <= 10.)
                    score = weights[0] * e + weights[1] * p
                    record.update(balanced_score=score, **{metric: score})
                    rows.append(record)
                history = pd.DataFrame(rows)
                chosen = selected_history_row(history, metric, policy)
                expected = min(rows, key=lambda r: (r['balanced_score'], r['epoch']))['epoch']
                self.assertEqual(int(chosen.epoch), expected)
                self.assertAlmostEqual(validation_score(rows[1], metric), rows[1]['balanced_score'])
                self.assertIn('not applied', selection_contract(metric, policy)['checkpoint_selection']['task_guards'])
                broken = history.assign(checkpoint_eligible=True)
                with self.assertRaises(ValueError):
                    selected_history_row(broken, metric, policy)
        # The adopted guarded rule still refuses the low-error epoch that fails a guard.
        guarded = pd.DataFrame([row(1, .3, .3), row(2, .1, .2, blind_fail_rate_edos=30., checkpoint_eligible=False)])
        self.assertEqual(int(selected_history_row(guarded, ENERGY_METRIC, policy).epoch), 1)

    def test_training_data_order_seed_defaults_to_historical_sampler(self):
        class Data(torch.utils.data.Dataset):
            def __len__(self):
                return 50

            def __getitem__(self, index):
                return index

        builder = ConfigBuilder.__new__(ConfigBuilder)
        orders = {}
        for seed in (None, 0, 43):
            sampler = builder.get_sampler(Data(), 'train', seed=seed)
            sampler.set_epoch(0)
            orders[seed] = list(sampler)
        self.assertEqual(orders[None], orders[0])
        self.assertNotEqual(orders[None], orders[43])
        with self.assertRaises(ValueError):
            builder.get_sampler(Data(), 'train', seed=-1)
        self.assertEqual(asdict(ExperimentConfig())['data_order_seed'], None)


if __name__ == '__main__':
    unittest.main()
