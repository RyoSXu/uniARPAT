"""LR2 E+S generalization arms: training-only strain augmentation (G) and local residual drop (D)."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from model.transformer import Transformer
from tests.data.test_spectral_balance import protocol
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_local_structure_controls import predict
from tests.pipeline.test_reference_controls import ReferenceControlTests, params
from tools.eval.local_global_compare import training_command
from tools.train import build_arg_parser
from utils.crystal import CrystalGeometry
from utils.local_global_experiment import comparison_contract, initialize_comparison_model, training_record
from utils.spectral_balance_experiment import validate_balance_config
from utils.structure_regularization import (
    DISPLACEMENT_CLIP_ANGSTROM, LOCAL_DROP_PROB, STRAIN_ISOTROPIC_HALF_WIDTH, STRAIN_SHEAR_SD,
    drop_local_residual, strain_perturb)


def drop_params():
    return {**params(), 'local_drop_prob': LOCAL_DROP_PROB}


class StrainPerturbTests(unittest.TestCase):
    def test_strain_is_small_reproducible_and_wraps(self):
        cell = torch.tensor([[4., 0., 0.], [1., 5., 0.], [0., .5, 6.]])
        frac = torch.rand(7, 3, generator=torch.Generator().manual_seed(0))
        first = strain_perturb(cell, frac, torch.Generator().manual_seed(3))
        again = strain_perturb(cell, frac, torch.Generator().manual_seed(3))
        other = strain_perturb(cell, frac, torch.Generator().manual_seed(4))
        for left, right in zip(first, again):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertFalse(torch.equal(first[0], other[0]))
        self.assertEqual((first[0].dtype, first[1].dtype), (cell.dtype, frac.dtype))
        self.assertTrue(((first[1] >= 0) & (first[1] < 1)).all())
        bound = (1 + STRAIN_ISOTROPIC_HALF_WIDTH + 6 * STRAIN_SHEAR_SD) ** 3
        for seed in range(50):
            strained, moved = strain_perturb(cell, frac, torch.Generator().manual_seed(seed))
            ratio = float(torch.det(strained.double()) / torch.det(cell.double()))
            self.assertTrue(1 / bound < ratio < bound)
            # Cartesian displacement relative to the strained original positions, modulo the lattice.
            delta = (moved.double() - frac.double()) % 1.0
            delta = torch.where(delta > .5, delta - 1, delta) @ strained.double()
            self.assertLessEqual(float(delta.abs().max()), DISPLACEMENT_CLIP_ANGSTROM + 1e-5)

    def test_dataset_perturbs_training_split_only(self):
        from utils.dataset import Dos_Dataset
        from utils.paths import input_path
        data_dir = 'data/crystal_q1_v1'
        try:
            input_path(data_dir)
            train = Dos_Dataset(data_dir, 'train', dos_sumnorm=True, strain_augment=True)
            valid = Dos_Dataset(data_dir, 'valid', dos_sumnorm=True, strain_augment=True)
            plain = Dos_Dataset(data_dir, 'train', dos_sumnorm=True)
        except (FileNotFoundError, OSError) as error:
            self.skipTest(f'crystal data unavailable: {error}')
        self.assertTrue(train.strain_augment)
        self.assertFalse(valid.strain_augment)
        torch.manual_seed(0)
        perturbed, reference = train[5][1], plain[5][1]
        self.assertFalse(torch.equal(perturbed.cell, reference.cell))
        self.assertEqual(perturbed.frac_coords.shape, reference.frac_coords.shape)
        self.assertTrue(torch.equal(valid[5][1].cell, Dos_Dataset(data_dir, 'valid', dos_sumnorm=True)[5][1].cell))
        with self.assertRaises(ValueError):
            Dos_Dataset(data_dir, 'train', dos_sumnorm=True, strain_augment=True, augment=True)


class LocalDropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_drop_path_rule(self):
        initial = torch.zeros(4000, 2, 3)
        updated = torch.ones(4000, 2, 3)
        out = drop_local_residual(initial, updated, LOCAL_DROP_PROB, torch.Generator().manual_seed(0))
        values = out[:, 0, 0]
        kept = values[values != 0]
        torch.testing.assert_close(kept, torch.full_like(kept, 1 / (1 - LOCAL_DROP_PROB)))
        self.assertAlmostEqual(float((values == 0).float().mean()), LOCAL_DROP_PROB, delta=.03)
        self.assertTrue((out == out[:, :1, :1]).all())  # one draw per sample
        for bad in (0., 1., -.1):
            with self.assertRaises(ValueError):
                drop_local_residual(initial, updated, bad)

    def test_eval_path_matches_lr2_and_init_is_unchanged(self):
        reference, model = Transformer(**params()), Transformer(**drop_params())
        first = initialize_comparison_model(reference, params(), 42)
        second = initialize_comparison_model(model, drop_params(), 42)
        self.assertEqual(first['state_sha256'], second['state_sha256'])
        reference.eval(), model.eval()
        z, geometry = fixture()
        expected = predict(reference, z, geometry)
        for key, value in predict(model, z, geometry).items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_dropped_samples_give_local_parameters_no_gradient(self):
        model = Transformer(**drop_params())
        initialize_comparison_model(model, drop_params(), 42)
        model.train()
        z, geometry = fixture()

        def drop_all(initial, updated, prob, generator=None):
            return initial + 0 * updated

        with patch('utils.structure_regularization.drop_local_residual', drop_all):
            loss = sum(value.float().sum() for value in predict(model, z, geometry).values()
                       if torch.is_tensor(value) and value.requires_grad)
            loss.backward()
        local = [p.grad for name, p in model.named_parameters() if name.startswith('encoder.local_layers.')]
        others = [p.grad for name, p in model.named_parameters() if name.startswith('encoder.global_layers.')]
        self.assertTrue(local and others)
        self.assertTrue(all(g is None or not torch.count_nonzero(g) for g in local))
        self.assertTrue(any(g is not None and torch.count_nonzero(g) for g in others))
        with self.assertRaises(ValueError):
            Transformer(**{**params(), 'local_route': 'I3', 'local_drop_prob': .25})


class RegularizationContractTests(ReferenceControlTests):
    def test_validation_accepts_only_the_fixed_arms(self):
        cfg = self.configuration()
        accepted = [replace(cfg, strain_augment=True), replace(cfg, local_drop_prob=LOCAL_DROP_PROB),
                    replace(cfg, strain_augment=True, seed=43, data_order_seed=43)]
        rejected = [replace(cfg, strain_augment=True, local_drop_prob=LOCAL_DROP_PROB),
                    replace(cfg, local_drop_prob=.5),
                    replace(cfg, strain_augment=True, single_task='edos', selection_metric='edos_blind_energy'),
                    replace(cfg, local_route='S0', local_depth=0, selection_metric='unguarded_blind_energy',
                            strain_augment=True),
                    replace(cfg, strain_augment=True, local_layout='midpoint')]
        with patch('utils.spectral_balance_experiment.load_balance_protocol', return_value=self.policy(cfg)):
            for item in accepted:
                validate_balance_config(item)
            for item in rejected:
                with self.subTest(rejected=(item.local_route, item.strain_augment, item.local_drop_prob)):
                    with self.assertRaises(ValueError):
                        validate_balance_config(item)

    def regularized_contract(self, cfg, transformer_params):
        with patch('utils.spectral_balance_experiment.validate_balance_config', return_value=protocol()):
            with patch('utils.spectral_balance_experiment.file_sha256', return_value='unit'):
                return comparison_contract(cfg, transformer_params)

    def test_contracts_record_the_arm_and_leave_defaults_unchanged(self):
        cfg = self.configuration()
        adopted = self.contract(cfg)
        self.assertNotIn('regularization', adopted)
        self.assertFalse({'strain_augment', 'local_drop_prob'} & set(training_record(cfg)))
        strain = self.regularized_contract(replace(cfg, strain_augment=True), params())
        self.assertEqual(strain['regularization']['kind'], 'strain_augment')
        self.assertEqual(strain['schema'], adopted['schema'])
        self.assertEqual(strain['backbone_reference'], adopted['backbone_reference'])
        self.assertNotEqual(strain['training'], adopted['training'])
        drop = self.regularized_contract(replace(cfg, local_drop_prob=LOCAL_DROP_PROB), drop_params())
        self.assertEqual((drop['regularization']['kind'], drop['regularization']['prob']), ('local_drop', .25))
        self.assertEqual(drop['backbone_reference'], adopted['backbone_reference'])
        with self.assertRaises(ValueError):
            self.regularized_contract(replace(cfg, local_drop_prob=LOCAL_DROP_PROB), params())
        parsed = build_arg_parser().parse_args(training_command(replace(cfg, local_drop_prob=LOCAL_DROP_PROB))[2:])
        self.assertEqual(parsed.local_drop_prob, LOCAL_DROP_PROB)
        parsed = build_arg_parser().parse_args(training_command(replace(cfg, strain_augment=True))[2:])
        self.assertTrue(parsed.strain_augment)

    # The inherited reference-control tests run once in their own module.
    test_validation_accepts_only_declared_controls = None
    test_contracts_isolate_controls_and_leave_default_records_unchanged = None
    test_structure_blind_copies_same_seed_lr2_tensors_and_ignores_geometry = None
    test_seed_replicate_changes_initial_tensors_deterministically = None
    test_single_task_loss_supervises_only_its_task = None
    test_reference_selection_ignores_guards_but_verifies_records = None
    test_training_data_order_seed_defaults_to_historical_sampler = None


if __name__ == '__main__':
    unittest.main()
