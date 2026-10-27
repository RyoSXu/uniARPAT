"""The single shape auxiliary changes supervision, never blind reconstruction or selection."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model.losses import nonzero_sumnorm_klw_loss, sumnorm_klw_loss
from tests.pipeline import test_spectral_balance as fixtures
from tools.checks.check_long_budget import captured_config
from tools.eval.lg_valid_export import load_run
from tools.eval.local_global_compare import training_command
from tools.train import build_arg_parser, evaluate_local_global_split
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.bucket_batch import crystal_collate
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import capture_training_rng, comparison_contract, training_record, validate_local_global_config
from utils.local_structure_experiment import FROZEN
from utils.spectral_balance import ENERGY_METRIC
from utils.spectral_balance_experiment import shape_supervision_contract


class SpectralShapeAuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_positive_spectra_reuse_exact_legacy_shape_terms_and_unit_invariance(self):
        logits = torch.tensor([[.1, -.2, .3], [-.4, .6, .1]], dtype=torch.float64)
        target = torch.tensor([[0., 2., 3.], [4., 0., 1.]], dtype=torch.float64)
        expected = sumnorm_klw_loss(logits, target/target.sum(-1, keepdim=True), None, False, 1., 1., .02)
        actual = nonzero_sumnorm_klw_loss(logits, target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(nonzero_sumnorm_klw_loss(logits, target*1000), actual)

    def test_zero_spectrum_has_no_shape_preference_or_gradient_and_keeps_batch_weight(self):
        logits = torch.tensor([[4., 0., -4.], [.2, -.1, .4]], requires_grad=True)
        target = torch.tensor([[0., 0., 0.], [0., 1., 2.]])
        values = nonzero_sumnorm_klw_loss(logits, target)
        self.assertEqual(float(values[0]), 0.)
        values.mean().backward()
        torch.testing.assert_close(logits.grad[0], torch.zeros(3), rtol=0, atol=0)
        self.assertGreater(float(logits.grad[1].abs().sum()), 0.)
        self.assertAlmostEqual(float(values.mean()), float(values[1])/2)

    def test_shape_gradcheck_and_invalid_physical_targets(self):
        logits = torch.tensor([[.2, -.1, .4]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[.5, 1., 2.]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda x: nonzero_sumnorm_klw_loss(x, target).mean(), (logits,)))
        for bad in (target * -1, target * float('nan')):
            with self.assertRaises(ValueError):
                nonzero_sumnorm_klw_loss(logits, bad)

    def models(self):
        helper = fixtures.BalancedPipelineTests()
        baseline = helper.training_model()
        candidate = helper.training_model(shape_aux=True)
        candidate.model['transformer'].load_state_dict(deepcopy(baseline.model['transformer'].state_dict()))
        return baseline, candidate

    def test_candidate_adds_only_fixed_shape_loss_and_preserves_blind_outputs_and_metric(self):
        baseline, candidate = self.models()
        items = [fixtures.grid_sample(3, 42), fixtures.grid_sample(2, 43)]
        for item in items:
            item[14] = None
        batch = crystal_collate(items)
        left, right = baseline.train_one_step(batch, 0), candidate.train_one_step(batch, 0)
        self.assertAlmostEqual(right['loss_energy_e'], left['loss_edos'], places=6)
        self.assertAlmostEqual(right['loss_energy_p'], left['loss_phdos'], places=6)
        self.assertAlmostEqual(right['loss']-left['loss'], .5*(right['loss_shape_e']+right['loss_shape_p']), places=6)
        self.assertGreater(right['loss_shape_e'], 0.)
        a = evaluate_local_global_split(baseline, [batch])
        b = evaluate_local_global_split(candidate, [batch])
        self.assertEqual(a[ENERGY_METRIC], b[ENERGY_METRIC])
        self.assertEqual(a['blind_r2_edos_median'], b['blind_r2_edos_median'])
        # Shape supervision cannot directly alter the mass-head gradient.
        for p, q in zip(baseline.model['transformer'].mass_head.parameters(), candidate.model['transformer'].mass_head.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p.grad, q.grad) for p, q in zip(
            baseline.model['transformer'].edos_out_head.parameters(), candidate.model['transformer'].edos_out_head.parameters())))

    def test_microbatches_preserve_sample_weight_and_one_optimizer_update(self):
        _, full = self.models()
        _, split = self.models()
        split.model['transformer'].load_state_dict(deepcopy(full.model['transformer'].state_dict()))
        batch = crystal_collate([fixtures.grid_sample(3, 42), fixtures.grid_sample(2, 43), fixtures.grid_sample(1, 44)])
        left = full.train_one_step(batch, 0, pair_budget=27)
        with patch.object(split.optimizer['transformer'], 'step', wraps=split.optimizer['transformer'].step) as step:
            right = split.train_one_step(batch, 0, pair_budget=9)
        self.assertEqual(step.call_count, 1)
        for key in ('loss', 'loss_energy_e', 'loss_energy_p', 'loss_shape_e', 'loss_shape_p'):
            self.assertAlmostEqual(left[key], right[key], places=5)
        for (name, p), (_, q) in zip(full.model['transformer'].named_parameters(), split.model['transformer'].named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=name)

    def test_assembly_opt_in_contract_isolation_and_legacy_keys(self):
        cfg = ExperimentConfig(**{**FROZEN, 'local_route': 'I3', 'scale_mode': 'per_atom',
            'selection_metric': ENERGY_METRIC, 'spectral_balance_config': '/tmp/unit-protocol.json',
            'tag': '_unit_shape_aux'})
        self.assertNotIn('spectral_shape_aux', training_record(cfg))
        with self.assertRaises(ValueError):
            validate_local_global_config(replace(cfg, spectral_balance_config='', spectral_shape_aux=True))
        for value in (1, 'true', None):
            with self.assertRaises(ValueError):
                shape_supervision_contract(value)
        with patch('utils.spectral_balance_experiment.validate_balance_config', return_value=fixtures.protocol()):
            baseline = captured_config(cfg)
            candidate = captured_config(replace(cfg, spectral_shape_aux=True))
            params = baseline['model']['params']['sub_model']['transformer']
            with patch('utils.spectral_balance_experiment.file_sha256', return_value='unit'):
                old = comparison_contract(cfg, params)
                new = comparison_contract(replace(cfg, spectral_shape_aux=True), params)
        expected = deepcopy(baseline)
        expected['model']['params']['spectral_shape_aux'] = True
        self.assertEqual(candidate, expected)
        self.assertNotIn('shape_supervision', old)
        self.assertEqual(old['schema'], 'joint-spectral-balance-model-v1')
        self.assertNotEqual(old, new)
        self.assertEqual(old['checkpoint_selection'], new['checkpoint_selection'])
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters())
        ck = build_ablation_checkpoint(1, 'M1', 42, False, model, opt, .3,
            model_contract=old, training_rng=capture_training_rng())
        with self.assertRaisesRegex(ValueError, 'different model contract'):
            restore_ablation_checkpoint(ck, model, opt, False, model_contract=new)
        parsed = build_arg_parser().parse_args(training_command(replace(cfg, spectral_shape_aux=True))[2:])
        self.assertTrue(parsed.spectral_shape_aux)

    def test_export_rejects_auxiliary_flag_or_formula_drift(self):
        policy = shape_supervision_contract(True)
        saved = {'model_contract': {'schema': 'joint-spectral-balance-shape-model-v1',
            'transformer': {'scale_mode': 'per_atom'}, 'training': {'spectral_shape_aux': True},
            'spectral_balance': fixtures.protocol(), **policy},
            'config': {'model': {'params': {'loss_form': ENERGY_METRIC, 'spectral_balance': fixtures.protocol(),
                'spectral_shape_aux': True, 'sub_model': {'transformer': {'scale_mode': 'per_atom'}}}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config_used.yaml'
            import yaml
            path.write_text(yaml.safe_dump(saved))
            load_run(Path(directory))
            for altered in ('flag', 'policy'):
                changed = deepcopy(saved)
                if altered == 'flag':
                    changed['config']['model']['params']['spectral_shape_aux'] = False
                else:
                    changed['model_contract']['shape_supervision']['weight'] = 2.
                path.write_text(yaml.safe_dump(changed))
                with self.assertRaisesRegex(ValueError, 'shape auxiliary'):
                    load_run(Path(directory))


if __name__ == '__main__':
    unittest.main()
