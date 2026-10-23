"""Scientific invariants of train-energy normalization and guarded selection."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import torch

from utils.checkpoint_selection import (
    initial_best_score, is_better_score, selected_history_row, selection_contract, validation_score,
)
from utils.spectral_balance import (
    BALANCE_SCHEMA, ENERGY_METRIC, balanced_checkpoint_eligible, fit_task_normalizer,
    normalized_spectral_error, reconstruct_balanced_spectra, selected_energy_history_row,
)


def protocol(bins=(2, 2)):
    return {"schema": BALANCE_SCHEMA, "fit_split": "train", "task_weights": {"edos": .5, "phdos": .5},
        "tasks": {task: {"energy": 5., "mass_per_atom_reference": 2., "bin_width": 1., "bins": n}
                  for task, n in zip(("edos", "phdos"), bins)},
        "reference": {"label": "A64", "split": "valid", "samples": 100,
                      "failures_edos": 10, "failures_phdos": 5,
                      "r2_edos_median": .5, "r2_phdos_median": .75},
        "failure_increase_limit_pp": 0., "r2_drop_limit": .02}


def row(epoch=1, e=.2, p=.4, **changes):
    return {"epoch": epoch, "blind_energy_edos_mean": e, "blind_energy_phdos_mean": p,
            ENERGY_METRIC: .5*(e+p), "balanced_score": .5*(e+p), "selection_samples": 100,
            "blind_fail_rate_edos": 10., "blind_fail_rate_phdos": 5.,
            "blind_r2_edos_median": .5, "blind_r2_phdos_median": .75,
            "checkpoint_eligible": True, **changes}


class SpectralBalanceTests(unittest.TestCase):
    def test_zero_predictor_has_unit_mean_training_error_in_each_task(self):
        for width, values in ((.1, [[1., 3.], [5., 2.], [0., 0.]]),
                              (20., [[0., 4.], [1., 2.], [0., 0.]])):
            target = torch.tensor(values, dtype=torch.float64)
            atoms = torch.tensor([2., 4., 1.], dtype=torch.float64)
            normalizer = fit_task_normalizer(target.numpy(), atoms.numpy(), width)
            error = normalized_spectral_error(torch.zeros_like(target), target, atoms, normalizer)
            self.assertAlmostEqual(float(error.mean()), 1., places=13)
            self.assertEqual(float(error[-1]), 0.)

    def test_physical_units_and_bin_width_change_cancel(self):
        target = torch.tensor([[1., 3.], [2., 5.]], dtype=torch.float64)
        prediction = target * .8
        atoms = np.array([1., 3.])
        norm = fit_task_normalizer(target.numpy(), atoms, .1)
        error = normalized_spectral_error(prediction, target, atoms, norm)
        # x'=1000*x and DOS'=DOS/1000 preserve integrated states/modes.
        changed = fit_task_normalizer(target.numpy()/1000, atoms, 100.)
        actual = normalized_spectral_error(prediction/1000, target/1000, atoms, changed)
        torch.testing.assert_close(actual, error, rtol=1e-13, atol=1e-13)

    def test_adding_correct_zero_background_preserves_error(self):
        target = torch.tensor([[1., 3.], [2., 4.]], dtype=torch.float64)
        prediction = torch.tensor([[2., 2.], [1., 3.]], dtype=torch.float64)
        norm = fit_task_normalizer(target.numpy(), np.ones(2), 1.)
        expected = normalized_spectral_error(prediction, target, [1, 1], norm)
        zeros = torch.zeros_like(target)
        padded_t = torch.cat((target, zeros), -1)
        padded_p = torch.cat((prediction, zeros), -1)
        padded_norm = fit_task_normalizer(padded_t.numpy(), np.ones(2), 1.)
        self.assertEqual(norm['energy'], padded_norm['energy'])
        torch.testing.assert_close(normalized_spectral_error(padded_p, padded_t, [1, 1], padded_norm), expected)
        padded_p[:, -1] = 1.
        self.assertTrue((normalized_spectral_error(padded_p, padded_t, [1, 1], padded_norm) > expected).all())

    def test_representation_as_supercell_preserves_per_atom_error(self):
        target = torch.tensor([[1., 3.]], dtype=torch.float64)
        prediction = target * .7
        norm = fit_task_normalizer(target.numpy(), [2], 1.)
        original = normalized_spectral_error(prediction, target, [2], norm)
        torch.testing.assert_close(normalized_spectral_error(prediction*4, target*4, [8], norm), original)

    def test_zero_target_is_finite_and_scale_gradient_can_reduce_it(self):
        logits = torch.zeros((1, 2), dtype=torch.float64)
        mass = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
        output = reconstruct_balanced_spectra({'edos': logits, 'phdos': logits, 'mass_logits': mass}, [3], protocol())
        loss = normalized_spectral_error(output['blind_phdos'], torch.zeros_like(logits), [3], protocol()['tasks']['phdos']).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(mass.grad[0, 1]), 0.)

    def test_mass_is_not_bounded_by_atoms_or_valence_and_empty_rows_are_zero(self):
        shapes = torch.zeros((2, 2), dtype=torch.float64)
        out = reconstruct_balanced_spectra({'edos': shapes, 'phdos': shapes,
            'mass_logits': torch.full((2, 2), 10., dtype=torch.float64)}, [2, 0], protocol())
        self.assertGreater(float(out['blind_edos'][0].sum()), 20.)
        self.assertGreater(float(out['blind_phdos'][0].sum()), 6.)
        torch.testing.assert_close(out['blind_edos'][1], torch.zeros(2, dtype=torch.float64))

    def test_loss_gradcheck_through_shape_and_mass(self):
        torch.manual_seed(42)
        e, p, mass = [torch.randn(1, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        target = torch.tensor([[1., 3.]], dtype=torch.float64)
        def loss(e, p, mass):
            out = reconstruct_balanced_spectra({'edos': e, 'phdos': p, 'mass_logits': mass}, [2], protocol())
            return sum(normalized_spectral_error(out[f'blind_{task}'], target, [2], protocol()['tasks'][task]).mean()
                       for task in ('edos', 'phdos')) * .5
        self.assertTrue(torch.autograd.gradcheck(loss, (e, p, mass)))

    def test_selection_is_equal_task_mean_error_minimum_not_r2_or_oracle(self):
        metrics = {**row(), 'balanced_oracle_mae': 0., 'balanced_blind_r2': 1.}
        self.assertAlmostEqual(validation_score(metrics, ENERGY_METRIC), .3)
        self.assertTrue(is_better_score(.3, initial_best_score(ENERGY_METRIC), ENERGY_METRIC))
        self.assertFalse(is_better_score(.3, .3, ENERGY_METRIC))

    def test_either_task_can_block_checkpoint_despite_joint_improvement(self):
        for change in ({'blind_fail_rate_edos': 11.}, {'blind_fail_rate_phdos': 6.},
                       {'blind_r2_edos_median': .479}, {'blind_r2_phdos_median': .729}):
            self.assertFalse(balanced_checkpoint_eligible(row(**change), protocol()))
        self.assertTrue(balanced_checkpoint_eligible(row(blind_r2_edos_median=.48), protocol()))
        with self.assertRaisesRegex(ValueError, 'complete valid'):
            balanced_checkpoint_eligible(row(selection_samples=99), protocol())
        with self.assertRaisesRegex(ValueError, 'integer count'):
            balanced_checkpoint_eligible(row(blind_fail_rate_edos=9.5), protocol())

    def test_history_filters_before_ranking_and_ties_choose_earliest_epoch(self):
        frame = pd.DataFrame([row(epoch=1, e=.1, p=.1, blind_fail_rate_edos=11., checkpoint_eligible=False),
                              row(epoch=3), row(epoch=2)])
        self.assertEqual(selected_energy_history_row(frame, protocol()).epoch, 2)
        self.assertEqual(selected_history_row(frame, ENERGY_METRIC, protocol()).epoch, 2)
        self.assertIsNone(selected_energy_history_row(frame.iloc[:1], protocol()))
        for altered in (frame.assign(balanced_score=99.), frame.assign(checkpoint_eligible=True),
                        frame.assign(checkpoint_eligible='False')):
            with self.assertRaises(ValueError):
                selected_energy_history_row(altered, protocol())

    def test_invalid_data_and_missing_new_protocol_fail_explicitly(self):
        for targets in (np.array([[-1., 2.]]), np.array([[np.nan, 2.]]), np.zeros((1, 2))):
            with self.assertRaises(ValueError):
                fit_task_normalizer(targets, [1.], 1.)
        with self.assertRaises(ValueError):
            selection_contract(ENERGY_METRIC)
        changed = deepcopy(protocol())
        changed['task_weights']['edos'] = .9
        with self.assertRaises(ValueError):
            selection_contract(ENERGY_METRIC, changed)


if __name__ == '__main__':
    unittest.main()
