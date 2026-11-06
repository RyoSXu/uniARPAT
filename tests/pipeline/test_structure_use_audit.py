"""Structure-use audit: label identity gate, binning, pair energy, contrasts and ensemble energies."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from tools.data.c2b_grids import box_average as reference_box_average
from tools.eval.structure_use_audit import (
    E0_EDGES, box_average, composition_features, element_table, ensemble_energies, group_contrast,
    label_gate, pair_energy, structure_features, total_variation)


class StructureUseAuditTests(unittest.TestCase):
    def test_box_average_matches_label_pipeline(self):
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(-8, 8, 900))
        y = rng.uniform(0, 3, 900)
        ours, cover = box_average(x, y, E0_EDGES)
        reference, reference_cover = reference_box_average(x, y, E0_EDGES)
        np.testing.assert_allclose(ours, reference, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(cover, reference_cover.astype(bool))
        shuffled = rng.permutation(len(x))
        np.testing.assert_allclose(box_average(x[shuffled], y[shuffled], E0_EDGES)[0], ours, rtol=1e-12)

    def test_label_gate_accepts_scale_and_winsorised_bins_only(self):
        rng = np.random.default_rng(1)
        recon = rng.uniform(.1, 2, 128)
        self.assertEqual(label_gate(recon, 2 * recon)['class'], 'exact')
        clipped = 2 * recon
        clipped[[10, 40]] *= .5
        gate = label_gate(recon, clipped)
        self.assertEqual((gate['class'], gate['off_bins']), ('clipped', 2))
        self.assertAlmostEqual(gate['scale'], 2.)
        raised = 2 * recon
        raised[10] *= 1.5
        self.assertEqual(label_gate(recon, raised)['class'], 'fail')
        many = 2 * recon
        many[:6] *= .5
        self.assertEqual(label_gate(recon, many)['class'], 'fail')
        self.assertEqual(label_gate(recon, rng.uniform(.1, 2, 128))['class'], 'fail')
        self.assertEqual(label_gate(np.zeros(8), np.zeros(8))['class'], 'exact')

    def test_pair_energy_is_symmetric_and_per_atom(self):
        y = np.arange(1., 9.)
        self.assertEqual(pair_energy(y, 2 * y, 1, 2, .5, 3.), 0.)
        left, right = y, y[::-1]
        self.assertAlmostEqual(pair_energy(left, right, 2, 3, .5, 3.), pair_energy(right, left, 3, 2, .5, 3.))
        self.assertAlmostEqual(pair_energy(y, np.zeros(8), 1, 1, .5, 2.), .5 * (y ** 2).sum() / 2.)
        self.assertAlmostEqual(float(total_variation(y, 3 * y)), 0.)

    def test_group_contrast_uses_only_mixed_groups(self):
        frame = pd.DataFrame({'composition': ['a', 'a', 'b', 'b', 'c'],
                              'flag': [True, False, True, False, True],
                              'value': [3., 1., 5., 4., 100.]})
        result = group_contrast(frame, 'flag', 'value')
        self.assertEqual(result['groups'], 2)
        self.assertAlmostEqual(result['mean_delta'], 1.5)
        self.assertEqual(group_contrast(frame[frame.composition == 'c'], 'flag', 'value'), {'groups': 0})

    def test_ensemble_energy_recovers_member_scale(self):
        rng = np.random.default_rng(2)
        ids = np.array(['m1', 'm2', 'm3'])
        target = {t: rng.uniform(0, 1, (3, n)) for t, n in (('edos', 6), ('phdos', 4))}
        scale = {'edos': np.array([.5, 1., 2.]), 'phdos': np.array([1., 3., .25])}
        with tempfile.TemporaryDirectory() as tmp:
            members = []
            preds = []
            for k in range(2):
                pred = {t: target[t] + rng.normal(0, .1, target[t].shape) for t in target}
                preds.append(pred)
                order = np.array([2, 0, 1]) if k else np.arange(3)
                frame = pd.DataFrame({'material_id': ids[order], **{
                    f'blind_energy_{t}': (scale[t] * ((pred[t] - target[t]) ** 2).sum(1))[order] for t in target}})
                samples, spectra = Path(tmp) / f's{k}.csv', Path(tmp) / f'p{k}.npz'
                frame.to_csv(samples, index=False)
                np.savez(spectra, material_ids=ids[order],
                         **{f'pred_blind_{t}': pred[t][order] for t in target},
                         **{f'tgt_{t}': target[t][order] for t in target})
                members.append((samples, spectra))
            out = Path(tmp) / 'e.csv'
            ensemble_energies({'valid': members}, out)
            result = pd.read_csv(out).set_index('material_id').loc[ids]
            for t in target:
                expected = scale[t] * (((preds[0][t] + preds[1][t]) / 2 - target[t]) ** 2).sum(1)
                np.testing.assert_allclose(result[f'ensemble_energy_{t}'], expected, rtol=1e-6)

    def test_composition_features_ignore_structure_and_structure_features_do_not(self):
        elements = element_table()
        numbers = np.array([11, 17])
        np.testing.assert_array_equal(composition_features(numbers, elements),
                                      composition_features(numbers[::-1], elements))
        cell = np.eye(3) * 5.6
        frac = np.array([[0, 0, 0], [.5, .5, .5]])
        first = structure_features(numbers, cell, frac, elements)
        second = structure_features(numbers, cell * 1.1, frac, elements)
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.allclose(first, second))


if __name__ == '__main__':
    unittest.main()
