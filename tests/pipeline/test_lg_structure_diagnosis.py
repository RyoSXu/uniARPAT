"""Scientific safeguards for the saved-checkpoint structure diagnosis."""
import unittest

import numpy as np
import pandas as pd
import torch

from model.local_global_encoder import LocalGlobalEncoder
from tools.eval.lg_structure_diagnosis import (
    attach_support, correlation, intervention, pair_errors, paired_group_interval,
    summarize_pairs,
)
from utils.periodic_geometry import build_periodic_records


class TestStructureDiagnosis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_correct_amplitude_wrong_direction_is_not_improvement(self):
        targets = np.array([[.8, .2], [.2, .8]])
        keys, ids = ['same', 'same'], ['one', 'two']
        exact = pair_errors(targets, targets, keys, ids)
        reversed_pair = pair_errors(targets[::-1], targets, keys, ids)
        zero = pair_errors(np.full((2, 2), .5), targets, keys, ids)
        self.assertAlmostEqual(exact.skill.iloc[0], 1.)
        self.assertAlmostEqual(zero.skill.iloc[0], 0.)
        self.assertAlmostEqual(reversed_pair.skill.iloc[0], -1.)
        self.assertAlmostEqual(reversed_pair.predicted_tv.iloc[0], exact.predicted_tv.iloc[0])
        self.assertAlmostEqual(reversed_pair.direction_cosine.iloc[0], -1.)

    def test_empty_and_identical_target_pairs_do_not_invent_skill(self):
        values = np.full((2, 2), .5)
        empty = pair_errors(values, values, ['a', 'b'], ['1', '2'])
        self.assertEqual(summarize_pairs(empty)['pairs'], 0)
        same = pair_errors(values, values, ['a', 'a'], ['1', '2'])
        self.assertIsNone(summarize_pairs(same)['skill_median'])
        self.assertIsNone(correlation([], []))

    def test_group_resampling_checks_pair_identity(self):
        y = np.array([[.8, .2], [.2, .8]])
        pairs = pair_errors(y, y, ['a', 'a'], ['1', '2'])
        result = paired_group_interval(pairs, pairs.copy())
        self.assertEqual(result['ci95'], [0., 0.])
        wrong = pairs.copy()
        wrong.loc[0, 'right_id'] = '3'
        with self.assertRaisesRegex(ValueError, 'identities'):
            paired_group_interval(pairs, wrong)

    def test_support_excludes_self_and_cannot_reference_query_labels(self):
        records = []
        for mid, comp, value in [('a', 'X', 0.), ('b', 'X', 1.), ('c', 'Y', 2.)]:
            records.append(dict(material_id=mid, composition=comp, chemical_system=comp,
                                degree_std=value, volume_per_atom=value + 1,
                                **{f'rbf{k}': value for k in range(16)}))
        train = pd.DataFrame(records)
        supported = attach_support(train, train)
        self.assertEqual(supported.same_composition_train_count.tolist(), [1, 1, 0])
        self.assertEqual(supported.nearest_train_id.tolist(), ['b', 'a', ''])
        self.assertTrue(np.isnan(supported.radial_support_distance.iloc[2]))
        query = train.iloc[[0]].copy()
        query['material_id'] = 'new'
        self.assertEqual(attach_support(train, query).same_composition_train_count.iloc[0], 2)

    def test_interventions_restore_hooks_on_error_without_changing_weights(self):
        from types import SimpleNamespace
        from utils.zproj_alignment import state_hash
        encoder = LocalGlobalEncoder('B+', 'staged', 2, d_model=16, nhead=4,
                                     dim_feedforward=32, dropout=0.)
        encoder.eval()
        # One asymmetric three-atom structure, with real periodic records.
        from utils.crystal import CrystalGeometry
        geometry = CrystalGeometry(torch.eye(3)[None] * 8,
                                   torch.tensor([[[0., 0., 0.], [.15, 0., 0.], [.05, .2, 0.]]]))
        mask = torch.zeros(1, 3, dtype=torch.bool)
        records = build_periodic_records(geometry, mask, radius=6.)
        values = torch.randn(1, 3, 16)
        wrapper = SimpleNamespace(encoder=encoder)
        digest = state_hash(encoder.state_dict())
        with torch.no_grad():
            expected = encoder(values, records, mask)
            for mode in ('identity', 'local_90pct', 'local_off', 'rbf_off',
                         'high_order_off', 'angular_off'):
                with intervention(wrapper, mode):
                    self.assertTrue(torch.isfinite(encoder(values, records, mask)).all())
                torch.testing.assert_close(encoder(values, records, mask), expected, rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, 'test exit'):
                with intervention(wrapper, 'rbf_off'):
                    raise RuntimeError('test exit')
            torch.testing.assert_close(encoder(values, records, mask), expected, rtol=0, atol=0)
        self.assertEqual(digest, state_hash(encoder.state_dict()))


if __name__ == '__main__':
    unittest.main()
