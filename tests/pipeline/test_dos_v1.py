"""Rules for choosing and attaching dos_v1 labels, without the full database."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/data/spectral_store'))

from build_dos_v1 import (  # noqa: E402
    _bin_label, _legacy_mp_id, _record_from_calc, attach_record, cluster_unmatched,
    imag_fraction, pick_representative,
)


def _row(record_id, kind, formula='AB', area=3.0, cs='cubic', volume=10.0, hint='mp-1'):
    return {'record_id': record_id, 'kind': kind, 'formula': formula, 'area_atom': area,
            'cs': cs, 'volume': volume, 'hint': hint, 'structure': record_id}


class DosV1RulesTest(unittest.TestCase):
    def test_representative_prefers_mp_then_mode_count(self):
        chosen = pick_representative([
            _row('pdb-b', 'phonondb', area=3.0),
            _row('mp-a', 'mp_dfpt', area=3.2),
            _row('mp-c', 'mp_pheasy', area=4.0),
            _row('mp-d', 'mp_pheasy', area=2.9),
        ])
        self.assertEqual(chosen['record_id'], 'mp-d')

    def test_cluster_does_not_chain_merge(self):
        rows = [_row('a', 'phonondb'), _row('b', 'phonondb'), _row('c', 'phonondb')]

        def fits(left, right):
            pair = {left['record_id'], right['record_id']}
            return pair != {'a', 'c'}

        groups = cluster_unmatched(rows, fits)
        self.assertEqual(sorted(len(g) for g in groups), [1, 2])

    def test_hint_within_volume_attaches_even_if_crystal_system_differs(self):
        materials = {'mp-1': {'formula': 'AB', 'cs': 'cubic', 'volume': 10.0, 'structure': 'm'}}
        record = _row('calc', 'mp_dfpt', cs='tetragonal', volume=10.5)
        decision = attach_record(record, materials, {'AB': ['mp-1']},
                                 lambda material, rec: True, lambda material, rec: False)
        self.assertEqual(decision['reason'], 'hint')
        self.assertTrue(decision['label_ok'])

    def test_near_volume_match_failure_is_counted_and_not_attached(self):
        materials = {
            'mp-1': {'formula': 'AB', 'cs': 'cubic', 'volume': 10.0, 'structure': 'hint'},
            'mp-2': {'formula': 'AB', 'cs': 'cubic', 'volume': 10.0, 'structure': 'other'},
        }
        record = _row('calc', 'phonondb', volume=10.2)

        def fit(material, rec):
            return False

        decision = attach_record(record, materials, {'AB': ['mp-1', 'mp-2']}, fit, fit)
        self.assertTrue(decision['near_miss'])
        self.assertIsNone(decision['material_id'])

    def test_two_matches_need_a_unique_tighter_hit(self):
        materials = {
            'mp-1': {'formula': 'AB', 'cs': 'hexagonal', 'volume': 20.0, 'structure': 'hint'},
            'mp-2': {'formula': 'AB', 'cs': 'cubic', 'volume': 10.0, 'structure': 'a'},
            'mp-3': {'formula': 'AB', 'cs': 'cubic', 'volume': 10.1, 'structure': 'b'},
        }
        record = _row('calc', 'phonondb', volume=10.0)

        def fit_default(material, rec):
            return material['structure'] in ('a', 'b')

        def fit_tight(material, rec):
            return material['structure'] == 'b'

        decision = attach_record(record, materials, {'AB': ['mp-1', 'mp-2', 'mp-3']}, fit_default, fit_tight)
        self.assertEqual(decision['material_id'], 'mp-3')
        self.assertFalse(decision['label_ok'])

    def test_area_uses_dos_norm_count_and_keeps_structure_count(self):
        lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
        record = _record_from_calc('phdos:phonondb:a', 'phonondb', None, lattice, [1], [[0.0, 0.0, 0.0]],
                                   x=[0.0, 10.0], y=[2.0, 2.0], dos_n=2)
        self.assertEqual(record['n'], 1)
        self.assertEqual(record['dos_n'], 2)
        self.assertAlmostEqual(record['area_atom'], 10.0)

    def test_label_scale_uses_dos_norm_count(self):
        spectra = {'phdos:phonondb:a': ([-280.0, 980.0], [1.0, 1.0])}
        record = {'record_id': 'phdos:phonondb:a', 'n': 4, 'dos_n': 2}
        values, _ = _bin_label(record, {'structure': {'n': 2}}, spectra, 'phdos', 64)
        self.assertAlmostEqual(float(np.sum(values) * (1260.0 / 64)), 1260.0, places=3)
        record['dos_n'] = 4
        values, _ = _bin_label(record, {'structure': {'n': 2}}, spectra, 'phdos', 64)
        self.assertAlmostEqual(float(np.sum(values) * (1260.0 / 64)), 630.0, places=3)

    def test_numeric_mp_id_uses_padded_base26(self):
        self.assertEqual(_legacy_mp_id('mp-1'), 'mp-aaaaaaab')
        self.assertEqual(_legacy_mp_id('mp-149'), 'mp-aaaaaaft')
        self.assertIsNone(_legacy_mp_id('mp-aaaaaaab'))
        self.assertIsNone(_legacy_mp_id(None))

    def test_imaginary_weight_is_the_area_below_minus_20(self):
        x = np.array([-40.0, -20.0, 0.0])
        y = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(imag_fraction(x, y), 0.5)


if __name__ == '__main__':
    unittest.main()
