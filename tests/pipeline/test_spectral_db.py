"""spectral_db_v1 keeps source spectra without dataset decisions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/data/spectral_store'))

from build_db_v1 import assemble_edos, jarvis_structure, site_facts  # noqa: E402


def _row():
    return {
        'identifier': 'task1',
        'structure': {
            'lattice': {'matrix': [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]},
            'sites': [{
                'species': [{'element': 'Si', 'occu': 1.0}],
                'abc': [0.0, 0.0, 0.0],
                'properties': {'magmom': 0.5},
            }],
        },
        'energies': [0.0, 1.0, 2.0],
        'spin_up_densities': [1.0, 2.0, 3.0],
        'spin_down_densities': [0.5, 0.5, 0.5],
        'efermi': 1.0,
    }


class EdosRecord(unittest.TestCase):
    def test_shift_and_spin_sum_keep_order(self):
        record = assemble_edos(_row(), 'GGA', 'mp-1', True, 'GGA Static', 'snap')
        self.assertEqual(record['x'], [-1.0, 0.0, 1.0])
        self.assertEqual(record['y'], [1.5, 2.5, 3.5])
        self.assertEqual(record['spin_channels'], 2)
        self.assertEqual(record['efermi'], 1.0)
        self.assertTrue(record['is_canonical'])
        self.assertIsNone(record['parse_error'])
        self.assertNotIn('trunc_ratio', record)
        self.assertNotIn('status', record)
        self.assertEqual(record['calc_elements'], ['Si:1.0'])
        self.assertEqual(record['calc_magmom'], [0.5])

    def test_unreadable_keeps_the_row(self):
        row = _row()
        row['efermi'] = None
        record = assemble_edos(row, 'GGA', None, False, None, 'snap')
        self.assertIsNone(record['material_id'])
        self.assertTrue(record['parse_error'].startswith('unreadable'))
        self.assertEqual(record['x'], [])

    def test_disordered_site_keeps_every_species(self):
        structure = {'sites': [{'species': [{'element': 'Fe', 'occu': 0.5}, {'element': 'Ni', 'occu': 0.5}],
                                'properties': {}}]}
        elements, magmom, disordered = site_facts(structure)
        self.assertEqual(elements, ['Fe:0.5,Ni:0.5'])
        self.assertEqual(disordered, 1)
        self.assertEqual(magmom, [None])


class JarvisCoordinates(unittest.TestCase):
    def test_cartesian_converts_to_fractional(self):
        atoms = {'lattice_mat': [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
                 'coords': [[1.0, 2.0, 2.5]], 'elements': ['Si'], 'cartesian': True}
        lattice, elements, frac = jarvis_structure(atoms)
        np.testing.assert_allclose(lattice, [2, 0, 0, 0, 4, 0, 0, 0, 5])
        np.testing.assert_allclose(frac, [0.5, 0.5, 0.5])
        self.assertEqual(elements, ['Si'])


if __name__ == '__main__':
    unittest.main()
