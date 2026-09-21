import unittest

import numpy as np

from tools.eval.d4b_negative_coordinate_provenance_audit import (
    negative_coordinate_mass_fraction,
    parse_bool,
    select_mp_record,
)


class D4bProvenanceAuditTests(unittest.TestCase):
    def test_csv_boolean_parser_does_not_accept_false_as_truthy(self):
        self.assertTrue(parse_bool("True"))
        self.assertFalse(parse_bool(" false "))
        with self.assertRaises(ValueError):
            parse_bool("yes")

    def test_negative_coordinate_mass_splits_crossing_interval(self):
        # y is constant over [-1, 1], so exactly half of its integral is negative.
        self.assertAlmostEqual(negative_coordinate_mass_fraction([-1.0, 1.0], [2.0, 2.0]), 0.5)
        self.assertEqual(negative_coordinate_mass_fraction([0.0, 1.0], [2.0, 2.0]), 0.0)

    def test_mp_reference_must_match_frozen_choice(self):
        records = [
            {"method": "dfpt", "frequencies_THz": [0.0, 1.0], "densities": [1.0, 1.0]},
            {"method": "pheasy", "frequencies_THz": [0.0, 1.0], "densities": [1.0, 1.0]},
        ]
        self.assertEqual(select_mp_record(records, "mp:pheasy")["method"], "pheasy")
        self.assertIsNone(select_mp_record(records, "mp:missing"))
        with self.assertRaises(ValueError):
            select_mp_record(records, "jv:JVASP-1")


if __name__ == "__main__":
    unittest.main()
