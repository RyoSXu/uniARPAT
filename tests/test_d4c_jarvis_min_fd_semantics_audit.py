import unittest

from tools.eval.d4c_jarvis_min_fd_semantics_audit import (
    classify_min_fd,
    exact_numeric_match,
    parse_numeric_csv,
)


class D4cJarvisMinFdSemanticTests(unittest.TestCase):
    def test_negative_serialized_zero_is_not_negative(self):
        self.assertEqual(classify_min_fd("-0.0"), (0.0, "zero_serialized_negative"))
        self.assertEqual(classify_min_fd("-0.1"), (-0.1, "negative"))
        self.assertEqual(classify_min_fd("0.0"), (0.0, "zero"))

    def test_csv_parser_requires_finite_numbers(self):
        self.assertEqual(parse_numeric_csv("'-1.0, 0.0, 2.5' ").tolist(), [-1.0, 0.0, 2.5])
        with self.assertRaises(ValueError):
            parse_numeric_csv("nan, 1.0")

    def test_field_equality_has_no_tolerance(self):
        self.assertTrue(exact_numeric_match(-2.0, -2.0))
        self.assertFalse(exact_numeric_match(-2.0, -2.0000001))


if __name__ == "__main__":
    unittest.main()
