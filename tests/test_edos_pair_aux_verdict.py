"""Pure decision contracts for the valid-only eDOS pair auxiliary verdict."""

import unittest

from tools.eval.edos_pair_aux_verdict import evaluate_verdict


def _metric(delta, fail=0.0):
    return {"delta_median": delta, "delta_fail_pp": fail}


class TestPairAuxVerdict(unittest.TestCase):
    def test_all_five_gates_are_required_for_win(self):
        overall = {
            "edos_blind": _metric(0.02, 0.9),
            "edos_oracle": _metric(0.0, 0.9),
            "phdos_oracle": _metric(-0.02, 0.9),
            "phdos_blind": _metric(-0.02, 0.9),
        }
        paired = {
            "edos_oracle": _metric(-0.02, 0.9),
            "edos_blind": _metric(-0.02, 0.9),
        }
        mechanism = {"relative_reduction": 0.05, "relative_reduction_ci": [0.001, 0.1]}
        result = evaluate_verdict(overall, paired, mechanism)
        self.assertEqual(result["verdict"], "win")
        self.assertTrue(all(result["gates"].values()))

    def test_boundary_failure_parks_without_opening_another_route(self):
        overall = {
            "edos_blind": _metric(0.0199),
            "edos_oracle": _metric(0.0),
            "phdos_oracle": _metric(0.0),
            "phdos_blind": _metric(0.0),
        }
        paired = {"edos_oracle": _metric(0.0), "edos_blind": _metric(0.0)}
        mechanism = {"relative_reduction": 0.2, "relative_reduction_ci": [0.1, 0.3]}
        result = evaluate_verdict(overall, paired, mechanism)
        self.assertEqual(result["verdict"], "park")
        self.assertFalse(result["gates"]["overall_main"])

    def test_one_percentage_point_is_not_accepted(self):
        overall = {
            "edos_blind": _metric(0.03, 1.0),
            "edos_oracle": _metric(0.01),
            "phdos_oracle": _metric(0.0),
            "phdos_blind": _metric(0.0),
        }
        paired = {"edos_oracle": _metric(0.0), "edos_blind": _metric(0.0)}
        mechanism = {"relative_reduction": 0.1, "relative_reduction_ci": [0.01, 0.2]}
        self.assertEqual(evaluate_verdict(overall, paired, mechanism)["verdict"], "park")


if __name__ == "__main__":
    unittest.main()
