import importlib
import unittest
from types import SimpleNamespace

import gurobipy as gp

from helper.stochastic_fjsp import weibull_down_probability


nonlinear = importlib.import_module("03_Gurobi.build_fjsp_with_nonlinear")


class NonlinearIntegerBetaTests(unittest.TestCase):
    def _solve_probability(self, beta):
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 2
        midpoint = model.addVar(lb=10.0, ub=10.0, name="midpoint")
        assignment = model.addVar(
            lb=1.0, ub=1.0, vtype=gp.GRB.BINARY, name="assignment"
        )
        variables = {
            "H": 50.0,
            "T": {0: midpoint},
            "Y": {(0, 0): assignment},
            "Y_index": [(0, 0)],
            "weibull_alpha": {0: 30.0},
            "weibull_beta": {0: beta},
            "repair_rate": {0: 0.5},
        }
        nonlinear._add_pd_quadrature(
            model,
            variables,
            None,
            SimpleNamespace(quadrature_points=12),
        )
        model.setObjective(variables["Pd"][0, 0], gp.GRB.MINIMIZE)
        model.optimize()
        self.assertEqual(model.Status, gp.GRB.OPTIMAL)
        self.assertEqual(model.NumQConstrs, 0)
        self.assertEqual(
            sum(
                constraint.GenConstrType == gp.GRB.GENCONSTR_NL
                for constraint in model.getGenConstrs()
            ),
            13,
        )
        value = variables["Pd"][0, 0].X
        model.dispose()
        return value

    def test_beta_two_probability_matches_reference_quadrature(self):
        actual = self._solve_probability(2.0)
        expected = weibull_down_probability(10.0, 30.0, 2.0, 0.5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_beta_three_probability_matches_reference_quadrature(self):
        actual = self._solve_probability(3.0)
        expected = weibull_down_probability(10.0, 30.0, 3.0, 0.5)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_fractional_beta_is_rejected_before_optimization(self):
        with self.assertRaisesRegex(ValueError, "supports only.*2.*3"):
            self._solve_probability(2.3)


if __name__ == "__main__":
    unittest.main()
