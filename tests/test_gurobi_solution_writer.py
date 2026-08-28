from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gurobipy as gp

from helper.gurobi_solution_writer import write_comparable_solution


class GurobiSolutionWriterTests(unittest.TestCase):
    def test_infeasible_model_writes_summary_without_variable_values(self):
        model = gp.Model()
        model.Params.OutputFlag = 0
        makespan = model.addVar(lb=0.0, name="C_max")
        model.addConstr(makespan >= 1.0)
        model.addConstr(makespan <= 0.0)
        model.optimize()
        self.assertEqual(model.SolCount, 0)

        with tempfile.TemporaryDirectory() as directory:
            path = write_comparable_solution(
                model,
                {
                    "C_max": makespan,
                    "formulation": "infeasible_test",
                    "objective_definition": "test",
                },
                Path(directory) / "solution.txt",
            )
            content = path.read_text(encoding="utf-8")

        self.assertIn("Status: INFEASIBLE", content)
        self.assertIn("Objective: \n", content)
        self.assertIn("Makespan: \n", content)
        self.assertIn("Operating cost: \n", content)
        model.dispose()


if __name__ == "__main__":
    unittest.main()
