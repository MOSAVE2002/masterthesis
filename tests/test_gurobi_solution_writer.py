from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gurobipy as gp

from helper.gurobi_solution_writer import write_comparable_solution
from helper.start_solve_ins import _optimize_with_telemetry


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
                    "solver_telemetry": {
                        "branch_and_bound_nodes": 12.0,
                        "root_node_bound": 3.5,
                        "time_to_first_incumbent_seconds": None,
                        "first_incumbent_objective": None,
                        "time_to_best_incumbent_seconds": None,
                        "best_incumbent_objective": None,
                        "progress_trace": [{
                            "runtime_seconds": 0.1,
                            "incumbent_objective": None,
                            "best_bound": 3.5,
                            "relative_gap": None,
                            "node_count": 12.0,
                            "solution_count": 0,
                            "event": "final",
                            "objective_sense": "minimize",
                        }],
                    },
                },
                Path(directory) / "solution.txt",
            )
            content = path.read_text(encoding="utf-8")
            progress = (
                Path(directory) / "solution_solver_progress.csv"
            ).read_text(encoding="utf-8")

        self.assertIn("Status: INFEASIBLE", content)
        self.assertIn("Objective: \n", content)
        self.assertIn("Makespan: \n", content)
        self.assertIn("Operating cost: \n", content)
        self.assertIn("Branch-and-bound nodes: 12.000000", content)
        self.assertIn("Root-node bound: 3.500000", content)
        self.assertIn("Time to first incumbent [s]: \n", content)
        self.assertIn(
            "Solver progress file: solution_solver_progress.csv", content
        )
        self.assertIn("runtime_seconds,incumbent_objective,best_bound", progress)
        self.assertIn("0.1,,3.5", progress)
        self.assertIn("linear_matrix_nonzeros:", content)
        model.dispose()

    def test_callback_records_incumbent_and_search_metrics(self):
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.Presolve = 0
        variable = model.addVar(vtype=gp.GRB.BINARY, name="x")
        model.setObjective(variable, gp.GRB.MAXIMIZE)

        telemetry = _optimize_with_telemetry(model)

        self.assertEqual(model.Status, gp.GRB.OPTIMAL)
        self.assertIsNotNone(telemetry["branch_and_bound_nodes"])
        self.assertIsNotNone(telemetry["linear_matrix_nonzeros"])
        self.assertIsNotNone(
            telemetry["time_to_first_incumbent_seconds"]
        )
        self.assertEqual(telemetry["first_incumbent_objective"], 1.0)
        self.assertEqual(telemetry["best_incumbent_objective"], 1.0)
        self.assertGreaterEqual(
            telemetry["time_to_best_incumbent_seconds"],
            telemetry["time_to_first_incumbent_seconds"],
        )
        self.assertGreaterEqual(len(telemetry["progress_trace"]), 1)
        final = telemetry["progress_trace"][-1]
        self.assertEqual(final["event"], "final")
        self.assertEqual(final["incumbent_objective"], 1.0)
        self.assertEqual(final["relative_gap"], 0.0)
        self.assertEqual(final["objective_sense"], "maximize")
        model.dispose()


if __name__ == "__main__":
    unittest.main()
