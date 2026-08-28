from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gurobipy as gp

from helper import solution_plots


ROOT = Path(__file__).resolve().parents[1]


def _instance():
    return SimpleNamespace(
        num_machines=2,
        real_operations=[1, 2, 3, 4],
        eligible_machines={
            1: [0, 1],
            2: [0],
            3: [1],
            4: [0, 1],
        },
        processing_times={
            (1, 0): 3.0,
            (1, 1): 2.0,
            (2, 0): 4.0,
            (3, 1): 3.0,
            (4, 0): 2.0,
            (4, 1): 4.0,
        },
        predecessors={1: [], 2: [1], 3: [], 4: [3]},
        jobs={1: [1, 2], 2: [3, 4]},
        job_end_operations={1: 2, 2: 4},
    )


class SolutionPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib

        base = importlib.import_module("03_Gurobi.build_fjsp")
        cls.instance = _instance()
        cls.model = gp.Model()
        cls.model.Params.OutputFlag = 0
        cls.model, cls.variables = base.build_fjsp(
            cls.model, cls.instance, include_makespan=True
        )
        cls.model.optimize()
        if cls.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("Plot test fixture did not solve to optimality.")

    def test_config_exposes_restored_plot_controls(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        nonlinear = config["solvers"]["gurobi"]["nonlinear"]
        for key in (
            "plot_solution_schedule",
            "plot_solution_graph",
            "plot_candidate_graph",
        ):
            self.assertIn(key, nonlinear)
            self.assertIsInstance(nonlinear[key], bool)
        self.assertEqual(
            nonlinear["plot_solution_graph_style"], "disjunctive"
        )
        self.assertEqual(
            nonlinear["plot_output_directory"],
            "plots/fjsp_solution_plots",
        )

    def test_all_restored_plots_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = solution_plots.write_solution_plots(
                self.model,
                self.variables,
                self.instance,
                instance_name="test_instance",
                solver="gurobi_nonlinear",
                output_directory=directory,
                plot_solution_schedule_enabled=True,
                plot_solution_graph_enabled=True,
                plot_candidate_graph_enabled=True,
                graph_style="disjunctive",
            )
            self.assertEqual(
                set(paths), {"schedule", "solution_graph", "candidate_graph"}
            )
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_machine_operation_graph_style_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = solution_plots.plot_solution_graph(
                self.model,
                self.variables,
                self.instance,
                Path(directory) / "machine_operation.png",
                style="machine_operation",
            )
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
