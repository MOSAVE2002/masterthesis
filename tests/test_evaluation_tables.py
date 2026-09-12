from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


evaluation = importlib.import_module("06_Evaluation.evaluate_solutions")


def _schedule_row():
    return {
        "instance_name": "i3_k3_o3-5_1",
        "evaluation_tier": "in_distribution",
        "solver": "gurobi_gnn",
        "model": "gnn_sage_layers1_hidden4",
        "formulation": "gnn_per_job_ontime_probability_v3",
        "status": "OPTIMAL",
        "postsolve_evaluation_status": "evaluated",
        "objective": 84.125,
        "best_bound": 84.125,
        "mip_gap": 0.0,
        "runtime_seconds": 0.125,
        "model_build_seconds": 0.025,
        "optimizer_wall_seconds": 0.1,
        "number_of_jobs": 3,
        "minimum_target_service_level": 0.8,
        "maximum_target_service_level": 0.8,
        "minimum_internal_probability": 0.83,
        "internal_all_jobs_feasible": True,
        "minimum_mc_ontime_probability": 0.82,
        "minimum_wilson_lower_bound": 0.8135,
        "minimum_bonferroni_wilson_lower_bound": 0.8102,
        "mc_all_jobs_point_feasible": True,
        "wilson_all_jobs_feasible": True,
        "bonferroni_wilson_all_jobs_feasible": True,
        "maximum_service_shortfall": 0.0,
        "simulation_replications": 10_000,
        "simulation_seed": 42,
        "simulation_mean_failures": 1.2,
        "simulation_mean_total_repair_delay": 2.5,
        "solution_file": "solution.txt",
    }


class EvaluationTableTests(unittest.TestCase):
    def test_parse_solution_resolves_solver_progress_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "solution_i3_k3_o3-5_1_gurobi.txt"
            path.write_text(
                "\n".join((
                    "Formulation: nominal_fjsp",
                    "Status: TIME_LIMIT",
                    "Solution count: 0",
                    "Solver progress file: solution_progress.csv",
                    "Solver progress points: 3",
                )),
                encoding="utf-8",
            )

            parsed = evaluation.parse_solution(path)

        self.assertEqual(
            parsed["solver_progress_path"],
            (root / "solution_progress.csv").resolve(),
        )
        self.assertEqual(parsed["solver_progress_points"], 3)

    def test_bonferroni_confidence(self):
        confidence = evaluation.bonferroni_confidence(0.95, 5)
        self.assertAlmostEqual(confidence, 0.99)
        individual = evaluation.wilson_lower_bound(8_200, 10_000, 0.95)
        corrected = evaluation.wilson_lower_bound(
            8_200, 10_000, confidence
        )
        self.assertLess(corrected, individual)

    def test_writes_csv_latex_and_pdf_tables(self):
        rows = [_schedule_row()]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "result_table.csv"
            latex_path = root / "result_table.tex"
            pdf_path = root / "result_table.pdf"

            evaluation._write_csv(csv_path, rows)
            evaluation._write_latex_table(latex_path, rows, 0.95)
            evaluation._write_pdf_table(pdf_path, rows, 0.95)

            self.assertIn("minimum_bonferroni_wilson_lower_bound", csv_path.read_text())
            latex = latex_path.read_text(encoding="utf-8")
            self.assertIn(r"\begin{longtable}", latex)
            self.assertIn(r"i3\_k3\_o3-5\_1", latex)
            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF"))
            self.assertGreater(pdf_path.stat().st_size, 1_000)

    def test_configured_evaluation_is_gated_by_workflow(self):
        config = {"workflow": {"evaluate": False}}
        with patch.object(evaluation, "run_evaluation") as run:
            result = evaluation.evaluate_from_config(config)
        self.assertIsNone(result)
        run.assert_not_called()

    def test_configured_evaluation_passes_gnn_diagnostics(self):
        config = {
            "workflow": {"evaluate": True},
            "evaluation": {
                "dataset_diagnostics": {"enabled": True},
                "prediction_diagnostics": {
                    "enabled": True,
                    "calibration_bins": 8,
                },
                "numerical_analysis": {
                    "enabled": True,
                    "strict": True,
                },
            },
        }
        with patch.object(
            evaluation, "run_evaluation", return_value={}
        ) as run:
            evaluation.evaluate_from_config(config)

        self.assertEqual(
            run.call_args.kwargs["dataset_diagnostics"],
            {"enabled": True},
        )
        self.assertEqual(
            run.call_args.kwargs["prediction_diagnostics"][
                "calibration_bins"
            ],
            8,
        )
        self.assertEqual(
            run.call_args.kwargs["numerical_analysis"],
            {"enabled": True, "strict": True},
        )

    def test_numerical_analysis_runs_as_part_of_evaluation(self):
        analysis_result = {"solver_summary": Path("solver_summary.csv")}
        fake_module = unittest.mock.Mock()
        fake_module.run_analysis.return_value = analysis_result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            solution_path = root / "solution.txt"
            with (
                patch.object(
                    evaluation, "_solution_paths", return_value=[solution_path]
                ),
                patch.object(
                    evaluation,
                    "evaluate_solution",
                    return_value=(
                        _schedule_row(),
                        [{
                            "evaluation_tier": "in_distribution",
                            "instance_name": "i3_k3_o3-5_1",
                            "model": "gnn_sage_layers1_hidden4",
                            "job_id": 0,
                        }],
                    ),
                ),
                patch.object(evaluation, "_write_latex_table"),
                patch.object(evaluation, "_write_pdf_table"),
                patch.object(
                    evaluation.importlib,
                    "import_module",
                    return_value=fake_module,
                ),
            ):
                result = evaluation.run_evaluation(
                    config={"evaluation": {}},
                    solutions=None,
                    solutions_root=root,
                    instances_root=root,
                    output_directory=root,
                    numerical_analysis={"enabled": True, "strict": True},
                )

        fake_module.run_analysis.assert_called_once_with(
            config={"evaluation": {}},
            result_table=root / "result_table.csv",
            job_table=root / "job_comparison.csv",
            manifest=root / "solve_manifest.json",
            instances_root=root,
            output_directory=root / "numerical_analysis",
            strict=True,
        )
        self.assertEqual(result["numerical_analysis"], analysis_result)

    def test_external_paths_remain_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "solution.txt"
            self.assertEqual(
                evaluation._portable_path(external), str(external.resolve())
            )


if __name__ == "__main__":
    unittest.main()
