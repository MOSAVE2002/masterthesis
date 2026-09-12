from __future__ import annotations

import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


dataset_evaluation = importlib.import_module(
    "06_Evaluation.evaluate_gnn_dataset"
)
prediction_evaluation = importlib.import_module(
    "06_Evaluation.evaluate_gnn_predictions"
)
from helper.local_buffer import TARGET_COLUMN


def _write_dataset(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance_name",
        "candidate_generation_mode",
        "due_date_factor",
        TARGET_COLUMN,
        "reliability_graph_parameters",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for values in rows:
            instance_name, mode, probabilities = values[:3]
            due_date_factor = values[3] if len(values) > 3 else ""
            writer.writerow({
                "instance_name": instance_name,
                "candidate_generation_mode": mode,
                "due_date_factor": due_date_factor,
                TARGET_COLUMN: json.dumps(probabilities),
                "reliability_graph_parameters": json.dumps({}),
            })


class GNNDiagnosticsTests(unittest.TestCase):
    def test_many_calibrated_due_factors_have_bounded_plot_size(self):
        rows = [{"due_date_factor": 1.0 + i/1000, "probabilities": [float(i), float(i)+.5]}
                for i in range(2126)]
        groups = dataset_evaluation._due_factor_plot_groups(rows)
        self.assertLessEqual(len(groups), 12)
        self.assertEqual(sum(len(values) for _, _, values in groups), 4252)
        self.assertEqual([v for _, _, values in groups for v in values],
                         [v for row in rows for v in row['probabilities']])

    def test_dataset_distribution_baseline_and_histogram(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "results"
            _write_dataset(
                dataset / "training" / "graphs_training.csv",
                [
                    ("i3_k3_o3-5_1", "unconstrained", [0.0, 1.0], 1.55),
                    ("i3_k3_o3-5_2", "nominal_ontime", [0.9, 0.95], 1.70),
                ],
            )
            _write_dataset(
                dataset / "valid" / "graphs_valid.csv",
                [("i5_k3_o3-5_1", "nominal_ontime", [0.8, 1.0])],
            )
            _write_dataset(
                dataset / "test" / "graphs_test.csv",
                [("i7_k5_o3-5_1", "unconstrained", [0.94, 0.96])],
            )

            result = dataset_evaluation.run_dataset_evaluation(
                dataset_directory=dataset,
                output_directory=output,
                expected_service_level=0.50,
                boundary_width=0.25,
                histogram_bins=10,
            )

            training = next(
                row for row in result["distribution_rows"]
                if row["split"] == "training"
                and row["group_type"] == "overall"
            )
            self.assertEqual(training["graphs"], 2)
            self.assertEqual(training["labels"], 4)
            baseline = next(
                row for row in result["baseline_rows"]
                if row["split"] == "training"
            )
            self.assertAlmostEqual(
                baseline["constant_repair_buffer"], 0.7125
            )
            self.assertTrue(result["histogram_pdf"].read_bytes().startswith(
                b"%PDF"
            ))
            self.assertGreater(result["histogram_png"].stat().st_size, 1000)
            self.assertTrue(result["distribution_csv"].exists())
            self.assertTrue(result["constant_baseline_csv"].exists())
            self.assertTrue(result["due_factor_histogram_pdf"].exists())
            self.assertTrue(result["due_factor_histogram_png"].exists())
            factor_rows = [
                row for row in result["distribution_rows"]
                if row["split"] == "training"
                and row["group_type"] == "effective_due_date_factor"
            ]
            self.assertEqual(
                {row["group"] for row in factor_rows}, {"1.55", "1.70"}
            )

    def test_prediction_metrics_and_calibration(self):
        rows = [
            {
                "solver": "gurobi_gnn",
                "model": "gnn_sage_layers1_hidden4",
                "instance_name": "instance_1",
                "job_id": str(index),
                "internal_expected_repair_buffer": prediction,
                "reference_expected_repair_buffer": observed,
                "postsolve_evaluation_status": "evaluated",
            }
            for index, (prediction, observed) in enumerate([
                (0.90, 0.80),
                (0.96, 0.97),
                (0.99, 0.90),
                (0.20, 0.30),
            ])
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = prediction_evaluation.run_prediction_evaluation(
                job_rows=rows,
                output_directory=directory,
                boundary_width=0.03,
                calibration_bins=5,
            )

            self.assertEqual(len(result["metric_rows"]), 1)
            metrics = result["metric_rows"][0]
            self.assertAlmostEqual(metrics["mae"], 0.075)
            self.assertNotIn("threshold_accuracy", metrics)
            self.assertGreater(len(result["calibration_rows"]), 0)
            self.assertTrue(result["calibration_pdf"].read_bytes().startswith(
                b"%PDF"
            ))
            self.assertGreater(result["calibration_png"].stat().st_size, 1000)

    def test_diagnostics_are_gated_by_evaluate_workflow(self):
        config = {"workflow": {"evaluate": False}}
        self.assertIsNone(dataset_evaluation.evaluate_from_config(config))
        self.assertIsNone(prediction_evaluation.evaluate_from_config(config))

    def test_direct_script_calls_ignore_evaluate_workflow(self):
        config = {
            "workflow": {"evaluate": False},
            "evaluation": {
                "output_directory": "evaluation-results",
                "dataset_diagnostics": {},
                "prediction_diagnostics": {},
            },
            "training": {
                "data_generation": {
                    "output_directory": "gnn-dataset",
                    "fixed_y": {
                        "label_distribution_center": 0.50,
                        "label_distribution_half_width": 0.25,
                    },
                }
            },
            "constraint": {"weibull": {"reliability_graph": {}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with (
                patch.object(dataset_evaluation, "CONFIG_PATH", config_path),
                patch.object(
                    dataset_evaluation,
                    "_parse_args",
                    return_value=SimpleNamespace(
                        dataset_directory=None,
                        output_directory=None,
                    ),
                ),
                patch.object(
                    dataset_evaluation,
                    "run_dataset_evaluation",
                    return_value={},
                ) as run_dataset,
            ):
                dataset_evaluation.main()

            with (
                patch.object(
                    prediction_evaluation, "CONFIG_PATH", config_path
                ),
                patch.object(
                    prediction_evaluation,
                    "_parse_args",
                    return_value=SimpleNamespace(
                        input_path=None,
                        output_directory=None,
                    ),
                ),
                patch.object(
                    prediction_evaluation,
                    "run_prediction_evaluation",
                    return_value={},
                ) as run_predictions,
            ):
                prediction_evaluation.main()

        run_dataset.assert_called_once()
        run_predictions.assert_called_once()


if __name__ == "__main__":
    unittest.main()
