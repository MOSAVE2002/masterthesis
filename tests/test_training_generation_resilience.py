from __future__ import annotations

import csv
import importlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


generator = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_fix_and_optimize_training_data"
)


class TrainingGenerationResilienceTests(unittest.TestCase):
    def test_skips_failed_instance_and_writes_incremental_json_summary(self):
        calls = Counter()

        def candidate():
            return {
                "row": {field: "" for field in generator.FIELDNAMES},
                "job_probabilities": [0.75],
            }

        def collect(
            _instance,
            instance_name,
            *_args,
            **_kwargs,
        ):
            calls[instance_name] += 1
            if instance_name == "always_bad":
                return [], []
            value = candidate()
            return [value], [{"candidate": value, "category": "test"}]

        with tempfile.TemporaryDirectory() as directory:
            generation = {
                "method": "fix_and_optimize",
                "instance_splits": {
                    "train": [
                        "always_bad",
                        "immediate_success",
                    ],
                    "valid": [],
                    "test": [],
                },
                "output_directory": directory,
                "random_seed": 42,
                "samples_per_instance": 1,
                "reliability_graph": {},
                "fixed_y": {"fix_ratios": [0.25]},
                "instance_failure_handling": {
                    "summary_filename": "generation_summary.json",
                },
            }
            with (
                patch.object(
                    generator,
                    "load_generated_instance",
                    return_value=object(),
                ),
                patch.object(
                    generator,
                    "_collect_and_select_instance_candidates",
                    side_effect=collect,
                ),
            ):
                returned = generator.generate_from_config(generation)

            summary_path = Path(directory) / "generation_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(returned["status"], "completed")
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["totals"]["configured_instances"], 2)
            self.assertEqual(summary["totals"]["successful_instances"], 1)
            self.assertEqual(summary["totals"]["skipped_instances"], 1)
            self.assertEqual(summary["totals"]["written_graphs"], 1)
            skipped = summary["splits"]["train"]["skipped_instances"]
            self.assertEqual(skipped[0]["instance_name"], "always_bad")
            self.assertEqual(skipped[0]["error_type"], "RuntimeError")
            self.assertEqual(calls["always_bad"], 1)
            self.assertFalse(summary_path.with_suffix(".json.tmp").exists())

            csv_path = Path(directory) / "training" / "graphs_training.csv"
            with csv_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertIn("nonlinear_expected_repair_buffer", reader.fieldnames)
            self.assertNotIn("simulation_parameters", reader.fieldnames)
            self.assertNotIn("job_ontime_probabilities", reader.fieldnames)
            self.assertEqual(
                summary["label"]["label_method"],
                "weibull_expected_repair_buffer_v1",
            )
            self.assertNotIn("simulation", summary["label"])
            self.assertEqual(
                summary["graph"]["machine_predecessor_edge_scope"],
                "direct",
            )
            self.assertTrue(
                summary["graph"]["include_machine_predecessor_edges"]
            )
            self.assertTrue(summary["graph"]["include_job_precedence_edges"])


if __name__ == "__main__":
    unittest.main()
