import copy
import tempfile
import unittest
from pathlib import Path

import main


def _config(instance_directory, *, create_instances):
    return {
        "instances": {
            "generation": {
                "operations_per_job": [1, 1],
                "processing_times": {"base_range": [2, 4]},
                "due_dates": {
                    "method": "fjsp_lower_bound_factors",
                    "factors": [1.6],
                    "assignment": "cyclic",
                },
            }
        },
        "solve": {
            "create_instances": create_instances,
            "evaluation": {
                "random_seed": 2026,
                "instance_directory": str(instance_directory),
                "benchmark": {
                    "enabled": True,
                    "reuse_existing_instances": True,
                    "num_jobs": [2, 3],
                    "num_machines": [2],
                    "instances_per_size": 1,
                    "due_date_factors": [1.6],
                },
            },
        },
    }


class SolveInstanceGenerationTests(unittest.TestCase):
    def test_create_true_generates_exact_selected_benchmark_set(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(temporary_directory, create_instances=True)
            plan = main._generated_tier_plan(config, "benchmark")

            self.assertEqual(len(plan), 2)
            generated = sorted(
                path.name
                for path in (
                    Path(temporary_directory) / "benchmark" / "df1p60"
                ).glob("*.pkl")
            )
            self.assertEqual(generated, [
                "i2_k2_o1-1_1_benchmark_df1p60.pkl",
                "i3_k2_o1-1_1_benchmark_df1p60.pkl",
            ])

    def test_create_false_reuses_only_complete_matching_set(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(temporary_directory, create_instances=True)
            created = main._generated_tier_plan(config, "benchmark")
            config = copy.deepcopy(config)
            config["solve"]["create_instances"] = False

            reused = main._generated_tier_plan(config, "benchmark")

            self.assertEqual(created, reused)

    def test_create_false_rejects_partial_set(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(temporary_directory, create_instances=True)
            main._generated_tier_plan(config, "benchmark")
            directory = (
                Path(temporary_directory) / "benchmark" / "df1p60"
            )
            next(directory.glob("*.pkl")).unlink()
            config["solve"]["create_instances"] = False

            with self.assertRaisesRegex(ValueError, "do not match"):
                main._generated_tier_plan(config, "benchmark")


if __name__ == "__main__":
    unittest.main()
