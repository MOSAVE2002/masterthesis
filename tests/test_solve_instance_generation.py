import copy
import importlib
import pickle
import tempfile
import unittest
from pathlib import Path

import main


evaluation = importlib.import_module("06_Evaluation.evaluate_solutions")


def _config(instance_directory, *, create_instances):
    return {
        "instances": {
            "generation": {
                "operations_per_job": [1, 1],
                "processing_times": {"base_range": [2, 4]},
                "due_dates": {
                    "method": "total_work_content",
                    "machine_aggregation": "mean",
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
                    "due_dates": {
                        "method": "calibrated_total_work_content",
                        "machine_aggregation": "mean",
                        "relative_makespan_offsets": [0.0, 0.15, 0.30],
                        "time_limit_seconds": 2,
                        "mip_gap": 0.01,
                        "output_flag": 0,
                        "seed": 42,
                    },
                },
            },
        },
    }


class SolveInstanceGenerationTests(unittest.TestCase):
    def test_create_true_generates_exact_selected_benchmark_set(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(temporary_directory, create_instances=True)
            plan = main._generated_tier_plan(config, "benchmark")

            self.assertEqual(len(plan), 6)
            expected_offsets = [0.0, 0.15, 0.30]
            self.assertEqual(
                sorted({
                    item["relative_makespan_offset"] for item in plan
                }),
                expected_offsets,
            )
            instances_by_offset = {}
            for offset, slug in (
                (0.0, "twk_d0p00"),
                (0.15, "twk_d0p15"),
                (0.30, "twk_d0p30"),
            ):
                directory = Path(temporary_directory) / "benchmark" / slug
                generated = sorted(
                    path.name for path in directory.glob("*.pkl")
                )
                self.assertEqual(generated, [
                    f"i2_k2_o1-1_1_benchmark_{slug}.pkl",
                    f"i3_k2_o1-1_1_benchmark_{slug}.pkl",
                ])
                instances_by_offset[offset] = (
                    main._instances.load_generated_instance(
                        generated[0].removesuffix(".pkl"), directory
                    )
                )
            instance = instances_by_offset[0.0]
            self.assertEqual(
                instance.due_date_method,
                "calibrated_total_work_content",
            )
            self.assertEqual(instance.due_date_relative_makespan_offset, 0.0)
            self.assertAlmostEqual(
                instance.due_date_factor,
                instance.nominal_twk_due_date_factor,
            )
            baseline = instances_by_offset[0.0]
            for offset, candidate in instances_by_offset.items():
                self.assertEqual(
                    candidate.physical_instance_id,
                    baseline.physical_instance_id,
                )
                for attribute in (
                    "jobs",
                    "processing_times",
                    "eligible_machines",
                    "machine_speed",
                    "machine_cost",
                    "machine_profiles",
                    "weibull_alpha",
                    "weibull_beta",
                    "repair_rate",
                    "repair_duration",
                ):
                    if hasattr(baseline, attribute):
                        self.assertEqual(
                            getattr(candidate, attribute),
                            getattr(baseline, attribute),
                        )
                self.assertEqual(
                    candidate.nominal_makespan_calibration,
                    baseline.nominal_makespan_calibration,
                )
                self.assertAlmostEqual(
                    candidate.due_date_factor,
                    (1.0 + offset) * baseline.nominal_twk_due_date_factor,
                )
            seeds = {
                evaluation._evaluation_seed(
                    config["solve"]["evaluation"]["random_seed"],
                    candidate.physical_instance_id,
                )
                for candidate in instances_by_offset.values()
            }
            self.assertEqual(len(seeds), 1)

    def test_legacy_due_date_variant_names_share_physical_seed_key(self):
        first = "i3_k3_o3-5_1_benchmark_twk_d0p00"
        second = "i3_k3_o3-5_1_benchmark_twk_d0p30"
        instance = type("LegacyInstance", (), {})()

        first_key = evaluation._physical_instance_id(instance, first)
        second_key = evaluation._physical_instance_id(instance, second)

        self.assertEqual(first_key, "benchmark/i3_k3_o3-5_1")
        self.assertEqual(first_key, second_key)
        self.assertEqual(
            evaluation._evaluation_seed(2026, first_key),
            evaluation._evaluation_seed(2026, second_key),
        )

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
                Path(temporary_directory) / "benchmark" / "twk_d0p00"
            )
            next(directory.glob("*.pkl")).unlink()
            config["solve"]["create_instances"] = False

            with self.assertRaisesRegex(ValueError, "do not match"):
                main._generated_tier_plan(config, "benchmark")

    def test_reuse_rejects_mismatching_calibration_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(temporary_directory, create_instances=True)
            plan = main._generated_tier_plan(config, "benchmark")
            first = plan[0]
            instance = main._instances.load_generated_instance(
                first["instance_name"], first["instance_directory"]
            )
            instance.due_date_relative_makespan_offset = 0.15
            path = (
                Path(first["instance_directory"])
                / f"{first['instance_name']}.pkl"
            )
            with path.open("wb") as output_file:
                pickle.dump(instance, output_file)
            config["solve"]["create_instances"] = False

            with self.assertRaisesRegex(ValueError, "calibrated TWK"):
                main._generated_tier_plan(config, "benchmark")

    def test_calibrated_offsets_reject_scalar_and_list_combination(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(temporary_directory, create_instances=True)
            due_dates = config["solve"]["evaluation"]["benchmark"][
                "due_dates"
            ]
            due_dates["relative_makespan_offset"] = 0.0

            with self.assertRaisesRegex(ValueError, "either"):
                main._generated_tier_plan(config, "benchmark")


if __name__ == "__main__":
    unittest.main()
