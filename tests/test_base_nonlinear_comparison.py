import importlib
import unittest


comparison = importlib.import_module(
    "06_Evaluation.compare_base_nonlinear"
)


class BaseNonlinearComparisonTests(unittest.TestCase):
    def test_project_uses_makespan_relative_offsets(self):
        import json

        config = json.loads(comparison.CONFIG_PATH.read_text(encoding="utf-8"))
        adaptive = config["training"]["data_generation"][
            "adaptive_due_dates"
        ]
        self.assertNotIn("relative_lower_bound_offsets", adaptive)
        self.assertEqual(
            adaptive["relative_makespan_offsets"],
            [0.00, 0.15, 0.30, 0.45],
        )

    def test_controlled_due_date_is_relative_to_nominal_makespan(self):
        self.assertEqual(comparison.controlled_due_date(45.0, 0.02), 46.0)
        self.assertEqual(comparison.controlled_due_date(45.0, 0.05), 48.0)
        self.assertEqual(comparison.controlled_due_date(45.0, 0.10), 50.0)
        self.assertEqual(comparison.controlled_due_date(45.0, 0.20), 54.0)
        self.assertEqual(comparison.controlled_due_date(45.0, -0.10), 41.0)

    def test_controlled_due_date_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            comparison.controlled_due_date(0.0, 0.10)
        with self.assertRaises(ValueError):
            comparison.controlled_due_date(45.0, -1.0)

    def test_buffer_alignment_is_unscaled(self):
        rows = []
        for instance_index in range(4):
            for offset in (0.02, 0.10):
                buffer = float(instance_index + 1)
                rows.append({
                    "instance_name": f"instance_{instance_index}",
                    "relative_due_date_offset": offset,
                    "formulation": "nonlinear",
                    "expected_repair_buffer": buffer,
                    "nominal_completion": 10.0,
                    "mc_mean_completion": 10.0 + buffer,
                })
        result = comparison._buffer_alignment(rows)
        self.assertEqual(len(result["instances"]), 4)
        self.assertAlmostEqual(result["overall"]["mae"], 0.0)
        self.assertNotIn("gamma", result["overall"])


if __name__ == "__main__":
    unittest.main()
