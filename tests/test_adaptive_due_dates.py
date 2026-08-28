from __future__ import annotations

import copy
import importlib
import math
import random
import unittest
from unittest.mock import patch


instances = importlib.import_module("01_generator.instance_generator")
generator = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_fix_and_optimize_training_data"
)


class AdaptiveDueDateTests(unittest.TestCase):
    def test_offsets_are_all_used_and_keep_nominal_schedule_feasible(self):
        instance = instances.FJSPData(
            nb_instance=1,
            num_jobs=2,
            num_machines=2,
            operations_per_job_min=2,
            operations_per_job_max=2,
            num_operations=[2, 2],
            flag_save_file=False,
            random_source=random.Random(17),
        )
        instance.instance_name = "adaptive_test"
        makespan = generator._fjsp_processing_lower_bound(instance) + 3.0
        generation = {
            "random_seed": 42,
            "adaptive_due_dates": {
                "enabled": True,
                "relative_makespan_offsets": [0.02, 0.05, 0.10, 0.20],
            },
        }
        observed = []
        with patch.object(
            generator,
            "_nominal_makespan_calibration",
            return_value={"makespan": makespan, "status": 2, "gap": 0.0},
        ):
            for run_index in range(4):
                sample = copy.deepcopy(instance)
                generator._sample_training_due_dates(
                    sample,
                    random.Random(run_index),
                    generation,
                    run_index,
                )
                observed.append(sample.training_due_date_delta)
                expected_due = math.ceil(
                    (1.0 + sample.training_due_date_delta) * makespan - 1e-12
                )
                self.assertEqual(sample.training_common_due_date, expected_due)
                self.assertTrue(all(
                    due_date >= makespan
                    for due_date in sample.due_dates.values()
                ))
        self.assertEqual(
            sorted(observed), [0.02, 0.05, 0.10, 0.20]
        )


if __name__ == "__main__":
    unittest.main()
