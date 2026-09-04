from __future__ import annotations

import copy
from collections import Counter, defaultdict
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
    @staticmethod
    def _balanced_generation():
        return {
            "adaptive_due_dates": {
                "enabled": True,
                "relative_makespan_offsets": [0.00, 0.15, 0.30, 0.45],
                "candidate_selection": {
                    "ensure_all_categories": True,
                    "rotate_repeated_categories": True,
                    "rotate_surplus_offsets": True,
                    "prefer_unique_structures": True,
                },
            },
            "fixed_y": {},
        }

    @staticmethod
    def _candidate_pool():
        candidates = []
        for offset_index, offset in enumerate([0.00, 0.15, 0.30, 0.45]):
            for index in range(12):
                probability = 0.20 + 0.06 * index
                structure = ("structure", offset_index, index)
                candidates.append({
                    "candidate_id": (offset_index, index),
                    "structure": structure,
                    "structure_tokens": frozenset({
                        ("machine", offset_index, index),
                        ("sequence", index % 5),
                    }),
                    "pool_objective": float(12 - index + offset_index),
                    "service_risk": float(1.0 - probability),
                    "job_probabilities": [probability, min(1.0, probability + 0.04)],
                    "simulated_job_expected_completion_delays": [
                        float(index + 1),
                        float(2 * index + offset_index + 1),
                    ],
                    "min_job_probability": probability,
                    "row": {"training_due_date_delta": offset},
                })
        return candidates

    def _select(self, selection_index):
        return generator._select_adaptive_due_date_candidates(
            copy.deepcopy(self._candidate_pool()),
            10,
            generator.DEFAULT_SELECTION_RATIOS,
            0.50,
            0.25,
            self._balanced_generation(),
            selection_index,
        )

    def test_ten_graphs_cover_all_eight_categories_per_instance(self):
        expected = set(generator.ADAPTIVE_SELECTION_CATEGORIES)
        for selection_index in range(8):
            selected = self._select(selection_index)
            categories = [
                entry["category"].split(":", 1)[1]
                for entry in selected
            ]
            self.assertEqual(len(selected), 10)
            self.assertEqual(set(categories), expected)
            self.assertEqual(
                sorted(Counter(categories).values()),
                [1, 1, 1, 1, 1, 1, 2, 2],
            )

    def test_repeated_categories_and_surplus_offsets_rotate_evenly(self):
        category_counts = Counter()
        for selection_index in range(8):
            category_counts.update(
                entry["category"].split(":", 1)[1]
                for entry in self._select(selection_index)
            )
        self.assertEqual(set(category_counts.values()), {10})

        offset_counts = Counter()
        for selection_index in range(4):
            selected = self._select(selection_index)
            per_instance = Counter(
                entry["candidate"]["row"]["training_due_date_delta"]
                for entry in selected
            )
            self.assertEqual(sorted(per_instance.values()), [2, 2, 3, 3])
            offset_counts.update(per_instance)
        self.assertEqual(set(offset_counts.values()), {10})

    def test_categories_are_not_coupled_to_one_due_date_offset(self):
        observed_offsets = defaultdict(set)
        for selection_index in range(28):
            for entry in self._select(selection_index):
                category = entry["category"].split(":", 1)[1]
                observed_offsets[category].add(
                    entry["candidate"]["row"]["training_due_date_delta"]
                )
        self.assertTrue(all(
            offsets == {0.00, 0.15, 0.30, 0.45}
            for offsets in observed_offsets.values()
        ))

    def test_category_coverage_requires_at_least_eight_graphs(self):
        with self.assertRaisesRegex(ValueError, "at least 8"):
            generator._select_adaptive_due_date_candidates(
                self._candidate_pool(),
                7,
                generator.DEFAULT_SELECTION_RATIOS,
                0.50,
                0.25,
                self._balanced_generation(),
            )

    def test_offsets_are_all_used_for_job_specific_calibrated_twk_dates(self):
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
        makespan = 100.0
        generation = {
            "random_seed": 42,
            "adaptive_due_dates": {
                "enabled": True,
                "method": "calibrated_total_work_content",
                "machine_aggregation": "mean",
                "relative_makespan_offsets": [0.00, 0.15, 0.30, 0.45],
            },
        }
        work_content = generator._total_work_content_by_job(instance)
        nominal_factor = makespan / (
            sum(work_content.values()) / len(work_content)
        )
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
                effective_factor = (
                    (1.0 + sample.training_due_date_delta) * nominal_factor
                )
                expected = {
                    job: float(math.ceil(
                        effective_factor * work_content[job] - 1e-12
                    ))
                    for job in instance.jobs
                }
                self.assertEqual(sample.due_dates, expected)
                self.assertEqual(sample.training_due_dates, expected)
                self.assertAlmostEqual(
                    sample.nominal_twk_due_date_factor, nominal_factor
                )
                self.assertAlmostEqual(
                    sample.training_effective_due_date_factor,
                    effective_factor,
                )
                self.assertAlmostEqual(
                    sample.training_mean_due_date,
                    sum(expected.values()) / len(expected),
                )
        self.assertEqual(
            sorted(observed), [0.00, 0.15, 0.30, 0.45]
        )

    def test_offsets_must_keep_effective_twk_factor_positive(self):
        instance = instances.FJSPData(
            nb_instance=1,
            num_jobs=2,
            num_machines=2,
            operations_per_job_min=2,
            operations_per_job_max=2,
            num_operations=[2, 2],
            flag_save_file=False,
            random_source=random.Random(21),
        )
        generation = {
            "adaptive_due_dates": {
                "enabled": True,
                "relative_makespan_offsets": [-1.0],
            },
        }
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            generator._sample_training_due_dates(
                instance, random.Random(1), generation, 0
            )

    def test_weibull_scale_factors_rotate_reproducibly_per_instance(self):
        generation = {
            "random_seed": 42,
            "weibull_scale_factors": [0.8, 1.0, 1.2],
        }
        base_instance = instances.FJSPData(
            nb_instance=1,
            num_jobs=2,
            num_machines=2,
            operations_per_job_min=2,
            operations_per_job_max=2,
            num_operations=[2, 2],
            flag_save_file=False,
            random_source=random.Random(31),
        )
        base_instance.instance_name = "scale_rotation_test"
        generator.ensure_stochastic_parameters(base_instance)
        base_alpha = dict(base_instance.weibull_alpha)
        factors = []
        for run_index in range(6):
            instance = copy.deepcopy(base_instance)
            generator._sample_reliability(
                instance,
                random.Random(run_index),
                generation,
                run_index,
            )
            factors.append(instance.training_weibull_scale_factor)
            for machine, alpha in base_alpha.items():
                self.assertAlmostEqual(
                    instance.weibull_alpha[machine],
                    alpha * instance.training_weibull_scale_factor,
                )

        self.assertEqual(
            sorted(factors[:3]), [0.8, 1.0, 1.2]
        )
        self.assertEqual(factors[:3], factors[3:])


if __name__ == "__main__":
    unittest.main()
