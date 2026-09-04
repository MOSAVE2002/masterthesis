from __future__ import annotations

import importlib
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch


simulation = importlib.import_module("05_Simulation.preempt_resume")
class MachineSingleFailureSimulationTests(unittest.TestCase):
    @staticmethod
    def _independent_machine_schedule(operation_order=(1, 2)):
        return simulation.FixedSchedule(
            operations=tuple(operation_order),
            selected_machines={1: 0, 2: 1},
            processing_times={1: 8.0, 2: 8.0},
            planned_starts={1: 10.0, 2: 10.0},
            job_predecessors={1: (), 2: ()},
            machine_edges=(),
            jobs={1: (1,), 2: (2,)},
            job_end_operations={1: 1, 2: 2},
            due_dates={1: 20.0, 2: 20.0},
            weibull_scale={0: 15.0, 1: 15.0},
            weibull_shape={0: 2.0, 1: 2.0},
            repair_rate={0: 0.5, 1: 0.5},
        )

    def test_only_replication_and_seed_parameters_are_configurable(self):
        config = simulation.normalize_simulation_config({
            "pilot_replications": 32,
            "label_replications": 64,
            "random_seed": 7,
        })
        self.assertEqual(config.pilot_replications, 32)
        self.assertEqual(config.label_replications, 64)
        self.assertEqual(config.random_seed, 7)
        with self.assertRaisesRegex(ValueError, "Unknown simulation parameters"):
            simulation.normalize_simulation_config({
                "failure_clock": "calendar_time"
            })

    def test_operation_affected_probability_matches_weibull_failure_cdf(self):
        schedule = simulation.FixedSchedule(
            operations=(1,),
            selected_machines={1: 0},
            processing_times={1: 10.0},
            planned_starts={1: 0.0},
            job_predecessors={1: ()},
            machine_edges=(),
            jobs={1: (1,)},
            job_end_operations={1: 1},
            due_dates={1: 100.0},
            weibull_scale={0: 30.0},
            weibull_shape={0: 2.0},
            repair_rate={0: 0.5},
        )
        probability = 1.0 - math.exp(-((10.0 / 30.0) ** 2.0))
        result = simulation.simulate_fixed_schedule(
            schedule,
            replications=30_000,
            seed=123,
        )
        self.assertAlmostEqual(
            result.operation_failure_probabilities[0], probability, delta=0.01
        )
        self.assertEqual(
            result.label_method,
            simulation.LABEL_METHOD,
        )

    def test_one_common_downtime_is_drawn_per_used_machine(self):
        schedule = self._independent_machine_schedule((1, 2))
        with patch.object(
            simulation,
            "_draw_machine_downtime",
            side_effect=((100.0, 1.0), (100.0, 1.0)),
        ) as draw:
            simulation.simulate_fixed_schedule(
                schedule,
                replications=1,
                seed=1,
            )
        self.assertEqual(draw.call_count, 2)

    def test_machine_downtime_draw_uses_weibull_and_exponential(self):
        rng = SimpleNamespace(
            weibull=lambda shape: 2.0,
            exponential=lambda scale: 3.0,
        )
        self.assertEqual(
            simulation._draw_machine_downtime(rng, 15.0, 2.0, 0.5),
            (30.0, 3.0),
        )

    def test_preempt_resume_and_waiting_rules(self):
        self.assertEqual(
            simulation._preempt_resume_completion(0.0, 5.0, 2.0, 4.0),
            (9.0, 4.0),
        )
        self.assertEqual(
            simulation._preempt_resume_completion(3.0, 2.0, 1.0, 5.0),
            (8.0, 3.0),
        )
        self.assertEqual(
            simulation._preempt_resume_completion(7.0, 2.0, 1.0, 5.0),
            (9.0, 0.0),
        )

    def test_streams_do_not_depend_on_topological_tie_order(self):
        forward = simulation.simulate_fixed_schedule(
            self._independent_machine_schedule((1, 2)),
            replications=2_000,
            seed=123,
        )
        reverse = simulation.simulate_fixed_schedule(
            self._independent_machine_schedule((2, 1)),
            replications=2_000,
            seed=123,
        )
        self.assertEqual(
            forward.job_ontime_probabilities,
            reverse.job_ontime_probabilities,
        )
        self.assertEqual(forward.mean_failures, reverse.mean_failures)

    def test_delay_propagates_over_job_and_machine_predecessors(self):
        schedule = simulation.FixedSchedule(
            operations=(1, 2, 3),
            selected_machines={1: 0, 2: 1, 3: 0},
            processing_times={1: 1.0, 2: 1.0, 3: 1.0},
            planned_starts={1: 0.0, 2: 1.0, 3: 1.0},
            job_predecessors={1: (), 2: (1,), 3: ()},
            machine_edges=((1, 3, 0),),
            jobs={1: (1, 2), 2: (3,)},
            job_end_operations={1: 2, 2: 3},
            due_dates={1: 6.5, 2: 6.5},
            weibull_scale={0: 20.0, 1: 20.0},
            weibull_shape={0: 2.0, 1: 2.0},
            repair_rate={0: 1.0, 1: 1.0},
        )
        with patch.object(
            simulation,
            "_draw_machine_downtime",
            side_effect=((0.5, 5.0), (100.0, 1.0)),
        ):
            result = simulation.simulate_fixed_schedule(
                schedule,
                replications=1,
                seed=7,
            )
        self.assertEqual(result.job_mean_completion_times, (7.0, 7.0))
        self.assertEqual(result.job_mean_completion_delays, (5.0, 5.0))
        self.assertEqual(
            result.job_completion_delay_standard_errors, (0.0, 0.0)
        )
        self.assertEqual(result.job_ontime_probabilities, (0.0, 0.0))
        self.assertEqual(result.operation_mean_repair_delays, (5.0, 0.0, 0.0))
        self.assertEqual(result.mean_failures, 1.0)
        self.assertEqual(result.mean_total_repair_duration, 5.0)


if __name__ == "__main__":
    unittest.main()
