from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch


simulation = importlib.import_module("05_Simulation.preempt_resume")
stochastic = importlib.import_module("helper.stochastic_fjsp")


class MidpointSnapshotSimulationTests(unittest.TestCase):
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

    def test_mc_mean_matches_nonlinear_expected_repair_delay(self):
        schedule = simulation.FixedSchedule(
            operations=(1,),
            selected_machines={1: 0},
            processing_times={1: 10.0},
            planned_starts={1: 15.0},
            job_predecessors={1: ()},
            machine_edges=(),
            jobs={1: (1,)},
            job_end_operations={1: 1},
            due_dates={1: 100.0},
            weibull_scale={0: 30.0},
            weibull_shape={0: 2.0},
            repair_rate={0: 0.5},
        )
        probability = stochastic.weibull_down_probability(
            20.0, 30.0, 2.0, 0.5
        )
        result = simulation.simulate_fixed_schedule(
            schedule,
            replications=30_000,
            seed=123,
        )
        self.assertAlmostEqual(
            result.operation_failure_probabilities[0], probability, delta=0.01
        )
        self.assertAlmostEqual(
            result.operation_mean_repair_delays[0],
            probability / 0.5,
            delta=0.04,
        )
        self.assertEqual(
            result.label_method,
            simulation.LABEL_METHOD,
        )

    def test_probability_uses_nominal_operation_midpoint(self):
        schedule = self._independent_machine_schedule((1, 2))
        calls = []

        def probability(t, alpha, beta, repair_rate):
            calls.append((t, alpha, beta, repair_rate))
            return 0.0

        with patch.object(
            simulation, "weibull_down_probability", side_effect=probability
        ):
            simulation.simulate_fixed_schedule(
                schedule,
                replications=1,
                seed=1,
            )
        self.assertEqual(calls, [
            (14.0, 15.0, 2.0, 0.5),
            (14.0, 15.0, 2.0, 0.5),
        ])

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
        disruptions = iter([(1, 5.0), (0, 0.0), (0, 0.0)])
        with patch.object(
            simulation,
            "_midpoint_disruption",
            side_effect=lambda *_args: next(disruptions),
        ):
            result = simulation.simulate_fixed_schedule(
                schedule,
                replications=1,
                seed=7,
            )
        self.assertEqual(result.job_mean_completion_times, (7.0, 7.0))
        self.assertEqual(result.job_ontime_probabilities, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
