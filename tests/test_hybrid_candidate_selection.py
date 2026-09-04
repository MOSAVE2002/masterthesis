from __future__ import annotations

import importlib
import json
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch


generator = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_fix_and_optimize_training_data"
)


def _candidate(
    identifier,
    probabilities,
    generation_mode="unconstrained",
):
    return {
        "candidate_id": identifier,
        "pool_objective": float(identifier),
        "service_risk": 1.0 - min(probabilities),
        "min_job_probability": min(probabilities),
        "job_probabilities": list(probabilities),
        "simulated_job_expected_completion_delays": [
            float(identifier + job_index + 1)
            for job_index in range(len(probabilities))
        ],
        "candidate_generation_mode": generation_mode,
        "structure": ((identifier,), ()),
        "structure_tokens": frozenset({
            ("candidate", identifier),
            ("group", identifier % 7),
        }),
    }


class HybridCandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        probability_groups = (
            (0.10, 0.20, 0.30, 0.40),
            (0.72, 0.74, 0.76, 0.78),
            (0.81, 0.83, 0.85, 0.87),
            (0.94, 0.96, 0.98, 1.00),
        )
        self.candidates = [
            _candidate(index, probability_groups[index % 4])
            for index in range(40)
        ]
        self.config = {
            "enabled": True,
            "anchor_fraction": 0.40,
            "anchor_ratios": {
                "good_nominal_objective": 0.20,
                "low_expected_completion_delay": 0.20,
                "high_expected_completion_delay": 0.20,
                "pareto_tradeoff": 0.20,
                "structurally_diverse": 0.20,
            },
            "job_probability_target_ratios": {
                "low": 0.20,
                "boundary_below": 0.30,
                "boundary_above": 0.30,
                "high": 0.20,
            },
        }

    def test_keeps_quality_anchors_and_selects_unique_candidates(self):
        selected = generator._select_candidates(
            self.candidates,
            25,
            None,
            0.80,
            0.10,
            hybrid_selection=self.config,
        )
        categories = Counter(entry["category"] for entry in selected)
        identifiers = {
            entry["candidate"]["candidate_id"] for entry in selected
        }

        self.assertEqual(len(selected), 25)
        self.assertEqual(len(identifiers), 25)
        self.assertEqual(categories["good_nominal_objective"], 2)
        self.assertEqual(categories["low_expected_completion_delay"], 2)
        self.assertEqual(categories["high_expected_completion_delay"], 2)
        self.assertEqual(categories["pareto_tradeoff"], 2)
        self.assertEqual(categories["structurally_diverse"], 2)
        self.assertIn(0, identifiers)
        self.assertIn(3, identifiers)

    def test_remaining_candidates_balance_job_probability_bins(self):
        selected = generator._select_candidates(
            self.candidates,
            25,
            None,
            0.80,
            0.10,
            hybrid_selection=self.config,
        )
        counts = generator._selection_probability_counts(
            selected, 0.80, 0.10
        )

        self.assertEqual(sum(counts.values()), 100)
        self.assertGreaterEqual(counts["boundary_below"], 28)
        self.assertGreaterEqual(counts["boundary_above"], 28)
        self.assertEqual(
            sum(
                entry["category"].startswith("job_probability_")
                for entry in selected
            ),
            15,
        )

    def test_delay_rankings_select_low_medium_and_high_targets(self):
        ratios = {
            name: float(name in {
                "low_expected_completion_delay",
                "medium_expected_completion_delay",
                "high_expected_completion_delay",
            })
            for name in generator.DEFAULT_SELECTION_RATIOS
        }
        selected = generator._select_candidates(
            self.candidates,
            3,
            ratios,
            0.80,
            0.10,
            hybrid_selection=None,
        )
        by_category = {
            entry["category"]: entry["candidate"]["candidate_id"]
            for entry in selected
        }
        self.assertEqual(
            by_category["low_expected_completion_delay"], 0
        )
        self.assertIn(
            by_category["medium_expected_completion_delay"], {19, 20}
        )
        self.assertEqual(
            by_category["high_expected_completion_delay"], 39
        )

    def test_hybrid_selection_is_limited_to_configured_splits(self):
        fixed = {
            "hybrid_selection": {
                **self.config,
                "apply_to_splits": ["training", "valid"],
            }
        }

        self.assertIsNotNone(
            generator._hybrid_selection_for_split(fixed, "training")
        )
        self.assertIsNotNone(
            generator._hybrid_selection_for_split(fixed, "valid")
        )
        self.assertIsNone(
            generator._hybrid_selection_for_split(fixed, "test")
        )

    def test_mixed_generation_is_limited_to_configured_splits(self):
        fixed = {
            "mixed_candidate_generation": {
                "enabled": True,
                "apply_to_splits": ["training", "valid"],
                "nominal_selected_fraction": 0.50,
                "nominal_minimum_pool_runs": 2,
                "nominal_maximum_pool_runs": 4,
            }
        }

        training = generator._mixed_candidate_generation_for_split(
            fixed, "training"
        )
        self.assertEqual(training["nominal_selected_fraction"], 0.50)
        self.assertIsNotNone(
            generator._mixed_candidate_generation_for_split(fixed, "valid")
        )
        self.assertIsNone(
            generator._mixed_candidate_generation_for_split(fixed, "test")
        )

    def test_mixed_selection_uses_requested_fraction(self):
        unconstrained = [
            _candidate(index, (0.10, 0.50, 0.99))
            for index in range(20)
        ]
        nominal = [
            _candidate(
                100 + index,
                (0.92, 0.95, 0.98),
                "nominal_ontime",
            )
            for index in range(20)
        ]

        selected = generator._select_mixed_candidates(
            unconstrained,
            nominal,
            10,
            None,
            0.95,
            0.03,
            self.config,
            0.50,
        )

        modes = Counter(
            entry["candidate"]["candidate_generation_mode"]
            for entry in selected
        )
        self.assertEqual(modes["unconstrained"], 5)
        self.assertEqual(modes["nominal_ontime"], 5)
        self.assertEqual(
            len({entry["candidate"]["structure"] for entry in selected}),
            10,
        )

    def test_mixed_selection_falls_back_to_unconstrained_pool(self):
        unconstrained = [
            _candidate(index, (0.10, 0.50, 0.99))
            for index in range(20)
        ]
        nominal = [
            _candidate(
                100 + index,
                (0.92, 0.95, 0.98),
                "nominal_ontime",
            )
            for index in range(3)
        ]

        selected = generator._select_mixed_candidates(
            unconstrained,
            nominal,
            10,
            None,
            0.95,
            0.03,
            self.config,
            0.50,
        )

        modes = Counter(
            entry["candidate"]["candidate_generation_mode"]
            for entry in selected
        )
        self.assertEqual(modes["nominal_ontime"], 3)
        self.assertEqual(modes["unconstrained"], 7)

    def test_nominal_only_generation_skips_unconstrained_pool(self):
        generation = {
            "fixed_y": {
                "mixed_candidate_generation": {
                    "enabled": True,
                    "apply_to_splits": ["training", "valid", "test"],
                    "nominal_selected_fraction": 1.0,
                    "nominal_minimum_pool_runs": 2,
                    "nominal_maximum_pool_runs": 4,
                },
                "service_boundary_width": 0.03,
            }
        }
        nominal = [
            _candidate(
                100 + index,
                (0.92, 0.95, 0.98),
                "nominal_ontime",
            )
            for index in range(8)
        ]

        with patch.object(
            generator,
            "_collect_instance_candidates",
            return_value=nominal,
        ) as collect:
            candidates, selected = (
                generator._collect_and_select_instance_candidates(
                    object(),
                    "instance",
                    generation,
                    SimpleNamespace(service_level=0.95),
                    4,
                    split="training",
                )
            )

        self.assertEqual(collect.call_count, 1)
        self.assertEqual(
            collect.call_args.kwargs["candidate_generation_mode"],
            "nominal_ontime",
        )
        self.assertEqual(len(candidates), 8)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(
            entry["candidate"]["candidate_generation_mode"]
            == "nominal_ontime"
            for entry in selected
        ))

    def test_single_nonlinear_mode_is_used_for_every_candidate_pool(self):
        generation = {
            "fixed_y": {
                "candidate_generation_mode": "nonlinear",
                "service_boundary_width": 0.03,
            }
        }
        nonlinear = [
            _candidate(
                200 + index,
                (0.94, 0.95, 0.96),
                "nonlinear",
            )
            for index in range(8)
        ]

        with patch.object(
            generator,
            "_collect_instance_candidates",
            return_value=nonlinear,
        ) as collect:
            candidates, selected = (
                generator._collect_and_select_instance_candidates(
                    object(),
                    "instance",
                    generation,
                    SimpleNamespace(service_level=0.95),
                    4,
                    split="test",
                )
            )

        self.assertEqual(collect.call_count, 1)
        self.assertEqual(
            collect.call_args.kwargs["candidate_generation_mode"],
            "nonlinear",
        )
        self.assertEqual(len(candidates), 8)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(
            entry["candidate"]["candidate_generation_mode"] == "nonlinear"
            for entry in selected
        ))

    def test_nonlinear_builder_uses_reference_formulation_and_graph_state(self):
        model = SimpleNamespace(update=Mock())
        variables = {"Y": object(), "X": object()}
        instance = object()
        graph_config = object()

        with (
            patch.object(generator.gp, "Model", return_value=model),
            patch.object(
                generator._nonlinear,
                "build_fjsp",
                return_value=(model, variables),
            ) as build_nonlinear,
            patch.object(
                generator,
                "add_reliability_graph_variables",
            ) as add_graph_state,
        ):
            result = generator._build_candidate_model(
                instance,
                graph_config,
                "nonlinear",
            )

        self.assertEqual(result, (model, variables))
        build_nonlinear.assert_called_once_with(
            model,
            instance,
            reliability_graph_config=graph_config,
            service_probability_band=None,
        )
        add_graph_state.assert_called_once_with(
            model, variables, instance, graph_config
        )
        model.update.assert_called_once()

    def test_fixed_schedule_uses_nonlinear_markov_reference_evaluation(self):
        schedule = generator.FixedSchedule(
            operations=(0,),
            selected_machines={0: 0},
            processing_times={0: 10.0},
            planned_starts={0: 5.0},
            job_predecessors={0: ()},
            machine_edges=(),
            jobs={0: (0,)},
            job_end_operations={0: 0},
            due_dates={0: 25.0},
            weibull_scale={0: 30.0},
            weibull_shape={0: 2.0},
            repair_rate={0: 0.5},
        )
        candidate = {
            "simulation_schedule": schedule,
            "row": {},
        }

        with patch.object(
            generator, "weibull_down_probability", return_value=0.2
        ) as probability:
            generator._evaluate_fixed_schedule_nonlinear(
                candidate,
                SimpleNamespace(
                    quadrature_points=12,
                    service_scope="job",
                ),
            )

        probability.assert_called_once_with(
            10.0, 30.0, 2.0, 0.5, order=12
        )
        self.assertAlmostEqual(
            candidate["nonlinear_job_probabilities"][0], 0.96
        )
        self.assertAlmostEqual(
            candidate["nonlinear_min_job_probability"], 0.96
        )
        self.assertNotIn(
            "nonlinear_job_ontime_probability_lbs", candidate["row"]
        )

    def test_fixed_schedule_without_positive_slack_has_zero_bound(self):
        schedule = generator.FixedSchedule(
            operations=(0,),
            selected_machines={0: 0},
            processing_times={0: 10.0},
            planned_starts={0: 5.0},
            job_predecessors={0: ()},
            machine_edges=(),
            jobs={0: (0,)},
            job_end_operations={0: 0},
            due_dates={0: 15.0},
            weibull_scale={0: 30.0},
            weibull_shape={0: 2.0},
            repair_rate={0: 0.5},
        )
        candidate = {"simulation_schedule": schedule, "row": {}}

        with patch.object(
            generator, "weibull_down_probability", return_value=0.2
        ):
            generator._evaluate_fixed_schedule_nonlinear(
                candidate,
                SimpleNamespace(
                    quadrature_points=12,
                    service_scope="job",
                ),
            )

        self.assertEqual(candidate["nonlinear_job_probabilities"], [0.0])

    def test_fixed_schedule_uses_propagated_simulation_delay_as_gnn_label(self):
        schedule = generator.FixedSchedule(
            operations=(0,),
            selected_machines={0: 0},
            processing_times={0: 10.0},
            planned_starts={0: 5.0},
            job_predecessors={0: ()},
            machine_edges=(),
            jobs={0: (0,)},
            job_end_operations={0: 0},
            due_dates={0: 25.0},
            weibull_scale={0: 30.0},
            weibull_shape={0: 2.0},
            repair_rate={0: 0.5},
        )
        candidate = {"simulation_schedule": schedule, "row": {}}
        result = SimpleNamespace(
            job_mean_completion_delays=(4.5,),
            job_ontime_probabilities=(0.875,),
            job_completion_delay_standard_errors=(0.25,),
            replications=8,
        )
        with patch.object(
            generator._simulation,
            "simulate_fixed_schedule",
            return_value=result,
        ) as simulate:
            generator._evaluate_fixed_schedule_simulation(
                candidate,
                {"label_replications": 8, "random_seed": 7},
            )

        simulate.assert_called_once()
        self.assertEqual(
            candidate["row"][generator.TARGET_COLUMN], "[4.5]"
        )
        self.assertEqual(candidate["job_probabilities"], [0.875])
        self.assertEqual(candidate["min_job_probability"], 0.875)
        self.assertEqual(candidate["row"]["simulation_replications"], 8)

    def test_candidate_uses_stored_gurobi_timing(self):
        def variable(value):
            return SimpleNamespace(Xn=float(value))

        instance = SimpleNamespace(
            jobs={1: (1,)},
            predecessors={1: ()},
            job_end_operations={1: 1},
            due_dates={1: 12.0},
            processing_times={(1, 0): 3.0},
        )
        variables = {
            "real_operations": (1,),
            "eligible_machines": {1: (0,)},
            "Y": {(1, 0): variable(1.0)},
            "U_index": (),
            "U": {},
            "C": {1: variable(9.0)},
            "weibull_alpha": {0: 24.0},
            "weibull_beta": {0: 2.0},
            "repair_rate": {0: 0.5},
        }
        model = SimpleNamespace(PoolObjVal=12.0, Runtime=0.1)

        candidate = generator._candidate_from_solution(
            model,
            variables,
            instance,
            generator.normalize_reliability_graph_config(None),
            solution_number=0,
            run_index=0,
            fix_stats={},
            candidate_generation_mode="nonlinear_evaluated",
        )

        self.assertEqual(
            candidate["row"]["schedule_timing_method"],
            "gurobi_solution",
        )
        self.assertEqual(
            candidate["simulation_schedule"].planned_starts[1], 6.0
        )
        self.assertEqual(
            json.loads(candidate["row"]["gnn_node_features"])[0][:2],
            [0.5, 0.75],
        )

    def test_candidate_uses_direct_u_machine_edges_for_gnn(self):
        def variable(value):
            return SimpleNamespace(Xn=float(value))

        operations = (1, 2, 3)
        directed = tuple(
            (source, target, 0)
            for source in operations
            for target in operations
            if source != target
        )
        instance = SimpleNamespace(
            jobs={1: operations},
            predecessors={1: (), 2: (1,), 3: (2,)},
            job_end_operations={1: 3},
            due_dates={1: 20.0},
            due_date_factor=1.70,
            processing_times={(operation, 0): 2.0 for operation in operations},
        )
        variables = {
            "real_operations": operations,
            "eligible_machines": {operation: (0,) for operation in operations},
            "Y": {(operation, 0): variable(1.0) for operation in operations},
            "U_index": directed,
            "U": {
                edge: variable(edge in {(1, 2, 0), (2, 3, 0)})
                for edge in directed
            },
            "C": {
                1: variable(2.0),
                2: variable(4.0),
                3: variable(6.0),
            },
            "weibull_alpha": {0: 24.0},
            "weibull_beta": {0: 2.0},
            "repair_rate": {0: 0.5},
        }
        candidate = generator._candidate_from_solution(
            SimpleNamespace(PoolObjVal=12.0, Runtime=0.1),
            variables,
            instance,
            generator.normalize_reliability_graph_config(None),
            solution_number=0,
            run_index=0,
            fix_stats={},
        )

        self.assertEqual(
            candidate["simulation_schedule"].machine_edges,
            ((1, 2, 0), (2, 3, 0)),
        )
        self.assertEqual(
            json.loads(candidate["row"]["gnn_active_edges"]),
            [[0, 1], [1, 2], [0, 1], [1, 2]],
        )
        self.assertEqual(candidate["row"]["due_date_factor"], 1.70)

    def test_hybrid_selection_can_balance_minimum_job_probability(self):
        config = {
            **self.config,
            "anchor_fraction": 0.0,
            "probability_target_basis": "min_job_probability",
            "job_probability_target_ratios": {
                "low": 0.25,
                "boundary_below": 0.25,
                "boundary_above": 0.25,
                "high": 0.25,
            },
        }

        selected = generator._select_candidates(
            self.candidates,
            8,
            None,
            0.80,
            0.10,
            hybrid_selection=config,
        )
        counts = Counter(
            generator._probability_bin(
                entry["candidate"]["min_job_probability"],
                0.80,
                0.10,
            )
            for entry in selected
        )

        self.assertEqual(
            counts,
            Counter({
                "low": 2,
                "boundary_below": 2,
                "boundary_above": 2,
                "high": 2,
            }),
        )

    def test_probability_band_limits_follow_service_boundary(self):
        below = generator._probability_band_limits(
            "boundary_below", 0.95, 0.03
        )
        self.assertAlmostEqual(below[0], 0.92)
        self.assertAlmostEqual(below[1], 0.95)
        self.assertEqual(
            generator._probability_band_limits(
                "boundary_above", 0.95, 0.03
            ),
            (0.95, 0.98),
        )

    def test_final_job_probability_coverage_checks_individual_labels(self):
        fixed = {
            "hybrid_selection": {
                "enabled": True,
                "final_job_probability_coverage": {
                    "apply_to_splits": ["training"],
                    "minimum_ratios": {
                        "boundary_below": 0.20,
                        "boundary_above": 0.20,
                    },
                },
            },
        }
        result = generator._validate_final_job_probability_coverage(
            fixed,
            "training",
            Counter({
                "low": 2,
                "boundary_below": 3,
                "boundary_above": 3,
                "high": 2,
            }),
        )
        self.assertEqual(result["labels"], 10)
        with self.assertRaisesRegex(RuntimeError, "boundary_below"):
            generator._validate_final_job_probability_coverage(
                fixed,
                "training",
                Counter({
                    "low": 5,
                    "boundary_below": 1,
                    "boundary_above": 3,
                    "high": 1,
                }),
            )

    def test_neighborhood_fixes_only_machine_assignments(self):
        class Variable:
            def __init__(self, value):
                self.X = float(value)
                self.lb = 0.0
                self.ub = 1.0

        selected = Variable(1.0)
        rejected = Variable(0.0)
        variables = {
            "Y": {(0, 0): selected, (0, 1): rejected},
        }
        instance = SimpleNamespace(
            real_operations=[0],
            eligible_machines={0: [0, 1]},
        )
        model = SimpleNamespace(update=Mock())

        stats = generator._fix_neighborhood(
            model,
            variables,
            instance,
            generator.random.Random(42),
            1.0,
            use_incumbent=True,
        )

        self.assertEqual(stats["fixed_operations"], 1)
        self.assertEqual(selected.lb, 1.0)
        self.assertEqual(selected.ub, 1.0)
        self.assertEqual(rejected.lb, 0.0)
        self.assertEqual(rejected.ub, 0.0)
        model.update.assert_called_once()

    def test_probability_bin_boundaries_follow_service_level(self):
        self.assertEqual(
            generator._probability_bin(0.70, 0.80, 0.10),
            "boundary_below",
        )
        self.assertEqual(
            generator._probability_bin(0.80, 0.80, 0.10),
            "boundary_above",
        )
        self.assertEqual(
            generator._probability_bin(0.90, 0.80, 0.10),
            "boundary_above",
        )

    def test_candidate_collection_uses_adaptive_maximum(self):
        generation = {
            "fixed_y": {
                "pool_candidates": 3,
                "minimum_candidate_pool_runs": 1,
                "fix_ratios": [0.25],
                "service_boundary_width": 0.10,
            },
            "random_seed": 42,
        }
        hybrid = {
            **self.config,
            "maximum_candidate_pool_runs": 5,
        }

        def generate_low_candidates(
            _instance,
            _instance_name,
            run_index,
            *_args,
        ):
            return [
                _candidate(run_index * 10 + offset, (0.10, 0.20, 0.30))
                for offset in range(3)
            ]

        with patch.object(
            generator, "_run_neighborhood", side_effect=generate_low_candidates
        ) as run:
            candidates = generator._collect_instance_candidates(
                object(),
                "instance",
                generation,
                SimpleNamespace(service_level=0.80),
                5,
                hybrid_selection=hybrid,
            )

        self.assertEqual(run.call_count, 5)
        self.assertEqual(len(candidates), 15)

    def test_nominal_constraints_cover_every_job_end_operation(self):
        class CompletionVariable:
            def __init__(self, operation):
                self.operation = operation

            def __le__(self, due_date):
                return self.operation, float(due_date)

        class Model:
            def __init__(self):
                self.constraints = []
                self.updated = False

            def addConstr(self, expression, name):
                self.constraints.append((expression, name))

            def update(self):
                self.updated = True

        model = Model()
        instance = SimpleNamespace(
            job_end_operations={0: 7, 1: 11},
            due_dates={0: 20.0, 1: 31.0},
        )
        variables = {
            "C": {
                7: CompletionVariable(7),
                11: CompletionVariable(11),
            }
        }

        generator._add_nominal_ontime_constraints(
            model, variables, instance
        )

        self.assertTrue(model.updated)
        self.assertEqual(
            model.constraints,
            [
                ((7, 20.0), "candidate_nominal_ontime[0]"),
                ((11, 31.0), "candidate_nominal_ontime[1]"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
