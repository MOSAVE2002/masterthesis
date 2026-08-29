import importlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
instances = importlib.import_module("01_generator.instance_generator")
training = importlib.import_module(
    "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
)
sequence_setup = importlib.import_module("helper.sequence_setup")


class ServiceLevelConfigurationTests(unittest.TestCase):
    def test_active_pipeline_uses_cost_and_repair_buffer_formulation(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(
            config["objective"],
            {
                "type": (
                    "minimize_processing_plus_operating_plus_tardiness_cost"
                ),
                "facility_cost_per_time": 1.0,
                "tardiness_cost_per_time": 1.0,
            },
        )
        profiles = config["instances"]["generation"]["machine_profiles"]
        self.assertIn("cost_rate", profiles["profiles"]["old"])
        self.assertIn("cost_rate", profiles["profiles"]["new"])
        configured_convolutions = {
            item["convolution"]
            for item in config["training"]["gnn"]["combinations"]
        }
        fixed = config["training"]["data_generation"]["fixed_y"]
        self.assertEqual(
            config["training"]["data_generation"][
                "samples_per_instance"
            ],
            4,
        )
        generation = config["instances"]["generation"]
        self.assertEqual(generation["time_unit_minutes"], 10)
        self.assertAlmostEqual(
            profiles["profiles"]["old"]["repair_rate"], 1.0 / 30.0
        )
        self.assertAlmostEqual(
            profiles["profiles"]["new"]["repair_rate"], 1.0 / 15.0
        )
        self.assertEqual(generation["num_jobs"], [3, 4, 5])
        self.assertEqual(generation["num_machines"], [3, 4, 5])
        split = instances.split_items(
            list(range(generation["instances_per_size"])),
            generation["split_ratios"],
            generation["random_seed"],
        )
        training_graphs_per_size = (
            len(split["train"])
            * config["training"]["data_generation"][
                "samples_per_instance"
            ]
        )
        self.assertEqual(training_graphs_per_size * 9, 2016)
        self.assertNotIn(
            "service_level",
            config["constraint"]["weibull"]["reliability_graph"],
        )
        self.assertEqual(
            config["instances"]["generation"]["due_dates"]["factors"],
            [1.25, 1.30, 1.40, 1.55],
        )
        self.assertEqual(
            config["solve"]["evaluation"]["benchmark"]["due_date_factors"],
            [1.60],
        )
        self.assertTrue(config["solve"]["create_instances"])
        self.assertEqual(
            config["solve"]["evaluation"]["stress"]["due_date_factor"],
            1.60,
        )
        self.assertEqual(
            config["solve"]["evaluation"]["extrapolation"][
                "due_date_factor"
            ],
            1.60,
        )
        for tier in ("benchmark", "extrapolation", "stress"):
            self.assertTrue(
                config["solve"]["evaluation"][tier][
                    "reuse_existing_instances"
                ]
            )
        nonlinear_solver = config["solvers"]["gurobi"]["nonlinear"]
        self.assertEqual(nonlinear_solver["Presolve"], 0)
        self.assertEqual(nonlinear_solver["NumericFocus"], 3)
        self.assertEqual(nonlinear_solver["MIPFocus"], 1)
        self.assertNotIn(
            "Presolve", config["solvers"]["gurobi"]["common"]
        )
        self.assertEqual(
            config["training"]["data_generation"]["adaptive_due_dates"]
            ["relative_makespan_offsets"],
            [0.02, 0.05, 0.10, 0.20],
        )
        self.assertNotIn(
            "repair_buffer_scale",
            config["constraint"]["weibull"]["reliability_graph"],
        )
        self.assertNotIn(
            "failure_clock", config["evaluation"]["simulation"]
        )
        self.assertEqual(
            config["training"]["data_generation"]["adaptive_due_dates"]
            ["maximum_candidate_pool_runs"],
            4,
        )
        self.assertEqual(configured_convolutions, {"linear", "sage", "job"})
        self.assertEqual(fixed["label_distribution_center"], 0.50)
        self.assertEqual(fixed["label_distribution_half_width"], 0.25)
        self.assertEqual(
            fixed["candidate_generation_mode"], "nonlinear_evaluated"
        )
        self.assertNotIn("mixed_candidate_generation", fixed)
        self.assertEqual(fixed["pool_candidates"], 10)
        self.assertEqual(fixed["minimum_candidate_pool_runs"], 4)
        self.assertEqual(fixed["time_limit_seconds"], 1)
        self.assertEqual(fixed["pool_search_mode"], 2)
        self.assertEqual(fixed["mip_focus"], 1)
        self.assertNotIn("nonlinear_solver_seed", fixed)
        self.assertNotIn(
            "apply_to_splits", fixed["hybrid_selection"]
        )
        self.assertEqual(
            fixed["hybrid_selection"]["probability_target_basis"],
            "job_probabilities",
        )
        self.assertEqual(
            fixed["hybrid_selection"]["anchor_fraction"], 0.10
        )
        self.assertFalse(fixed["nonlinear_probability_bands"]["enabled"])
        self.assertEqual(
            fixed["hybrid_selection"]["maximum_candidate_pool_runs"],
            4,
        )
        self.assertEqual(
            training.TARGET_COLUMN, "nonlinear_expected_repair_buffer"
        )
        self.assertEqual(
            sequence_setup.RELIABILITY_GNN_OUTPUT_HEAD,
            "per_job_expected_repair_buffer_relu_v4",
        )
        self.assertFalse(
            hasattr(sequence_setup.ReliabilityGraphConfig(), "service_level")
        )

        instance = instances.FJSPData(
            nb_instance=1,
            num_jobs=1,
            num_machines=1,
            operations_per_job_min=1,
            operations_per_job_max=1,
            num_operations=[1],
            flag_save_file=False,
        )
        self.assertFalse(hasattr(instance, "service_levels"))

    def test_gnn_variants_have_descriptive_names(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        configured_variants = {
            item["variant"]: item["convolution"]
            for item in config["training"]["gnn"]["combinations"]
        }
        self.assertEqual(configured_variants, {
            "linear_baseline": "linear",
            "fixed_job_graphsage": "job",
            "decision_dependent_direct_machine_job_graphsage": "sage",
        })

    def test_gnn_initial_repair_buffer_is_configurable(self):
        self.assertEqual(
            sequence_setup.reliability_node_feature_names(),
            [
                "nominal_start_over_horizon",
                "nominal_completion_over_horizon",
                "processing_time_over_weibull_alpha",
                "repair_rate_times_weibull_alpha_over_30",
                "weibull_beta_over_5",
            ],
        )
        model = training.FJSPGraphSAGE(
            input_size=len(sequence_setup.reliability_node_feature_names()),
            hidden_channels=2,
            num_graphsage_layers=1,
            convolution="linear",
            initial_repair_buffer=0.50,
        )
        self.assertAlmostEqual(float(model.out.bias.item()), 0.50, places=6)


if __name__ == "__main__":
    unittest.main()
