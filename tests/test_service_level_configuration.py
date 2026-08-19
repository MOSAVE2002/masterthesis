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
    def test_active_pipeline_defaults_to_point_ninety_five(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        configured = config["constraint"]["weibull"]["reliability_graph"][
            "service_level"
        ]
        boundary_width = config["training"]["data_generation"]["fixed_y"][
            "service_boundary_width"
        ]
        fixed = config["training"]["data_generation"]["fixed_y"]
        self.assertEqual(
            config["training"]["data_generation"][
                "samples_per_instance"
            ],
            20,
        )
        targets = config["training"]["data_generation"]["fixed_y"][
            "hybrid_selection"
        ]["job_probability_target_ratios"]
        self.assertEqual(configured, 0.95)
        self.assertLess(configured + boundary_width, 1.0)
        self.assertEqual(
            fixed["candidate_generation_mode"], "nonlinear_evaluated"
        )
        self.assertNotIn("mixed_candidate_generation", fixed)
        self.assertEqual(fixed["pool_candidates"], 30)
        self.assertEqual(fixed["minimum_candidate_pool_runs"], 4)
        self.assertEqual(fixed["time_limit_seconds"], 4)
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
            12,
        )
        self.assertEqual(
            targets,
            {
                "low": 0.10,
                "boundary_below": 0.40,
                "boundary_above": 0.40,
                "high": 0.10,
            },
        )
        self.assertEqual(
            sequence_setup.ReliabilityGraphConfig().service_level, 0.95
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
        self.assertEqual(instance.service_levels, {1: 0.95})

    def test_gnn_initial_probability_is_configurable(self):
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
            initial_probability=0.95,
        )
        self.assertAlmostEqual(float(model.out.bias.item()), 0.95, places=6)


if __name__ == "__main__":
    unittest.main()
