import importlib
import json
import math
import random
import unittest

import gurobipy as gp

from helper.sequence_setup import ReliabilityGraphConfig, reliability_node_feature_names
from helper.stochastic_fjsp import DEFAULT_MACHINE_PROFILE_CONFIG, weibull_mean_lifetime
from helper.surrogate_constraint import CONSTRAINT_WEIBULL


instances = importlib.import_module("01_generator.instance_generator")
generator = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_fix_and_optimize_training_data"
)
embedding = importlib.import_module("03_Gurobi.build_fjsp_with_gnn")


class GNNInputFeatureTests(unittest.TestCase):
    def test_weibull_mean_lifetime_known_distributions(self):
        self.assertAlmostEqual(weibull_mean_lifetime(120, 1), 120)
        self.assertAlmostEqual(weibull_mean_lifetime(200, 2), 100 * math.sqrt(math.pi))

    def test_dataset_and_milp_features_match_for_either_machine(self):
        instance = instances.FJSPData(
            nb_instance=1, num_jobs=1, num_machines=2,
            operations_per_job_min=2, operations_per_job_max=2,
            flag_save_file=False, random_source=random.Random(42),
            machine_profile_config=DEFAULT_MACHINE_PROFILE_CONFIG,
        )
        graph_config = ReliabilityGraphConfig(service_scope="job")
        horizon = max(instance.due_dates.values())
        for chosen_machine in range(2):
            with self.subTest(machine=chosen_machine):
                model, variables = generator._build_candidate_model(instance, graph_config)
                try:
                    model.Params.OutputFlag = 0
                    model.Params.TimeLimit = 5
                    for operation in instance.real_operations:
                        model.addConstr(variables["Y"][operation, chosen_machine] == 1)
                    variables["service_horizon"] = horizon
                    embedding._add_midpoint_state(model, variables, instance)
                    model.update()
                    original_size = (model.NumVars, model.NumBinVars, model.NumConstrs)
                    expressions, bounds = embedding._build_node_feature_expressions(
                        instance, variables, CONSTRAINT_WEIBULL
                    )
                    model.update()
                    self.assertEqual(
                        (model.NumVars, model.NumBinVars, model.NumConstrs), original_size
                    )
                    model.optimize()
                    self.assertGreater(model.SolCount, 0)
                    row = generator._candidate_from_solution(
                        model, variables, instance, graph_config,
                        solution_number=None, run_index=0, fix_stats={},
                    )["row"]
                    self.assertEqual(json.loads(row["gnn_feature_names"]), reliability_node_feature_names())
                    values = json.loads(row["gnn_node_features"])
                    for operation, vector, terms, intervals in zip(instance.real_operations, values, expressions, bounds):
                        self.assertEqual(len(vector), 4)
                        for value, term, (lower, upper) in zip(vector, terms, intervals):
                            self.assertAlmostEqual(value, term.getValue())
                            self.assertGreaterEqual(value + 1e-8, lower)
                            self.assertLessEqual(value - 1e-8, upper)
                        alpha = instance.weibull_alpha[chosen_machine]
                        rate = instance.repair_rate[chosen_machine]
                        self.assertAlmostEqual(vector[0], variables["T"][operation].X / alpha)
                        self.assertAlmostEqual(vector[3], 1.0 / (60.0 * rate))
                finally:
                    model.dispose()


if __name__ == "__main__":
    unittest.main()
