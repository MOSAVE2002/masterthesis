import importlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import gurobipy as gp
import torch


instances = importlib.import_module("01_generator.instance_generator")
training = importlib.import_module(
    "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
)
embedding = importlib.import_module("03_Gurobi.build_fjsp_with_gnn")
sequence_setup = importlib.import_module("helper.sequence_setup")
stochastic = importlib.import_module("helper.stochastic_fjsp")


class GNNEdgeFormulationTests(unittest.TestCase):
    def _instance(self):
        return instances.FJSPData(
            nb_instance=1,
            num_jobs=1,
            num_machines=1,
            operations_per_job_min=2,
            operations_per_job_max=2,
            num_operations=[2],
            flag_save_file=False,
            machine_profile_config=stochastic.DEFAULT_MACHINE_PROFILE_CONFIG,
            time_unit_minutes=10.0,
            random_source=random.Random(42),
        )

    def _metadata(self, convolution, graph_config):
        message_passing = (
            "none"
            if convolution == "linear"
            else "source_node_states_only"
        )
        return {
            "target_column": training.TARGET_COLUMN,
            "input_size": len(
                sequence_setup.reliability_node_feature_names()
            ),
            "feature_names": (
                sequence_setup.reliability_node_feature_names()
            ),
            "graph_mode": "fixed_candidate",
            "convolution": convolution,
            "aggregation": "none" if convolution == "linear" else "sum",
            "pooling": "global_add",
            "num_graphsage_layers": 1,
            "hidden_channels": 2,
            "output_head": sequence_setup.RELIABILITY_GNN_OUTPUT_HEAD,
            "job_target": "job_expected_completion_delay",
            "graph_schema": sequence_setup.RELIABILITY_GNN_GRAPH_SCHEMA,
            "message_passing": message_passing,
            "include_machine_predecessor_edges": convolution == "sage",
            "machine_predecessor_edge_scope": (
                "direct" if convolution == "sage" else "none"
            ),
            "include_job_precedence_edges": convolution in {"sage", "job"},
            "reliability_graph_config": (
                sequence_setup.reliability_graph_config_dict(graph_config)
            ),
            "machine_profile_config": (
                stochastic.normalize_machine_profile_config()
            ),
            "time_unit_minutes": 10.0,
        }

    def _build(self, directory, convolution):
        graph_config = sequence_setup.ReliabilityGraphConfig(
            service_scope="job",
        )
        aggregation = "none" if convolution == "linear" else "sum"
        network = training.FJSPGraphSAGE(
            input_size=len(
                sequence_setup.reliability_node_feature_names()
            ),
            hidden_channels=2,
            num_graphsage_layers=1,
            convolution=convolution,
        )
        model_path = Path(directory) / f"{convolution}.pt"
        metadata_path = Path(directory) / f"{convolution}.json"
        torch.save(network.state_dict(), model_path)
        metadata_path.write_text(
            json.dumps(self._metadata(convolution, graph_config)),
            encoding="utf-8",
        )

        model = gp.Model()
        model.Params.OutputFlag = 0
        _, variables = embedding.build_fjsp(
            model,
            self._instance(),
            model_path=model_path,
            metadata_path=metadata_path,
            convolution=convolution,
            aggregation=aggregation,
            pooling="global_add",
            layers=1,
            hidden_channels=2,
            reliability_graph_config=graph_config,
        )
        return model, variables

    def test_gnn_constraint_scales_expected_delay_for_service_level(self):
        with tempfile.TemporaryDirectory() as directory:
            model, variables = self._build(directory, "linear")
            try:
                self.assertIn("job_expected_delays", variables)
                self.assertIn("job_service_level_buffers", variables)
                self.assertIn("job_service_level_violation", variables)
                self.assertIn("service_violation_cost", variables)
                self.assertAlmostEqual(variables["service_level"], 0.90)
                self.assertAlmostEqual(
                    variables["service_buffer_scale"], 10.0
                )
            finally:
                model.dispose()

    def test_only_direct_u_sage_creates_machine_precedence_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            for convolution, expects_u in (
                ("linear", False),
                ("job", False),
                ("sage", True),
            ):
                with self.subTest(convolution=convolution):
                    model, variables = self._build(directory, convolution)
                    try:
                        self.assertEqual("U" in variables, expects_u)
                        self.assertEqual("U_index" in variables, expects_u)
                        self.assertEqual("A_plus" in variables, expects_u)
                        self.assertEqual("A_minus" in variables, expects_u)
                        if expects_u:
                            incoming = embedding._relational_incoming_edges(
                                self._instance(), variables, "sage"
                            )
                            machine_gates = [
                                gate
                                for edges in incoming.values()
                                for _source_idx, gate, *_rest in edges
                                if not isinstance(gate, float)
                            ]
                            self.assertTrue(machine_gates)
                            self.assertTrue(all(
                                gate.VarName.startswith("U_direct")
                                for gate in machine_gates
                            ))
                    finally:
                        model.dispose()


if __name__ == "__main__":
    unittest.main()
