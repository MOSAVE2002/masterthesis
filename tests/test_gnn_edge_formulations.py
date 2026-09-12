import importlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gurobipy as gp
import torch


instances = importlib.import_module("01_generator.instance_generator")
training = importlib.import_module(
    "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
)
embedding = importlib.import_module("03_Gurobi.build_fjsp_with_gnn")
sequence_setup = importlib.import_module("helper.sequence_setup")
stochastic = importlib.import_module("helper.stochastic_fjsp")
validator = importlib.import_module("tests.validate_gnn_embedding")


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
            time_unit="ZE",
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
            "job_target": training.JOB_TARGET,
            "job_repair_buffer_label_method": training.SIMULATION_LABEL_METHOD,
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
            "time_unit": "ZE",
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

    def test_gnn_constraint_uses_unscaled_repair_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            model, variables = self._build(directory, "linear")
            try:
                self.assertIn("job_expected_delays", variables)
                self.assertIn("job_service_level_buffers", variables)
                self.assertIn("job_service_level_violation", variables)
                self.assertIn("service_violation_cost", variables)
                self.assertNotIn("service_level", variables)
                self.assertNotIn("service_buffer_scale", variables)
                self.assertEqual(variables["service_constraint_bound"],
                                 "unscaled_expected_local_repair_buffer")
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

    def test_legacy_units_produce_identical_gnn_optimization_models(self):
        original_metadata = self._metadata

        def legacy_metadata(convolution, graph_config):
            metadata = original_metadata(convolution, graph_config)
            metadata.pop("time_unit")
            metadata["time_unit_minutes"] = 10.0
            return metadata

        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng():
            for convolution in ("linear", "job", "sage"):
                with self.subTest(convolution=convolution):
                    fingerprints = []
                    for metadata_factory in (original_metadata, legacy_metadata):
                        torch.manual_seed(42)
                        with patch.object(self, "_metadata", metadata_factory):
                            model, _variables = self._build(directory, convolution)
                        try:
                            model.update()
                            fingerprints.append(model.Fingerprint)
                        finally:
                            model.dispose()
                    self.assertEqual(fingerprints[0], fingerprints[1])

    def test_four_input_gnn_matches_pytorch_for_all_architectures(self):
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng():
            for convolution in ("linear", "job", "sage"):
                with self.subTest(convolution=convolution):
                    torch.manual_seed(42)
                    model, variables = self._build(directory, convolution)
                    try:
                        model.Params.TimeLimit = 5
                        model.optimize()
                        self.assertGreater(model.SolCount, 0)
                        graph, _jobs = validator._pytorch_graph(
                            variables, self._instance()
                        )
                        self.assertEqual(graph.x.shape[1], 4)
                        validator.validate_result({
                            "solution_count": model.SolCount,
                            "variables": variables,
                            "instance": self._instance(),
                        })
                    finally:
                        model.dispose()


if __name__ == "__main__":
    unittest.main()
