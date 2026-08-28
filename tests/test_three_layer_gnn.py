import importlib
import unittest

import torch
from torch_geometric.data import Batch, Data


architectures = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)
training = importlib.import_module(
    "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
)
sequence_setup = importlib.import_module("helper.sequence_setup")


class ThreeLayerGNNTests(unittest.TestCase):
    def test_architecture_expansion_accepts_three_layers(self):
        variants = architectures.expand_architecture_variants({
            "convolution": "sage",
            "layers": [2, 3],
            "hidden_channels": [4, 6],
        })
        self.assertEqual(
            [(item["layers"], item["hidden_channels"]) for item in variants],
            [(2, 4), (2, 6), (3, 4), (3, 6)],
        )

    def test_job_architecture_uses_sum_aggregation(self):
        variants = architectures.expand_architecture_variants({
            "convolution": "job",
            "layers": [1],
            "hidden_channels": [4],
        })
        self.assertEqual(variants[0]["aggregation"], "sum")
        self.assertIn("job", architectures.MESSAGE_PASSING_CONVOLUTIONS)

    def test_job_precedence_edges_follow_each_job_chain(self):
        self.assertEqual(
            training._job_precedence_edges([0, 0, 0, 1, 1]),
            [(0, 1), (1, 2), (3, 4)],
        )

    def test_three_layer_network_runs_all_convolutions(self):
        input_size = len(sequence_setup.reliability_node_feature_names())
        graph = Batch.from_data_list([Data(
            x=torch.ones((2, input_size), dtype=torch.float32),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            job_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            job_membership=torch.zeros(2, dtype=torch.long),
            num_jobs_tensor=torch.tensor([1], dtype=torch.long),
        )])
        for convolution in ("linear", "sage", "job"):
            with self.subTest(convolution=convolution):
                model = training.FJSPGraphSAGE(
                    input_size=input_size,
                    hidden_channels=4,
                    num_graphsage_layers=3,
                    convolution=convolution,
                )
                prediction = model(graph)
                self.assertEqual(tuple(prediction.shape), (1,))
                self.assertIsNotNone(model.conv2)
                self.assertIsNotNone(model.conv3)
                self.assertTrue(any(
                    key.startswith("conv3.")
                    for key in model.state_dict()
                ))

    def test_job_model_uses_only_fixed_job_edges(self):
        input_size = len(sequence_setup.reliability_node_feature_names())
        model = training.FJSPGraphSAGE(
            input_size=input_size,
            hidden_channels=1,
            num_graphsage_layers=1,
            convolution="job",
        )
        with torch.no_grad():
            model.conv1.lin_root.weight.zero_()
            model.conv1.lin_root.bias.zero_()
            model.conv1.lin_message.weight.fill_(1.0)
            model.out.weight.fill_(1.0)
            model.out.bias.zero_()
            model.out_input.weight.zero_()
            model.out_input.bias.zero_()

        def prediction(job_edges):
            graph = Batch.from_data_list([Data(
                x=torch.ones((2, input_size), dtype=torch.float32),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                job_edge_index=job_edges,
                job_membership=torch.zeros(2, dtype=torch.long),
                num_jobs_tensor=torch.tensor([1], dtype=torch.long),
            )])
            return float(model(graph).item())

        empty = torch.empty((2, 0), dtype=torch.long)
        fixed = torch.tensor([[0], [1]], dtype=torch.long)
        self.assertEqual(prediction(empty), 0.0)
        self.assertEqual(prediction(fixed), 5.0)


if __name__ == "__main__":
    unittest.main()
