import importlib
import unittest
from types import SimpleNamespace

import torch


training = importlib.import_module(
    "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
)


class GNNLossTests(unittest.TestCase):
    def setUp(self):
        self.prediction = torch.tensor([0.90, 0.97])
        self.batch = SimpleNamespace(job_y=torch.tensor([0.95, 0.95]))

    def loss(self, name, **kwargs):
        return float(training._loss(
            self.prediction,
            self.prediction,
            self.batch,
            loss_name=name,
            **kwargs,
        ))

    def test_mse(self):
        self.assertAlmostEqual(
            self.loss(training.LOSS_MSE), 0.00145, places=7
        )

    def test_asymmetric_mse_penalizes_overestimation(self):
        self.assertAlmostEqual(
            self.loss(
                training.LOSS_ASYMMETRIC_MSE,
                overestimation_weight=2.0,
            ),
            0.00165,
            places=7,
        )

    def test_boundary_weighting(self):
        self.assertAlmostEqual(
            self.loss(
                training.LOSS_BOUNDARY_WEIGHTED_MSE,
                boundary_width=0.03,
                boundary_weight=4.0,
            ),
            4.0 * 0.00145,
            places=7,
        )

    def test_huber(self):
        self.assertAlmostEqual(
            self.loss(training.LOSS_HUBER, huber_delta=0.05),
            0.000725,
            places=7,
        )

    def test_unknown_loss_is_rejected(self):
        with self.assertRaises(ValueError):
            self.loss("unknown")


if __name__ == "__main__":
    unittest.main()
