import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


VALIDATOR_PATH = Path(__file__).with_name("validate_gnn_embedding.py")
SPEC = importlib.util.spec_from_file_location(
    "validate_gnn_embedding", VALIDATOR_PATH
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class GNNEmbeddingValidatorTests(unittest.TestCase):
    def test_expected_repair_buffer_above_one_is_not_clipped(self):
        result = {
            "solution_count": 1,
            "instance": object(),
            "variables": {
                "gnn_metadata": {},
                "gnn_model_path": "unused.pt",
                "gnn_job_output_expressions": {0: 1.75},
            },
        }
        with (
            patch.object(validator, "_pytorch_graph", return_value=(None, (0,))),
            patch.object(validator, "_pytorch_prediction", return_value=[1.75]),
        ):
            records = validator.validate_result(result)

        self.assertEqual(records[0]["gurobi"], 1.75)
        self.assertEqual(records[0]["pytorch"], 1.75)
        self.assertEqual(records[0]["absolute_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
