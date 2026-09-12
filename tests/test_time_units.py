import copy
import importlib
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path

from helper.stochastic_fjsp import DEFAULT_MACHINE_PROFILE_CONFIG
from helper.time_units import normalize_time_unit


instances = importlib.import_module("01_generator.instance_generator")
training = importlib.import_module(
    "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
)


class TimeUnitTests(unittest.TestCase):
    def test_legacy_pickle_keeps_all_numerical_values(self):
        instance = instances.FJSPData(
            nb_instance=1, num_jobs=2, num_machines=2,
            operations_per_job_min=2, operations_per_job_max=2,
            flag_save_file=False, random_source=random.Random(42),
            machine_profile_config=DEFAULT_MACHINE_PROFILE_CONFIG,
        )
        expected = copy.deepcopy(vars(instance))
        del instance.time_unit
        instance.time_unit_minutes = 10.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{instance.instance_name}.pkl"
            path.write_bytes(pickle.dumps(instance))
            loaded = instances.load_generated_instance(
                instance.instance_name, instance_directory=directory
            )
        self.assertEqual(vars(loaded), expected)

    def test_training_accepts_legacy_and_ze_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "training" / "graphs_training.csv"
            for unit_fields in ({"time_unit": "ZE"}, {"time_unit_minutes": 10}):
                (root / "generation_summary.json").write_text(json.dumps({
                    "machine_profile_config": DEFAULT_MACHINE_PROFILE_CONFIG,
                    **unit_fields,
                }))
                training._validate_dataset_context(
                    csv_path, machine_profile_config=DEFAULT_MACHINE_PROFILE_CONFIG,
                    time_unit="ZE",
                )

    def test_missing_or_unsupported_artifact_units_are_rejected(self):
        for metadata in (
            {}, {"time_unit": "min"}, {"time_unit": "s"},
            {"time_unit_minutes": 0}, {"time_unit_minutes": float("nan")},
        ):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                normalize_time_unit(metadata, require_metadata=True)


if __name__ == "__main__":
    unittest.main()
