import importlib
import json
import random
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
instances = importlib.import_module("01_generator.instance_generator")
stochastic = importlib.import_module("helper.stochastic_fjsp")


class IndependentInstanceGeneratorTests(unittest.TestCase):
    def _instance(self, seed=123):
        return instances.FJSPData(
            nb_instance=1,
            num_jobs=3,
            num_machines=5,
            operations_per_job_min=3,
            operations_per_job_max=5,
            flag_save_file=False,
            processing_time_range=[1, 10],
            processing_time_deviation=0.2,
            machine_parameter_ranges={
                "hourly_cost": [1.0, 2.25],
                "weibull_alpha": [24.0, 42.0],
                "weibull_beta": [1.6, 3.0],
                "repair_rate": [0.3, 0.7],
            },
            random_source=random.Random(seed),
        )

    def test_structure_is_not_rebuilt_by_a_modernity_rule(self):
        instance = self._instance()
        self.assertEqual(
            instance.instance_generation_model,
            stochastic.INDEPENDENT_GENERATION_MODEL,
        )
        self.assertFalse(hasattr(instance, "machine_modernity"))
        self.assertFalse(hasattr(instance, "operation_requirement"))
        before_eligible = {
            operation: list(machines)
            for operation, machines in instance.eligible_machines.items()
        }
        before_processing = dict(instance.processing_times)
        stochastic.ensure_stochastic_parameters(
            instance, rebuild_modernity=True
        )
        self.assertEqual(instance.eligible_machines, before_eligible)
        self.assertEqual(instance.processing_times, before_processing)

    def test_flat_and_derived_fjsp_encodings_match(self):
        instance = self._instance()
        offset = 0
        for index, operation in enumerate(instance.real_operations):
            count = instance.nums_option[index]
            machines = instance.ope_machine[offset:offset + count]
            durations = instance.processing_time[offset:offset + count]
            self.assertEqual(machines, instance.eligible_machines[operation])
            self.assertEqual(
                durations,
                [
                    instance.processing_times[operation, machine]
                    for machine in machines
                ],
            )
            offset += count

    def test_machine_parameters_lie_in_configured_ranges(self):
        instance = self._instance()
        for value in instance.machine_cost.values():
            self.assertTrue(1.0 <= value <= 2.25)
        for value in instance.weibull_alpha.values():
            self.assertTrue(24.0 <= value <= 42.0)
        for value in instance.weibull_beta.values():
            self.assertTrue(1.6 <= value <= 3.0)
        for value in instance.repair_rate.values():
            self.assertTrue(0.3 <= value <= 0.7)

    def test_same_seed_reproduces_complete_instance(self):
        first = self._instance(seed=77)
        second = self._instance(seed=77)
        for name in (
            "nums_operation",
            "nums_option",
            "ope_machine",
            "processing_time",
            "machine_cost",
            "weibull_alpha",
            "weibull_beta",
            "repair_rate",
        ):
            self.assertEqual(getattr(first, name), getattr(second, name))

    def test_project_config_uses_neutral_generator_parameters(self):
        config = json.loads(
            (ROOT / "config.json").read_text(encoding="utf-8")
        )
        generation = config["instances"]["generation"]
        self.assertEqual(generation["processing_times"]["base_range"], [1, 10])
        self.assertEqual(
            set(generation["machine_parameters"]),
            {"hourly_cost", "weibull_alpha", "weibull_beta", "repair_rate"},
        )

    def test_saved_instances_are_validated_against_generator_config(self):
        specs = [{
            "num_jobs": 3,
            "num_machines": 3,
            "operations_per_job": [3, 5],
            "count": 4,
        }]
        ranges = {
            "hourly_cost": [1.0, 2.25],
            "weibull_alpha": [24.0, 42.0],
            "weibull_beta": [1.6, 3.0],
            "repair_rate": [0.3, 0.7],
        }
        with tempfile.TemporaryDirectory() as directory:
            instances.generate_instance_specs(
                specs,
                random_seed=69,
                output_directory=directory,
                processing_time_range=[1, 10],
                processing_time_deviation=0.2,
                machine_parameter_ranges=ranges,
            )
            selected = instances.configured_instance_names_by_split(
                specs,
                random_seed=69,
                instance_directory=directory,
                processing_time_range=[1, 10],
                processing_time_deviation=0.2,
                machine_parameter_ranges=ranges,
            )
            self.assertEqual(sum(map(len, selected.values())), 4)
            with self.assertRaisesRegex(ValueError, "Maschinenparameter"):
                instances.configured_instance_names_by_split(
                    specs,
                    random_seed=69,
                    instance_directory=directory,
                    machine_parameter_ranges={
                        **ranges,
                        "hourly_cost": [2.0, 3.0],
                    },
                )


if __name__ == "__main__":
    unittest.main()
