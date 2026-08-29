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
            "weibull_alpha",
            "weibull_beta",
            "repair_rate",
        ):
            self.assertEqual(getattr(first, name), getattr(second, name))

    def test_project_config_uses_machine_profiles(self):
        config = json.loads(
            (ROOT / "config.json").read_text(encoding="utf-8")
        )
        generation = config["instances"]["generation"]
        self.assertEqual(generation["processing_times"]["base_range"], [5, 15])
        self.assertEqual(
            set(generation["machine_profiles"]["profiles"]),
            {"old", "new"},
        )
        self.assertEqual(
            generation["machine_profiles"]["profiles"]["old"],
            {
                "cost_rate": 1.0,
                "speed": 0.8,
                "weibull_alpha": 60.0,
                "weibull_beta": 3.0,
                "repair_rate": 1.0 / 30.0,
            },
        )
        self.assertEqual(
            generation["machine_profiles"]["profiles"]["new"],
            {
                "cost_rate": 1.6,
                "speed": 1.25,
                "weibull_alpha": 100.0,
                "weibull_beta": 2.0,
                "repair_rate": 1.0 / 15.0,
            },
        )
        self.assertTrue(all(
            value == 0.0
            for value in generation["machine_profiles"][
                "parameter_jitter"
            ].values()
        ))
        self.assertEqual(
            generation["machine_profiles"]["profiles"]["old"][
                "weibull_beta"
            ],
            3.0,
        )
        self.assertEqual(
            generation["machine_profiles"]["profiles"]["new"][
                "weibull_beta"
            ],
            2.0,
        )
        self.assertNotIn(
            "weibull_beta_range", generation["machine_profiles"]
        )
        self.assertNotIn("machine_parameters", generation)
        self.assertNotIn(
            "reliability_ranges",
            config["training"]["data_generation"],
        )
        self.assertEqual(
            config["training"]["data_generation"][
                "weibull_scale_factors"
            ],
            [1.0],
        )

    def test_saved_instances_are_validated_against_generator_config(self):
        specs = [{
            "num_jobs": 3,
            "num_machines": 3,
            "operations_per_job": [3, 5],
            "count": 4,
        }]
        ranges = {
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
                        "weibull_alpha": [30.0, 35.0],
                    },
                )


class MachineProfileGeneratorTests(unittest.TestCase):
    def _instance(self, seed=123, number=1):
        return instances.FJSPData(
            nb_instance=number,
            num_jobs=3,
            num_machines=3,
            operations_per_job_min=3,
            operations_per_job_max=5,
            flag_save_file=False,
            processing_time_range=[5, 15],
            machine_profile_config=stochastic.DEFAULT_MACHINE_PROFILE_CONFIG,
            due_date_config={
                "method": "fjsp_lower_bound_factors",
                "factors": [1.25, 1.35, 1.45],
                "assignment": "cyclic",
                "service_level": 0.95,
            },
            random_source=random.Random(seed),
        )

    def test_old_and_new_machine_types_are_present_and_ordered(self):
        instance = self._instance()
        self.assertEqual(
            instance.instance_generation_model,
            stochastic.PROFILE_GENERATION_MODEL,
        )
        self.assertEqual(
            set(instance.machine_profiles.values()),
            {"old", "new"},
        )
        machine_by_profile = {
            profile: machine
            for machine, profile in instance.machine_profiles.items()
        }
        old = machine_by_profile["old"]
        new = machine_by_profile["new"]
        self.assertLess(instance.machine_cost[old], instance.machine_cost[new])
        self.assertLess(
            instance.machine_speed[old],
            instance.machine_speed[new],
        )
        self.assertLess(
            instance.weibull_alpha[old],
            instance.weibull_alpha[new],
        )
        self.assertEqual(instance.weibull_beta[old], 3.0)
        self.assertEqual(instance.weibull_beta[new], 2.0)
        self.assertLess(instance.repair_rate[old], instance.repair_rate[new])

    def test_profile_parameters_are_constant_across_instances(self):
        expected = stochastic.DEFAULT_MACHINE_PROFILE_CONFIG["profiles"]
        field_names = {
            "machine_cost": "cost_rate",
            "machine_speed": "speed",
            "weibull_alpha": "weibull_alpha",
            "weibull_beta": "weibull_beta",
            "repair_rate": "repair_rate",
        }
        for seed in (11, 22, 33):
            instance = self._instance(seed=seed)
            for machine, profile_name in instance.machine_profiles.items():
                for attribute, profile_field in field_names.items():
                    self.assertEqual(
                        getattr(instance, attribute)[machine],
                        expected[profile_name][profile_field],
                    )

    def test_each_operation_has_at_least_two_profile_classes(self):
        instance = self._instance()
        for operation in instance.real_operations:
            eligible_profiles = {
                instance.machine_profiles[machine]
                for machine in instance.eligible_machines[operation]
            }
            self.assertGreaterEqual(len(eligible_profiles), 2)
            for machine in instance.eligible_machines[operation]:
                self.assertGreater(
                    instance.processing_times[operation, machine], 0
                )

    def test_profile_flat_and_derived_encodings_match(self):
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

    def test_due_date_factor_is_assigned_cyclically(self):
        first = self._instance(number=1)
        second = self._instance(number=2)
        self.assertEqual(first.due_date_factor, 1.25)
        self.assertEqual(second.due_date_factor, 1.35)
        self.assertFalse(hasattr(first, "service_levels"))


if __name__ == "__main__":
    unittest.main()
