"""Entry point for per-graph vectors of job-specific repair buffers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from helper.time_units import normalize_time_unit


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config.json"
_instances = importlib.import_module("01_generator.instance_generator")


def _generation_from_project_config(config):
    source = config["training"]["data_generation"]
    instance_generation = config["instances"]["generation"]
    specs = [
        {
            "num_jobs": jobs,
            "num_machines": machines,
            "operations_per_job": instance_generation["operations_per_job"],
            "count": instance_generation["instances_per_size"],
        }
        for jobs in instance_generation["num_jobs"]
        for machines in instance_generation["num_machines"]
    ]
    ranges = source.get("reliability_ranges") or {}
    return {
        "method": source["method"],
        "instance_splits": _instances.configured_instance_names_by_split(
            specs,
            split_ratios=instance_generation.get("split_ratios"),
            random_seed=instance_generation.get("random_seed", 42),
            processing_time_range=instance_generation.get(
                "processing_times", {}
            ).get("base_range"),
            processing_time_deviation=instance_generation.get(
                "processing_times", {}
            ).get("relative_deviation"),
            machine_parameter_ranges=instance_generation.get(
                "machine_parameters"
            ),
            machine_profile_config=instance_generation.get(
                "machine_profiles"
            ),
            training_parameter_jitter=instance_generation.get(
                "training_parameter_jitter"
            ),
            time_unit=normalize_time_unit(instance_generation),
        ),
        "generate_splits": source.get("generate_splits"),
        "output_directory": source["output_directory"],
        "random_seed": int(source.get("random_seed", 42)),
        "samples_per_instance": int(source["samples_per_instance"]),
        "labels": source.get("labels"),
        "instance_failure_handling": source.get(
            "instance_failure_handling"
        ),
        "alpha_range": ranges.get("alpha"),
        "beta_range": ranges.get("beta"),
        "repair_rate_range": ranges.get("repair_rate"),
        "reliability_graph": config["constraint"]["weibull"][
            "reliability_graph"
        ],
        "weibull_scale_factors": source.get("weibull_scale_factors"),
        "machine_profile_config": instance_generation.get(
            "machine_profiles"
        ),
        "training_parameter_jitter": instance_generation.get(
            "training_parameter_jitter"
        ),
        "time_unit": normalize_time_unit(instance_generation),
        "adaptive_due_dates": source.get("adaptive_due_dates"),
        "fixed_y": source["fixed_y"],
    }


def generate_from_config(generation=None):
    if generation is None:
        with CONFIG_PATH.open(encoding="utf-8") as file:
            generation = _generation_from_project_config(json.load(file))
    if str(generation.get("method", "")).lower() != "fix_and_optimize":
        raise ValueError(
            "The repair-buffer pipeline supports only fix_and_optimize."
        )
    generator = importlib.import_module(
        "04_GraphNeuralNetworks.models."
        "generate_fix_and_optimize_training_data"
    ).generate_from_config
    return generator(generation)


if __name__ == "__main__":
    generate_from_config()
