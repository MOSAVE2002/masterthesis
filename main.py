"""Orchestrate the complete FJSP and GNN experiment pipeline.

This module is the central entry point of the project. It reads ``config.json``,
determines which parts of the pipeline must be executed and uses the workflow
flags from the configuration file.

The pipeline consists of five optional phases:

1. Generate FJSP instances and divide them into training, validation and test
   splits.
2. Generate graph-based training data from the configured FJSP instances.
3. Train the configured graph neural network architectures.
4. Solve the configured benchmark and extrapolation instances with the
   nominal, nonlinear and GNN-embedded Gurobi formulations.
5. Post-evaluate stored production plans with the thesis Monte Carlo model.

The phases are executed in dependency order because each phase may consume the
artifacts produced by the preceding phase. This module prepares the required
configuration values, resolves project-relative paths, constructs instance
specifications and validates the due-date metadata of existing training
instances before delegating the actual work.

Detailed responsibilities are separated into dedicated helper modules. The
experiment-plan module creates or validates the benchmark and extrapolation
instance plans. The model registry locates trained GNN artifacts and verifies
that their metadata matches the current configuration. The solve runner then
executes all requested solver and GNN-model combinations and incrementally
writes their results to a manifest.

``main.py`` only coordinates the complete workflow and passes data between its phases.
"""

# Packages beginning with a number must be imported dynamically.
import importlib
import json
from pathlib import Path

from helper.experiment_plan import generated_tier_plan
from helper.gnn_model_registry import trained_models
from helper.solve_runner import solve_plan


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json"

instances = importlib.import_module("01_generator.instance_generator")


def absolute(path):
    """Resolve a configured path against the repository root.

    Args:
        path: Absolute or project-relative path-like value.

    Returns:
        Absolute paths unchanged; otherwise a path below ``ROOT_DIR``.
    """
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def instance_specs(generation):
    """Expand generation settings into all configured instance-size pairs.

    Returns:
        Specification dictionaries for the Cartesian product of job and machine
        counts, sharing the operation range and instances-per-size count.
    """
    return [
        {
            "num_jobs": int(jobs),
            "num_machines": int(machines),
            "operations_per_job": list(generation["operations_per_job"]),
            "count": int(generation["instances_per_size"]),
        }
        for jobs in generation["num_jobs"]
        for machines in generation["num_machines"]
    ]


def configured_instance_splits(config, specs):
    """Reconstruct and validate the configured generated-instance splits.

    The detailed validation is delegated to the instance generator, which
    checks dimensions, profiles, jitter, processing times and due dates.

    Returns:
        Mapping from train, validation and test to validated instance names.
    """
    generation = config["instances"]["generation"]
    return instances.configured_instance_names_by_split(
        specs,
        machine_profile_config=generation["machine_profiles"],
        due_date_config=generation["due_dates"],
        split_ratios=generation["split_ratios"],
        random_seed=generation["random_seed"],
        processing_time_range=generation["processing_times"]["base_range"],
        training_parameter_jitter=generation["training_parameter_jitter"],
    )


def generate_data(config, splits):
    """Generate labelled GNN graph data from configured instance splits.

    Only the settings required by fix-and-optimize generation are extracted
    from the full project configuration and passed to the dedicated module.
    """
    data = config["training"]["data_generation"]

    generator = importlib.import_module(
        "04_GraphNeuralNetworks.models."
        "generate_fix_and_optimize_training_data"
    ).generate_from_config

    generator({
        "instance_splits": splits,
        "output_directory": data["output_directory"],
        "random_seed": int(data["random_seed"]),
        "samples_per_instance": int(data["samples_per_instance"]),
        "weibull_scale_factors": data["weibull_scale_factors"],
        "machine_profile_config": config["instances"]["generation"]["machine_profiles"],
        "training_parameter_jitter": config["instances"]["generation"]["training_parameter_jitter"],
        "adaptive_due_dates": data["adaptive_due_dates"],
        "fixed_y": data["fixed_y"],
    })


def train(config):
    """Train every unique GNN architecture declared in the configuration.

    Returns:
        Artifact path pairs returned by the dedicated training module.
    """
    trainer = importlib.import_module(
        "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
    ).train_from_config
    return trainer(config)


def solve(config):
    """Assemble experiment tiers and execute all requested solver cases.

    Common Gurobi and economic settings are combined with trained GNN artifacts
    before benchmark and extrapolation plans are dispatched sequentially.

    Returns:
        Compact result records accumulated across all enabled tiers.
    """
    # Prepare solver config
    solver_config = config["solvers"]["gurobi"]
    common = dict(solver_config.get("common", {}))
    common["solutions_directory"] = str(absolute(config["solve"]["solutions_directory"]))
    common["facility_cost_per_time"] = float(config["objective"].get("facility_cost_per_time", 1.0))
    common["tardiness_cost_per_time"] = float(config["objective"]["tardiness_cost_per_time"])
    # Get requested solvers.
    requested = [name.lower() for name in config["solve"]["solvers"]]
    # Load gnn models
    models = trained_models(config) if "gurobi_gnn" in requested else []

    planned_tiers = [
        (tier_name, plan)
        for tier_name in ("benchmark", "extrapolation")
        if (plan := generated_tier_plan(config, tier_name))
    ]
    if not planned_tiers:
        raise ValueError("At least one benchmark or extrapolation tier must be enabled.")

    # Collect results
    records = []
    for tier_label, plan in planned_tiers:
        print(f"[Solve] starting tier={tier_label} | instances={len(plan)}")
        solve_plan(
            config,
            plan,
            requested,
            models,
            common,
            records,
        )
    return records


def simulate(config):
    """Post-evaluate stored solutions using common random numbers.

    Returns:
        Output CSV path and evaluation rows returned by the simulation module.
    """
    simulation_config = config["simulation"]
    simulation = importlib.import_module("05_Simulation.simulation")
    return simulation.simulate_solution_files(
        [absolute(simulation_config["solutions_directory"])],
        instances_root=absolute(simulation_config["instances_directory"]),
        output_file=absolute(simulation_config["output_file"]),
        replications=int(simulation_config.get("replications", 1_000)),
        random_seed=int(simulation_config.get("random_seed", 900_042)),
    )


def main():
    """Load ``config.json`` and execute enabled workflow phases in order.

    Instance specifications are prepared once. Existing instances are validated
    only when graph data is requested, while later phases consume their own
    persisted artifacts according to the workflow flags.
    """
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    workflow = config["workflow"]
    generation = config["instances"]["generation"]
    specs = instance_specs(generation)

    if workflow.get("create_instances", False):
        instances.generate_instance_specs(
            specs,
            machine_profile_config=generation["machine_profiles"],
            due_date_config=generation["due_dates"],
            split_ratios=generation["split_ratios"],
            random_seed=generation["random_seed"],
            processing_time_range=generation["processing_times"]["base_range"],
            training_parameter_jitter=generation["training_parameter_jitter"],
        )

    splits = (configured_instance_splits(config, specs)
        if workflow.get("generate_training_data", False)
        else None
    )

    print( "Cost plus expected-repair-buffer pipeline | "+ " | ".join( f"{key}={bool(value)}" for key, value in workflow.items()) )
    if workflow.get("generate_training_data", False):
        generate_data(config, splits)
    if workflow.get("train_gnn", False):
        train(config)
    if workflow.get("solve", False):
        solve(config)
    if workflow.get("simulate", False):
        simulate(config)


if __name__ == "__main__":
    main()
