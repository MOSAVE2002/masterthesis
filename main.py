"""Entry point for the jobspecific service-probability pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from pathlib import Path

from helper.sequence_setup import normalize_reliability_graph_config
from helper.start_solve_ins import solve_instances_with_solver
from helper.surrogate_constraint import target_column


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json"

WORKFLOW_PHASES = (
    "create-instances",
    "generate-training-data",
    "train-gnn",
    "solve",
    "evaluate",
)

_instances = importlib.import_module("01_generator.instance_generator")
_architectures = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run selected phases of the FJSP pipeline."
    )
    parser.add_argument(
        "--workflow",
        nargs="+",
        choices=(*WORKFLOW_PHASES, "all"),
        metavar="PHASE",
        help=(
            "pipeline phases to run; omit this option to use config.json. "
            f"Available phases: {', '.join(WORKFLOW_PHASES)}, all"
        ),
    )
    args = parser.parse_args(argv)
    if args.workflow and "all" in args.workflow and len(args.workflow) > 1:
        parser.error("'all' cannot be combined with other workflow phases")
    return args


def _resolve_workflow(config_workflow, selected_phases):
    if selected_phases is None:
        return dict(config_workflow)

    selected = (
        set(WORKFLOW_PHASES)
        if selected_phases == ["all"]
        else set(selected_phases)
    )
    return {
        phase.replace("-", "_"): phase in selected
        for phase in WORKFLOW_PHASES
    }


def _absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def _instance_specs(generation):
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


def _configured_instance_splits(config, specs):
    data = config["training"]["data_generation"]
    explicit = data.get("instance_names")
    if explicit:
        key_map = {"training": "train", "valid": "valid", "test": "test"}
        unknown = set(explicit) - set(key_map)
        if unknown:
            raise ValueError(f"Unknown instance split names: {sorted(unknown)}")
        splits = {
            split: list(explicit.get(config_key, []))
            for config_key, split in key_map.items()
        }
        all_names = [name for names in splits.values() for name in names]
        if len(all_names) != len(set(all_names)):
            raise ValueError("Training, validation and test instances overlap.")
    else:
        generation = config["instances"]["generation"]
        splits = _instances.configured_instance_names_by_split(
            specs,
            split_ratios=generation.get("split_ratios"),
            random_seed=generation.get("random_seed", 42),
            processing_time_range=generation.get(
                "processing_times", {}
            ).get("base_range"),
            processing_time_deviation=generation.get(
                "processing_times", {}
            ).get("relative_deviation"),
            machine_parameter_ranges=generation.get("machine_parameters"),
        )

    expected_service_level = float(
        config["constraint"]["weibull"]["reliability_graph"][
            "service_level"
        ]
    )
    for name in (
        instance_name
        for split_names in splits.values()
        for instance_name in split_names
    ):
        instance = _instances.load_generated_instance(name)
        mismatches = {
            job: float(instance.service_levels[job])
            for job in instance.jobs
            if not math.isclose(
                float(instance.service_levels[job]),
                expected_service_level,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        }
        if mismatches:
            raise ValueError(
                f"Instance {name} contains service levels {mismatches}, "
                f"but config requires {expected_service_level}. Regenerate "
                "or migrate the instances before running the pipeline."
            )
    return splits


def _generate_data(config, splits):
    data = config["training"]["data_generation"]
    ranges = data["reliability_ranges"]
    generator = importlib.import_module(
        "04_GraphNeuralNetworks.models.generate_weibull_training_data"
    ).generate_from_config
    generator({
        "method": data["method"],
        "instance_splits": splits,
        "generate_splits": data.get("generate_splits"),
        "output_directory": data["output_directory"],
        "random_seed": int(data.get("random_seed", 42)),
        "samples_per_instance": int(data["samples_per_instance"]),
        "alpha_range": ranges["alpha"],
        "beta_range": ranges["beta"],
        "repair_rate_range": ranges["repair_rate"],
        "reliability_graph": config["constraint"]["weibull"][
            "reliability_graph"
        ],
        "simulation": data["simulation"],
        "fixed_y": data.get("fixed_y"),
    })


def _train(config):
    gnn = config["training"]["gnn"]
    trainer = importlib.import_module(
        "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
    ).train_from_config
    return trainer(seed=int(gnn.get("seed", 42)))


def _in_distribution_plan(config, splits):
    evaluation = config["solve"].get("evaluation", {})
    tier = evaluation.get("in_distribution", {})
    if not tier.get("enabled", True):
        return []
    count = int(tier.get("instances_per_size", 1))
    if count <= 0:
        raise ValueError(
            "in_distribution.instances_per_size must be positive."
        )
    generation = config["instances"]["generation"]
    jobs = [
        int(value)
        for value in tier.get("num_jobs", generation["num_jobs"])
    ]
    machines = [
        int(value)
        for value in tier.get("num_machines", generation["num_machines"])
    ]
    if not jobs or not machines or min(jobs + machines) <= 0:
        raise ValueError(
            "in_distribution.num_jobs and num_machines must be non-empty "
            "lists of positive integers."
        )
    if len(jobs) != len(set(jobs)) or len(machines) != len(set(machines)):
        raise ValueError(
            "in_distribution.num_jobs and num_machines must not contain "
            "duplicates."
        )

    candidates_by_size = {}
    for name in splits["test"]:
        instance = _instances.load_generated_instance(name)
        size = (int(instance.num_jobs), int(instance.num_machines))
        candidates_by_size.setdefault(size, []).append(name)

    plan = []
    base_seed = int(evaluation.get("random_seed", 2026))
    for size_index, (num_jobs, num_machines) in enumerate(
        (job_count, machine_count)
        for job_count in jobs
        for machine_count in machines
    ):
        candidates = candidates_by_size.get((num_jobs, num_machines), [])
        if count > len(candidates):
            raise ValueError(
                "in_distribution.instances_per_size requests "
                f"{count} instances for {num_jobs} jobs and "
                f"{num_machines} machines, but the test split contains "
                f"only {len(candidates)}."
            )
        rng = random.Random(base_seed + size_index)
        plan.extend(
            {
                "tier": "in_distribution",
                "instance_name": name,
                "num_jobs": num_jobs,
                "num_machines": num_machines,
            }
            for name in sorted(rng.sample(candidates, count))
        )
    return plan


def _generated_tier_plan(config, tier_name):
    if tier_name not in {"extrapolation", "stress"}:
        raise ValueError(f"Unknown generated evaluation tier: {tier_name}")
    evaluation = config["solve"].get("evaluation", {})
    generation = config["instances"]["generation"]
    root = _absolute(
        evaluation.get(
            "instance_directory", "02_data/fjsp_instances/evaluation"
        )
    )
    tier = evaluation.get(tier_name, {})
    if not tier.get("enabled", False):
        return []
    specs = [
        {
            "num_jobs": int(jobs),
            "num_machines": int(machines),
            "operations_per_job": list(
                tier.get(
                    "operations_per_job",
                    generation["operations_per_job"],
                )
            ),
            "count": int(tier["instances_per_size"]),
        }
        for jobs in tier["num_jobs"]
        for machines in tier["num_machines"]
    ]
    directory = root / tier_name
    paths = _instances.generate_evaluation_instance_specs(
        specs,
        random_seed=int(evaluation.get("random_seed", 2026)),
        output_directory=directory,
        processing_time_range=generation.get(
            "processing_times", {}
        ).get("base_range"),
        processing_time_deviation=generation.get(
            "processing_times", {}
        ).get("relative_deviation", 0.2),
        machine_parameter_ranges=generation.get("machine_parameters"),
    )
    return [
        {
            "tier": tier_name,
            "instance_name": path.stem,
            "instance_directory": str(directory),
        }
        for path in paths
    ]


def _trained_models(config):
    training = config["training"]
    gnn = training["gnn"]
    model_root = _absolute(gnn["model_directory"])
    seed = int(gnn.get("seed", 42))
    target = target_column(config["constraint"]["type"])
    models = []
    for raw in gnn["combinations"]:
        for architecture in _architectures.expand_architecture_variants(raw):
            directory = _architectures.architecture_model_dir(
                model_root,
                architecture["graph_mode"],
                architecture["convolution"],
                architecture["aggregation"],
                architecture["pooling"],
                layers=architecture["layers"],
                hidden_channels=architecture["hidden_channels"],
            )
            stem = _architectures.architecture_stem(
                architecture["graph_mode"],
                architecture["convolution"],
                architecture["aggregation"],
                architecture["pooling"],
                target,
                seed,
                layers=architecture["layers"],
                hidden_channels=architecture["hidden_channels"],
            )
            model_path = directory / f"{stem}.pt"
            metadata_path = directory / f"{stem}_meta.json"
            if not model_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(
                    f"Trained GNN missing: {model_path} / {metadata_path}"
                )
            models.append({
                "convolution": architecture["convolution"],
                "aggregation": architecture["aggregation"],
                "pooling": architecture["pooling"],
                "layers": architecture["layers"],
                "hidden_channels": architecture["hidden_channels"],
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
            })
    return models


def _solve_plan(
    config,
    plan,
    requested,
    models,
    common,
    stochastic,
):
    solver_config = config["solvers"]["gurobi"]
    for item in plan:
        instance_args = {
            "instance_name": item["instance_name"],
            **(
                {"instance_directory": item["instance_directory"]}
                if item.get("instance_directory") else {}
            ),
        }
        for solver in requested:
            if solver == "gurobi":
                solve_instances_with_solver(
                    solver=solver,
                    **instance_args,
                    **common,
                )
            elif solver == "gurobi_nonlinear":
                solve_instances_with_solver(
                    solver=solver,
                    **instance_args,
                    **common,
                    **stochastic,
                    **solver_config.get("nonlinear", {}),
                )
            elif solver == "gurobi_gnn":
                for trained in models:
                    solve_instances_with_solver(
                        solver=solver,
                        **instance_args,
                        **common,
                        **stochastic,
                        **solver_config.get("gnn", {}),
                        **trained,
                    )
            else:
                raise ValueError(f"Unknown solver: {solver}")


def _solve(config, splits):
    evaluation = config["solve"].get("evaluation", {})
    enabled = {
        "in_distribution": evaluation.get("in_distribution", {}).get(
            "enabled", True
        ),
        "extrapolation": evaluation.get("extrapolation", {}).get(
            "enabled", False
        ),
        "stress": evaluation.get("stress", {}).get("enabled", False),
    }
    if not any(enabled.values()):
        raise ValueError("At least one solve evaluation tier must be enabled.")

    solver_config = config["solvers"]["gurobi"]
    common = dict(solver_config.get("common", {}))
    constraint = config["constraint"]
    stochastic = {
        "constraint_type": constraint["type"],
        "reliability_graph_config": constraint["weibull"][
            "reliability_graph"
        ],
    }
    requested = [name.lower() for name in config["solve"]["solvers"]]
    models = _trained_models(config) if "gurobi_gnn" in requested else []

    tier_plans = (
        ("solve", lambda: _in_distribution_plan(config, splits)),
        (
            "extrapolation",
            lambda: _generated_tier_plan(config, "extrapolation"),
        ),
        ("stress", lambda: _generated_tier_plan(config, "stress")),
    )
    for tier_label, build_plan in tier_plans:
        plan = build_plan()
        if not plan:
            continue
        print(
            f"[Solve] starting tier={tier_label} | instances={len(plan)}",
            flush=True,
        )
        _solve_plan(
            config,
            plan,
            requested,
            models,
            common,
            stochastic,
        )


def _evaluate(config):
    evaluator = importlib.import_module(
        "06_Evaluation.evaluate_solutions"
    ).evaluate_from_config
    return evaluator(config)


def main(argv=None):
    args = _parse_args(argv)
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    workflow = _resolve_workflow(config["workflow"], args.workflow)
    generation = config["instances"]["generation"]
    specs = _instance_specs(generation)
    normalize_reliability_graph_config(
        config["constraint"]["weibull"]["reliability_graph"]
    )
    if workflow.get("create_instances", False):
        _instances.generate_instance_specs(
            specs,
            split_ratios=generation.get("split_ratios"),
            random_seed=generation.get("random_seed", 42),
            processing_time_range=generation.get(
                "processing_times", {}
            ).get("base_range"),
            processing_time_deviation=generation.get(
                "processing_times", {}
            ).get("relative_deviation", 0.2),
            machine_parameter_ranges=generation.get("machine_parameters"),
        )
    splits = _configured_instance_splits(config, specs)
    print(
        "Per-job probability pipeline | "
        + " | ".join(f"{key}={bool(value)}" for key, value in workflow.items())
    )
    if workflow.get("generate_training_data", False):
        _generate_data(config, splits)
    if workflow.get("train_gnn", False):
        _train(config)
    if workflow.get("solve", False):
        _solve(config, splits)
    if workflow.get("evaluate", False):
        _evaluate(config)


if __name__ == "__main__":
    main()
