"""Entry point for the job-specific expected-repair-buffer pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from pathlib import Path

from helper.sequence_setup import normalize_reliability_graph_config
from helper.stochastic_fjsp import normalize_machine_profile_config
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


def _due_date_generation_config(config):
    """Return due-date generation settings."""
    return dict(
        config["instances"]["generation"].get("due_dates") or {}
    )


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
            machine_profile_config=generation.get("machine_profiles"),
            time_unit_minutes=generation.get("time_unit_minutes", 1.0),
        )

    due_date_config = _due_date_generation_config(config)
    expected_due_date_method = str(
        due_date_config.get("method", "total_work_content")
    )
    expected_due_date_aggregation = str(
        due_date_config.get("machine_aggregation", "mean")
    )
    expected_due_date_assignment = str(
        due_date_config.get("assignment", "cyclic")
    )
    due_date_factors = [
        float(value)
        for value in due_date_config.get("factors", [])
    ]
    if not due_date_factors:
        raise ValueError("instances.generation.due_dates.factors is empty.")
    for name in (
        instance_name
        for split_names in splits.values()
        for instance_name in split_names
    ):
        instance = _instances.load_generated_instance(name)
        expected_factor = due_date_factors[
            (int(instance.nb_instance) - 1) % len(due_date_factors)
        ]
        actual_factor = getattr(instance, "due_date_factor", None)
        due_date_context_matches = (
            getattr(instance, "due_date_method", None)
            == expected_due_date_method
            and getattr(instance, "due_date_machine_aggregation", None)
            == expected_due_date_aggregation
            and getattr(instance, "due_date_assignment", None)
            == expected_due_date_assignment
        )
        if (
            actual_factor is None
            or not math.isclose(
                float(actual_factor),
                expected_factor,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not due_date_context_matches
        ):
            raise ValueError(
                f"Instance {name} contains due_date_factor="
                f"{actual_factor} and due_date_method="
                f"{getattr(instance, 'due_date_method', None)!r}, but config "
                f"requires factor={expected_factor}, method="
                f"{expected_due_date_method!r}, machine_aggregation="
                f"{expected_due_date_aggregation!r}, assignment="
                f"{expected_due_date_assignment!r}. "
                "Regenerate the instances before running the pipeline."
            )
    return splits


def _generate_data(config, splits):
    data = config["training"]["data_generation"]
    ranges = data.get("reliability_ranges") or {}
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
        "simulation": data.get("simulation"),
        "instance_failure_handling": data.get(
            "instance_failure_handling"
        ),
        "alpha_range": ranges.get("alpha"),
        "beta_range": ranges.get("beta"),
        "repair_rate_range": ranges.get("repair_rate"),
        "weibull_scale_factors": data.get("weibull_scale_factors"),
        "machine_profile_config": config["instances"]["generation"][
            "machine_profiles"
        ],
        "time_unit_minutes": config["instances"]["generation"].get(
            "time_unit_minutes", 1.0
        ),
        "adaptive_due_dates": data.get("adaptive_due_dates"),
        "reliability_graph": config["constraint"]["weibull"][
            "reliability_graph"
        ],
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
    if tier_name not in {"benchmark", "extrapolation", "stress"}:
        raise ValueError(f"Unknown generated evaluation tier: {tier_name}")
    evaluation = config["solve"].get("evaluation", {})
    create_instances = bool(
        config["solve"].get("create_instances", False)
    )
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
    calibrated_config = tier.get("due_dates")
    if calibrated_config is not None:
        calibrated_config = dict(calibrated_config)
        if (
            tier.get("due_date_factors") is not None
            or tier.get("due_date_factor") is not None
        ):
            raise ValueError(
                f"{tier_name}.due_dates cannot be combined with legacy "
                "due-date factors."
            )
        method = str(calibrated_config.get("method", ""))
        aggregation = str(
            calibrated_config.get("machine_aggregation", "mean")
        )
        if method != "calibrated_total_work_content":
            raise ValueError(
                f"{tier_name}.due_dates.method must be "
                "'calibrated_total_work_content'."
            )
        if aggregation != "mean":
            raise ValueError(
                f"{tier_name}.due_dates.machine_aggregation must be 'mean'."
            )
        has_scalar_offset = "relative_makespan_offset" in calibrated_config
        has_offset_list = "relative_makespan_offsets" in calibrated_config
        if has_scalar_offset and has_offset_list:
            raise ValueError(
                f"{tier_name}.due_dates must use either "
                "relative_makespan_offset or relative_makespan_offsets, "
                "not both."
            )
        raw_offsets = calibrated_config.get("relative_makespan_offsets")
        if raw_offsets is None:
            raw_offsets = [
                calibrated_config.get("relative_makespan_offset", 0.0)
            ]
        if not isinstance(raw_offsets, (list, tuple)) or not raw_offsets:
            raise ValueError(
                f"{tier_name}.due_dates.relative_makespan_offsets must be "
                "a non-empty list."
            )
        offsets = [float(value) for value in raw_offsets]
        if any(
            not math.isfinite(offset) or offset <= -1.0
            for offset in offsets
        ):
            raise ValueError(
                f"{tier_name}.due_dates relative makespan offsets must be "
                "finite and greater than -1."
            )
        if len(set(offsets)) != len(offsets):
            raise ValueError(
                f"{tier_name}.due_dates.relative_makespan_offsets must not "
                "contain duplicates."
            )
        variants = []
        factor_slugs = set()
        for offset in offsets:
            offset_slug = (
                f"{offset:.2f}".replace("-", "m").replace(".", "p")
            )
            factor_slug = f"twk_d{offset_slug}"
            if factor_slug in factor_slugs:
                raise ValueError(
                    f"{tier_name}.due_dates relative makespan offsets must "
                    "remain distinct when rounded to two decimals for "
                    "directory names."
                )
            factor_slugs.add(factor_slug)
            variants.append({
                "factor": None,
                "factor_slug": factor_slug,
                "calibrated_config": calibrated_config,
                "relative_makespan_offset": offset,
            })
    else:
        raw_factors = tier.get("due_date_factors")
        if raw_factors is None:
            scalar = tier.get("due_date_factor")
            raw_factors = [scalar] if scalar is not None else [None]
        factors = []
        for raw_factor in raw_factors:
            factor = None if raw_factor is None else float(raw_factor)
            if factor is not None and factor <= 0.0:
                raise ValueError(
                    f"{tier_name}.due_date_factors must be positive."
                )
            if factor in factors:
                raise ValueError(
                    f"{tier_name}.due_date_factors must not contain "
                    "duplicates."
                )
            factors.append(factor)
        variants = [{
            "factor": factor,
            "factor_slug": (
                "configured"
                if factor is None else f"df{factor:.2f}".replace(".", "p")
            ),
            "calibrated_config": None,
            "relative_makespan_offset": None,
        } for factor in factors]

    plan = []
    for variant in variants:
        factor = variant["factor"]
        factor_slug = variant["factor_slug"]
        relative_offset = variant["relative_makespan_offset"]
        directory = root / tier_name / factor_slug
        due_date_config = _due_date_generation_config(config)
        if factor is not None:
            due_date_config["factors"] = [factor]
        instance_postprocessor = None
        if variant["calibrated_config"] is not None:
            benchmark_calibration = importlib.import_module(
                "04_GraphNeuralNetworks.models."
                "generate_fix_and_optimize_training_data"
            )

            def instance_postprocessor(instance, *, _variant=variant):
                calibrated = (
                    benchmark_calibration
                    .calibrated_total_work_content_due_dates(
                        instance,
                        _variant["calibrated_config"],
                        _variant["relative_makespan_offset"],
                    )
                )
                calibration = calibrated["calibration"]
                instance.due_dates = dict(calibrated["due_dates"])
                instance.due_date_method = "calibrated_total_work_content"
                instance.due_date_machine_aggregation = "mean"
                instance.due_date_assignment = "nominal_makespan_calibrated"
                instance.due_date_factor = calibrated["effective_factor"]
                instance.due_date_work_content = dict(
                    calibrated["work_content"]
                )
                instance.due_date_relative_makespan_offset = calibrated[
                    "relative_offset"
                ]
                instance.nominal_makespan_calibration = calibration[
                    "makespan"
                ]
                instance.nominal_twk_due_date_factor = calibrated[
                    "nominal_factor"
                ]
                instance.due_date_calibration_status = calibration["status"]
                instance.due_date_calibration_gap = calibration["gap"]
                instance.due_date_calibration_runtime_seconds = calibration[
                    "runtime_seconds"
                ]
                instance.nominal_calibration_assignment = dict(
                    calibration.get("assignment") or {}
                )

        suffix = f"{tier_name}_{factor_slug}"
        expected_paths = [
            directory / (
                f"i{spec['num_jobs']}_k{spec['num_machines']}_"
                f"o{min(spec['operations_per_job'])}-"
                f"{max(spec['operations_per_job'])}_{instance_number}_"
                f"{suffix}.pkl"
            )
            for spec in specs
            for instance_number in range(1, spec["count"] + 1)
        ]
        existing_paths = (
            sorted(directory.glob("*.pkl")) if directory.exists() else []
        )
        reuse_existing = bool(
            tier.get("reuse_existing_instances", True)
        )
        if create_instances:
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
                machine_parameter_ranges=generation.get(
                    "machine_parameters"
                ),
                machine_profile_config=generation.get("machine_profiles"),
                due_date_config=due_date_config,
                instance_postprocessor=instance_postprocessor,
                instance_name_suffix=suffix,
                time_unit_minutes=generation.get("time_unit_minutes", 1.0),
            )
        elif reuse_existing and existing_paths:
            expected_set = {path.resolve() for path in expected_paths}
            existing_set = {path.resolve() for path in existing_paths}
            if existing_set != expected_set:
                missing = sorted(
                    path.name for path in expected_set - existing_set
                )
                unexpected = sorted(
                    path.name for path in existing_set - expected_set
                )
                raise ValueError(
                    f"Existing {tier_name} instances do not match the "
                    "configured evaluation plan. To preserve benchmark "
                    "instances, no files were overwritten. "
                    f"Missing={missing}; unexpected={unexpected}. Set "
                    "solve.create_instances to true to replace this folder "
                    "with the exactly configured set, or move "
                    f"{directory} explicitly."
                )
            paths = expected_paths
            if variant["calibrated_config"] is not None:
                for path in paths:
                    instance = _instances.load_generated_instance(
                        path.stem, directory
                    )
                    actual_offset = getattr(
                        instance, "due_date_relative_makespan_offset", None
                    )
                    nominal_factor = getattr(
                        instance, "nominal_twk_due_date_factor", None
                    )
                    actual_factor = getattr(instance, "due_date_factor", None)
                    work_content = getattr(
                        instance, "due_date_work_content", {}
                    )
                    stored_due_dates = getattr(instance, "due_dates", {})
                    context_matches = (
                        getattr(instance, "due_date_method", None)
                        == "calibrated_total_work_content"
                        and getattr(
                            instance,
                            "due_date_machine_aggregation",
                            None,
                        ) == "mean"
                        and getattr(instance, "due_date_assignment", None)
                        == "nominal_makespan_calibrated"
                    )
                    factor_matches = (
                        nominal_factor is not None
                        and actual_factor is not None
                        and math.isclose(
                            float(actual_factor),
                            (1.0 + relative_offset)
                            * float(nominal_factor),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    )
                    due_dates_match = (
                        factor_matches
                        and set(work_content) == set(instance.jobs)
                        and set(stored_due_dates) == set(instance.jobs)
                        and all(
                            math.isclose(
                                float(stored_due_dates[job]),
                                float(math.ceil(
                                    float(actual_factor)
                                    * float(work_content[job])
                                    - 1e-12
                                )),
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                            for job in instance.jobs
                        )
                    )
                    if (
                        actual_offset is None
                        or not math.isclose(
                            float(actual_offset),
                            relative_offset,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        or not context_matches
                        or not due_dates_match
                    ):
                        raise ValueError(
                            f"Existing benchmark instance {path.name} does "
                            "not match the calibrated TWK configuration. "
                            "Regenerate the benchmark instances explicitly."
                        )
            print(
                f"Reusing {len(paths)} unchanged {tier_name} instances "
                f"from: {directory}",
                flush=True,
            )
        elif not existing_paths:
            raise FileNotFoundError(
                f"No {tier_name} instances exist in {directory}. Set "
                "solve.create_instances to true once to generate exactly "
                "the configured evaluation set."
            )
        else:
            raise ValueError(
                f"Existing {tier_name} instances may not be regenerated "
                "while solve.create_instances is false. Either enable "
                "reuse_existing_instances or set solve.create_instances "
                "to true explicitly."
            )
        plan.extend({
            "tier": tier_name,
            "instance_name": path.stem,
            "instance_directory": str(directory),
            "due_date_factor": factor,
            "relative_makespan_offset": relative_offset,
        } for path in paths)
    return plan


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
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            configured_profiles = normalize_machine_profile_config(
                config["instances"]["generation"]["machine_profiles"]
            )
            metadata_profiles = metadata.get("machine_profile_config")
            configured_time_unit = float(
                config["instances"]["generation"].get(
                    "time_unit_minutes", 1.0
                )
            )
            if (
                metadata_profiles is None
                or normalize_machine_profile_config(metadata_profiles)
                != configured_profiles
                or not math.isclose(
                    float(metadata.get("time_unit_minutes", -1.0)),
                    configured_time_unit,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Configured machine profiles or time units differ from "
                    f"the trained GNN metadata {metadata_path}. Regenerate "
                    "the training data and retrain all GNN models before "
                    "running gurobi_gnn."
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
        "benchmark": evaluation.get("benchmark", {}).get(
            "enabled", False
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
    common["facility_cost_per_time"] = float(
        config["objective"].get("facility_cost_per_time", 1.0)
    )
    common["service_violation_cost_per_time"] = float(
        config["objective"].get("service_violation_cost_per_time", 1.0)
    )
    constraint = config["constraint"]
    stochastic = {
        "constraint_type": constraint["type"],
        "reliability_graph_config": constraint["weibull"][
            "reliability_graph"
        ],
        "service_level": float(
            constraint["weibull"].get("service_level", 0.90)
        ),
    }
    requested = [name.lower() for name in config["solve"]["solvers"]]
    models = _trained_models(config) if "gurobi_gnn" in requested else []

    tier_plans = (
        ("solve", lambda: _in_distribution_plan(config, splits)),
        ("benchmark", lambda: _generated_tier_plan(config, "benchmark")),
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
    result = evaluator(config)
    return result


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
            machine_profile_config=generation.get("machine_profiles"),
            due_date_config=_due_date_generation_config(config),
            time_unit_minutes=generation.get("time_unit_minutes", 1.0),
        )
    splits = _configured_instance_splits(config, specs)
    print(
        "Cost plus expected-repair-buffer pipeline | "
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
