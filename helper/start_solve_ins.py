"""Small solver dispatcher for the per-job service-probability pipeline."""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import gurobipy as gp


ROOT_DIR = Path(__file__).resolve().parents[1]

_instances = importlib.import_module("01_generator.instance_generator")
load_generated_instance = _instances.load_generated_instance

_base = importlib.import_module("03_Gurobi.build_fjsp")
_nonlinear = importlib.import_module("03_Gurobi.build_fjsp_with_nonlinear")
_gnn = importlib.import_module("03_Gurobi.build_fjsp_with_gnn")
_architectures = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)


BUILD_KEYS = {
    "gurobi": set(),
    "gurobi_nonlinear": {
        "constraint_type",
        "reliability_graph_config",
    },
    "gurobi_gnn": {
        "model_path",
        "metadata_path",
        "convolution",
        "aggregation",
        "pooling",
        "layers",
        "hidden_channels",
        "add_schedule_upper_bounds",
        "constraint_type",
        "reliability_graph_config",
        "analytic_bounds",
    },
}

BOOLEAN_BUILD_KEYS = {
    "add_schedule_upper_bounds",
    "analytic_bounds",
}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _split_parameters(solver, values):
    build, gurobi = {}, {}
    write_solution = _as_bool(values.pop("write_solution", True))
    aliases = {"type": "constraint_type"}
    for raw_key, value in values.items():
        key = aliases.get(raw_key.lower(), raw_key)
        if key in BUILD_KEYS[solver]:
            if key in BOOLEAN_BUILD_KEYS:
                value = _as_bool(value)
            elif key in {"layers", "hidden_channels"}:
                value = int(value)
            build[key] = value
        else:
            gurobi[raw_key] = value
    return build, gurobi, write_solution


def _apply_gurobi_parameters(model, parameters):
    unknown = []
    for key, value in parameters.items():
        if hasattr(model.Params, key):
            setattr(model.Params, key, value)
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"Unknown Gurobi parameters: {sorted(unknown)}")


def _solution_directory(solver, variables=None):
    directory = ROOT_DIR / "02_data" / "fjsp_solutions" / solver
    if solver == "gurobi_gnn":
        metadata = variables["gnn_metadata"]
        directory /= "fixed_candidate"
        directory /= _architectures.architecture_slug(
            graph_mode="fixed_candidate",
            convolution=metadata["convolution"],
            aggregation=metadata["aggregation"],
            pooling=metadata["pooling"],
            layers=int(metadata["num_graphsage_layers"]),
            hidden_channels=int(metadata["hidden_channels"]),
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _solution_path(solver, instance_name, variables):
    directory = _solution_directory(solver, variables)
    if solver != "gurobi_gnn":
        return directory / f"solution_{instance_name}_{solver}.txt"
    metadata = variables["gnn_metadata"]
    return directory / (
        f"solution_{instance_name}_gurobi_gnn_"
        f"layers{int(metadata['num_graphsage_layers'])}_"
        f"hidden{int(metadata['hidden_channels'])}_"
        f"seed{int(metadata['seed'])}.txt"
    )


def _minimum_service_slack(variables, instance):
    probabilities = variables.get("job_ontime_probabilities")
    if probabilities:
        def probability_value(item):
            value = (
                float(item.X)
                if hasattr(item, "X")
                else float(item.getValue())
                if hasattr(item, "getValue")
                else float(item)
            )
            if variables.get("job_probability_postprocess") == (
                "relu_clip_0_1_outside_model"
            ):
                return min(1.0, max(0.0, value))
            return value

        return min(
            probability_value(probabilities[job])
            - float(variables["service_levels"][job])
            for job in instance.jobs
        )
    if not variables.get("service_buffers"):
        return None
    return min(
        float(variables["due_dates"][job])
        - float(variables["C"][instance.job_end_operations[job]].X)
        - float(variables["service_buffers"][job].getValue())
        for job in instance.jobs
    )


def solveModel(**kwargs):
    """Load and solve one instance with one active Gurobi formulation."""
    request_started = time.perf_counter()
    values = dict(kwargs)
    try:
        instance_name = values.pop("instance_name")
    except KeyError as exc:
        raise ValueError("instance_name is required.") from exc
    solver = str(values.pop("solver", "gurobi_nonlinear")).lower()
    if solver not in BUILD_KEYS:
        raise ValueError(
            "solver must be 'gurobi', 'gurobi_nonlinear' or 'gurobi_gnn'."
        )
    instance_directory = values.pop("instance_directory", None)
    instance = load_generated_instance(
        instance_name, instance_directory=instance_directory
    )
    instance_loaded = time.perf_counter()
    build_kwargs, gurobi_params, write_solution = _split_parameters(
        solver, values
    )
    build_started = time.perf_counter()
    model = gp.Model(f"{solver}:{instance_name}")
    _apply_gurobi_parameters(model, gurobi_params)
    if solver == "gurobi":
        model, variables = _base.build_fjsp(model, instance)
        writer = _base.write_solution_file
    elif solver == "gurobi_nonlinear":
        model, variables = _nonlinear.build_fjsp(
            model, instance, **build_kwargs
        )
        writer = _nonlinear.write_solution_file
    else:
        model, variables = _gnn.build_fjsp(
            model, instance, **build_kwargs
        )
        writer = _gnn.write_solution_file
    build_seconds = time.perf_counter() - build_started
    optimize_started = time.perf_counter()
    model.optimize()
    optimize_wall_seconds = time.perf_counter() - optimize_started
    model_total_seconds = build_seconds + optimize_wall_seconds
    variables["timing"] = {
        "instance_load_seconds": instance_loaded - request_started,
        "model_build_seconds": build_seconds,
        "optimizer_wall_seconds": optimize_wall_seconds,
        "gurobi_runtime_seconds": float(model.Runtime),
        "build_plus_optimizer_seconds": model_total_seconds,
    }

    status_name = _base.STATUS_NAMES.get(model.Status, str(model.Status))
    print(
        f"{solver} | {instance_name} | status={status_name} | "
        f"solutions={model.SolCount} | build={build_seconds:.4f}s | "
        f"solve={model.Runtime:.4f}s | total={model_total_seconds:.4f}s"
    )
    if model.SolCount:
        print(f"  objective={model.ObjVal:.6f}")
        service_slack = _minimum_service_slack(variables, instance)
        if service_slack is not None:
            print(f"  minimum alpha-service slack={service_slack:.6f}")
    if write_solution:
        path = _solution_path(solver, instance_name, variables)
        writer(model, variables, instance, path)
        print(f"  solution={path}")
    result = {
        "status": int(model.Status),
        "solution_count": int(model.SolCount),
        "objective": float(model.ObjVal) if model.SolCount else None,
        "best_bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap) if model.SolCount else None,
        "runtime": float(model.Runtime),
        "model_build_runtime": float(build_seconds),
        "optimizer_wall_runtime": float(optimize_wall_seconds),
        "build_plus_optimizer_runtime": float(model_total_seconds),
        "variables": variables,
        "model": model,
        "instance": instance,
    }
    return result


def solve_instances_with_solver(**kwargs):
    """Compatibility wrapper for the single-instance active workflow."""
    return solveModel(**kwargs)
