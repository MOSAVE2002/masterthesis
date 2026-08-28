"""Small solver dispatcher for the cost/repair-buffer FJSP pipeline."""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import gurobipy as gp

from helper.solution_plots import write_solution_plots


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
    "gurobi": {"facility_cost_per_time"},
    "gurobi_nonlinear": {
        "constraint_type",
        "reliability_graph_config",
        "facility_cost_per_time",
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
        "facility_cost_per_time",
    },
}

BOOLEAN_BUILD_KEYS = {
    "add_schedule_upper_bounds",
    "analytic_bounds",
}

PLOT_KEYS = {
    "plot_solution_schedule",
    "plot_solution_graph",
    "plot_candidate_graph",
    "plot_solution_graph_style",
    "plot_output_directory",
}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _model_float(model, name):
    try:
        return float(getattr(model, name))
    except (AttributeError, gp.GurobiError):
        return None


def _split_parameters(solver, values):
    build, gurobi = {}, {}
    write_solution = _as_bool(values.pop("write_solution", True))
    plot_config = {
        key: values.pop(key) for key in tuple(values) if key in PLOT_KEYS
    }
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
    return build, gurobi, write_solution, plot_config


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
    build_kwargs, gurobi_params, write_solution, plot_config = _split_parameters(
        solver, values
    )
    build_started = time.perf_counter()
    model = gp.Model(f"{solver}:{instance_name}")
    _apply_gurobi_parameters(model, gurobi_params)
    if solver == "gurobi":
        model, variables = _base.build_fjsp(
            model, instance, **build_kwargs
        )
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
        if variables.get("processing_cost") is not None:
            print(
                f"  processing cost={variables['processing_cost'].getValue():.6f} | "
                f"operating cost={variables['operating_cost'].getValue():.6f}"
            )
    if write_solution:
        path = _solution_path(solver, instance_name, variables)
        writer(model, variables, instance, path)
        print(f"  solution={path}")
    plot_paths = write_solution_plots(
        model,
        variables,
        instance,
        instance_name=instance_name,
        solver=solver,
        output_directory=plot_config.get(
            "plot_output_directory", "plots/fjsp_solution_plots"
        ),
        plot_solution_schedule_enabled=_as_bool(
            plot_config.get("plot_solution_schedule", False)
        ),
        plot_solution_graph_enabled=_as_bool(
            plot_config.get("plot_solution_graph", False)
        ),
        plot_candidate_graph_enabled=_as_bool(
            plot_config.get("plot_candidate_graph", False)
        ),
        graph_style=plot_config.get(
            "plot_solution_graph_style", "disjunctive"
        ),
    )
    result = {
        "status": int(model.Status),
        "solution_count": int(model.SolCount),
        "objective": float(model.ObjVal) if model.SolCount else None,
        "best_bound": _model_float(model, "ObjBound"),
        "mip_gap": _model_float(model, "MIPGap") if model.SolCount else None,
        "runtime": float(model.Runtime),
        "model_build_runtime": float(build_seconds),
        "optimizer_wall_runtime": float(optimize_wall_seconds),
        "build_plus_optimizer_runtime": float(model_total_seconds),
        "variables": variables,
        "model": model,
        "instance": instance,
        "plot_paths": plot_paths,
    }
    return result


def solve_instances_with_solver(**kwargs):
    """Compatibility wrapper for the single-instance active workflow."""
    return solveModel(**kwargs)
