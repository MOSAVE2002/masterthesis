"""Build, solve and export one FJSP instance with a selected formulation.

The dispatcher separates model-building options from native Gurobi parameters,
loads the requested generated instance, measures build and optimization times
and writes comparable solution and optional plot artifacts.
"""

import importlib
import time
from pathlib import Path

import gurobipy as gp

from helper.gurobi_solution_writer import (
    status_name,
    write_comparable_solution,
)
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

SOLVER_MODULES = {
    "gurobi": _base,
    "gurobi_nonlinear": _nonlinear,
    "gurobi_gnn": _gnn,
}


BUILD_KEYS = {
    "gurobi": {
        "facility_cost_per_time",
        "tardiness_cost_per_time",
    },
    "gurobi_nonlinear": {
        "facility_cost_per_time",
        "tardiness_cost_per_time",
    },
    "gurobi_gnn": {
        "model_path",
        "metadata_path",
        "facility_cost_per_time",
        "tardiness_cost_per_time",
    },
}

PLOT_KEYS = {
    "plot_solution_schedule",
    "plot_solution_graph",
    "plot_candidate_graph",
    "plot_solution_graph_style",
    "plot_output_directory",
}


def _as_bool(value):
    """Interpret booleans and common truthy configuration strings.

    Other values fall back to Python's normal truth-value conversion.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _model_float(model, name):
    """Read an optional numeric Gurobi attribute as ``float``.

    Returns ``None`` when the attribute is unavailable for the model state.
    """
    try:
        return float(getattr(model, name))
    except (AttributeError, gp.GurobiError):
        return None


def _optimize(model):
    """Optimize a built model and collect search-size metrics.

    Returns:
        Branch-and-bound node count and linear-matrix nonzero count.
    """
    model.update()
    model.optimize()
    return {
        "branch_and_bound_nodes": _model_float(model, "NodeCount"),
        "linear_matrix_nonzeros": _model_float(model, "NumNZs"),
    }


def _split_parameters(solver, values):
    """Separate builder, native Gurobi, export and plotting parameters.

    Args:
        solver: Active formulation identifier.
        values: Mutable copy of supplied keyword arguments.

    Returns:
        Builder kwargs, Gurobi parameters, write flag and plot configuration.
    """
    build, gurobi = {}, {}
    write_solution = _as_bool(values.pop("write_solution", True))
    plot_config = {
        key: values.pop(key) for key in tuple(values) if key in PLOT_KEYS
    }
    for raw_key, value in values.items():
        key = raw_key
        if key in BUILD_KEYS[solver]:
            build[key] = value
        else:
            gurobi[raw_key] = value
    return build, gurobi, write_solution, plot_config


def _apply_gurobi_parameters(model, parameters):
    """Apply validated native Gurobi parameters to a model.

    Raises:
        ValueError: If one or more parameter names are unknown.
    """
    unknown = []
    for key, value in parameters.items():
        if hasattr(model.Params, key):
            setattr(model.Params, key, value)
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"Unknown Gurobi parameters: {sorted(unknown)}")


def _solution_directory(solver, variables=None, solutions_directory=None):
    """Create and return the canonical output directory for a solver variant.

    GNN solutions receive an additional architecture-specific directory derived
    from the validated model metadata.
    """
    root = Path(solutions_directory) if solutions_directory else ROOT_DIR / '02_data' / 'fjsp_solutions'
    if not root.is_absolute():
        root = ROOT_DIR / root
    directory = root / solver
    if solver == "gurobi_gnn":
        metadata = variables["gnn_metadata"]
        directory /= "fixed_candidate"
        directory /= _architectures.architecture_slug(
            convolution=metadata["convolution"],
            layers=int(metadata["num_graphsage_layers"]),
            hidden_channels=int(metadata["hidden_channels"]),
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _solution_path(solver, instance_name, variables, solutions_directory=None):
    """Construct the deterministic comparable-solution filename.

    GNN filenames include layer count, hidden width and seed to prevent model
    combinations from overwriting one another.
    """
    directory = _solution_directory(solver, variables, solutions_directory)
    if solver != "gurobi_gnn":
        return directory / f"solution_{instance_name}_{solver}.txt"
    metadata = variables["gnn_metadata"]
    return directory / (
        f"solution_{instance_name}_gurobi_gnn_"
        f"layers{int(metadata['num_graphsage_layers'])}_"
        f"hidden{int(metadata['hidden_channels'])}_"
        f"seed{int(metadata['seed'])}.txt"
    )


def solve_instance(**kwargs):
    """Load, build, optimize and optionally export one solver case.

    Keyword arguments contain the instance and solver identifiers plus builder,
    Gurobi, output and plotting settings. The returned Gurobi model remains live
    so the caller can inspect it and is responsible for disposal.

    Returns:
        Dictionary containing status, objective, bounds, timings, model size,
        artifacts, variables, instance and live Gurobi model.
    """
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
    solutions_directory = values.pop('solutions_directory', None)
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
    model, variables = SOLVER_MODULES[solver].build_fjsp(
        model, instance, **build_kwargs
    )
    build_seconds = time.perf_counter() - build_started
    optimize_started = time.perf_counter()
    solver_metrics = _optimize(model)
    optimize_wall_seconds = time.perf_counter() - optimize_started
    model_total_seconds = build_seconds + optimize_wall_seconds
    variables["timing"] = {
        "instance_load_seconds": instance_loaded - request_started,
        "model_build_seconds": build_seconds,
        "optimizer_wall_seconds": optimize_wall_seconds,
        "gurobi_runtime_seconds": float(model.Runtime),
        "build_plus_optimizer_seconds": model_total_seconds,
    }
    variables["solver_metrics"] = solver_metrics

    readable_status = status_name(model.Status)
    print(
        f"{solver} | {instance_name} | status={readable_status} | "
        f"solutions={model.SolCount} | build={build_seconds:.4f}s | "
        f"solve={model.Runtime:.4f}s | total={model_total_seconds:.4f}s"
    )
    if model.SolCount:
        print(f"  objective={model.ObjVal:.6f}")
        if variables.get("processing_cost") is not None:
            print(
                f"  processing cost={variables['processing_cost'].getValue():.6f} | "
                f"operating cost={variables['operating_cost'].getValue():.6f} | "
                "tardiness cost="
                f"{variables['tardiness_cost'].getValue():.6f}"
            )
    path = None
    if write_solution:
        path = _solution_path(solver, instance_name, variables, solutions_directory)
        write_comparable_solution(
            model, variables, path, instance=instance
        )
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
        **solver_metrics,
        "runtime": float(model.Runtime),
        "model_build_runtime": float(build_seconds),
        "optimizer_wall_runtime": float(optimize_wall_seconds),
        "build_plus_optimizer_runtime": float(model_total_seconds),
        "variables": variables,
        "model": model,
        "instance": instance,
        "plot_paths": plot_paths,
        "solution_path": str(path) if path is not None else None,
    }
    return result
