"""Small solver dispatcher for the cost/repair-buffer FJSP pipeline."""

from __future__ import annotations

import importlib
import math
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from helper.gurobi_solution_writer import solver_progress_path
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
    "gurobi": {
        "facility_cost_per_time",
        "service_violation_cost_per_time",
        "tardiness_cost_per_time",
    },
    "gurobi_nonlinear": {
        "constraint_type",
        "reliability_graph_config",
        "facility_cost_per_time",
        "service_violation_cost_per_time",
        "tardiness_cost_per_time",
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
        "service_violation_cost_per_time",
        "tardiness_cost_per_time",
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


def _optimize_with_telemetry(model):
    """Optimize once and retain scalar and time-resolved MIP telemetry."""
    model.update()
    progress_sample_interval_seconds = 0.25
    telemetry = {
        "root_node_bound": None,
        "time_to_first_incumbent_seconds": None,
        "first_incumbent_objective": None,
        "time_to_best_incumbent_seconds": None,
        "best_incumbent_objective": None,
        "progress_sample_interval_seconds": progress_sample_interval_seconds,
        "progress_trace": [],
    }
    objective_sense = int(model.ModelSense)
    progress_state = {
        "runtime_seconds": None,
        "incumbent_objective": None,
        "best_bound": None,
        "solution_count": None,
    }

    def finite_callback_value(callback_model, code, *, objective=False):
        try:
            value = float(callback_model.cbGet(code))
        except (AttributeError, gp.GurobiError, TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        if objective and abs(value) >= 0.5 * float(GRB.INFINITY):
            return None
        return value

    def relative_gap(incumbent, bound):
        if incumbent is None or bound is None:
            return None
        difference = abs(incumbent - bound)
        if abs(incumbent) <= 1e-10:
            return 0.0 if difference <= 1e-10 else None
        return difference / abs(incumbent)

    def changed(current, previous):
        if current is None or previous is None:
            return current != previous
        return not math.isclose(
            float(current), float(previous), rel_tol=1e-12, abs_tol=1e-12
        )

    def record_progress(
        runtime, incumbent, bound, nodes, solution_count, event, *, force=False
    ):
        if runtime is None:
            return
        last_runtime = progress_state["runtime_seconds"]
        material_change = any((
            changed(incumbent, progress_state["incumbent_objective"]),
            changed(bound, progress_state["best_bound"]),
            solution_count != progress_state["solution_count"],
        ))
        interval_elapsed = (
            last_runtime is None
            or runtime - last_runtime >= progress_sample_interval_seconds
        )
        if not (force or material_change or interval_elapsed):
            return
        telemetry["progress_trace"].append({
            "runtime_seconds": runtime,
            "incumbent_objective": incumbent,
            "best_bound": bound,
            "relative_gap": relative_gap(incumbent, bound),
            "node_count": nodes,
            "solution_count": solution_count,
            "event": event,
            "objective_sense": (
                "minimize" if objective_sense >= 0 else "maximize"
            ),
        })
        progress_state.update({
            "runtime_seconds": runtime,
            "incumbent_objective": incumbent,
            "best_bound": bound,
            "solution_count": solution_count,
        })

    def retain_root_bound(value):
        if value is None:
            return
        current = telemetry["root_node_bound"]
        stronger = (
            current is None
            or (objective_sense >= 0 and value > current)
            or (objective_sense < 0 and value < current)
        )
        if stronger:
            telemetry["root_node_bound"] = value

    def callback(callback_model, where):
        if where == GRB.Callback.MIP:
            node_count = finite_callback_value(
                callback_model, GRB.Callback.MIP_NODCNT
            )
            incumbent = finite_callback_value(
                callback_model, GRB.Callback.MIP_OBJBST, objective=True
            )
            bound = finite_callback_value(
                callback_model, GRB.Callback.MIP_OBJBND, objective=True
            )
            solution_count = finite_callback_value(
                callback_model, GRB.Callback.MIP_SOLCNT
            )
            runtime = finite_callback_value(
                callback_model, GRB.Callback.RUNTIME
            )
            record_progress(
                runtime,
                incumbent,
                bound,
                node_count,
                int(solution_count) if solution_count is not None else None,
                "mip_update",
            )
            if node_count is not None and node_count <= 0.5:
                retain_root_bound(bound)
        elif where == GRB.Callback.MIPNODE:
            node_count = finite_callback_value(
                callback_model, GRB.Callback.MIPNODE_NODCNT
            )
            incumbent = finite_callback_value(
                callback_model, GRB.Callback.MIPNODE_OBJBST, objective=True
            )
            bound = finite_callback_value(
                callback_model, GRB.Callback.MIPNODE_OBJBND, objective=True
            )
            solution_count = finite_callback_value(
                callback_model, GRB.Callback.MIPNODE_SOLCNT
            )
            runtime = finite_callback_value(
                callback_model, GRB.Callback.RUNTIME
            )
            record_progress(
                runtime,
                incumbent,
                bound,
                node_count,
                int(solution_count) if solution_count is not None else None,
                "mipnode_update",
            )
            if node_count is not None and node_count <= 0.5:
                retain_root_bound(bound)
        elif where == GRB.Callback.MIPSOL:
            runtime = finite_callback_value(
                callback_model, GRB.Callback.RUNTIME
            )
            objective = finite_callback_value(
                callback_model, GRB.Callback.MIPSOL_OBJ, objective=True
            )
            if runtime is None or objective is None:
                return
            if telemetry["time_to_first_incumbent_seconds"] is None:
                telemetry["time_to_first_incumbent_seconds"] = runtime
                telemetry["first_incumbent_objective"] = objective
            best = telemetry["best_incumbent_objective"]
            improved = (
                best is None
                or (objective_sense >= 0 and objective < best - 1e-9)
                or (objective_sense < 0 and objective > best + 1e-9)
            )
            if improved:
                telemetry["best_incumbent_objective"] = objective
                telemetry["time_to_best_incumbent_seconds"] = runtime
            incumbent = finite_callback_value(
                callback_model, GRB.Callback.MIPSOL_OBJBST, objective=True
            )
            bound = finite_callback_value(
                callback_model, GRB.Callback.MIPSOL_OBJBND, objective=True
            )
            node_count = finite_callback_value(
                callback_model, GRB.Callback.MIPSOL_NODCNT
            )
            solution_count = finite_callback_value(
                callback_model, GRB.Callback.MIPSOL_SOLCNT
            )
            if incumbent is None:
                incumbent = objective
            elif objective_sense >= 0:
                incumbent = min(incumbent, objective)
            else:
                incumbent = max(incumbent, objective)
            record_progress(
                runtime,
                incumbent,
                bound,
                node_count,
                int(solution_count) + 1 if solution_count is not None else None,
                "incumbent",
                force=True,
            )

    model.optimize(callback)
    final_incumbent = (
        _model_float(model, "ObjVal") if int(model.SolCount) > 0 else None
    )
    final_bound = _model_float(model, "ObjBound")
    final_gap = (
        _model_float(model, "MIPGap") if int(model.SolCount) > 0 else None
    )
    record_progress(
        _model_float(model, "Runtime"),
        final_incumbent,
        final_bound,
        _model_float(model, "NodeCount"),
        int(model.SolCount),
        "final",
        force=True,
    )
    if telemetry["progress_trace"]:
        telemetry["progress_trace"][-1]["relative_gap"] = final_gap
    telemetry.update({
        "branch_and_bound_nodes": _model_float(model, "NodeCount"),
        "linear_matrix_nonzeros": _model_float(model, "NumNZs"),
    })
    return telemetry


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


def _solution_directory(solver, variables=None, solutions_directory=None):
    root = Path(solutions_directory) if solutions_directory else ROOT_DIR / '02_data' / 'fjsp_solutions'
    if not root.is_absolute():
        root = ROOT_DIR / root
    directory = root / solver
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


def _solution_path(solver, instance_name, variables, solutions_directory=None):
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


def _apply_warm_start(variables, warm_start):
    selected = dict(warm_start['selected_machines'])
    completions = dict(warm_start['completions'])
    starts = dict(warm_start['starts'])
    direct = {tuple(edge) for edge in warm_start['machine_edges']}
    for (operation, machine), value in variables['Y'].items():
        value.Start = float(selected.get(operation) == machine)
    for operation, completion in completions.items():
        variables['C'][operation].Start = completion
    for (a, b, machine), value in variables.get('X', {}).items():
        value.Start = float(selected.get(a) == machine and selected.get(b) == machine
                            and completions[a] <= starts[b] + 1e-5)
    for edge, value in variables.get('U', {}).items():
        value.Start = float(edge in direct)


def _nominal_warm_start(variables, instance):
    selected = {o: max(instance.eligible_machines[o], key=lambda m: variables['Y'][o, m].X)
                for o in instance.real_operations}
    completions = {o: float(variables['C'][o].X) for o in selected}
    starts = {o: completions[o] - instance.processing_times[o, m] for o, m in selected.items()}
    edges = []
    for machine in set(selected.values()):
        chain = sorted((o for o, m in selected.items() if m == machine), key=lambda o: (starts[o], o))
        edges.extend((a, b, machine) for a, b in zip(chain, chain[1:]))
    return {'selected_machines': list(selected.items()), 'completions': list(completions.items()),
            'starts': list(starts.items()), 'machine_edges': edges}


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
    solutions_directory = values.pop('solutions_directory', None)
    warm_start = values.pop('warm_start', None)
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
    if warm_start is not None:
        _apply_warm_start(variables, warm_start)
    optimize_started = time.perf_counter()
    solver_telemetry = _optimize_with_telemetry(model)
    optimize_wall_seconds = time.perf_counter() - optimize_started
    model_total_seconds = build_seconds + optimize_wall_seconds
    variables["timing"] = {
        "instance_load_seconds": instance_loaded - request_started,
        "model_build_seconds": build_seconds,
        "optimizer_wall_seconds": optimize_wall_seconds,
        "gurobi_runtime_seconds": float(model.Runtime),
        "build_plus_optimizer_seconds": model_total_seconds,
    }
    variables["solver_telemetry"] = solver_telemetry

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
                f"operating cost={variables['operating_cost'].getValue():.6f} | "
                "buffered due-date violation cost="
                f"{variables['service_violation_cost'].getValue():.6f}"
            )
    path = None
    progress_path = None
    if write_solution:
        path = _solution_path(solver, instance_name, variables, solutions_directory)
        writer(model, variables, instance, path)
        candidate_progress_path = solver_progress_path(path)
        if candidate_progress_path.exists():
            progress_path = candidate_progress_path
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
        **solver_telemetry,
        "runtime": float(model.Runtime),
        "model_build_runtime": float(build_seconds),
        "optimizer_wall_runtime": float(optimize_wall_seconds),
        "build_plus_optimizer_runtime": float(model_total_seconds),
        "variables": variables,
        "model": model,
        "instance": instance,
        "plot_paths": plot_paths,
        "solution_path": str(path) if path is not None else None,
        "solver_progress_path": (
            str(progress_path) if progress_path is not None else None
        ),
        'warm_start': _nominal_warm_start(variables, instance) if solver == 'gurobi' and model.SolCount else None,
    }
    return result


def solve_instances_with_solver(**kwargs):
    """Compatibility wrapper for the single-instance active workflow."""
    return solveModel(**kwargs)
