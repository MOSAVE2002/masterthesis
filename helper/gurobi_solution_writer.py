"""Write solver-independent text summaries of solved FJSP formulations.

The common format records model size, timing, costs, job buffers, operation
timings and active direct machine edges. It is both human-readable and the
canonical input consumed by the Monte Carlo post-evaluation module.
"""

from collections import Counter
from itertools import pairwise
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
}


def status_name(status):
    """Return a readable Gurobi status name or preserve an unknown code.

    Args:
        status: Numeric Gurobi model status.

    Returns:
        Stable uppercase name or the decimal status text.
    """
    return STATUS_NAMES.get(status, str(status))


def _value(item):
    """Safely evaluate a numeric value, Gurobi variable or expression.

    Returns ``None`` when solution attributes are unavailable, allowing the
    writer to produce diagnostic files even without an incumbent.
    """
    if item is None:
        return None
    if hasattr(item, "X"):
        return float(item.X)
    if hasattr(item, "getValue"):
        try:
            return float(item.getValue())
        except (AttributeError, gp.GurobiError):
            return None
    if isinstance(item, (gp.Var, gp.LinExpr, gp.QuadExpr)):
        # Gurobi solution attributes are unavailable when no incumbent exists.
        return None
    return float(item)


def _number(item, digits=6):
    """Format an optional model value with a fixed number of decimal places.

    Unavailable values become empty fields so diagnostic output stays parseable.
    """
    value = _value(item)
    return "" if value is None else f"{value:.{digits}f}"


def _model_attribute(model, name):
    """Read an optional Gurobi model attribute without propagating API errors.

    Attributes unavailable for the current solver state are represented by
    ``None`` in the comparable output.
    """
    try:
        return getattr(model, name)
    except (AttributeError, gp.GurobiError):
        return None


def _selected_machine(variables, operation):
    """Return the machine with the largest incumbent assignment value.

    Binary solutions therefore yield the unique selected eligible machine.
    """
    return max(
        variables["eligible_machines"][operation],
        key=lambda machine: _value(variables["Y"][operation, machine]),
    )


def _operation_start(variables, operation):
    """Return an operation start from explicit or derived timing data.

    Models without start variables derive the value from completion and the
    selected machine's processing duration.
    """
    if variables.get("S") is not None:
        return _value(variables["S"][operation])
    machine = _selected_machine(variables, operation)
    return (
        _value(variables["C"][operation])
        - float(variables["processing_times"][operation, machine])
    )


def _active_machine_edges(variables):
    """Return immediate machine-predecessor edges of the incumbent schedule.

    Embedded graph models expose direct ``U`` variables. For nominal models,
    the same edges are reconstructed by sorting selected operations by start.
    """
    if variables.get("U") is not None:
        return [
            index
            for index in variables.get("U_index", [])
            if _value(variables["U"][index]) > 0.5
        ]

    operations_by_machine = {
        machine: [] for machine in variables["machines"]
    }
    for operation in variables["real_operations"]:
        machine = _selected_machine(variables, operation)
        operations_by_machine[machine].append(operation)

    edges = []
    for machine, operations in operations_by_machine.items():
        ordered = sorted(
            operations,
            key=lambda operation: (
                _operation_start(variables, operation),
                _value(variables["C"][operation]),
                operation,
            ),
        )
        edges.extend(
            (source, target, machine)
            for source, target in pairwise(ordered)
        )
    return edges


def _model_structure(model):
    """Summarize variables, constraints and linear-matrix nonzeros.

    Returns:
        Dictionary of counts shared by all solver result files.
    """
    variable_types = Counter(variable.VType for variable in model.getVars())
    general_types = Counter(
        constraint.GenConstrType for constraint in model.getGenConstrs()
    )
    return {
        "variables": len(model.getVars()),
        "continuous": variable_types[GRB.CONTINUOUS],
        "binary": variable_types[GRB.BINARY],
        "integer": variable_types[GRB.INTEGER],
        "linear_matrix_nonzeros": int(
            _model_attribute(model, "NumNZs") or 0
        ),
        "linear_constraints": len(model.getConstrs()),
        "quadratic_constraints": len(model.getQConstrs()),
        "general_constraints": len(model.getGenConstrs()),
        "nonlinear_constraints": general_types[GRB.GENCONSTR_NL],
    }


def write_comparable_solution(
    model,
    variables,
    filename="solution.txt",
    instance=None,
):
    """Write a complete comparable solution and diagnostic summary.

    Args:
        model: Optimized Gurobi model, with or without an incumbent.
        variables: Shared formulation dictionary containing reportable values.
        filename: Destination text path.
        instance: Solved instance required for per-job and operation output.

    Returns:
        Path of the written text file.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_solution = model.SolCount > 0
    metadata = variables.get("gnn_metadata") or {}
    structure = _model_structure(model)
    solver_metrics = variables.get("solver_metrics", {})

    with path.open("w", encoding="utf-8") as file:
        file.write("Solution summary:\n")
        file.write("Model time unit: ZE (times and buffers); rates: 1/ZE; cost rates: GE/ZE\n")
        file.write(f"Formulation: {variables.get('formulation', '')}\n")
        file.write(f"Status: {status_name(model.Status)}\n")
        file.write(
            "Objective definition: "
            f"{variables.get('objective_definition', '')}\n"
        )
        file.write(f"Objective: {_number(model.ObjVal if has_solution else None)}\n")
        file.write(f"Makespan: {_number(variables.get('C_max'))}\n")
        file.write(f"Processing cost: {_number(variables.get('processing_cost'))}\n")
        file.write(f"Operating cost: {_number(variables.get('operating_cost'))}\n")
        file.write(
            "Tardiness cost: "
            f"{_number(variables.get('tardiness_cost'))}\n"
        )
        file.write(
            "Total tardiness: "
            f"{_number(variables.get('total_tardiness'))}\n"
        )
        file.write(f"Total cost: {_number(variables.get('total_cost'))}\n")
        file.write(
            "Facility cost per time: "
            f"{_number(variables.get('facility_cost_per_time'))}\n"
        )
        file.write(
            "Tardiness cost per time: "
            f"{_number(variables.get('tardiness_cost_per_time'))}\n"
        )
        file.write(f"Objective mode: {variables.get('objective_mode', '')}\n")
        file.write(
            f"Best bound: {_number(_model_attribute(model, 'ObjBound'))}\n"
        )
        file.write(
            f"MIP gap: {_number(_model_attribute(model, 'MIPGap'))}\n"
        )
        file.write(f"Runtime [s]: {_number(model.Runtime)}\n")
        file.write(
            "Branch-and-bound nodes: "
            f"{_number(solver_metrics.get('branch_and_bound_nodes'))}\n"
        )
        timing = variables.get("timing", {})
        file.write(
            "Model build runtime [s]: "
            f"{_number(timing.get('model_build_seconds'))}\n"
        )
        file.write(
            "Optimizer wall runtime [s]: "
            f"{_number(timing.get('optimizer_wall_seconds'))}\n"
        )
        file.write(
            "Build plus optimizer runtime [s]: "
            f"{_number(timing.get('build_plus_optimizer_seconds'))}\n"
        )
        file.write(f"Solution count: {model.SolCount}\n")
        for name, count in structure.items():
            file.write(f"{name}: {count}\n")

        file.write("\nStochastic formulation:\n")
        file.write(f"Service scope: {variables.get('service_scope', '')}\n")
        file.write(f"Due dates: {variables.get('due_dates', {})}\n")
        file.write(f"GNN convolution: {metadata.get('convolution', '')}\n")
        file.write(f"GNN layers: {metadata.get('num_graphsage_layers', '')}\n")
        file.write(f"GNN hidden channels: {metadata.get('hidden_channels', '')}\n")

        if not has_solution:
            return path

        file.write("\nExpected local repair-buffer summary:\n")
        buffers = variables.get("job_repair_buffers", {})
        buffer_values = {
            job: _value(buffers[job]) for job in sorted(buffers)
        }
        if buffer_values:
            file.write(
                "Maximum expected local job repair buffer: "
                f"{max(buffer_values.values()):.6f}\n"
            )
        file.write(
            "Repair buffer label method: "
            f"{variables.get('job_repair_buffer_label_method', '')}\n"
        )

        file.write("\nPer-job buffered due dates:\n")
        for job in sorted(buffers):
            completion = variables["C"][instance.job_end_operations[job]]
            due_date = float(variables["due_dates"][job])
            repair_buffer = buffer_values[job]
            tardiness = variables.get("job_tardiness", {}).get(job)
            file.write(
                f"job {job}: completion={_number(completion)}, "
                f"due_date={due_date:.6f}, "
                f"expected_local_repair_buffer={repair_buffer:.6f}, "
                f"repair_buffer={repair_buffer:.6f}, "
                "buffered_completion="
                f"{_value(completion) + repair_buffer:.6f}, "
                "buffered_slack="
                f"{due_date - _value(completion) - repair_buffer:.6f}, "
                f"tardiness={_number(tardiness)}\n"
            )

        file.write("\nOperation values:\n")
        for operation in variables["real_operations"]:
            machine = _selected_machine(variables, operation)
            pd_model = None
            if variables.get("Pd") is not None:
                pd_model = variables["Pd"].get((operation, machine))
            pd_gnn = None
            if variables.get("Pd_gnn") is not None:
                pd_gnn = variables["Pd_gnn"].get(operation)
            file.write(
                f"op {operation}: machine={machine}, "
                f"S={_number(_operation_start(variables, operation))}, "
                f"C={_number(variables['C'][operation])}, "
                f"T={_number(variables.get('T', {}).get(operation))}, "
                f"Pd_model={_number(pd_model)}, "
                f"Pd_gnn={_number(pd_gnn)}\n"
            )

        file.write("\nActive U edges:\n")
        for edge in _active_machine_edges(variables):
            file.write(f"U{edge}=1\n")
    return path
