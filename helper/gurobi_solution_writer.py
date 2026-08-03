"""Shared, directly comparable text format for Gurobi FJSP solutions."""

from __future__ import annotations

import heapq
import math
from collections import Counter

from gurobipy import GRB, GurobiError


STATUS_NAMES = {
    GRB.LOADED: "LOADED", GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE", GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED", GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT", GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED", GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL", GRB.INPROGRESS: "INPROGRESS",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}

VARIABLE_TYPE_NAMES = {
    GRB.CONTINUOUS: "continuous",
    GRB.BINARY: "binary",
    GRB.INTEGER: "integer",
    GRB.SEMICONT: "semi-continuous",
    GRB.SEMIINT: "semi-integer",
}

GENERAL_CONSTRAINT_TYPE_NAMES = {
    GRB.GENCONSTR_MAX: "MAX",
    GRB.GENCONSTR_MIN: "MIN",
    GRB.GENCONSTR_ABS: "ABS",
    GRB.GENCONSTR_AND: "AND",
    GRB.GENCONSTR_OR: "OR",
    GRB.GENCONSTR_NORM: "NORM",
    GRB.GENCONSTR_NL: "NL",
    GRB.GENCONSTR_INDICATOR: "INDICATOR",
    GRB.GENCONSTR_PWL: "PWL",
    GRB.GENCONSTR_POLY: "POLY",
    GRB.GENCONSTR_EXP: "EXP",
    GRB.GENCONSTR_EXPA: "EXPA",
    GRB.GENCONSTR_LOG: "LOG",
    GRB.GENCONSTR_LOGA: "LOGA",
    GRB.GENCONSTR_POW: "POW",
    GRB.GENCONSTR_SIN: "SIN",
    GRB.GENCONSTR_COS: "COS",
    GRB.GENCONSTR_TAN: "TAN",
    GRB.GENCONSTR_LOGISTIC: "LOGISTIC",
}


def _format_counts(counts, names):
    """Format a stable, exhaustive list of model-element type counts."""
    return ", ".join(
        f"{name}={counts.get(code, 0)}" for code, name in names.items()
    )


def _format_constraint_senses(counts):
    return ", ".join(
        (
            f"equality={counts.get(GRB.EQUAL, 0)}",
            f"less-or-equal={counts.get(GRB.LESS_EQUAL, 0)}",
            f"greater-or-equal={counts.get(GRB.GREATER_EQUAL, 0)}",
        )
    )


def write_model_structure(file, model):
    """Write variable and constraint counts, split by their Gurobi types."""
    variable_types = Counter(variable.VType for variable in model.getVars())
    linear_senses = Counter(constraint.Sense for constraint in model.getConstrs())
    quadratic_senses = Counter(
        constraint.QCSense for constraint in model.getQConstrs()
    )
    general_types = Counter(
        constraint.GenConstrType for constraint in model.getGenConstrs()
    )

    num_linear = len(model.getConstrs())
    num_quadratic = len(model.getQConstrs())
    num_general = len(model.getGenConstrs())
    num_sos = int(model.NumSOS)
    num_constraints = num_linear + num_quadratic + num_general + num_sos

    unknown_variable_types = {
        code: count
        for code, count in variable_types.items()
        if code not in VARIABLE_TYPE_NAMES
    }
    variable_type_text = _format_counts(
        variable_types,
        VARIABLE_TYPE_NAMES,
    )
    if unknown_variable_types:
        variable_type_text += ", " + ", ".join(
            f"unknown({code})={count}"
            for code, count in sorted(unknown_variable_types.items())
        )

    unknown_general_types = {
        code: count
        for code, count in general_types.items()
        if code not in GENERAL_CONSTRAINT_TYPE_NAMES
    }
    general_type_text = ", ".join(
        f"{GENERAL_CONSTRAINT_TYPE_NAMES[code]}={count}"
        for code, count in sorted(general_types.items())
        if code in GENERAL_CONSTRAINT_TYPE_NAMES
    )
    if unknown_general_types:
        unknown_text = ", ".join(
            f"UNKNOWN({code})={count}"
            for code, count in sorted(unknown_general_types.items())
        )
        general_type_text = ", ".join(
            part for part in (general_type_text, unknown_text) if part
        )
    if not general_type_text:
        general_type_text = "none"

    file.write("\nModel structure:\n")
    file.write(f"Variables total: {len(model.getVars())}\n")
    file.write(f"Variable types: {variable_type_text}\n")
    file.write(f"Constraints total: {num_constraints}\n")
    file.write(
        f"Linear constraints: {num_linear} "
        f"({_format_constraint_senses(linear_senses)})\n"
    )
    file.write(
        f"Quadratic constraints: {num_quadratic} "
        f"({_format_constraint_senses(quadratic_senses)})\n"
    )
    file.write(
        f"General constraints: {num_general} (types: {general_type_text})\n"
    )
    file.write(f"SOS constraints: {num_sos}\n")


def _number(value, digits=6):
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def _model_statistic(model, attribute):
    """Return an optimization statistic when Gurobi exposes it."""
    try:
        return float(getattr(model, attribute))
    except (AttributeError, TypeError, ValueError, GurobiError):
        # Some attributes are unavailable when no incumbent exists.
        return None


def _solution_value(value):
    if value is None:
        return None
    if hasattr(value, "X"):
        return float(value.X)
    if hasattr(value, "getValue"):
        return float(value.getValue())
    return float(value)


def _format_indexed(values, prefix):
    if not values:
        return ""
    items = sorted(values.items())
    unique = {round(float(value), 10) for _, value in items}
    if len(unique) == 1:
        return _number(items[0][1], 4)
    return ", ".join(f"{prefix}{idx}={float(value):.4f}" for idx, value in items)


def _format_repair_durations(values):
    if not values:
        return ""
    unique = {round(float(value), 10) for value in values.values()}
    if len(unique) == 1:
        return _number(next(iter(values.values())), 4)
    return ", ".join(
        f"({operation},{machine})={float(value):.4f}"
        for (operation, machine), value in sorted(values.items())
    )


def _selected_machine(variables, operation):
    return max(
        variables["eligible_machines"][operation],
        key=lambda candidate: _solution_value(
            variables["Y"][operation, candidate]
        ),
    )


def _operation_start(variables, operation):
    starts = variables.get("S")
    if starts is not None:
        return _solution_value(starts[operation])

    completion = _solution_value(variables["C"][operation])
    durations = variables.get("D")
    if durations is not None and operation in durations:
        return completion - _solution_value(durations[operation])

    machine = _selected_machine(variables, operation)
    processing = float(variables["processing_times"][operation, machine])
    return completion - processing


def _exact_weibull_operation_values(variables):
    if variables.get("constraint_type") != "weibull":
        return None, None
    if not variables.get("R"):
        return None, None

    probabilities = {}
    delays = {}
    for operation in variables["real_operations"]:
        machine = _selected_machine(variables, operation)
        repair_duration = float(
            variables.get("repair_durations", {}).get(
                (operation, machine), 0.0
            )
        )
        age = _solution_value(variables["R"][operation, machine])
        processing = float(variables["processing_times"][operation, machine])
        eta = float(variables["weibull_eta"][machine])
        beta = float(variables["weibull_beta"][machine])
        transition_gamma = float(
            variables.get("reliability_graph_config", {}).get(
                "transition_gamma", 0.0
            )
        )
        transition_load = _solution_value(
            variables.get("transition_load", {}).get(operation, 0.0)
        )
        hazard_increment = (
            ((age + processing) / eta) ** beta
            - (age / eta) ** beta
            + transition_gamma * transition_load
        )
        probability = min(
            1.0,
            max(0.0, 1.0 - math.exp(-hazard_increment)),
        )
        probabilities[operation] = probability
        delays[operation] = repair_duration * probability
    return probabilities, delays


def _exact_weibull_value(variables):
    _probabilities, delays = _exact_weibull_operation_values(variables)
    return sum(delays.values()) if delays is not None else None


def _exact_fixed_schedule_makespan(variables, instance):
    """Evaluate the exact makespan of one fixed GNN/exact schedule.

    Machine age depends on the selected direct machine-predecessor chain, not
    on wall-clock idle time.  Once Y and U are fixed, exact operation durations
    are therefore constants and the minimum makespan is the longest path over
    job and machine-predecessor arcs.
    """
    if instance is None:
        return None
    _probabilities, exact_delays = _exact_weibull_operation_values(variables)
    if exact_delays is None:
        return None

    operations = list(variables["real_operations"])
    operation_set = set(operations)
    predecessors = {operation: set() for operation in operations}
    for operation in operations:
        predecessors[operation].update(
            predecessor
            for predecessor in instance.predecessors.get(operation, [])
            if predecessor in operation_set
        )

    for (source, target, _machine), direct in variables.get("U", {}).items():
        if _solution_value(direct) > 0.5:
            predecessors[target].add(source)

    durations = {}
    for operation in operations:
        machine = _selected_machine(variables, operation)
        processing = float(variables["processing_times"][operation, machine])
        durations[operation] = processing + exact_delays[operation]

    successors = {operation: [] for operation in operations}
    indegree = {}
    for operation, operation_predecessors in predecessors.items():
        indegree[operation] = len(operation_predecessors)
        for predecessor in operation_predecessors:
            successors[predecessor].append(operation)

    ready = [operation for operation in operations if indegree[operation] == 0]
    heapq.heapify(ready)
    completion = {}
    while ready:
        operation = heapq.heappop(ready)
        start = max(
            (completion[predecessor] for predecessor in predecessors[operation]),
            default=0.0,
        )
        completion[operation] = start + durations[operation]
        for successor in successors[operation]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)

    if len(completion) != len(operations):
        return None
    end_operations = list(instance.job_end_operations.values())
    if not end_operations:
        end_operations = operations
    return max(completion[operation] for operation in end_operations)


def write_comparable_solution(
    model,
    variables,
    filename="solution.txt",
    instance=None,
):
    """Write the same schema for exact and fixed-candidate GNN models."""
    has_solution = model.SolCount > 0
    metadata = variables.get("gnn_metadata") or {}
    graph_mode = variables.get("gnn_graph_mode", "")
    formulation = variables.get("formulation") or (
        "gnn_fixed_graph" if graph_mode else "nonlinear_reliability"
    )

    objective = float(model.ObjVal) if has_solution else None
    makespan = _solution_value(variables.get("C_max")) if has_solution else None
    exact_weibull = _exact_weibull_value(variables) if has_solution else None
    exact_objective = (
        _exact_fixed_schedule_makespan(variables, instance)
        if has_solution else None
    )
    objective_error = (
        objective - exact_objective
        if objective is not None and exact_objective is not None
        else None
    )
    predicted = (
        _solution_value(variables.get("predicted_total_failure_delay"))
        if has_solution else None
    )
    predicted_probability = (
        _solution_value(
            variables.get("predicted_total_failure_probability")
        )
        if has_solution else None
    )
    raw_predicted_probability = (
        _solution_value(
            variables.get("predicted_total_failure_probability_raw")
        )
        if has_solution else None
    )
    constraint_type = variables.get("constraint_type", "")
    exact_constraint = (
        _solution_value(variables.get("total_failure_delay"))
        if has_solution and variables.get("total_failure_delay") is not None
        else exact_weibull
    )
    constraint_value = predicted if graph_mode else exact_constraint
    best_bound = _model_statistic(model, "ObjBound")
    mip_gap = _model_statistic(model, "MIPGap") if has_solution else None
    node_count = _model_statistic(model, "NodeCount")
    solution_count = int(model.SolCount)

    with open(filename, "w", encoding="utf-8") as file:
        file.write("Solution summary:\n")
        file.write(f"Formulation: {formulation}\n")
        file.write(f"Status: {STATUS_NAMES.get(model.Status, model.Status)}\n")
        file.write(f"Objective: {_number(objective, 4)}\n")
        file.write(
            "Exact objective (fixed decisions): "
            f"{_number(exact_objective, 4)}\n"
        )
        file.write(
            "Objective error (model - exact): "
            f"{_number(objective_error, 6)}\n"
        )
        file.write(f"Makespan: {_number(makespan, 4)}\n")
        file.write(
            "Exact makespan (fixed decisions): "
            f"{_number(exact_objective, 4)}\n"
        )
        file.write(f"Runtime [s]: {_number(model.Runtime, 4)}\n")
        file.write(f"Best bound: {_number(best_bound, 6)}\n")
        file.write(f"MIP gap: {_number(mip_gap, 8)}\n")
        file.write(
            "MIP gap [%]: "
            f"{_number(100.0 * mip_gap if mip_gap is not None else None, 6)}\n"
        )
        file.write(f"Explored nodes: {_number(node_count, 0)}\n")
        file.write(f"Solution count: {solution_count}\n")
        file.write(f"Big M: {_number(variables.get('H'), 4)}\n")

        write_model_structure(file, model)

        file.write("\nReliability summary:\n")
        file.write(f"Distribution: {constraint_type}\n")
        file.write(
            f"Constraint enforced: {variables.get('constraint_enforced', '')}\n"
        )
        file.write(
            f"Constraint budget: {_number(variables.get('constraint_budget'))}\n"
        )
        file.write(f"Total expected failure delay: {_number(constraint_value)}\n")
        file.write(f"Exact total failure delay: {_number(exact_constraint)}\n")
        file.write(f"Predicted total failure delay: {_number(predicted)}\n")
        file.write(
            "Sum of predicted operation failure probabilities: "
            f"{_number(predicted_probability)}\n"
        )
        file.write(
            "Raw probability-head sum before ReLU/clipping: "
            f"{_number(raw_predicted_probability)}\n"
        )
        file.write(f"Exact Weibull failure delay: {_number(exact_weibull)}\n")

        file.write("\nModel parameters:\n")
        file.write(f"GNN model path: {variables.get('gnn_model_path', '')}\n")
        file.write(f"GNN metadata path: {variables.get('gnn_metadata_path', '')}\n")
        file.write(f"Graph mode: {graph_mode}\n")
        file.write(f"Convolution: {metadata.get('convolution', '')}\n")
        file.write(f"Aggregation: {metadata.get('aggregation', '')}\n")
        file.write(f"Pooling: {metadata.get('pooling', '')}\n")
        file.write(f"GNN layers: {metadata.get('num_graphsage_layers', '')}\n")
        file.write(f"Hidden channels: {metadata.get('hidden_channels', '')}\n")
        file.write(f"GNN used in objective: {variables.get('gnn_used_in_objective', '')}\n")
        file.write(f"Schedule upper bounds: {variables.get('gnn_schedule_upper_bounds', '')}\n")
        file.write(f"ReLU formulation: {variables.get('gnn_relu_formulation', '')}\n")
        file.write(f"Edge type: {variables.get('gnn_edge_type', '')}\n")
        file.write(
            "Job precedence edges: "
            f"{variables.get('gnn_include_job_precedence_edges', '')}\n"
        )
        file.write(
            "Machine predecessor edges: "
            f"{variables.get('gnn_include_machine_predecessor_edges', '')}\n"
        )
        file.write(f"Number GNN edges: {len(variables.get('gnn_edges', [])) if graph_mode else ''}\n")
        file.write(
            "Reliability-graph schema: "
            f"{variables.get('reliability_graph_schema', '')}\n"
        )
        file.write(
            "Reliability-graph parameters: "
            f"{variables.get('reliability_graph_config', '')}\n"
        )

        file.write("\nReliability parameters:\n")
        file.write(
            f"machine_initial_age: {_format_indexed(variables.get('machine_initial_age'), 'M')}\n"
        )
        file.write(f"mu_fail: {_number(variables.get('mu_fail'), 4)}\n")
        file.write(f"weibull_eta: {_format_indexed(variables.get('weibull_eta'), 'M')}\n")
        file.write(f"weibull_beta: {_format_indexed(variables.get('weibull_beta'), 'M')}\n")
        file.write(
            "repair_duration: "
            f"{_format_repair_durations(variables.get('repair_durations'))}\n"
        )

        if has_solution:
            file.write("\nOperation values:\n")
            for operation in variables["real_operations"]:
                start = _operation_start(variables, operation)
                file.write(
                    f"op {operation}: C={variables['C'][operation].X:.4f}, "
                    f"S={start:.4f}, "
                    f"D={_number(_solution_value(variables.get('D', {}).get(operation)) if variables.get('D') else None, 4)}, "
                    f"Delta={_number(_solution_value(variables.get('Delta', {}).get(operation)) if variables.get('Delta') else None, 4)}\n"
                )

            file.write("\nY values:\n")
            for operation, machine in variables["Y_index"]:
                y_value = int(round(variables["Y"][operation, machine].X))
                processing = float(variables["processing_times"][operation, machine])
                age = (
                    _solution_value(variables["R"][operation, machine])
                    if variables.get("R") else None
                )
                probability = (
                    _solution_value(variables["pi_fail"][operation, machine])
                    if variables.get("pi_fail") else None
                )
                repair_duration = variables.get("repair_durations", {}).get(
                    (operation, machine)
                )
                transition = (
                    _solution_value(variables["transition_load"][operation])
                    if variables.get("transition_load") and y_value else None
                )
                file.write(
                    f"Y[{operation},{machine}] = {y_value}, p={processing:.4f}, "
                    f"transition={_number(transition)}, R={_number(age)}, "
                    f"pi_fail={_number(probability)}, "
                    f"tau={_number(repair_duration)}\n"
                )
