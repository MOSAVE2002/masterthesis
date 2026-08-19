"""Comparable text output for the nonlinear and embedded-GNN models."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

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


def _value(item):
    if item is None:
        return None
    if hasattr(item, "X"):
        return float(item.X)
    if hasattr(item, "getValue"):
        return float(item.getValue())
    return float(item)


def _number(item, digits=6):
    value = _value(item)
    return "" if value is None else f"{value:.{digits}f}"


def _job_probability_value(variables, item):
    value = _value(item)
    if variables.get("job_probability_postprocess") == (
        "relu_clip_0_1_outside_model"
    ):
        return min(1.0, max(0.0, value))
    return value


def _selected_machine(variables, operation):
    return max(
        variables["eligible_machines"][operation],
        key=lambda machine: _value(variables["Y"][operation, machine]),
    )


def _active_machine_edges(variables):
    """Return immediate machine-predecessor edges of the incumbent schedule."""
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
                _value(variables["S"][operation]),
                _value(variables["C"][operation]),
                operation,
            ),
        )
        edges.extend(
            (source, target, machine)
            for source, target in zip(ordered, ordered[1:])
        )
    return edges


def _model_structure(model):
    variable_types = Counter(variable.VType for variable in model.getVars())
    general_types = Counter(
        constraint.GenConstrType for constraint in model.getGenConstrs()
    )
    return {
        "variables": len(model.getVars()),
        "continuous": variable_types[GRB.CONTINUOUS],
        "binary": variable_types[GRB.BINARY],
        "integer": variable_types[GRB.INTEGER],
        "linear_constraints": len(model.getConstrs()),
        "quadratic_constraints": len(model.getQConstrs()),
        "general_constraints": len(model.getGenConstrs()),
        "nonlinear_constraints": general_types[GRB.GENCONSTR_NL],
    }


def write_model_structure(file, model):
    """Write the compact structure block used by the nominal FJSP writer."""
    file.write("\nModel structure:\n")
    for name, count in _model_structure(model).items():
        file.write(f"{name}: {count}\n")


def write_comparable_solution(
    model,
    variables,
    filename="solution.txt",
    instance=None,
):
    """Write the active model, service, Pd, Y and U values."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_solution = model.SolCount > 0
    metadata = variables.get("gnn_metadata") or {}
    structure = _model_structure(model)

    with path.open("w", encoding="utf-8") as file:
        file.write("Solution summary:\n")
        file.write(f"Formulation: {variables.get('formulation', '')}\n")
        file.write(f"Status: {STATUS_NAMES.get(model.Status, model.Status)}\n")
        file.write(
            "Objective definition: "
            f"{variables.get('objective_definition', '')}\n"
        )
        file.write(f"Objective: {_number(model.ObjVal if has_solution else None)}\n")
        file.write(f"Best bound: {_number(model.ObjBound)}\n")
        file.write(f"MIP gap: {_number(model.MIPGap if has_solution else None)}\n")
        file.write(f"Runtime [s]: {_number(model.Runtime)}\n")
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
        file.write(f"Constraint type: {variables.get('constraint_type', '')}\n")
        file.write(f"Service scope: {variables.get('service_scope', '')}\n")
        file.write(f"Service levels: {variables.get('service_levels', {})}\n")
        file.write(f"Due dates: {variables.get('due_dates', {})}\n")
        file.write(f"GNN convolution: {metadata.get('convolution', '')}\n")
        file.write(f"GNN layers: {metadata.get('num_graphsage_layers', '')}\n")
        file.write(f"GNN hidden channels: {metadata.get('hidden_channels', '')}\n")

        if not has_solution:
            return path

        file.write("\nAlpha service level summary:\n")
        probabilities = variables.get("job_ontime_probabilities", {})
        realized_alpha = {
            job: _job_probability_value(variables, probabilities[job])
            for job in sorted(probabilities)
        }
        alpha_slacks = {
            job: realized_alpha[job]
            - float(variables["service_levels"][job])
            for job in realized_alpha
        }
        if realized_alpha:
            file.write(
                "Minimum realized per-job alpha service level: "
                f"{min(realized_alpha.values()):.6f}\n"
            )
            file.write(
                "Minimum alpha service slack: "
                f"{min(alpha_slacks.values()):.6f}\n"
            )
            file.write(
                "All per-job alpha targets satisfied: "
                f"{'yes' if min(alpha_slacks.values()) >= -1e-6 else 'no'}\n"
            )
        file.write(
            "Probability label method: "
            f"{variables.get('job_probability_label_method', '')}\n"
        )

        file.write("\nPer-job alpha service levels:\n")
        for job in sorted(probabilities):
            completion = variables["C"][instance.job_end_operations[job]]
            due_date = float(variables["due_dates"][job])
            probability = realized_alpha[job]
            target = float(variables["service_levels"][job])
            slack = alpha_slacks[job]
            file.write(
                f"job {job}: completion={_number(completion)}, "
                f"due_date={due_date:.6f}, "
                f"realized_alpha={probability:.6f}, "
                f"target_alpha={target:.6f}, "
                f"alpha_slack={slack:.6f}, "
                f"satisfied={'yes' if slack >= -1e-6 else 'no'}\n"
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
                f"S={_number(variables['S'][operation])}, "
                f"C={_number(variables['C'][operation])}, "
                f"T={_number(variables.get('T', {}).get(operation))}, "
                f"Pd_model={_number(pd_model)}, "
                f"Pd_gnn={_number(pd_gnn)}\n"
            )

        file.write("\nActive U edges:\n")
        for edge in _active_machine_edges(variables):
            file.write(f"U{edge}=1\n")
    return path
