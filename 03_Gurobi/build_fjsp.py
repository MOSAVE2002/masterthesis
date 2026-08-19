"""Nominal flexible job-shop MILP shared by both stochastic extensions."""

from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from helper.gurobi_solution_writer import write_model_structure


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


def build_fjsp(model, instance, *, include_makespan=True):
    operations = list(instance.real_operations)
    machines = list(range(instance.num_machines))
    horizon = sum(
        max(
            float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        for operation in operations
    )
    completion = model.addVars(
        operations, lb=0.0, vtype=GRB.CONTINUOUS, name="C"
    )
    makespan = (
        model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")
        if include_makespan
        else None
    )
    assignment_index = [
        (operation, machine)
        for operation in operations
        for machine in instance.eligible_machines[operation]
    ]
    assignment = model.addVars(
        assignment_index, vtype=GRB.BINARY, name="Y"
    )
    order_index = [
        (operation_i, operation_j, machine)
        for position, operation_i in enumerate(operations)
        for operation_j in operations[position + 1:]
        for machine in sorted(
            set(instance.eligible_machines[operation_i])
            & set(instance.eligible_machines[operation_j])
        )
    ]
    order = model.addVars(order_index, vtype=GRB.BINARY, name="X")

    for operation in operations:
        model.addConstr(
            gp.quicksum(
                assignment[operation, machine]
                for machine in instance.eligible_machines[operation]
            ) == 1,
            name=f"assignment[{operation}]",
        )
        duration = gp.quicksum(
            float(instance.processing_times[operation, machine])
            * assignment[operation, machine]
            for machine in instance.eligible_machines[operation]
        )
        model.addConstr(
            completion[operation] >= duration,
            name=f"completion_lb[{operation}]",
        )
        for predecessor in instance.predecessors.get(operation, []):
            model.addConstr(
                completion[operation] >= completion[predecessor] + duration,
                name=f"precedence[{predecessor}_before_{operation}]",
            )

    for operation_i, operation_j, machine in order_index:
        x = order[operation_i, operation_j, machine]
        yi = assignment[operation_i, machine]
        yj = assignment[operation_j, machine]
        model.addConstr(
            completion[operation_i]
            >= completion[operation_j]
            + float(instance.processing_times[operation_i, machine])
            - horizon * (2 + x - yi - yj),
            name=f"nonoverlap_j_before_i[{operation_j}_{operation_i}_{machine}]",
        )
        model.addConstr(
            completion[operation_j]
            >= completion[operation_i]
            + float(instance.processing_times[operation_j, machine])
            - horizon * (3 - x - yi - yj),
            name=f"nonoverlap_i_before_j[{operation_i}_{operation_j}_{machine}]",
        )

    if include_makespan:
        for job, end_operation in instance.job_end_operations.items():
            model.addConstr(
                makespan >= completion[end_operation],
                name=f"makespan[{job}]",
            )
        model.setObjective(makespan, GRB.MINIMIZE)
    model.update()
    variables = {
        "C": completion,
        "Y": assignment,
        "X": order,
        "H": horizon,
        "operations": operations,
        "real_operations": operations,
        "eligible_machines": instance.eligible_machines,
        "processing_times": instance.processing_times,
        "machines": machines,
        "jobs": instance.jobs,
        "X_index": order_index,
        "Y_index": assignment_index,
    }
    if include_makespan:
        variables["C_max"] = makespan
    return model, variables


def write_solution_file(model, variables, instance, filename="solution.txt"):
    del instance
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(f"Status: {STATUS_NAMES.get(model.Status, model.Status)}\n")
        file.write(
            f"Makespan: {model.ObjVal:.4f}\n" if model.SolCount else "Makespan:\n"
        )
        file.write(f"Big M: {float(variables['H']):.4f}\n")
        file.write(f"Runtime [s]: {float(model.Runtime):.4f}\n")
        file.write(f"Solution count: {int(model.SolCount)}\n")
        write_model_structure(file, model)
        if model.SolCount:
            file.write("\nY values:\n")
            for index in variables["Y_index"]:
                file.write(f"Y{index}={int(round(variables['Y'][index].X))}\n")
    return path
