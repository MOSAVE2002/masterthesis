"""Nominal flexible job-shop MILP shared by both stochastic extensions."""

import gurobipy as gp
from gurobipy import GRB

from helper.gurobi_solution_writer import write_comparable_solution
from helper.economic_objective import (
    add_economic_cost_objective,
    add_nominal_due_date_constraints,
)


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


def build_fjsp(
    model,
    instance,
    *,
    include_makespan=True,
    horizon_upper_bound=None,
    enforce_due_dates=None,
    economic_objective=None,
    facility_cost_per_time=1.0,
    service_violation_cost_per_time=1.0,
    tardiness_cost_per_time=None,
):
    operations = list(instance.real_operations)
    machines = list(range(instance.num_machines))
    serial_horizon = sum(
        max(
            float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        for operation in operations
    )
    horizon = serial_horizon
    if horizon_upper_bound is not None:
        horizon_upper_bound = float(horizon_upper_bound)
        if horizon_upper_bound <= 0.0:
            raise ValueError("horizon_upper_bound must be positive.")
        horizon = min(horizon, horizon_upper_bound)
    completion = model.addVars(
        operations,
        lb=0.0,
        ub=horizon,
        vtype=GRB.CONTINUOUS,
        name="C",
    )
    makespan = (
        model.addVar(
            lb=0.0,
            ub=horizon,
            vtype=GRB.CONTINUOUS,
            name="C_max",
        )
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
    if enforce_due_dates is None:
        enforce_due_dates = hasattr(instance, "due_dates")
    if economic_objective is None:
        economic_objective = hasattr(instance, "due_dates")
    if enforce_due_dates:
        add_nominal_due_date_constraints(model, variables, instance)
    if economic_objective:
        add_economic_cost_objective(
            model,
            variables,
            instance,
            facility_cost_per_time=facility_cost_per_time,
            service_violation_cost_per_time=service_violation_cost_per_time,
            tardiness_cost_per_time=tardiness_cost_per_time,
        )
    elif include_makespan:
        model.setObjective(makespan, GRB.MINIMIZE)
    variables.update({
        "formulation": "nominal_fjsp_with_soft_due_date_violation_v2",
        "constraint_type": "nominal_completion_plus_soft_violation",
        "service_scope": "none",
    })
    model.update()
    return model, variables


def write_solution_file(model, variables, instance, filename="solution.txt"):
    return write_comparable_solution(
        model,
        variables,
        filename,
        instance=instance,
    )
