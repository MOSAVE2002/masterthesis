"""Build the nominal completion-time formulation shared by all FJSP models.

The module creates machine assignments, pairwise machine orders, technological
precedence and completion times. Separate entry points add either a makespan
objective or the project's soft due dates and economic cost objective.
"""

import gurobipy as gp
from gurobipy import GRB

from helper.economic_objective import (
    add_economic_cost_objective,
    add_makespan,
    add_nominal_due_date_constraints,
)


def build_core_fjsp(model, instance):
    """Build the shared assignment and scheduling constraints.

    Add machine assignments, operation ordering and completion times without
    choosing an objective or adding due-date constraints.
    """
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
        operations,
        lb=0.0,
        ub=horizon,
        vtype=GRB.CONTINUOUS,
        name="C",
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
            )
            == 1,
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
                completion[operation]
                >= completion[predecessor] + duration,
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

    variables = {
        "C": completion,
        "Y": assignment,
        "X": order,
        "H": horizon,
        "real_operations": operations,
        "eligible_machines": instance.eligible_machines,
        "processing_times": instance.processing_times,
        "machines": machines,
        "X_index": order_index,
        "Y_index": assignment_index,
    }
    model.update()
    return model, variables


def build_makespan_fjsp(model, instance):
    """Build the core FJSP and minimize its nominal makespan.

    Args:
        model: Gurobi model to extend.
        instance: FJSP instance defining operations and eligible machines.

    Returns:
        The updated model and its shared variable/metadata dictionary.
    """
    model, variables = build_core_fjsp(model, instance)
    makespan = add_makespan(model, variables, instance)
    model.setObjective(makespan, GRB.MINIMIZE)
    model.update()
    return model, variables


def build_fjsp(
    model,
    instance,
    *,
    facility_cost_per_time=1.0,
    tardiness_cost_per_time=1.0,
):
    """Build the nominal economic FJSP with soft due dates.

    Args:
        model: Gurobi model to extend.
        instance: FJSP instance to optimize.
        facility_cost_per_time: Cost assigned to one makespan time unit.
        tardiness_cost_per_time: Cost assigned to one tardiness time unit.

    Returns:
        The completed nominal model and its variable/metadata dictionary.
    """
    model, variables = build_core_fjsp(model, instance)
    add_nominal_due_date_constraints(model, variables, instance)
    add_economic_cost_objective(
        model,
        variables,
        instance,
        facility_cost_per_time=facility_cost_per_time,
        tardiness_cost_per_time=tardiness_cost_per_time,
    )
    variables.update({
        "formulation": "nominal_fjsp_with_soft_due_date_violation_v2",
        "service_scope": "none",
    })
    model.update()
    return model, variables
