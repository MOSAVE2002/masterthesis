"""Shared economic objective and due-date constraints for all FJSP models."""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from helper.stochastic_fjsp import ensure_stochastic_parameters


def add_nominal_due_date_constraints(model, variables, instance):
    """Require every nominal job completion to respect its due date."""
    constraints = {}
    for job, end_operation in instance.job_end_operations.items():
        constraints[job] = model.addConstr(
            variables["C"][end_operation] <= float(instance.due_dates[job]),
            name=f"nominal_due_date[{job}]",
        )
    variables["nominal_due_date_constraints"] = constraints
    variables["due_dates"] = dict(instance.due_dates)
    return constraints


def add_robust_due_date_constraints(
    model,
    variables,
    instance,
    job_repair_buffers,
):
    """Protect each due date by its unscaled expected repair buffer."""
    constraints = {}
    for job, end_operation in instance.job_end_operations.items():
        constraints[job] = model.addConstr(
            variables["C"][end_operation] + job_repair_buffers[job]
            <= float(instance.due_dates[job]),
            name=f"due_date_with_expected_repair_buffer[{job}]",
        )
    variables.update({
        "job_expected_repair_buffers": job_repair_buffers,
        "robust_due_date_constraints": constraints,
    })
    return constraints


def add_economic_cost_objective(
    model, variables, instance, *, facility_cost_per_time=1.0
):
    """Minimize K = sum(i,k) c_k p_ik Y_ik + c_B C_max."""
    ensure_stochastic_parameters(instance)
    facility_cost_per_time = float(facility_cost_per_time)
    if facility_cost_per_time < 0.0:
        raise ValueError("facility_cost_per_time must be nonnegative.")
    machine_cost = {
        machine: float(instance.machine_cost[machine])
        for machine in variables["machines"]
    }
    if any(value < 0.0 for value in machine_cost.values()):
        raise ValueError("Machine processing costs must be nonnegative.")

    makespan = variables.get("C_max")
    if makespan is None:
        makespan = model.addVar(
            lb=0.0,
            ub=float(variables["H"]),
            vtype=GRB.CONTINUOUS,
            name="C_max",
        )
        for job, end_operation in instance.job_end_operations.items():
            model.addConstr(
                makespan >= variables["C"][end_operation],
                name=f"economic_makespan[{job}]",
            )
        variables["C_max"] = makespan

    processing_cost = gp.quicksum(
        machine_cost[machine]
        * float(instance.processing_times[operation, machine])
        * variables["Y"][operation, machine]
        for operation, machine in variables["Y_index"]
    )
    operating_cost = facility_cost_per_time * makespan
    total_cost = processing_cost + operating_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    variables.update({
        "machine_cost": machine_cost,
        "facility_cost_per_time": facility_cost_per_time,
        "processing_cost": processing_cost,
        "operating_cost": operating_cost,
        "total_cost": total_cost,
        "objective_mode": "minimize_economic_cost",
        "objective_definition": "sum_{i,k} c_k*p_ik*Y_ik + c_B*C_max",
    })
    return total_cost
