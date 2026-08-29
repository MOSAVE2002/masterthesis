"""Shared economic objective and due-date constraints for all FJSP models."""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from helper.stochastic_fjsp import ensure_stochastic_parameters


def add_due_date_tardiness_constraints(
    model,
    variables,
    instance,
    job_repair_buffers=None,
):
    """Model nonnegative job tardiness for a nominal or protected completion."""
    buffers = job_repair_buffers or {
        job: 0.0 for job in instance.job_end_operations
    }
    jobs = list(instance.job_end_operations)
    tardiness = model.addVars(
        jobs,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="L",
    )
    constraints = {}
    for job, end_operation in instance.job_end_operations.items():
        constraints[job] = model.addConstr(
            variables["C"][end_operation] + buffers[job]
            <= float(instance.due_dates[job]) + tardiness[job],
            name=f"due_date_with_tardiness[{job}]",
        )
    variables.update({
        "due_date_constraints": constraints,
        "job_tardiness": tardiness,
        "job_expected_repair_buffers": buffers,
        "due_dates": dict(instance.due_dates),
    })
    return constraints


def add_nominal_due_date_constraints(model, variables, instance):
    """Compatibility wrapper for soft nominal due dates."""
    constraints = add_due_date_tardiness_constraints(
        model, variables, instance
    )
    variables["nominal_due_date_constraints"] = constraints
    return constraints


def add_robust_due_date_constraints(
    model,
    variables,
    instance,
    job_repair_buffers,
):
    """Model tardiness of completion plus the unscaled repair buffer."""
    constraints = add_due_date_tardiness_constraints(
        model,
        variables,
        instance,
        job_repair_buffers,
    )
    variables.update({
        "robust_due_date_constraints": constraints,
    })
    return constraints


def add_economic_cost_objective(
    model,
    variables,
    instance,
    *,
    facility_cost_per_time=1.0,
    tardiness_cost_per_time=1.0,
):
    """Minimize processing, operating and total job-tardiness costs."""
    ensure_stochastic_parameters(instance)
    facility_cost_per_time = float(facility_cost_per_time)
    if facility_cost_per_time < 0.0:
        raise ValueError("facility_cost_per_time must be nonnegative.")
    tardiness_cost_per_time = float(tardiness_cost_per_time)
    if tardiness_cost_per_time < 0.0:
        raise ValueError("tardiness_cost_per_time must be nonnegative.")
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
    job_tardiness = variables.get("job_tardiness", {})
    total_tardiness = gp.quicksum(job_tardiness.values())
    tardiness_cost = tardiness_cost_per_time * total_tardiness
    total_cost = processing_cost + operating_cost + tardiness_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    variables.update({
        "machine_cost": machine_cost,
        "facility_cost_per_time": facility_cost_per_time,
        "tardiness_cost_per_time": tardiness_cost_per_time,
        "processing_cost": processing_cost,
        "operating_cost": operating_cost,
        "total_tardiness": total_tardiness,
        "tardiness_cost": tardiness_cost,
        "total_cost": total_cost,
        "objective_mode": "minimize_economic_cost_with_tardiness",
        "objective_definition": (
            "sum_{i,k} c_k*p_ik*Y_ik + c_B*C_max + c_L*sum_j L_j"
        ),
    })
    return total_cost
