"""Shared economic objective and soft service constraints for all FJSP models."""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from helper.stochastic_fjsp import ensure_stochastic_parameters


def validate_service_level(service_level):
    """Return an admissible chance-constraint service level."""
    service_level = float(service_level)
    if not 0.0 < service_level < 1.0:
        raise ValueError("service_level must lie strictly between 0 and 1.")
    return service_level


def add_soft_service_level_constraints(
    model,
    variables,
    instance,
    job_expected_delays=None,
    *,
    service_level=0.90,
):
    """Add soft Markov service constraints based on expected job delays.

    For nonnegative stochastic completion delay D_j, Markov's inequality gives
    P(D_j <= s_j) >= 1 - E[D_j] / s_j.  Consequently, zero violation in the
    constraint below requires the conservative slack E[D_j] / (1 - alpha).
    """
    service_level = validate_service_level(service_level)
    expected_delays = job_expected_delays or {
        job: 0.0 for job in instance.job_end_operations
    }
    scale = 1.0 / (1.0 - service_level)
    service_buffers = {
        job: scale * expected_delays[job]
        for job in instance.job_end_operations
    }
    jobs = list(instance.job_end_operations)
    violation = model.addVars(
        jobs,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="L_service",
    )
    constraints = {}
    for job, end_operation in instance.job_end_operations.items():
        constraints[job] = model.addConstr(
            variables["C"][end_operation] + service_buffers[job]
            <= float(instance.due_dates[job]) + violation[job],
            name=f"soft_service_level[{job}]",
        )
    variables.update({
        "service_level_constraints": constraints,
        "due_date_constraints": constraints,
        "job_service_level_violation": violation,
        "job_expected_delays": expected_delays,
        "job_service_level_buffers": service_buffers,
        # Compatibility aliases for evaluation code written before the
        # service-level formulation replaced ordinary tardiness.
        "job_tardiness": violation,
        "job_expected_repair_buffers": expected_delays,
        "service_level": service_level,
        "service_tail_probability": 1.0 - service_level,
        "service_buffer_scale": scale,
        "service_constraint_is_soft": True,
        "service_constraint_bound": "markov_expected_completion_delay",
        "due_dates": dict(instance.due_dates),
    })
    return constraints


def add_due_date_tardiness_constraints(
    model,
    variables,
    instance,
    job_repair_buffers=None,
):
    """Compatibility wrapper for the former unscaled soft due-date model."""
    buffers = job_repair_buffers or {
        job: 0.0 for job in instance.job_end_operations
    }
    jobs = list(instance.job_end_operations)
    tardiness = model.addVars(
        jobs, lb=0.0, vtype=GRB.CONTINUOUS, name="L"
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
        "job_service_level_violation": tardiness,
        "job_expected_delays": buffers,
        "job_service_level_buffers": buffers,
        "job_tardiness": tardiness,
        "job_expected_repair_buffers": buffers,
        "service_constraint_is_soft": True,
        "service_constraint_bound": "nominal_or_unscaled_compatibility",
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
    """Compatibility wrapper for the former unscaled repair-buffer model."""
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
    service_violation_cost_per_time=1.0,
    tardiness_cost_per_time=None,
):
    """Minimize processing, operating and soft service-violation costs."""
    ensure_stochastic_parameters(instance)
    facility_cost_per_time = float(facility_cost_per_time)
    if facility_cost_per_time < 0.0:
        raise ValueError("facility_cost_per_time must be nonnegative.")
    if tardiness_cost_per_time is not None:
        service_violation_cost_per_time = tardiness_cost_per_time
    service_violation_cost_per_time = float(
        service_violation_cost_per_time
    )
    if service_violation_cost_per_time < 0.0:
        raise ValueError(
            "service_violation_cost_per_time must be nonnegative."
        )
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
    job_violation = variables.get("job_service_level_violation", {})
    total_violation = gp.quicksum(job_violation.values())
    service_violation_cost = (
        service_violation_cost_per_time * total_violation
    )
    total_cost = processing_cost + operating_cost + service_violation_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    variables.update({
        "machine_cost": machine_cost,
        "facility_cost_per_time": facility_cost_per_time,
        "service_violation_cost_per_time": service_violation_cost_per_time,
        "tardiness_cost_per_time": service_violation_cost_per_time,
        "processing_cost": processing_cost,
        "operating_cost": operating_cost,
        "total_service_level_violation": total_violation,
        "service_violation_cost": service_violation_cost,
        "total_tardiness": total_violation,
        "tardiness_cost": service_violation_cost,
        "total_cost": total_cost,
        "objective_mode": "minimize_economic_cost_with_soft_service_level",
        "objective_definition": (
            "sum_{i,k} c_k*p_ik*Y_ik + c_B*C_max + "
            "c_SL*sum_j L_service_j"
        ),
    })
    return total_cost
