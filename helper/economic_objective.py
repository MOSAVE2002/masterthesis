"""Add the shared economic objective and soft due dates to FJSP models.

Nominal, nonlinear and GNN formulations use these helpers so processing,
facility and tardiness costs remain directly comparable. Optional expected
repair buffers shift job completion requirements without hard infeasibility.
"""

import gurobipy as gp
from gurobipy import GRB

from helper.stochastic_fjsp import ensure_profile_parameters


def add_due_date_tardiness_constraints(
    model,
    variables,
    instance,
    job_repair_buffers=None,
):
    """Add soft due-date constraints and one tardiness variable per job.

    For every job, the constraint compares the completion time of its final
    operation plus an optional expected repair buffer with the configured due
    date. The nonnegative variable ``L_job`` absorbs any violation, so the due
    date remains feasible but its violation can be penalized in the objective.
    If no repair buffers are supplied, every buffer is set to zero.

    Parameters
    ----------
    model : gurobipy.Model
        Gurobi model to which the variables and constraints are added.
    variables : dict
        Shared model data containing the operation-completion variables under
        ``"C"``. The new constraints, tardiness variables and buffers are
        added to this dictionary.
    instance : FJSPData
        Problem instance containing job-end operations and due dates.
    job_repair_buffers : dict, optional
        Mapping from job identifiers to constant or Gurobi expressions for
        expected repair delays. The default is zero for every job.

    Returns
    -------
    dict
        Mapping from each job to its generated due-date constraint.
    """
    buffers = job_repair_buffers or dict.fromkeys(
        instance.job_end_operations, 0.0
    )
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
        "job_tardiness": tardiness,
        "job_repair_buffers": buffers,
        "due_dates": dict(instance.due_dates),
    })
    return constraints


def add_nominal_due_date_constraints(model, variables, instance):
    """Add soft due-date constraints without repair buffers.

    This is the nominal special case of
    :func:`add_due_date_tardiness_constraints`, in which every job-specific
    repair buffer is zero.

    Parameters
    ----------
    model : gurobipy.Model
        Gurobi model to which the constraints are added.
    variables : dict
        Shared model data containing the completion-time variables.
    instance : FJSPData
        Problem instance containing job-end operations and due dates.

    Returns
    -------
    dict
        Mapping from each job to its generated nominal due-date constraint.
    """
    constraints = add_due_date_tardiness_constraints(
        model, variables, instance
    )
    variables["nominal_due_date_constraints"] = constraints
    return constraints


def add_makespan(model, variables, instance):
    """Create and link the maximum job-completion variable.

    Args:
        model: Gurobi model receiving the makespan constraints.
        variables: Shared dictionary containing completion times and horizon.
        instance: FJSP instance defining every job's final operation.

    Returns:
        The created continuous makespan variable, also stored as ``C_max``.
    """
    makespan = model.addVar(
        lb=0.0,
        ub=float(variables["H"]),
        vtype=GRB.CONTINUOUS,
        name="C_max",
    )
    for job, end_operation in instance.job_end_operations.items():
        model.addConstr(
            makespan >= variables["C"][end_operation],
            name=f"makespan[{job}]",
        )
    variables["C_max"] = makespan
    return makespan


def add_economic_cost_objective(
    model,
    variables,
    instance,
    *,
    facility_cost_per_time=1.0,
    tardiness_cost_per_time=1.0,
):
    """Set the economic objective for the nominal or buffered FJSP model.

    The objective minimizes the sum of assignment-dependent machine
    processing costs, facility operating costs over the makespan and tardiness
    penalties. An existing makespan variable is reused; otherwise, the
    function creates one and links it to every job's final operation. Due-date
    constraints must already have created the job-tardiness variables.

    Parameters
    ----------
    model : gurobipy.Model
        Gurobi model whose minimization objective is set.
    variables : dict
        Shared model data containing machine assignments, processing times,
        the time horizon and job-tardiness variables. The objective components
        are added to this dictionary for reporting.
    instance : FJSPData
        Problem instance providing machine costs and job-end operations.
    facility_cost_per_time : float, optional
        Nonnegative cost incurred per unit of makespan.
    tardiness_cost_per_time : float, optional
        Nonnegative penalty incurred per unit of total job tardiness.

    Returns
    -------
    gurobipy.LinExpr
        Linear expression representing total economic cost.

    Raises
    ------
    ValueError
        If a facility, tardiness or machine-processing cost is negative.
    """
    ensure_profile_parameters(instance)
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
        makespan = add_makespan(model, variables, instance)

    processing_cost = gp.quicksum(
        machine_cost[machine]
        * float(instance.processing_times[operation, machine])
        * variables["Y"][operation, machine]
        for operation, machine in variables["Y_index"]
    )
    operating_cost = facility_cost_per_time * makespan
    total_tardiness = gp.quicksum(variables["job_tardiness"].values())
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
        "objective_mode": "minimize_economic_cost_with_buffered_tardiness",
        "objective_definition": (
            "sum_{i,k} c_k*p_ik*Y_ik + c_B*C_max + "
            "c_D*sum_j L_j"
        ),
    })
    return total_cost
