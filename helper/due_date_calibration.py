"""Calibrate total-work-content due dates against a nominal FJSP schedule.

The helper solves and caches a nominal makespan problem for each physical
instance structure. Its makespan anchors a total-work-content factor that can
be shifted reproducibly for training, benchmark and extrapolation scenarios.
"""

import importlib
import math

import gurobipy as gp


_base = importlib.import_module("03_Gurobi.build_fjsp")
_NOMINAL_MAKESPAN_CACHE = {}


def calibrated_total_work_content_due_dates(instance, config, relative_offset):
    """Calculate job due dates from calibrated total work content.

    Args:
        instance: FJSP instance whose nominal makespan is calibrated.
        config: Gurobi settings for the calibration solve.
        relative_offset: Relative increase or decrease from the nominal factor.

    Returns:
        Calibration result, work contents, factors, offset and derived dates.
    """
    relative_offset = float(relative_offset)
    if not math.isfinite(relative_offset) or relative_offset <= -1.0:
        raise ValueError("relative_offset must be finite and greater than -1.")
    calibration = _nominal_makespan_calibration(instance, config)
    work_content = _total_work_content_by_job(instance)
    mean_work_content = sum(work_content.values()) / len(work_content)
    nominal_factor = calibration["makespan"] / mean_work_content
    effective_factor = (1.0 + relative_offset) * nominal_factor
    due_dates = {
        job: float(math.ceil(effective_factor * work_content[job] - 1e-12))
        for job in instance.jobs
    }
    return {
        "calibration": calibration,
        "work_content": work_content,
        "nominal_factor": nominal_factor,
        "effective_factor": effective_factor,
        "relative_offset": relative_offset,
        "due_dates": due_dates,
    }


def _total_work_content_by_job(instance):
    """Compute each job's work content from mean eligible-machine durations.

    Returns:
        Positive total work content indexed by job identifier.

    Raises:
        ValueError: If no valid positive job work content can be derived.
    """
    work_content = {
        job: sum(
            sum(
                float(instance.processing_times[operation, machine])
                for machine in instance.eligible_machines[operation]
            ) / len(instance.eligible_machines[operation])
            for operation in operations
        )
        for job, operations in instance.jobs.items()
    }
    if not work_content or any(
        not math.isfinite(value) or value <= 0.0
        for value in work_content.values()
    ):
        raise ValueError("Every job must have positive total work content.")
    return work_content


def _nominal_makespan_calibration(instance, config):
    """Solve or retrieve the nominal makespan calibration of an instance.

    A structural signature makes the cache independent of mutable due dates.
    New signatures are optimized with the configured Gurobi limits and retain
    both the makespan and selected machine assignment.

    Returns:
        Dictionary containing nominal makespan and operation assignments.
    """
    operations = set(instance.real_operations)
    signature = (
        tuple(sorted(
            (job, tuple(job_operations))
            for job, job_operations in instance.jobs.items()
        )),
        tuple(sorted(
            (operation, tuple(sorted(machines)))
            for operation, machines in instance.eligible_machines.items()
            if operation in operations
        )),
        tuple(sorted(
            (operation, machine, float(value))
            for (operation, machine), value in instance.processing_times.items()
            if operation in operations
            and machine in instance.eligible_machines[operation]
        )),
    )
    if signature in _NOMINAL_MAKESPAN_CACHE:
        return dict(_NOMINAL_MAKESPAN_CACHE[signature])

    model = gp.Model("adaptive_due_date_nominal_makespan")
    model.Params.OutputFlag = int(config["output_flag"])
    model.Params.TimeLimit = float(config["time_limit_seconds"])
    model.Params.MIPGap = float(config["mip_gap"])
    model.Params.Seed = int(config["seed"])
    model, variables = _base.build_makespan_fjsp(model, instance)
    model.optimize()
    if model.SolCount == 0:
        status = int(model.Status)
        model.dispose()
        raise RuntimeError(
            "Nominal makespan calibration found no feasible schedule; "
            f"Gurobi status={status}."
        )
    result = {
        "makespan": float(variables["C_max"].X),
        "assignment": {
            operation: max(
                instance.eligible_machines[operation],
                key=lambda machine: float(
                    variables["Y"][operation, machine].X
                ),
            )
            for operation in instance.real_operations
        },
    }
    model.dispose()
    _NOMINAL_MAKESPAN_CACHE[signature] = dict(result)
    return result
