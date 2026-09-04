"""Controlled comparison of the nominal and nonlinear FJSP formulations.

For every physical instance, the script first solves an unconstrained nominal
makespan problem. It then sets a common due date to
``ceil((1 + offset) * nominal_makespan)`` and solves both economic models with
the identical instance, due date and Gurobi settings. Every incumbent is
post-evaluated with common-random-number Monte Carlo replications.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import gurobipy as gp


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config.json"
DEFAULT_OUTPUT_DIRECTORY = ROOT_DIR / "06_Evaluation" / "results"

_generator = importlib.import_module("01_generator.instance_generator")
_base = importlib.import_module("03_Gurobi.build_fjsp")
_nonlinear = importlib.import_module("03_Gurobi.build_fjsp_with_nonlinear")
_simulation = importlib.import_module("05_Simulation.preempt_resume")


def controlled_due_date(nominal_makespan, relative_offset):
    """Return the integer common due date used by both formulations."""
    nominal_makespan = float(nominal_makespan)
    relative_offset = float(relative_offset)
    if nominal_makespan <= 0.0:
        raise ValueError("nominal_makespan must be positive.")
    if not math.isfinite(relative_offset) or relative_offset <= -1.0:
        raise ValueError("relative_offset must be finite and greater than -1.")
    return float(math.ceil((1.0 + relative_offset) * nominal_makespan - 1e-12))


def _value(item):
    if hasattr(item, "X"):
        return float(item.X)
    if hasattr(item, "getValue"):
        return float(item.getValue())
    return float(item)


def _status_name(model):
    return _base.STATUS_NAMES.get(int(model.Status), str(model.Status))


def _selected_machines(instance, variables):
    return {
        operation: max(
            instance.eligible_machines[operation],
            key=lambda machine: _value(
                variables["Y"][operation, machine]
            ),
        )
        for operation in instance.real_operations
    }


def _fixed_schedule(instance, variables, selected):
    operations = list(instance.real_operations)
    durations = {
        operation: float(
            instance.processing_times[operation, selected[operation]]
        )
        for operation in operations
    }
    completions = {
        operation: _value(variables["C"][operation])
        for operation in operations
    }
    starts = {
        operation: max(
            0.0, completions[operation] - durations[operation]
        )
        for operation in operations
    }
    machine_edges = []
    for machine in range(instance.num_machines):
        ordered = sorted(
            (
                operation
                for operation in operations
                if selected[operation] == machine
            ),
            key=lambda operation: (
                starts[operation], completions[operation], operation
            ),
        )
        machine_edges.extend(
            (source, target, machine)
            for source, target in zip(ordered, ordered[1:])
        )
    schedule = _simulation.FixedSchedule(
        operations=tuple(operations),
        selected_machines=dict(selected),
        processing_times=durations,
        planned_starts=starts,
        job_predecessors={
            operation: tuple(instance.predecessors.get(operation, ()))
            for operation in operations
        },
        machine_edges=tuple(machine_edges),
        jobs={job: tuple(values) for job, values in instance.jobs.items()},
        job_end_operations=dict(instance.job_end_operations),
        due_dates={job: float(value) for job, value in instance.due_dates.items()},
        weibull_scale=dict(instance.weibull_alpha),
        weibull_shape=dict(instance.weibull_beta),
        repair_rate=dict(instance.repair_rate),
    )
    return schedule, completions


def _new_model(name, solver_config):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(solver_config["time_limit"])
    model.Params.MIPGap = float(solver_config["mip_gap"])
    model.Params.Seed = int(solver_config["seed"])
    return model


def _calibrate_nominal_makespan(instance, solver_config):
    model = _new_model(
        f"nominal_calibration_{instance.instance_name}", solver_config
    )
    model, variables = _base.build_fjsp(
        model,
        instance,
        include_makespan=True,
        enforce_due_dates=False,
        economic_objective=False,
    )
    model.optimize()
    if model.SolCount == 0:
        status = _status_name(model)
        model.dispose()
        raise RuntimeError(
            f"Nominal calibration failed for {instance.instance_name}: {status}."
        )
    result = {
        "makespan": _value(variables["C_max"]),
        "status": _status_name(model),
        "gap": float(model.MIPGap),
        "runtime_seconds": float(model.Runtime),
    }
    model.dispose()
    return result


def _solve_formulation(
    instance,
    formulation,
    solver_config,
    graph_config,
    simulation_config,
    replications,
    simulation_seed,
):
    model = _new_model(
        f"{formulation}_{instance.instance_name}", solver_config
    )
    if formulation == "base":
        model, variables = _base.build_fjsp(
            model,
            instance,
            facility_cost_per_time=solver_config["facility_cost_per_time"],
            tardiness_cost_per_time=solver_config["tardiness_cost_per_time"],
        )
    elif formulation == "nonlinear":
        model, variables = _nonlinear.build_fjsp(
            model,
            instance,
            reliability_graph_config=graph_config,
            facility_cost_per_time=solver_config["facility_cost_per_time"],
            tardiness_cost_per_time=solver_config["tardiness_cost_per_time"],
        )
    else:
        raise ValueError(f"Unknown formulation: {formulation!r}.")
    started = time.perf_counter()
    model.optimize()
    optimizer_wall_seconds = time.perf_counter() - started
    result = {
        "formulation": formulation,
        "status": _status_name(model),
        "solution_count": int(model.SolCount),
        "runtime_seconds": float(model.Runtime),
        "optimizer_wall_seconds": optimizer_wall_seconds,
    }
    if model.SolCount == 0:
        model.dispose()
        return result, []

    selected = _selected_machines(instance, variables)
    schedule, completions = _fixed_schedule(instance, variables, selected)
    simulation_result = _simulation.simulate_fixed_schedule(
        schedule,
        replications=replications,
        seed=simulation_seed,
        config=simulation_config,
    )
    buffers = variables.get("job_expected_repair_buffers", {})
    job_rows = []
    for position, job in enumerate(simulation_result.job_ids):
        completion = completions[instance.job_end_operations[job]]
        buffer = _value(buffers[job]) if job in buffers else 0.0
        optimization_tardiness = _value(
            variables["job_tardiness"][job]
        )
        due_date = float(instance.due_dates[job])
        job_rows.append({
            "job": job,
            "nominal_completion": completion,
            "due_date": due_date,
            "nominal_slack": due_date - completion,
            "expected_repair_buffer": buffer,
            "optimization_tardiness": optimization_tardiness,
            "protected_slack": due_date - completion - buffer,
            "mc_ontime_probability": float(
                simulation_result.job_ontime_probabilities[position]
            ),
            "mc_standard_error": float(
                simulation_result.job_probability_standard_errors[position]
            ),
            "mc_mean_completion": float(
                simulation_result.job_mean_completion_times[position]
            ),
        })
    profile_counts = {"old": 0, "new": 0}
    for machine in selected.values():
        profile_counts[instance.machine_profiles[machine]] += 1
    result.update({
        "objective": float(model.ObjVal),
        "best_bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap),
        "processing_cost": _value(variables["processing_cost"]),
        "operating_cost": _value(variables["operating_cost"]),
        "tardiness_cost": _value(variables["tardiness_cost"]),
        "total_tardiness": _value(variables["total_tardiness"]),
        "nominal_makespan": _value(variables["C_max"]),
        "old_operations": profile_counts["old"],
        "new_operations": profile_counts["new"],
        "minimum_nominal_slack": min(
            row["nominal_slack"] for row in job_rows
        ),
        "maximum_expected_repair_buffer": max(
            row["expected_repair_buffer"] for row in job_rows
        ),
        "minimum_protected_slack": min(
            row["protected_slack"] for row in job_rows
        ),
        "minimum_mc_ontime_probability": min(
            row["mc_ontime_probability"] for row in job_rows
        ),
        "mean_mc_ontime_probability": statistics.fmean(
            row["mc_ontime_probability"] for row in job_rows
        ),
        "simulation_mean_failures": float(simulation_result.mean_failures),
        "simulation_mean_total_repair_delay": float(
            simulation_result.mean_total_repair_delay
        ),
        "selected_machines": selected,
    })
    model.dispose()
    return result, job_rows


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def _summary(comparisons):
    solved = [
        row for row in comparisons
        if row["base_status"] in {"OPTIMAL", "TIME_LIMIT"}
        and row["nonlinear_status"] in {"OPTIMAL", "TIME_LIMIT"}
        and row["base_solution_count"] > 0
        and row["nonlinear_solution_count"] > 0
    ]
    return {
        "comparisons": len(comparisons),
        "jointly_solved": len(solved),
        "operations": sum(row["operations"] for row in solved),
        "changed_machine_assignments": sum(
            row["changed_machine_assignments"] for row in solved
        ),
        "changed_profile_assignments": sum(
            row["changed_profile_assignments"] for row in solved
        ),
        "mean_base_minimum_mc_ontime_probability": _mean(
            solved, "base_minimum_mc_ontime_probability"
        ),
        "mean_nonlinear_minimum_mc_ontime_probability": _mean(
            solved, "nonlinear_minimum_mc_ontime_probability"
        ),
        "mean_minimum_mc_probability_delta": _mean(
            solved, "minimum_mc_probability_delta"
        ),
        "mean_objective_delta": _mean(solved, "objective_delta"),
        "mean_base_runtime_seconds": _mean(
            solved, "base_runtime_seconds"
        ),
        "mean_nonlinear_runtime_seconds": _mean(
            solved, "nonlinear_runtime_seconds"
        ),
    }


def _offset_summaries(comparisons):
    summaries = []
    for offset in sorted({
        float(row["relative_due_date_offset"]) for row in comparisons
    }):
        rows = [
            row for row in comparisons
            if float(row["relative_due_date_offset"]) == offset
            and row["base_solution_count"] > 0
            and row["nonlinear_solution_count"] > 0
        ]
        summaries.append({
            "relative_due_date_offset": offset,
            "comparisons": len(rows),
            "operations": sum(row["operations"] for row in rows),
            "changed_machine_assignments": sum(
                row["changed_machine_assignments"] for row in rows
            ),
            "changed_profile_assignments": sum(
                row["changed_profile_assignments"] for row in rows
            ),
            "mean_base_new_operations": _mean(rows, "base_new_operations"),
            "mean_nonlinear_new_operations": _mean(
                rows, "nonlinear_new_operations"
            ),
            "mean_base_minimum_nominal_slack": _mean(
                rows, "base_minimum_nominal_slack"
            ),
            "mean_nonlinear_minimum_nominal_slack": _mean(
                rows, "nonlinear_minimum_nominal_slack"
            ),
            "mean_nonlinear_maximum_expected_repair_buffer": _mean(
                rows, "nonlinear_maximum_expected_repair_buffer"
            ),
            "mean_nonlinear_minimum_protected_slack": _mean(
                rows, "nonlinear_minimum_protected_slack"
            ),
            "mean_base_minimum_mc_ontime_probability": _mean(
                rows, "base_minimum_mc_ontime_probability"
            ),
            "mean_nonlinear_minimum_mc_ontime_probability": _mean(
                rows, "nonlinear_minimum_mc_ontime_probability"
            ),
            "mean_minimum_mc_probability_delta": _mean(
                rows, "minimum_mc_probability_delta"
            ),
            "mean_objective_delta": _mean(rows, "objective_delta"),
            "mean_base_total_repair_delay": _mean(
                rows, "base_simulation_mean_total_repair_delay"
            ),
            "mean_nonlinear_total_repair_delay": _mean(
                rows, "nonlinear_simulation_mean_total_repair_delay"
            ),
            "mean_base_runtime_seconds": _mean(
                rows, "base_runtime_seconds"
            ),
            "mean_nonlinear_runtime_seconds": _mean(
                rows, "nonlinear_runtime_seconds"
            ),
        })
    return summaries


def _job_delay_summaries(job_results):
    summaries = []
    keys = sorted({
        (float(row["relative_due_date_offset"]), row["formulation"])
        for row in job_results
    })
    for offset, formulation in keys:
        rows = [
            row for row in job_results
            if float(row["relative_due_date_offset"]) == offset
            and row["formulation"] == formulation
        ]
        mean_buffer = statistics.fmean(
            float(row["expected_repair_buffer"]) for row in rows
        )
        mean_realized_delay = statistics.fmean(
            float(row["mc_mean_completion"])
            - float(row["nominal_completion"])
            for row in rows
        )
        summaries.append({
            "relative_due_date_offset": offset,
            "formulation": formulation,
            "jobs": len(rows),
            "mean_expected_repair_buffer": mean_buffer,
            "mean_mc_job_completion_delay": mean_realized_delay,
            "mean_mc_delay_to_buffer_ratio": (
                mean_realized_delay / mean_buffer
                if mean_buffer > 0.0 else None
            ),
        })
    return summaries


def _pearson(first, second):
    if len(first) != len(second) or len(first) < 2:
        return None
    mean_first = statistics.fmean(first)
    mean_second = statistics.fmean(second)
    numerator = sum(
        (left - mean_first) * (right - mean_second)
        for left, right in zip(first, second)
    )
    denominator = math.sqrt(
        sum((value - mean_first) ** 2 for value in first)
        * sum((value - mean_second) ** 2 for value in second)
    )
    return numerator / denominator if denominator > 0.0 else None


def _buffer_metrics(rows):
    buffers = [float(row["expected_repair_buffer"]) for row in rows]
    delays = [
        float(row["mc_mean_completion"])
        - float(row["nominal_completion"])
        for row in rows
    ]
    errors = [buffer - delay for buffer, delay in zip(buffers, delays)]
    return {
        "jobs": len(rows),
        "mean_buffer": statistics.fmean(buffers),
        "mean_mc_job_completion_delay": statistics.fmean(delays),
        "pearson_buffer_mc_delay": _pearson(buffers, delays),
        "mae": statistics.fmean(abs(value) for value in errors),
        "rmse": math.sqrt(statistics.fmean(value ** 2 for value in errors)),
        "mean_error": statistics.fmean(errors),
    }


def _buffer_alignment(job_results):
    nonlinear_rows = [
        row for row in job_results if row["formulation"] == "nonlinear"
    ]
    if not nonlinear_rows:
        return None
    return {
        "method": "unscaled_expected_buffer_vs_mc_completion_delay",
        "instances": sorted({
            row["instance_name"] for row in nonlinear_rows
        }),
        "overall": _buffer_metrics(nonlinear_rows),
    }


def run_analysis(
    *,
    instances=1,
    num_jobs=3,
    num_machines=3,
    offsets=None,
    replications=None,
    time_limit=None,
    mip_gap=None,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    generation = config["instances"]["generation"]
    adaptive = config["training"]["data_generation"]["adaptive_due_dates"]
    offsets = [
        float(value) for value in (
            offsets
            if offsets is not None
            else adaptive["relative_makespan_offsets"]
        )
    ]
    if not offsets or any(
        not math.isfinite(value) or value <= -1.0 for value in offsets
    ):
        raise ValueError("offsets must be finite and greater than -1.")
    replications = int(
        config["evaluation"]["replications"]
        if replications is None else replications
    )
    common = config["solvers"]["gurobi"]["common"]
    solver_config = {
        "time_limit": float(
            common["TimeLimit"] if time_limit is None else time_limit
        ),
        "mip_gap": float(common["MIPGap"] if mip_gap is None else mip_gap),
        "seed": int(common["Seed"]),
        "facility_cost_per_time": float(
            config["objective"]["facility_cost_per_time"]
        ),
        "tardiness_cost_per_time": float(
            config["objective"].get("tardiness_cost_per_time", 1.0)
        ),
    }
    graph_config = config["constraint"]["weibull"]["reliability_graph"]
    simulation_config = config["evaluation"]["simulation"]
    comparisons, job_results, calibrations = [], [], []
    started = time.perf_counter()
    for instance_number in range(1, int(instances) + 1):
        rng_seed = int(generation["random_seed"]) + 100_000 + instance_number
        instance = _generator.FJSPData(
            nb_instance=instance_number,
            num_jobs=int(num_jobs),
            num_machines=int(num_machines),
            operations_per_job_min=generation["operations_per_job"][0],
            operations_per_job_max=generation["operations_per_job"][1],
            flag_save_file=False,
            processing_time_range=generation["processing_times"]["base_range"],
            machine_profile_config=generation["machine_profiles"],
            due_date_config=generation["due_dates"],
            random_source=random.Random(rng_seed),
        )
        calibration = _calibrate_nominal_makespan(instance, solver_config)
        calibrations.append({
            "instance_name": instance.instance_name,
            "random_seed": rng_seed,
            **calibration,
        })
        print(
            f"[Base-vs-NL] {instance.instance_name} | "
            f"Cmax*={calibration['makespan']:.3f}",
            flush=True,
        )
        for offset in offsets:
            due_date = controlled_due_date(calibration["makespan"], offset)
            instance.due_dates = {job: due_date for job in instance.jobs}
            simulation_seed = (
                int(config["evaluation"]["random_seed"])
                + 10_000 * instance_number
                + int(round(100_000 * offset))
            )
            outcomes = {}
            outcome_jobs = {}
            for formulation in ("base", "nonlinear"):
                outcomes[formulation], outcome_jobs[formulation] = (
                    _solve_formulation(
                        instance,
                        formulation,
                        solver_config,
                        graph_config,
                        simulation_config,
                        replications,
                        simulation_seed,
                    )
                )
            base_result = outcomes["base"]
            nonlinear_result = outcomes["nonlinear"]
            both_solved = (
                base_result.get("selected_machines") is not None
                and nonlinear_result.get("selected_machines") is not None
            )
            changed_machines = changed_profiles = None
            if both_solved:
                changed_machines = sum(
                    base_result["selected_machines"][operation]
                    != nonlinear_result["selected_machines"][operation]
                    for operation in instance.real_operations
                )
                changed_profiles = sum(
                    instance.machine_profiles[
                        base_result["selected_machines"][operation]
                    ]
                    != instance.machine_profiles[
                        nonlinear_result["selected_machines"][operation]
                    ]
                    for operation in instance.real_operations
                )
            comparison = {
                "instance_name": instance.instance_name,
                "random_seed": rng_seed,
                "operations": len(instance.real_operations),
                "relative_due_date_offset": offset,
                "calibrated_nominal_makespan": calibration["makespan"],
                "common_due_date": due_date,
                "simulation_replications": replications,
                "simulation_seed": simulation_seed,
                "changed_machine_assignments": changed_machines,
                "changed_profile_assignments": changed_profiles,
            }
            exported_fields = (
                "status",
                "solution_count",
                "objective",
                "mip_gap",
                "runtime_seconds",
                "processing_cost",
                "operating_cost",
                "tardiness_cost",
                "total_tardiness",
                "nominal_makespan",
                "old_operations",
                "new_operations",
                "minimum_nominal_slack",
                "maximum_expected_repair_buffer",
                "minimum_protected_slack",
                "minimum_mc_ontime_probability",
                "mean_mc_ontime_probability",
                "simulation_mean_failures",
                "simulation_mean_total_repair_delay",
            )
            for formulation, result in outcomes.items():
                for field in exported_fields:
                    comparison[f"{formulation}_{field}"] = result.get(field)
            comparison["objective_delta"] = (
                nonlinear_result["objective"] - base_result["objective"]
                if both_solved else None
            )
            comparison["minimum_mc_probability_delta"] = (
                nonlinear_result["minimum_mc_ontime_probability"]
                - base_result["minimum_mc_ontime_probability"]
                if both_solved else None
            )
            comparisons.append(comparison)
            for formulation, rows in outcome_jobs.items():
                for row in rows:
                    job_results.append({
                        "instance_name": instance.instance_name,
                        "relative_due_date_offset": offset,
                        "formulation": formulation,
                        "simulation_replications": replications,
                        **row,
                    })
            print(
                f"  offset={offset:.3f} due={due_date:.1f} | "
                f"base={base_result['status']} | "
                f"nonlinear={nonlinear_result['status']} | "
                f"machine changes={changed_machines} | "
                f"profile changes={changed_profiles} | "
                f"alpha delta={comparison['minimum_mc_probability_delta']}",
                flush=True,
            )

    summary = _summary(comparisons)
    offset_summaries = _offset_summaries(comparisons)
    job_delay_summaries = _job_delay_summaries(job_results)
    buffer_alignment = _buffer_alignment(job_results)
    output_directory = Path(output_directory)
    comparison_path = output_directory / "base_nonlinear_comparison.csv"
    job_path = output_directory / "base_nonlinear_job_comparison.csv"
    offset_summary_path = (
        output_directory / "base_nonlinear_offset_summary.csv"
    )
    delay_summary_path = (
        output_directory / "base_nonlinear_job_delay_summary.csv"
    )
    summary_path = output_directory / "base_nonlinear_comparison.json"
    _write_csv(comparison_path, comparisons)
    _write_csv(job_path, job_results)
    _write_csv(offset_summary_path, offset_summaries)
    _write_csv(delay_summary_path, job_delay_summaries)
    payload = {
        "method": "controlled_relative_nominal_makespan_due_dates_v1",
        "due_date_definition": "ceil((1 + offset) * nominal_makespan)",
        "offsets": offsets,
        "replications": replications,
        "num_jobs": int(num_jobs),
        "num_machines": int(num_machines),
        "solver": solver_config,
        "simulation": simulation_config,
        "machine_profiles": generation["machine_profiles"]["profiles"],
        "calibrations": calibrations,
        "summary": summary,
        "offset_summaries": offset_summaries,
        "job_delay_summaries": job_delay_summaries,
        "buffer_alignment": buffer_alignment,
        "wall_seconds": time.perf_counter() - started,
        "comparison_csv": str(comparison_path),
        "job_comparison_csv": str(job_path),
        "offset_summary_csv": str(offset_summary_path),
        "job_delay_summary_csv": str(delay_summary_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload, comparisons, job_results


def _arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the base and nonlinear FJSP models under controlled "
            "due-date slack and Monte Carlo post-evaluation."
        )
    )
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--num-jobs", type=int, default=3)
    parser.add_argument("--num-machines", type=int, default=3)
    parser.add_argument("--offsets", type=float, nargs="+")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--mip-gap", type=float)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    run_analysis(
        instances=arguments.instances,
        num_jobs=arguments.num_jobs,
        num_machines=arguments.num_machines,
        offsets=arguments.offsets,
        replications=arguments.replications,
        time_limit=arguments.time_limit,
        mip_gap=arguments.mip_gap,
        output_directory=arguments.output_directory,
    )
