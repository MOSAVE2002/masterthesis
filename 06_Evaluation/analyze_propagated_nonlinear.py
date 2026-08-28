"""Compare non-propagated and propagated nonlinear repair buffers.

This is an experimental evaluator. It does not change the production solver.
The propagated formulation augments every operation duration by its direct
nonlinear expected repair delay and propagates that delay over the selected
job and machine precedence graph through robust start/completion variables.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
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
DEFAULT_OUTPUT_DIRECTORY = (
    ROOT_DIR / "06_Evaluation" / "results" / "propagated_nonlinear"
)

_base = importlib.import_module("03_Gurobi.build_fjsp")
_nonlinear = importlib.import_module("03_Gurobi.build_fjsp_with_nonlinear")
_comparison = importlib.import_module("06_Evaluation.compare_base_nonlinear")
_generator = importlib.import_module("01_generator.instance_generator")
_simulation = importlib.import_module("05_Simulation.preempt_resume")

from helper.economic_objective import add_economic_cost_objective
from helper.sequence_setup import (
    normalize_reliability_graph_config,
    reliability_graph_config_dict,
)
from helper.stochastic_fjsp import (
    ensure_stochastic_parameters,
    stochastic_parameters,
)


def _value(item):
    if hasattr(item, "X"):
        return float(item.X)
    return float(item.getValue())


def build_propagated_nonlinear(
    model,
    instance,
    *,
    reliability_graph_config=None,
    facility_cost_per_time=1.0,
):
    """Build the experimental propagated expected-buffer MINLP."""
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)
    ensure_stochastic_parameters(instance)
    parameters = stochastic_parameters(instance)
    model, variables = _base.build_fjsp(
        model,
        instance,
        include_makespan=False,
        horizon_upper_bound=max(instance.due_dates.values()),
        enforce_due_dates=True,
        economic_objective=False,
    )
    model.Params.NonConvex = 2
    horizon = float(variables["H"])
    for operation in variables["real_operations"]:
        variables["C"][operation].ub = horizon
    variables.update({
        "machine_modernity": parameters["theta"],
        "machine_speed": parameters["speed"],
        "weibull_alpha": parameters["alpha"],
        "weibull_beta": parameters["beta"],
        "repair_rate": parameters["repair_rate"],
        "reliability_graph_config": reliability_graph_config_dict(graph_cfg),
    })
    _nonlinear._add_midpoint_state(model, variables, instance)
    _nonlinear._add_pd_quadrature(model, variables, instance, graph_cfg)

    direct_delay = {}
    for operation in variables["real_operations"]:
        direct_delay[operation] = gp.quicksum(
            variables["pi_fail"][operation, machine]
            / parameters["repair_rate"][machine]
            for machine in instance.eligible_machines[operation]
        )

    robust_start = model.addVars(
        variables["real_operations"],
        lb=0.0,
        ub=horizon,
        name="propagated_start",
    )
    robust_completion = model.addVars(
        variables["real_operations"],
        lb=0.0,
        ub=horizon,
        name="propagated_completion",
    )
    for operation in variables["real_operations"]:
        model.addConstr(
            robust_start[operation] >= variables["S"][operation],
            name=f"propagated_not_before_nominal[{operation}]",
        )
        model.addConstr(
            robust_completion[operation]
            == robust_start[operation]
            + variables["D"][operation]
            + direct_delay[operation],
            name=f"propagated_completion_def[{operation}]",
        )
        for predecessor in instance.predecessors.get(operation, []):
            model.addConstr(
                robust_start[operation] >= robust_completion[predecessor],
                name=(
                    f"propagated_job_precedence["
                    f"{predecessor},{operation}]"
                ),
            )

    for operation_i, operation_j, machine in variables["X_index"]:
        x = variables["X"][operation_i, operation_j, machine]
        yi = variables["Y"][operation_i, machine]
        yj = variables["Y"][operation_j, machine]
        model.addConstr(
            robust_start[operation_i]
            >= robust_completion[operation_j]
            - horizon * (2 + x - yi - yj),
            name=(
                f"propagated_machine_j_before_i["
                f"{operation_j},{operation_i},{machine}]"
            ),
        )
        model.addConstr(
            robust_start[operation_j]
            >= robust_completion[operation_i]
            - horizon * (3 - x - yi - yj),
            name=(
                f"propagated_machine_i_before_j["
                f"{operation_i},{operation_j},{machine}]"
            ),
        )

    job_buffers = {}
    robust_due_dates = {}
    for job, end_operation in instance.job_end_operations.items():
        job_buffers[job] = (
            robust_completion[end_operation] - variables["C"][end_operation]
        )
        robust_due_dates[job] = model.addConstr(
            robust_completion[end_operation] <= float(instance.due_dates[job]),
            name=f"propagated_due_date[{job}]",
        )
    add_economic_cost_objective(
        model,
        variables,
        instance,
        facility_cost_per_time=facility_cost_per_time,
    )
    variables.update({
        "Delta": direct_delay,
        "propagated_start": robust_start,
        "propagated_completion": robust_completion,
        "job_expected_repair_buffers": job_buffers,
        "robust_due_date_constraints": robust_due_dates,
        "service_scope": graph_cfg.service_scope,
        "due_dates": dict(instance.due_dates),
        "job_repair_buffer_label_method": (
            "propagated_weibull_expected_repair_buffer_v1"
        ),
        "constraint_type": "weibull",
        "formulation": (
            "nonlinear_propagated_expected_repair_buffer_cost_v1"
        ),
    })
    model.update()
    return model, variables


def _model(solver, name, settings):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(settings["time_limit"])
    model.Params.MIPGap = float(settings["mip_gap"])
    model.Params.Seed = int(settings["seed"])
    if solver in {"nonlinear", "propagated"}:
        model.Params.NonConvex = 2
    return model


def _solve(
    instance,
    formulation,
    settings,
    graph_config,
    simulation_config,
    replications,
    simulation_seed,
):
    model = _model(
        formulation, f"{formulation}_{instance.instance_name}", settings
    )
    build_started = time.perf_counter()
    if formulation == "base":
        model, variables = _base.build_fjsp(
            model,
            instance,
            facility_cost_per_time=settings["facility_cost_per_time"],
        )
    elif formulation == "nonlinear":
        model, variables = _nonlinear.build_fjsp(
            model,
            instance,
            reliability_graph_config=graph_config,
            facility_cost_per_time=settings["facility_cost_per_time"],
        )
    elif formulation == "propagated":
        model, variables = build_propagated_nonlinear(
            model,
            instance,
            reliability_graph_config=graph_config,
            facility_cost_per_time=settings["facility_cost_per_time"],
        )
    else:
        raise ValueError(f"Unknown formulation: {formulation}.")
    build_seconds = time.perf_counter() - build_started
    model.optimize()
    result = {
        "formulation": formulation,
        "status": _base.STATUS_NAMES.get(model.Status, str(model.Status)),
        "solution_count": int(model.SolCount),
        "build_seconds": build_seconds,
        "runtime_seconds": float(model.Runtime),
    }
    if not model.SolCount:
        model.dispose()
        return result

    selected = _comparison._selected_machines(instance, variables)
    schedule, completions = _comparison._fixed_schedule(
        instance, variables, selected
    )
    simulated = _simulation.simulate_fixed_schedule(
        schedule,
        replications=replications,
        seed=simulation_seed,
        config=simulation_config,
    )
    buffers = variables.get("job_expected_repair_buffers", {})
    buffer_values = [
        _value(buffers[job]) for job in sorted(buffers)
    ] or [0.0]
    profile_counts = {"old": 0, "new": 0}
    for machine in selected.values():
        profile_counts[instance.machine_profiles[machine]] += 1
    protected_slacks = []
    for job, end_operation in instance.job_end_operations.items():
        protected_slacks.append(
            float(instance.due_dates[job])
            - completions[end_operation]
            - (
                _value(buffers[job]) if job in buffers else 0.0
            )
        )
    result.update({
        "objective": float(model.ObjVal),
        "mip_gap": float(model.MIPGap),
        "processing_cost": _value(variables["processing_cost"]),
        "operating_cost": _value(variables["operating_cost"]),
        "nominal_makespan": _value(variables["C_max"]),
        "old_operations": profile_counts["old"],
        "new_operations": profile_counts["new"],
        "maximum_raw_buffer": max(buffer_values),
        "maximum_applied_buffer": max(buffer_values),
        "minimum_protected_slack": min(protected_slacks),
        "minimum_mc_ontime_probability": min(
            simulated.job_ontime_probabilities
        ),
        "mean_mc_ontime_probability": statistics.fmean(
            simulated.job_ontime_probabilities
        ),
        "mean_total_repair_delay": float(simulated.mean_total_repair_delay),
        "mean_failures": float(simulated.mean_failures),
        "selected_machines": selected,
    })
    model.dispose()
    return result


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_size(value):
    jobs, machines = value.lower().split("x", 1)
    jobs, machines = int(jobs), int(machines)
    if jobs <= 0 or machines <= 0:
        raise ValueError("Problem sizes must be positive.")
    return jobs, machines


def run_analysis(
    *,
    sizes=("3x3",),
    instances_per_size=1,
    offsets=None,
    replications=2000,
    time_limit=30.0,
    mip_gap=0.01,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    generation = config["instances"]["generation"]
    if offsets is None:
        offsets = config["training"]["data_generation"][
            "adaptive_due_dates"
        ]["relative_makespan_offsets"]
    offsets = [float(value) for value in offsets]
    graph_config = config["constraint"]["weibull"]["reliability_graph"]
    simulation_config = config["evaluation"]["simulation"]
    settings = {
        "time_limit": float(time_limit),
        "mip_gap": float(mip_gap),
        "seed": int(config["solvers"]["gurobi"]["common"]["Seed"]),
        "facility_cost_per_time": float(
            config["objective"]["facility_cost_per_time"]
        ),
    }
    rows = []
    started = time.perf_counter()
    for raw_size in sizes:
        num_jobs, num_machines = _parse_size(raw_size)
        size = f"{num_jobs}x{num_machines}"
        for instance_number in range(1, int(instances_per_size) + 1):
            seed = (
                int(generation["random_seed"])
                + 1_000_000
                + 10_000 * num_jobs
                + 100 * num_machines
                + instance_number
            )
            instance = _generator.FJSPData(
                nb_instance=instance_number,
                num_jobs=num_jobs,
                num_machines=num_machines,
                operations_per_job_min=generation["operations_per_job"][0],
                operations_per_job_max=generation["operations_per_job"][1],
                flag_save_file=False,
                processing_time_range=generation["processing_times"][
                    "base_range"
                ],
                machine_profile_config=generation["machine_profiles"],
                due_date_config=generation["due_dates"],
                random_source=random.Random(seed),
            )
            calibration = _comparison._calibrate_nominal_makespan(
                instance, settings
            )
            print(
                f"[Propagated-NL] {size} {instance.instance_name} | "
                f"Cmax*={calibration['makespan']:.3f}",
                flush=True,
            )
            for offset in offsets:
                due_date = _comparison.controlled_due_date(
                    calibration["makespan"], offset
                )
                instance.due_dates = {
                    job: due_date for job in instance.jobs
                }
                simulation_seed = (
                    int(config["evaluation"]["random_seed"])
                    + 100_000 * num_jobs
                    + 10_000 * num_machines
                    + 100 * instance_number
                    + int(round(1000 * offset))
                )
                results = {
                    formulation: _solve(
                        instance,
                        formulation,
                        settings,
                        graph_config,
                        simulation_config,
                        int(replications),
                        simulation_seed,
                    )
                    for formulation in ("base", "nonlinear", "propagated")
                }
                base_assignment = results["base"].get("selected_machines")
                for formulation, result in results.items():
                    assignment = result.pop("selected_machines", None)
                    assignment_changes = profile_changes = None
                    if base_assignment is not None and assignment is not None:
                        assignment_changes = sum(
                            assignment[operation]
                            != base_assignment[operation]
                            for operation in instance.real_operations
                        )
                        profile_changes = sum(
                            instance.machine_profiles[assignment[operation]]
                            != instance.machine_profiles[
                                base_assignment[operation]
                            ]
                            for operation in instance.real_operations
                        )
                    rows.append({
                        "size": size,
                        "instance_name": instance.instance_name,
                        "random_seed": seed,
                        "operations": len(instance.real_operations),
                        "relative_due_date_offset": offset,
                        "common_due_date": due_date,
                        "calibrated_nominal_makespan": calibration["makespan"],
                        "simulation_replications": int(replications),
                        "assignment_changes_from_base": assignment_changes,
                        "profile_changes_from_base": profile_changes,
                        **result,
                    })
                print(
                    f"  offset={offset:.3f} due={due_date:.1f} | "
                    + " | ".join(
                        f"{name}={result['status']}"
                        + (
                            f", alpha={result['minimum_mc_ontime_probability']:.3f}"
                            if result.get("minimum_mc_ontime_probability")
                            is not None else ""
                        )
                        for name, result in results.items()
                    ),
                    flush=True,
                )

    summary_rows = []
    for size in sorted({row["size"] for row in rows}):
        for offset in sorted({
            float(row["relative_due_date_offset"])
            for row in rows if row["size"] == size
        }):
            for formulation in ("base", "nonlinear", "propagated"):
                selected = [
                    row for row in rows
                    if row["size"] == size
                    and float(row["relative_due_date_offset"]) == offset
                    and row["formulation"] == formulation
                ]
                solved = [
                    row for row in selected if int(row["solution_count"]) > 0
                ]
                mean = lambda field: (
                    statistics.fmean(float(row[field]) for row in solved)
                    if solved else None
                )
                summary_rows.append({
                    "size": size,
                    "relative_due_date_offset": offset,
                    "formulation": formulation,
                    "instances": len(selected),
                    "solved_instances": len(solved),
                    "mean_objective": mean("objective"),
                    "mean_minimum_mc_ontime_probability": mean(
                        "minimum_mc_ontime_probability"
                    ),
                    "mean_machine_assignment_changes": mean(
                        "assignment_changes_from_base"
                    ),
                    "mean_profile_changes": mean(
                        "profile_changes_from_base"
                    ),
                    "mean_maximum_applied_buffer": mean(
                        "maximum_applied_buffer"
                    ),
                    "mean_runtime_seconds": mean("runtime_seconds"),
                })
    output_directory = Path(output_directory)
    raw_path = output_directory / "propagated_nonlinear_comparison.csv"
    summary_path = output_directory / "propagated_nonlinear_summary.csv"
    metadata_path = output_directory / "propagated_nonlinear_analysis.json"
    _write_csv(raw_path, rows)
    _write_csv(summary_path, summary_rows)
    payload = {
        "status": "complete",
        "experimental_only": True,
        "formulations": ["base", "nonlinear", "propagated"],
        "sizes": list(sizes),
        "instances_per_size": int(instances_per_size),
        "offsets": offsets,
        "replications": int(replications),
        "wall_seconds": time.perf_counter() - started,
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload, rows, summary_rows


def _arguments():
    parser = argparse.ArgumentParser(
        description="Compare scaled and propagated nonlinear formulations."
    )
    parser.add_argument("--sizes", nargs="+", default=["3x3"])
    parser.add_argument("--instances-per-size", type=int, default=1)
    parser.add_argument("--offsets", nargs="+", type=float)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    run_analysis(
        sizes=arguments.sizes,
        instances_per_size=arguments.instances_per_size,
        offsets=arguments.offsets,
        replications=arguments.replications,
        time_limit=arguments.time_limit,
        mip_gap=arguments.mip_gap,
        output_directory=arguments.output_directory,
    )
