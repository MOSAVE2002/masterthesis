"""Generate GNN graphs from fix-and-optimize scheduling neighborhoods.

For every configured instance, the module solves reproducible restricted FJSP
neighborhoods, extracts distinct assignment and sequence graphs, computes
deterministic job-level repair-buffer labels and selects a diverse subset for
the train, validation and test CSV files. A generation summary records progress
and failures so interrupted or incomplete datasets remain identifiable.
"""

import copy
import csv
import hashlib
import importlib
import json
import math
import random
from pathlib import Path

import gurobipy as gp

from helper.sequence_setup import (
    RELIABILITY_GNN_GRAPH_SCHEMA,
    RELIABILITY_SERVICE_SCOPE,
    add_reliability_graph_variables,
    reliability_graph_config_dict,
    reliability_node_feature_names,
    local_buffer_node_features,
)
from helper.due_date_calibration import (
    calibrated_total_work_content_due_dates,
)
from helper.stochastic_fjsp import (
    ensure_profile_parameters,
    stochastic_parameters,
)
from helper.training_data_quality import (
    utc_timestamp,
    write_generation_summary,
)
ROOT_DIR = Path(__file__).resolve().parents[2]
_instances = importlib.import_module("01_generator.instance_generator")
_base = importlib.import_module("03_Gurobi.build_fjsp")
_simulation = importlib.import_module("05_Simulation.simulation")
FixedSchedule = _simulation.FixedSchedule
LOCAL_BUFFER_LABEL_METHOD = _simulation.LABEL_METHOD
LABEL_SOURCE = _simulation.LABEL_SOURCE
expected_job_buffers = _simulation.expected_job_buffers
label_config_dict = _simulation.label_config_dict
InvalidScheduleError = _simulation.InvalidScheduleError
TARGET_COLUMN = _simulation.TARGET_COLUMN
from helper.buffer_candidate_selection import (
    input_signature, select_buffer_candidates,
)
SPLIT_DIRECTORIES = _instances.SPLIT_DIRECTORIES
SPLIT_CSV_FILENAMES = _instances.SPLIT_CSV_FILENAMES
load_generated_instance = _instances.load_generated_instance

FIELDNAMES = [
    "instance_name",
    "job_ids",
    TARGET_COLUMN,
    "operation_job_indices",
    "gnn_feature_names",
    "gnn_node_features",
    "gnn_active_edges",
    "reliability_graph_parameters",
]


def _compact(value):
    """Serialize a structured value without unnecessary CSV whitespace.

    Args:
        value: JSON-compatible object stored in one dataset cell.

    Returns:
        Compact JSON text preserving the complete nested value.
    """
    return json.dumps(value, separators=(",", ":"))


def _sample_seed(instance_name, run_index, base_seed, purpose="schedule"):
    """Derive a stable independent seed for one sampling context.

    The SHA-256 digest prevents results from depending on Python's randomized
    hash implementation while separating instances, runs and random purposes.

    Returns:
        An unsigned 64-bit integer derived from the supplied identifiers.
    """
    source = f"{instance_name}:{run_index}:{base_seed}:{purpose}"
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _validate_ratio(value, name):
    """Convert a configured ratio and require a value in the unit interval.

    Args:
        value: Numeric ratio to normalize.
        name: Configuration name used in error messages.

    Returns:
        The ratio as ``float``.
    """
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return value


def _sample_reliability(instance, generation, run_index):
    """Apply a reproducibly rotated Weibull scale factor to an instance copy.

    Args:
        instance: Mutable copied instance used only for one candidate run.
        generation: Data-generation settings containing scale factors and seed.
        run_index: Zero-based neighborhood-run index.

    Side Effects:
        Replaces every machine's Weibull scale on ``instance``.
    """
    ensure_profile_parameters(instance)
    scale_factors = [
        float(value) for value in generation["weibull_scale_factors"]
    ]
    if not scale_factors or any(value <= 0.0 for value in scale_factors):
        raise ValueError("weibull_scale_factors must be positive.")
    rotation = _sample_seed(
        instance.instance_name,
        0,
        generation["random_seed"],
        "weibull_scale_rotation",
    ) % len(scale_factors)
    scale_factor = scale_factors[
        (int(run_index) + rotation) % len(scale_factors)
    ]
    instance.weibull_alpha = {
        machine: scale_factor * float(value)
        for machine, value in instance.weibull_alpha.items()
    }


def _sample_training_due_dates(instance, generation, run_index):
    """Calibrate one reproducible due-date scenario for a candidate run.

    A rotated relative makespan offset is selected and passed to the shared
    total-work-content calibration. The resulting due dates and nominal
    reference assignment are stored on the temporary instance copy.
    """
    adaptive = dict(generation["adaptive_due_dates"])
    offsets = [
        float(value) for value in adaptive["relative_makespan_offsets"]
    ]
    if not offsets or any(
        not math.isfinite(value) or value <= -1.0 for value in offsets
    ):
        raise ValueError(
            "adaptive_due_dates.relative_makespan_offsets must contain "
            "finite values greater than -1."
        )
    rotation = _sample_seed(
        instance.instance_name,
        0,
        generation["random_seed"],
        "adaptive_due_date_rotation",
    ) % len(offsets)
    offset_index = (
        int(run_index)
        + int(rotation)
        + int(run_index) // len(offsets)
    ) % len(offsets)
    offset = offsets[offset_index]
    calibrated = calibrated_total_work_content_due_dates(
        instance, adaptive, offset
    )
    calibration = calibrated["calibration"]
    instance.due_dates = dict(calibrated["due_dates"])
    instance.nominal_calibration_assignment = dict(
        calibration["assignment"]
    )


def _fix_neighborhood(
    model,
    variables,
    instance,
    rng,
    fix_ratio,
    reference_assignment,
):
    """Restrict a neighborhood around a reference machine assignment.

    A seeded sample of flexible operations is fixed by setting both bounds of
    every corresponding assignment variable. Nonselected operations remain
    free, allowing the solver to explore a controlled local neighborhood.

    Args:
        model: Candidate-pool Gurobi model.
        variables: Shared model-variable dictionary.
        instance: Temporary sampled FJSP instance.
        rng: Seeded random-number generator.
        fix_ratio: Share of flexible operations to fix.
        reference_assignment: Machine selected for every operation.
    """
    assignment = {
        operation: reference_assignment[operation]
        for operation in instance.real_operations
    }
    flexible = [
        operation for operation in instance.real_operations
        if len(instance.eligible_machines[operation]) > 1
    ]
    fixed_count = min(
        len(flexible),
        max(1, round(len(flexible) * fix_ratio)) if fix_ratio else 0,
    )
    fixed_operations = set(
        rng.sample(flexible, fixed_count) if fixed_count else []
    )
    for operation in fixed_operations:
        selected = assignment[operation]
        for machine in instance.eligible_machines[operation]:
            value = 1.0 if machine == selected else 0.0
            variables["Y"][operation, machine].lb = value
            variables["Y"][operation, machine].ub = value

    model.update()


def _build_candidate_model(instance):
    """Build the candidate MILP with direct graph edges and midpoints.

    Args:
        instance: Sampled FJSP instance used for this neighborhood.

    Returns:
        The Gurobi model and its variable dictionary, extended with stochastic
        parameters, direct-predecessor graph variables and operation midpoints.
    """
    model = gp.Model("midpoint_pd_fix_and_optimize")
    model, variables = _base.build_fjsp(
        model,
        instance,
    )
    parameters = stochastic_parameters(instance)
    variables.update({
        "weibull_alpha": parameters["alpha"],
        "weibull_beta": parameters["beta"],
        "repair_rate": parameters["repair_rate"],
    })
    add_reliability_graph_variables(model, variables, instance)
    horizon = float(variables["H"])
    midpoints = {}
    for operation in variables["real_operations"]:
        duration = gp.quicksum(
            float(instance.processing_times[operation, machine])
            * variables["Y"][operation, machine]
            for machine in instance.eligible_machines[operation]
        )
        midpoint = model.addVar(
            lb=0.0, ub=horizon, name=f"t_midpoint[{operation}]"
        )
        model.addConstr(
            midpoint == variables["C"][operation] - 0.5 * duration
        )
        midpoints[operation] = midpoint
    variables["T"] = midpoints
    return model, variables


def _solution_value(variable):
    """Return one variable value from Gurobi's active solution-pool entry.

    Args:
        variable: Gurobi variable whose ``Xn`` attribute is available.

    Returns:
        The pool value converted to ``float``.
    """
    return float(variable.Xn)


def _selected_machine(variables, operation):
    """Identify the machine with the largest assignment value for an operation.

    Args:
        variables: Model-variable dictionary containing eligible machines and
            binary assignment variables.
        operation: Operation identifier to inspect.

    Returns:
        The selected zero-based machine identifier.
    """
    return max(
        variables["eligible_machines"][operation],
        key=lambda machine: _solution_value(variables["Y"][operation, machine]),
    )


def _candidate_from_solution(
    variables,
    instance,
):
    """Convert one Gurobi pool solution into a labelled-graph candidate.

    The extraction reconstructs machine assignments, direct machine edges,
    planned operation times, normalized node features and a fixed schedule for
    deterministic label calculation. It also computes the nominal economic
    cost used during diversity selection.

    Returns:
        Candidate dictionary containing its structure, fixed schedule, nominal
        cost and partially populated CSV row.
    """
    operations = list(variables["real_operations"])
    selected = {
        operation: _selected_machine(variables, operation)
        for operation in operations
    }
    job_ids = sorted(instance.jobs)
    job_to_index = {job: index for index, job in enumerate(job_ids)}
    operation_job = {
        operation: job
        for job, job_operations in instance.jobs.items()
        for operation in job_operations
    }
    operation_to_index = {
        operation: index for index, operation in enumerate(operations)
    }
    active_edges = [
        edge for edge in variables["U_index"]
        if _solution_value(variables["U"][edge]) > 0.5
    ]
    processing_times = {
        operation: float(instance.processing_times[
            operation, selected[operation]
        ])
        for operation in operations
    }
    completions = {
        operation: _solution_value(variables["C"][operation])
        for operation in operations
    }
    planned_starts = {
        operation: max(
            0.0,
            completions[operation] - processing_times[operation],
        )
        for operation in operations
    }
    node_features = []
    for operation in operations:
        machine = selected[operation]
        completion = completions[operation]
        start = planned_starts[operation]
        alpha = variables["weibull_alpha"][machine]
        beta = variables["weibull_beta"][machine]
        repair_rate = variables["repair_rate"][machine]
        node_features.append(local_buffer_node_features(
            0.5 * (start + completion), alpha, beta, repair_rate
        ))

    graph_edges = [
        (operation_to_index[source], operation_to_index[target])
        for source, target, _machine in active_edges
    ]
    for target in operations:
        for source in instance.predecessors.get(target, []):
            if source not in operation_to_index:
                continue
            graph_edges.append((
                operation_to_index[source], operation_to_index[target]
            ))

    structure = (
        tuple(sorted(selected.items())),
        tuple(sorted(active_edges)),
    )
    simulation_schedule = FixedSchedule(
        operations=tuple(operations),
        selected_machines=dict(selected),
        processing_times=processing_times,
        planned_starts=planned_starts,
        job_predecessors={
            operation: tuple(
                predecessor
                for predecessor in instance.predecessors.get(operation, [])
                if predecessor in operation_to_index
            )
            for operation in operations
        },
        machine_edges=tuple(active_edges),
        jobs={
            job: tuple(instance.jobs[job])
            for job in job_ids
        },
        job_end_operations=dict(instance.job_end_operations),
        due_dates={job: float(instance.due_dates[job]) for job in job_ids},
        weibull_scale=dict(variables["weibull_alpha"]),
        weibull_shape=dict(variables["weibull_beta"]),
        repair_rate=dict(variables["repair_rate"]),
    )
    nominal_ends = {
        j: planned_starts[instance.job_end_operations[j]]
        + processing_times[instance.job_end_operations[j]] for j in job_ids
    }
    nominal_schedule_cost = (
        sum(
            float(instance.machine_cost[selected[operation]])
            * processing_times[operation]
            for operation in operations
        )
        + max(planned_starts[o] + processing_times[o] for o in operations)
        + sum(
            max(0.0, nominal_ends[job] - instance.due_dates[job])
            for job in job_ids
        )
    )
    return {
        "nominal_schedule_cost": nominal_schedule_cost,
        "structure": structure,
        "simulation_schedule": simulation_schedule,
        "row": {
            "instance_name": "",
            "job_ids": _compact(job_ids),
            TARGET_COLUMN: "",
            "operation_job_indices": _compact([
                job_to_index[operation_job[operation]]
                for operation in operations
            ]),
            "gnn_feature_names": _compact(reliability_node_feature_names()),
            "gnn_node_features": _compact(node_features),
            "gnn_active_edges": _compact(graph_edges),
            "reliability_graph_parameters": _compact(
                reliability_graph_config_dict()
            ),
        },
    }


def _compute_deterministic_local_buffer_labels(candidate):
    """Attach deterministic job repair-buffer labels to one candidate.

    The temporary fixed schedule is removed from the candidate, evaluated by
    the checked quadrature routine and serialized into the configured target
    column. The numeric vector remains available for candidate selection.
    """
    schedule = candidate.pop("simulation_schedule")
    buffers, _ = expected_job_buffers(schedule)
    jobs = sorted(schedule.jobs)
    values = [buffers[j] for j in jobs]
    candidate["local_job_repair_buffers"] = values
    candidate["row"][TARGET_COLUMN] = _compact(values)


def _configured_fix_ratios(fixed):
    """Validate configured neighborhood-fixing ratios.

    Args:
        fixed: Fixed-neighborhood configuration containing ``fix_ratios``.

    Returns:
        A nonempty list of unique ratios in ``[0, 1]``.
    """
    fix_ratios = [
        _validate_ratio(value, "fix_ratio") for value in fixed["fix_ratios"]
    ]
    if not fix_ratios:
        raise ValueError("fixed_y.fix_ratios must not be empty.")
    if len(fix_ratios) != len(set(fix_ratios)):
        raise ValueError("fixed_y.fix_ratios must not contain duplicates.")
    return fix_ratios


def _run_neighborhood(
    instance,
    instance_name,
    run_index,
    generation,
    fix_ratio,
):
    """Solve and label the distinct graphs from one local neighborhood.

    Reliability parameters and due dates are sampled on a copy of the instance,
    assignments are partially fixed and Gurobi's solution pool is searched.
    Duplicate graph inputs and invalid schedules are discarded.

    Returns:
        List of unique feasible candidate dictionaries produced by the run.
    """
    rng = random.Random(_sample_seed(
        instance_name, run_index, generation["random_seed"]
    ))
    sample_instance = copy.deepcopy(instance)
    _sample_reliability(sample_instance, generation, run_index)
    _sample_training_due_dates(sample_instance, generation, run_index)
    fixed = generation["fixed_y"]
    model, variables = _build_candidate_model(sample_instance)
    model.Params.TimeLimit = float(fixed["time_limit_seconds"])
    model.Params.MIPGap = float(fixed["mip_gap"])
    model.Params.Seed = int(_sample_seed(
        instance_name,
        run_index,
        generation["random_seed"],
        "gurobi",
    ) % 2_000_000_000)
    _fix_neighborhood(
        model,
        variables,
        sample_instance,
        rng,
        fix_ratio,
        sample_instance.nominal_calibration_assignment,
    )
    model.Params.PoolSolutions = int(fixed["pool_candidates"])
    for variable in model.getVars():
        variable.PoolIgnore = 1
    for variable in variables["Y"].values():
        variable.PoolIgnore = 0
    for variable in variables["U"].values():
        variable.PoolIgnore = 0
    model.optimize()
    if model.SolCount == 0:
        model.dispose()
        return []

    candidates = []
    local_structures = set()
    for solution_number in range(model.SolCount):
        model.Params.SolutionNumber = solution_number
        candidate = _candidate_from_solution(
            variables,
            sample_instance,
        )
        signature = input_signature(candidate)
        if signature in local_structures:
            continue
        candidate["row"]["instance_name"] = instance_name
        try:
            _compute_deterministic_local_buffer_labels(candidate)
        except InvalidScheduleError as error:
            print(f"[Invalid candidate] {instance_name}: {error}", flush=True)
            continue
        local_structures.add(signature)
        candidates.append(candidate)
    print(
        f"[Local buffer labels] {instance_name} | run={run_index + 1} | "
        f"candidates={len(candidates)}",
        flush=True,
    )
    model.dispose()
    return candidates


def _collect_instance_candidates(
    instance,
    instance_name,
    generation,
    *,
    progress=None,
):
    """Collect the pooled candidates of all runs for one instance.

    Args:
        instance: Base generated FJSP instance.
        instance_name: Stable name written to every graph row.
        generation: Complete data-generation configuration.
        progress: Optional callback receiving run index, ratio and current
            candidate count.

    Returns:
        Combined candidate list before global diversity selection.
    """
    fixed = generation["fixed_y"]
    fix_ratios = _configured_fix_ratios(fixed)
    pool_candidates = int(fixed["pool_candidates"])
    if pool_candidates <= 0:
        raise ValueError("fixed_y.pool_candidates must be positive.")
    candidate_pool_runs = int(fixed["candidate_pool_runs"])
    if candidate_pool_runs < len(fix_ratios):
        raise ValueError(
            "fixed_y.candidate_pool_runs must be at least the number of "
            "fix ratios."
        )
    candidates = []
    for run_index in range(candidate_pool_runs):
        fix_ratio = fix_ratios[run_index % len(fix_ratios)]
        candidates.extend(_run_neighborhood(
            instance,
            instance_name,
            run_index,
            generation,
            fix_ratio,
        ))
        if progress is not None:
            progress(
                run_index,
                fix_ratio,
                len(candidates),
            )
    return candidates


def _collect_and_select_instance_candidates(
    instance,
    instance_name,
    generation,
    count,
    *,
    progress=None,
):
    """Generate a candidate pool and select a diverse fixed-size subset.

    Returns:
        A pair containing all collected candidates and the categorized selected
        entries returned by the shared selection helper.
    """
    candidates = _collect_instance_candidates(
        instance,
        instance_name,
        generation,
        progress=progress,
    )
    return candidates, select_buffer_candidates(candidates, count)


def generate_from_config(generation):
    """Generate all configured graph splits and their progress metadata.

    Args:
        generation: Resolved generation configuration including instance names,
            solver neighborhoods, stochastic sampling and output directory.

    Side Effects:
        Replaces split CSV files, writes ``generation_summary.json`` after each
        material step and logs skipped instances without aborting other splits.

    Raises:
        Exception: Fatal setup or output errors are recorded in the summary and
            re-raised; per-instance failures are recorded and skipped.
    """
    generation = dict(generation)
    samples_per_instance = int(generation["samples_per_instance"])
    if samples_per_instance <= 0:
        raise ValueError("samples_per_instance must be positive.")
    fixed = dict(generation["fixed_y"])
    _configured_fix_ratios(fixed)
    generation["fixed_y"] = fixed
    generation["random_seed"] = int(generation["random_seed"])
    output_root = Path(generation["output_directory"])
    if not output_root.is_absolute():
        output_root = ROOT_DIR / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    splits = generation["instance_splits"]
    summary_path = output_root / "generation_summary.json"
    summary = {
        "schema_version": 16,
        "status": "running",
        "started_at_utc": utc_timestamp(),
        "completed_at_utc": None,
        "output_directory": str(output_root),
        "samples_per_instance": samples_per_instance,
        "adaptive_due_dates": dict(generation["adaptive_due_dates"]),
        "training_weibull_scale_factors": generation["weibull_scale_factors"],
        "machine_profile_config": generation["machine_profile_config"],
        "training_parameter_jitter": generation["training_parameter_jitter"],
        "label": {
            "target_column": TARGET_COLUMN,
            "label_method": LOCAL_BUFFER_LABEL_METHOD,
            "source": LABEL_SOURCE,
            "parameters": label_config_dict(),
            "semantics": "Expected sum of residual repairs at nominal operation midpoints; no propagation or gap absorption.",
        },
        "candidate_selection": {
            "method": "buffer_structure_cost",
            "deduplication": "features_rounded7_job_membership_directed_multiedges",
            "nominal_cost": "processing_cost_plus_nominal_makespan_plus_nominal_tardiness",
        },
        "graph": {
            "graph_schema": RELIABILITY_GNN_GRAPH_SCHEMA,
            "feature_names": reliability_node_feature_names(),
            "service_scope": RELIABILITY_SERVICE_SCOPE,
            "include_machine_predecessor_edges": True,
            "machine_predecessor_edge_scope": "direct",
            "include_job_precedence_edges": True,
        },
        "splits": {
            split: {
                "configured_count": len(instance_names),
                "configured_instances": list(instance_names),
                "successful_instances": [],
                "skipped_instances": [],
                "written_graphs": 0,
            }
            for split, instance_names in splits.items()
        },
    }
    write_generation_summary(summary_path, summary)

    try:
        for split, instance_names in splits.items():
            output_path = (
                output_root / SPLIT_DIRECTORIES[split]
                / SPLIT_CSV_FILENAMES[split]
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            split_summary = summary["splits"][split]
            with output_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                file.flush()
                for instance_name in instance_names:
                    def report(run_index, fix_ratio, candidate_count):
                        """Print neighborhood progress for the current instance.

                        The callback exposes run number, fixing ratio and the
                        cumulative candidate count during long generation jobs.
                        """
                        print(
                            f"[Fix-and-optimize] {split} | "
                            f"{instance_name} | run={run_index + 1} | "
                            f"graphs={candidate_count} candidates | "
                            f"fix_ratio={fix_ratio}",
                            flush=True,
                        )

                    try:
                        instance = load_generated_instance(instance_name)
                        candidates, selected = (
                            _collect_and_select_instance_candidates(
                                instance,
                                instance_name,
                                generation,
                                samples_per_instance,
                                progress=report,
                            )
                        )
                        rows = []
                        for entry in selected:
                            candidate = entry["candidate"]
                            rows.append(candidate["row"])
                    except Exception as error:
                        split_summary["skipped_instances"].append({
                            "instance_name": instance_name,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        })
                        write_generation_summary(summary_path, summary)
                        print(
                            f"[Fix-and-optimize:skip] {split} | "
                            f"{instance_name} | {type(error).__name__}: "
                            f"{error}",
                            flush=True,
                        )
                        continue

                    writer.writerows(rows)
                    file.flush()
                    split_summary["written_graphs"] += len(rows)
                    split_summary["successful_instances"].append({
                        "instance_name": instance_name,
                        "training_parameter_jitter_applied": bool(
                            instance.training_parameter_jitter_applied
                        ),
                        "candidate_count": len(candidates),
                        "written_graphs": len(rows),
                    })
                    write_generation_summary(summary_path, summary)
                    print(
                        f"[Fix-and-optimize] {split} | {instance_name} | "
                        f"selected={len(rows)}/{len(candidates)} candidates",
                        flush=True,
                    )

            print(
                f"[Fix-and-optimize] finished {split}: "
                f"{split_summary['written_graphs']} graphs, "
                f"{len(split_summary['skipped_instances'])} skipped -> "
                f"{output_path}",
                flush=True,
            )
            write_generation_summary(summary_path, summary)
        summary["status"] = "completed"
        summary["completed_at_utc"] = utc_timestamp()
        write_generation_summary(summary_path, summary)
        print(
            f"[Fix-and-optimize] generation summary -> {summary_path}",
            flush=True,
        )
    except Exception as error:
        summary["status"] = "failed"
        summary["completed_at_utc"] = utc_timestamp()
        summary["fatal_error"] = {
            "error_type": type(error).__name__,
            "message": str(error),
        }
        write_generation_summary(summary_path, summary)
        raise
