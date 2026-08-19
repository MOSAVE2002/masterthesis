"""Fix-and-optimize graphs with Monte-Carlo job probabilities per row."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import math
import random
from collections import Counter, deque
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from helper.sequence_setup import (
    add_reliability_graph_variables,
    normalize_reliability_graph_config,
    reliability_graph_config_dict,
    reliability_node_feature_names,
)
from helper.stochastic_fjsp import (
    ensure_stochastic_parameters,
    stochastic_parameters,
    weibull_down_probability,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
_instances = importlib.import_module("01_generator.instance_generator")
_base = importlib.import_module("03_Gurobi.build_fjsp")
_nonlinear = importlib.import_module("03_Gurobi.build_fjsp_with_nonlinear")
_simulation = importlib.import_module("05_Simulation.preempt_resume")
FixedSchedule = _simulation.FixedSchedule
normalize_simulation_config = _simulation.normalize_simulation_config
simulate_fixed_schedule = _simulation.simulate_fixed_schedule
simulation_config_dict = _simulation.simulation_config_dict

SPLIT_DIRECTORIES = _instances.SPLIT_DIRECTORIES
SPLIT_CSV_FILENAMES = _instances.SPLIT_CSV_FILENAMES
load_generated_instance = _instances.load_generated_instance
selected_instance_splits = _instances.selected_instance_splits

FIELDNAMES = [
    "instance_name",
    "candidate_generation_mode",
    "schedule_timing_method",
    "candidate_probability_band",
    "job_ids",
    "nonlinear_job_ontime_probability_lbs",
    "nonlinear_min_job_ontime_probability_lb",
    "job_ontime_probabilities",
    "job_probability_standard_errors",
    "job_mean_completion_times",
    "job_probability_label_method",
    "operation_job_indices",
    "total_failure_delay",
    "gnn_feature_names",
    "gnn_node_features",
    "gnn_active_edges",
    "reliability_graph_parameters",
    "fix_ratio",
    "sequence_fix_ratio",
    "fixed_operations",
    "fixed_predecessor_edges",
    "optimization_run",
    "pool_solution_number",
    "pool_objective",
    "effective_objective",
    "operation_failure_delays",
    "operation_failure_probabilities",
    "operation_repair_durations",
    "simulation_replications",
    "simulation_mean_failures",
    "simulation_parameters",
    "pool_selection_category",
    "solver_runtime_seconds",
]

DEFAULT_SELECTION_RATIOS = {
    "service_boundary_below": 0.25,
    "service_boundary_above": 0.15,
    "high_min_job_probability": 0.15,
    "low_min_job_probability": 0.10,
    "good_nominal_objective": 0.15,
    "pareto_tradeoff": 0.10,
    "structurally_diverse": 0.10,
}

PROBABILITY_BINS = (
    "low",
    "boundary_below",
    "boundary_above",
    "high",
)
ANCHOR_CATEGORIES = (
    "good_nominal_objective",
    "high_min_job_probability",
    "pareto_tradeoff",
    "structurally_diverse",
)
DEFAULT_HYBRID_ANCHOR_RATIOS = {
    "good_nominal_objective": 0.30,
    "high_min_job_probability": 0.20,
    "pareto_tradeoff": 0.20,
    "structurally_diverse": 0.30,
}
DEFAULT_JOB_PROBABILITY_TARGET_RATIOS = {
    "low": 0.20,
    "boundary_below": 0.30,
    "boundary_above": 0.30,
    "high": 0.20,
}

CANDIDATE_GENERATION_MODES = (
    "unconstrained",
    "nominal_ontime",
    "nonlinear_evaluated",
    "nonlinear",
)
DEFAULT_NOMINAL_RUN_INDEX_OFFSET = 1_000_000


def _compact(value):
    return json.dumps(value, separators=(",", ":"))


def _sample_seed(instance_name, run_index, base_seed, purpose="schedule"):
    source = f"{instance_name}:{run_index}:{base_seed}:{purpose}"
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _validate_ratio(value, name):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return value


def _validate_candidate_generation_mode(value):
    value = str(value).strip().lower()
    if value not in CANDIDATE_GENERATION_MODES:
        raise ValueError(
            "candidate_generation_mode must be one of "
            f"{CANDIDATE_GENERATION_MODES}."
        )
    return value


def _probability_band_limits(name, service_level, boundary_width):
    name = str(name)
    service_level = float(service_level)
    boundary_width = float(boundary_width)
    lower = max(0.0, service_level - boundary_width)
    upper = min(1.0, service_level + boundary_width)
    bands = {
        "low": (0.0, lower),
        "boundary_below": (lower, service_level),
        "boundary_above": (service_level, upper),
        "high": (upper, 1.0),
    }
    if name not in bands:
        raise ValueError(
            f"Unknown candidate probability band {name!r}; "
            f"expected one of {PROBABILITY_BINS}."
        )
    return bands[name]


def _sample_range(rng, raw, defaults):
    if raw is None:
        return dict(defaults)
    low, high = map(float, raw)
    if low > high:
        raise ValueError("Reliability range lower bound exceeds upper bound.")
    return {
        machine: rng.uniform(low, high)
        for machine in defaults
    }


def _sample_reliability(instance, rng, generation):
    ensure_stochastic_parameters(instance)
    instance.weibull_alpha = _sample_range(
        rng, generation.get("alpha_range"), instance.weibull_alpha
    )
    instance.weibull_beta = _sample_range(
        rng, generation.get("beta_range"), instance.weibull_beta
    )
    instance.repair_rate = _sample_range(
        rng, generation.get("repair_rate_range"), instance.repair_rate
    )
    instance.repair_duration = {
        machine: 1.0 / rate
        for machine, rate in instance.repair_rate.items()
    }


def _random_topological_order(instance, rng):
    operations = list(instance.real_operations)
    predecessors = {
        operation: set(instance.predecessors.get(operation, []))
        for operation in operations
    }
    successors = {operation: [] for operation in operations}
    for operation, required in predecessors.items():
        for predecessor in required:
            successors[predecessor].append(operation)
    available = [
        operation for operation in operations if not predecessors[operation]
    ]
    result = []
    while available:
        operation = available.pop(rng.randrange(len(available)))
        result.append(operation)
        for successor in successors[operation]:
            predecessors[successor].discard(operation)
            if not predecessors[successor]:
                available.append(successor)
    if len(result) != len(operations):
        raise ValueError("Job precedence graph contains a cycle.")
    return result


def _random_assignment_and_edges(instance, rng):
    assignment = {
        operation: rng.choice(instance.eligible_machines[operation])
        for operation in instance.real_operations
    }
    by_machine = {
        machine: [] for machine in range(instance.num_machines)
    }
    for operation in _random_topological_order(instance, rng):
        by_machine[assignment[operation]].append(operation)
    edges = [
        (source, target, machine)
        for machine, operations in by_machine.items()
        for source, target in zip(operations, operations[1:])
    ]
    return assignment, edges


def _fix_neighborhood(
    model,
    variables,
    instance,
    rng,
    fix_ratio,
    sequence_fix_ratio,
    *,
    use_incumbent=False,
):
    if use_incumbent:
        assignment = {
            operation: max(
                instance.eligible_machines[operation],
                key=lambda machine: float(
                    variables["Y"][operation, machine].X
                ),
            )
            for operation in instance.real_operations
        }
        direct_edges = [
            edge for edge in variables["U_index"]
            if float(variables["U"][edge].X) > 0.5
        ]
    else:
        assignment, direct_edges = _random_assignment_and_edges(instance, rng)
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

    candidate_edges = [
        edge for edge in direct_edges if edge in variables["U"]
    ]
    edge_count = min(
        len(candidate_edges),
        max(1, round(len(candidate_edges) * sequence_fix_ratio))
        if sequence_fix_ratio else 0,
    )
    fixed_edges = rng.sample(candidate_edges, edge_count) if edge_count else []
    for edge in fixed_edges:
        if use_incumbent:
            source, target, machine = edge
            order_index = (
                (source, target, machine)
                if source < target
                else (target, source, machine)
            )
            order_value = 1.0 if source < target else 0.0
            variables["X"][order_index].lb = order_value
            variables["X"][order_index].ub = order_value
        else:
            variables["U"][edge].lb = 1.0
            variables["U"][edge].ub = 1.0
    model.update()
    return {
        "fix_ratio": len(fixed_operations) / len(flexible) if flexible else 0.0,
        "sequence_fix_ratio": (
            len(fixed_edges) / len(candidate_edges) if candidate_edges else 0.0
        ),
        "fixed_operations": len(fixed_operations),
        "fixed_predecessor_edges": len(fixed_edges),
    }


def _build_candidate_model(
    instance,
    graph_config,
    candidate_generation_mode="unconstrained",
    service_probability_band=None,
):
    candidate_generation_mode = _validate_candidate_generation_mode(
        candidate_generation_mode
    )
    if candidate_generation_mode == "nonlinear":
        model = gp.Model("nonlinear_fix_and_optimize")
        model, variables = _nonlinear.build_fjsp(
            model,
            instance,
            reliability_graph_config=graph_config,
            service_probability_band=service_probability_band,
        )
        add_reliability_graph_variables(
            model, variables, instance, graph_config
        )
        model.update()
        return model, variables

    model = gp.Model("midpoint_pd_fix_and_optimize")
    model, variables = _base.build_fjsp(
        model,
        instance,
        include_makespan=False,
    )
    parameters = stochastic_parameters(instance)
    variables.update({
        "weibull_alpha": parameters["alpha"],
        "weibull_beta": parameters["beta"],
        "repair_rate": parameters["repair_rate"],
        "machine_cost": parameters["cost"],
    })
    add_reliability_graph_variables(
        model, variables, instance, graph_config
    )
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
    assignment_cost = gp.quicksum(
        parameters["cost"][machine]
        * float(instance.processing_times[operation, machine])
        * variables["Y"][operation, machine]
        for operation, machine in variables["Y_index"]
    )
    model.setObjective(assignment_cost, GRB.MINIMIZE)
    model.update()
    return model, variables


def _add_nominal_ontime_constraints(model, variables, instance):
    """Require every job to meet its due date without stochastic failures."""
    for job, end_operation in instance.job_end_operations.items():
        model.addConstr(
            variables["C"][end_operation] <= float(instance.due_dates[job]),
            name=f"candidate_nominal_ontime[{job}]",
        )
    model.update()


def _solution_value(variable, solution_number):
    return float(variable.Xn if solution_number is not None else variable.X)


def _selected_machine(variables, operation, solution_number):
    return max(
        variables["eligible_machines"][operation],
        key=lambda machine: _solution_value(
            variables["Y"][operation, machine], solution_number
        ),
    )


def _earliest_start_times(
    instance,
    operations,
    processing_times,
    machine_edges,
):
    """Return the canonical left-shifted timing for a fixed graph."""
    operation_set = set(operations)
    predecessors = {
        operation: {
            predecessor
            for predecessor in instance.predecessors.get(operation, [])
            if predecessor in operation_set
        }
        for operation in operations
    }
    for source, target, _machine in machine_edges:
        if source not in operation_set or target not in operation_set:
            raise ValueError("Machine edge references an unknown operation.")
        predecessors[target].add(source)
    successors = {operation: set() for operation in operations}
    for target, required in predecessors.items():
        for source in required:
            successors[source].add(target)

    indegree = {
        operation: len(predecessors[operation]) for operation in operations
    }
    available = deque(
        operation for operation in operations if indegree[operation] == 0
    )
    starts, completions = {}, {}
    while available:
        operation = available.popleft()
        starts[operation] = max(
            (
                completions[predecessor]
                for predecessor in predecessors[operation]
            ),
            default=0.0,
        )
        completions[operation] = (
            starts[operation] + float(processing_times[operation])
        )
        for successor in successors[operation]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                available.append(successor)
    if len(starts) != len(operations):
        raise ValueError(
            "Fixed job and machine predecessor graph contains a cycle."
        )
    return starts, completions


def _candidate_from_solution(
    model,
    variables,
    instance,
    graph_config,
    solution_number,
    run_index,
    fix_stats,
    candidate_generation_mode="unconstrained",
    candidate_probability_band=None,
):
    candidate_generation_mode = _validate_candidate_generation_mode(
        candidate_generation_mode
    )
    operations = list(variables["real_operations"])
    selected = {
        operation: _selected_machine(variables, operation, solution_number)
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
        if _solution_value(variables["U"][edge], solution_number) > 0.5
    ]
    processing_times = {
        operation: float(instance.processing_times[
            operation, selected[operation]
        ])
        for operation in operations
    }
    if candidate_generation_mode == "nonlinear_evaluated":
        timing_method = "earliest_start"
        planned_starts, completions = _earliest_start_times(
            instance,
            operations,
            processing_times,
            active_edges,
        )
    else:
        timing_method = "gurobi_solution"
        completions = {
            operation: _solution_value(
                variables["C"][operation], solution_number
            )
            for operation in operations
        }
        planned_starts = {
            operation: max(
                0.0,
                completions[operation] - processing_times[operation],
            )
            for operation in operations
        }
    horizon = max(float(value) for value in instance.due_dates.values())
    node_features = []
    for operation in operations:
        machine = selected[operation]
        completion = completions[operation]
        processing_time = processing_times[operation]
        start = planned_starts[operation]
        alpha = variables["weibull_alpha"][machine]
        beta = variables["weibull_beta"][machine]
        repair_rate = variables["repair_rate"][machine]
        node_features.append([
            start / horizon,
            completion / horizon,
            processing_time / alpha,
            repair_rate * alpha / 30.0,
            beta / 5.0,
        ])

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
    nonlinear_probability_variables = variables.get(
        "job_ontime_probabilities", {}
    )
    nonlinear_probabilities = [
        _solution_value(
            nonlinear_probability_variables[job], solution_number
        )
        for job in job_ids
        if job in nonlinear_probability_variables
    ]
    nonlinear_min_probability = (
        min(nonlinear_probabilities) if nonlinear_probabilities else None
    )
    pool_objective = float(model.PoolObjVal)
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
    return {
        "candidate_id": (int(run_index), int(solution_number or 0)),
        "candidate_generation_mode": candidate_generation_mode,
        "candidate_probability_band": candidate_probability_band,
        "nonlinear_job_probabilities": nonlinear_probabilities,
        "nonlinear_min_job_probability": nonlinear_min_probability,
        "solution_number": solution_number,
        "structure": structure,
        "structure_tokens": frozenset(
            [("Y", *item) for item in structure[0]]
            + [("U", *edge) for edge in structure[1]]
        ),
        "pool_objective": pool_objective,
        "total_failure_delay": 0.0,
        "min_job_probability": 0.0,
        "service_risk": 1.0,
        "effective_objective": pool_objective,
        "simulation_schedule": simulation_schedule,
        "row": {
            "instance_name": "",
            "candidate_generation_mode": candidate_generation_mode,
            "schedule_timing_method": timing_method,
            "candidate_probability_band": (
                "" if candidate_probability_band is None
                else candidate_probability_band
            ),
            "job_ids": _compact(job_ids),
            "nonlinear_job_ontime_probability_lbs": (
                _compact(nonlinear_probabilities)
                if nonlinear_probabilities else ""
            ),
            "nonlinear_min_job_ontime_probability_lb": (
                "" if nonlinear_min_probability is None
                else nonlinear_min_probability
            ),
            "job_ontime_probabilities": "",
            "job_probability_standard_errors": "",
            "job_mean_completion_times": "",
            "job_probability_label_method": _simulation.LABEL_METHOD,
            "operation_job_indices": _compact([
                job_to_index[operation_job[operation]]
                for operation in operations
            ]),
            "total_failure_delay": 0.0,
            "gnn_feature_names": _compact(reliability_node_feature_names()),
            "gnn_node_features": _compact(node_features),
            "gnn_active_edges": _compact(graph_edges),
            "operation_failure_delays": "",
            "operation_failure_probabilities": "",
            "operation_repair_durations": "",
            "simulation_replications": 0,
            "simulation_mean_failures": 0.0,
            "simulation_parameters": "",
            "reliability_graph_parameters": _compact(
                reliability_graph_config_dict(graph_config)
            ),
            **fix_stats,
            "optimization_run": run_index,
            "pool_solution_number": solution_number,
            "pool_objective": pool_objective,
            "effective_objective": pool_objective,
            "pool_selection_category": "",
            "solver_runtime_seconds": float(model.Runtime),
        },
    }


def _simulation_seed(candidate, instance_name, simulation_config, purpose):
    run_index, solution_number = candidate["candidate_id"]
    return _sample_seed(
        instance_name,
        run_index,
        simulation_config.random_seed,
        f"simulation:{purpose}:solution={solution_number}",
    )


def _evaluate_fixed_schedule_nonlinear(candidate, graph_config):
    """Evaluate the reference nonlinear Markov bound on a fixed schedule."""
    schedule = candidate["simulation_schedule"]
    operation_probabilities = {}
    for operation in schedule.operations:
        machine = schedule.selected_machines[operation]
        midpoint = (
            float(schedule.planned_starts[operation])
            + 0.5 * float(schedule.processing_times[operation])
        )
        operation_probabilities[operation] = weibull_down_probability(
            midpoint,
            schedule.weibull_scale[machine],
            schedule.weibull_shape[machine],
            schedule.repair_rate[machine],
            order=graph_config.quadrature_points,
        )

    all_operations = tuple(schedule.operations)
    job_probabilities = []
    for job in sorted(schedule.jobs):
        service_operations = (
            tuple(schedule.jobs[job])
            if graph_config.service_scope == "job"
            else all_operations
        )
        expected_disruption = sum(
            operation_probabilities[operation]
            / float(schedule.repair_rate[
                schedule.selected_machines[operation]
            ])
            for operation in service_operations
        )
        end_operation = schedule.job_end_operations[job]
        nominal_completion = (
            float(schedule.planned_starts[end_operation])
            + float(schedule.processing_times[end_operation])
        )
        slack = float(schedule.due_dates[job]) - nominal_completion
        probability = (
            0.0
            if slack <= 0.0
            else max(0.0, min(1.0, 1.0 - expected_disruption / slack))
        )
        job_probabilities.append(probability)

    candidate["nonlinear_job_probabilities"] = job_probabilities
    candidate["nonlinear_min_job_probability"] = min(job_probabilities)
    candidate["row"].update({
        "nonlinear_job_ontime_probability_lbs": _compact(
            job_probabilities
        ),
        "nonlinear_min_job_ontime_probability_lb": min(
            job_probabilities
        ),
    })
    return candidate


def _apply_simulation_result(candidate, result, simulation_config):
    probabilities = list(result.job_ontime_probabilities)
    total_delay = float(result.mean_total_repair_delay)
    candidate.update({
        "job_probabilities": probabilities,
        "total_failure_delay": total_delay,
        "min_job_probability": min(probabilities),
        "service_risk": 1.0 - min(probabilities),
        "effective_objective": candidate["pool_objective"] + total_delay,
    })
    candidate["row"].update({
        "job_ontime_probabilities": _compact(probabilities),
        "job_probability_standard_errors": _compact(
            list(result.job_probability_standard_errors)
        ),
        "job_mean_completion_times": _compact(
            list(result.job_mean_completion_times)
        ),
        "job_probability_label_method": result.label_method,
        "total_failure_delay": total_delay,
        "operation_failure_delays": _compact(
            list(result.operation_mean_repair_delays)
        ),
        "operation_failure_probabilities": _compact(
            list(result.operation_failure_probabilities)
        ),
        "operation_repair_durations": _compact(
            list(result.operation_mean_repair_durations)
        ),
        "simulation_replications": int(result.replications),
        "simulation_mean_failures": float(result.mean_failures),
        "simulation_parameters": _compact(
            simulation_config_dict(simulation_config)
        ),
        "effective_objective": candidate["pool_objective"] + total_delay,
    })
    return candidate


def _simulate_candidate(
    candidate,
    instance_name,
    simulation_config,
    *,
    replications,
    purpose,
):
    result = simulate_fixed_schedule(
        candidate["simulation_schedule"],
        replications=int(replications),
        seed=_simulation_seed(
            candidate, instance_name, simulation_config, purpose
        ),
        config=simulation_config,
    )
    return _apply_simulation_result(candidate, result, simulation_config)


def _normalized(value, values):
    lower, upper = min(values), max(values)
    return 0.0 if upper - lower <= 1e-12 else (
        (float(value) - lower) / (upper - lower)
    )


def _pareto_ranks(candidates):
    remaining = set(range(len(candidates)))
    ranks = [0] * len(candidates)
    rank = 0
    while remaining:
        front = []
        for index in sorted(remaining):
            candidate = candidates[index]
            if not any(
                other != index
                and candidates[other]["pool_objective"]
                <= candidate["pool_objective"] + 1e-12
                and candidates[other]["service_risk"]
                <= candidate["service_risk"] + 1e-12
                and (
                    candidates[other]["pool_objective"]
                    < candidate["pool_objective"] - 1e-12
                    or candidates[other]["service_risk"]
                    < candidate["service_risk"] - 1e-12
                )
                for other in remaining
            ):
                front.append(index)
        for index in front:
            ranks[index] = rank
        remaining.difference_update(front)
        rank += 1
    return ranks


def _structure_distance(left, right):
    union = left["structure_tokens"] | right["structure_tokens"]
    return 0.0 if not union else 1.0 - (
        len(left["structure_tokens"] & right["structure_tokens"])
        / len(union)
    )


def _validated_ratios(raw, defaults, *, name):
    ratios = dict(defaults if raw is None else raw)
    unknown = set(ratios) - set(defaults)
    if unknown or any(float(value) < 0.0 for value in ratios.values()):
        raise ValueError(f"Invalid {name}: {sorted(unknown)}")
    ratios = {
        category: float(ratios.get(category, 0.0))
        for category in defaults
    }
    total = sum(ratios.values())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass.")
    return {
        category: value / total
        for category, value in ratios.items()
    }


def _quota_counts(total, ratios, categories):
    counts = Counter()
    for position in range(int(total)):
        category = max(
            categories,
            key=lambda name: (
                (position + 1) * ratios[name] - counts[name],
                -categories.index(name),
            ),
        )
        counts[category] += 1
    return counts


def _quota_sequence(total, ratios, categories):
    counts = Counter()
    result = []
    for position in range(int(total)):
        category = max(
            categories,
            key=lambda name: (
                (position + 1) * ratios[name] - counts[name],
                -categories.index(name),
            ),
        )
        counts[category] += 1
        result.append(category)
    return result


def _probability_bin(value, service_level, boundary_width):
    value = float(value)
    service_level = float(service_level)
    lower = max(0.0, service_level - float(boundary_width))
    upper = min(1.0, service_level + float(boundary_width))
    tolerance = 1e-12
    if value < lower - tolerance:
        return "low"
    if value < service_level - tolerance:
        return "boundary_below"
    if value <= upper + tolerance:
        return "boundary_above"
    return "high"


def _probability_bin_counts(values, service_level, boundary_width):
    return Counter(
        _probability_bin(value, service_level, boundary_width)
        for value in values
    )


def _candidate_probability_counts(
    candidate,
    service_level,
    boundary_width,
    target_basis="job_probabilities",
):
    if target_basis == "min_job_probability":
        values = [candidate["min_job_probability"]]
    elif target_basis == "job_probabilities":
        values = candidate["job_probabilities"]
    else:
        raise ValueError(
            "hybrid_selection.probability_target_basis must be "
            "'job_probabilities' or 'min_job_probability'."
        )
    return _probability_bin_counts(
        values, service_level, boundary_width
    )


def _selection_probability_counts(
    selected,
    service_level,
    boundary_width,
    target_basis="job_probabilities",
):
    counts = Counter()
    for entry in selected:
        counts.update(_candidate_probability_counts(
            entry["candidate"],
            service_level,
            boundary_width,
            target_basis,
        ))
    return counts


def _distribution_text(counts):
    total = sum(counts.values())
    return " | ".join(
        f"{name}={counts[name]}/{total} "
        f"({(counts[name] / total if total else 0.0):.1%})"
        for name in PROBABILITY_BINS
    )


def _prepare_candidate_scores(candidates):
    objectives = [item["pool_objective"] for item in candidates]
    risks = [item["service_risk"] for item in candidates]
    for item, rank in zip(candidates, _pareto_ranks(candidates)):
        item["objective_norm"] = _normalized(
            item["pool_objective"], objectives
        )
        item["risk_norm"] = _normalized(item["service_risk"], risks)
        item["balanced_score"] = 0.5 * (
            item["objective_norm"] + item["risk_norm"]
        )
        item["pareto_rank"] = rank


def _select_hybrid_candidates(
    candidates,
    limit,
    service_level,
    boundary_width,
    hybrid_selection,
):
    limit = min(int(limit), len(candidates))
    if limit <= 0:
        return []
    service_level = float(service_level)
    boundary_width = float(boundary_width)
    if boundary_width <= 0.0:
        raise ValueError("service_boundary_width must be positive.")

    config = dict(hybrid_selection or {})
    anchor_fraction = _validate_ratio(
        config.get("anchor_fraction", 0.40),
        "hybrid_selection.anchor_fraction",
    )
    anchor_ratios = _validated_ratios(
        config.get("anchor_ratios"),
        DEFAULT_HYBRID_ANCHOR_RATIOS,
        name="hybrid_selection.anchor_ratios",
    )
    target_ratios = _validated_ratios(
        config.get("job_probability_target_ratios"),
        DEFAULT_JOB_PROBABILITY_TARGET_RATIOS,
        name="hybrid_selection.job_probability_target_ratios",
    )
    target_basis = str(config.get(
        "probability_target_basis", "job_probabilities"
    ))
    if target_basis not in {
        "job_probabilities", "min_job_probability"
    }:
        raise ValueError(
            "hybrid_selection.probability_target_basis must be "
            "'job_probabilities' or 'min_job_probability'."
        )
    if target_basis == "job_probabilities":
        job_counts = {
            len(item.get("job_probabilities") or [])
            for item in candidates
        }
        if len(job_counts) != 1 or not next(iter(job_counts), 0):
            raise ValueError(
                "Hybrid selection requires the same nonzero number of job "
                "probabilities for every candidate."
            )
        labels_per_candidate = next(iter(job_counts))
    else:
        labels_per_candidate = 1

    _prepare_candidate_scores(candidates)
    anchor_limit = min(limit, int(round(limit * anchor_fraction)))
    requested = _quota_counts(
        anchor_limit,
        anchor_ratios,
        list(ANCHOR_CATEGORIES),
    )
    rankings = {
        "good_nominal_objective": sorted(
            candidates, key=lambda item: item["pool_objective"]
        ),
        "high_min_job_probability": sorted(
            candidates, key=lambda item: -item["min_job_probability"]
        ),
        "pareto_tradeoff": sorted(
            candidates,
            key=lambda item: (
                item["pareto_rank"],
                max(item["objective_norm"], item["risk_norm"]),
                item["balanced_score"],
            ),
        ),
    }
    selected, used = [], set()
    for category in ANCHOR_CATEGORIES[:-1]:
        for item in rankings[category]:
            if sum(
                entry["category"] == category for entry in selected
            ) >= requested[category]:
                break
            if item["candidate_id"] in used:
                continue
            selected.append({"candidate": item, "category": category})
            used.add(item["candidate_id"])

    for _ in range(requested["structurally_diverse"]):
        remaining = [
            item for item in candidates if item["candidate_id"] not in used
        ]
        if not remaining:
            break
        chosen = [entry["candidate"] for entry in selected]
        item = max(
            remaining,
            key=lambda candidate: (
                min(
                    (
                        _structure_distance(candidate, existing)
                        for existing in chosen
                    ),
                    default=1.0,
                ),
                -candidate["balanced_score"],
            ),
        )
        selected.append({
            "candidate": item,
            "category": "structurally_diverse",
        })
        used.add(item["candidate_id"])

    target_counts = {
        name: limit * labels_per_candidate * target_ratios[name]
        for name in PROBABILITY_BINS
    }
    current = _selection_probability_counts(
        selected, service_level, boundary_width, target_basis
    )
    while len(selected) < limit:
        remaining = [
            item for item in candidates if item["candidate_id"] not in used
        ]
        if not remaining:
            break
        chosen = [entry["candidate"] for entry in selected]

        def selection_key(item):
            item_counts = _candidate_probability_counts(
                item,
                service_level,
                boundary_width,
                target_basis,
            )
            contributions = {
                name: min(
                    item_counts[name],
                    max(0.0, target_counts[name] - current[name]),
                ) / target_counts[name]
                if target_counts[name] > 0.0 else 0.0
                for name in PROBABILITY_BINS
            }
            return (
                sum(contributions.values()),
                contributions["boundary_below"]
                + contributions["boundary_above"],
                min(
                    (
                        _structure_distance(item, existing)
                        for existing in chosen
                    ),
                    default=1.0,
                ),
                -item["balanced_score"],
            )

        item = max(remaining, key=selection_key)
        item_counts = _candidate_probability_counts(
            item,
            service_level,
            boundary_width,
            target_basis,
        )
        contributions = {
            name: min(
                item_counts[name],
                max(0.0, target_counts[name] - current[name]),
            ) / target_counts[name]
            if target_counts[name] > 0.0 else 0.0
            for name in PROBABILITY_BINS
        }
        dominant = max(
            PROBABILITY_BINS,
            key=lambda name: (
                contributions[name],
                item_counts[name],
                -PROBABILITY_BINS.index(name),
            ),
        )
        category = (
            f"job_probability_{dominant}"
            if contributions[dominant] > 0.0
            else "job_probability_structural_fill"
        )
        selected.append({"candidate": item, "category": category})
        used.add(item["candidate_id"])
        current.update(item_counts)
    return selected


def _select_candidates_legacy(
    candidates,
    limit,
    ratios,
    service_level=None,
    boundary_width=0.10,
    quota_offset=0,
):
    limit = min(int(limit), len(candidates))
    if limit <= 0:
        return []
    ratios = dict(DEFAULT_SELECTION_RATIOS if ratios is None else ratios)
    unknown = set(ratios) - set(DEFAULT_SELECTION_RATIOS)
    if unknown or any(float(value) < 0.0 for value in ratios.values()):
        raise ValueError(f"Invalid pool selection ratios: {sorted(unknown)}")
    ratios = {
        category: float(ratios.get(category, 0.0))
        for category in DEFAULT_SELECTION_RATIOS
    }
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0.0:
        raise ValueError("pool_selection_ratios must have positive mass.")

    objectives = [item["pool_objective"] for item in candidates]
    risks = [item["service_risk"] for item in candidates]
    for item, rank in zip(candidates, _pareto_ranks(candidates)):
        item["objective_norm"] = _normalized(item["pool_objective"], objectives)
        item["risk_norm"] = _normalized(item["service_risk"], risks)
        item["balanced_score"] = 0.5 * (
            item["objective_norm"] + item["risk_norm"]
        )
        item["pareto_rank"] = rank

    categories = list(DEFAULT_SELECTION_RATIOS)
    counts = Counter()
    allocation = []
    for position in range(int(quota_offset) + limit):
        category = max(
            categories,
            key=lambda name: (
                (position + 1) * ratios[name] / ratio_sum - counts[name],
                -categories.index(name),
            ),
        )
        counts[category] += 1
        allocation.append(category)
    requested = Counter(allocation[int(quota_offset):])
    selected, used = [], set()

    service_level = float(service_level)
    boundary_width = float(boundary_width)
    if boundary_width <= 0.0:
        raise ValueError("service_boundary_width must be positive.")

    def probabilities(item):
        values = item.get("job_probabilities")
        if not values:
            raise ValueError(
                "Candidate selection requires pilot probabilities for every job."
            )
        return [float(value) for value in values]

    def boundary_key(item, *, side):
        values = probabilities(item)
        if side == "below":
            relevant = [value for value in values if value < service_level]
        elif side == "above":
            relevant = [value for value in values if value >= service_level]
        else:
            relevant = values
        distances = [abs(value - service_level) for value in relevant]
        if not distances:
            return (math.inf, math.inf, item["balanced_score"])
        boundary_mass = sum(
            max(0.0, 1.0 - distance / boundary_width)
            for distance in distances
        )
        return (
            -boundary_mass,
            min(distances),
            item["balanced_score"],
        )

    boundary = sorted(
        candidates,
        key=lambda item: boundary_key(item, side="all"),
    )
    below = sorted(
        (
            item for item in candidates
            if any(value < service_level for value in probabilities(item))
        ),
        key=lambda item: boundary_key(item, side="below"),
    )
    above = sorted(
        (item for item in candidates if item["min_job_probability"] >= service_level),
        key=lambda item: boundary_key(item, side="above"),
    )
    def boundary_fallback(primary):
        identifiers = {item["candidate_id"] for item in primary}
        return primary + [
            item for item in boundary if item["candidate_id"] not in identifiers
        ]

    rankings = {
        "good_nominal_objective": sorted(
            candidates, key=lambda item: item["pool_objective"]
        ),
        "high_min_job_probability": sorted(
            candidates, key=lambda item: -item["min_job_probability"]
        ),
        "service_boundary_below": boundary_fallback(below),
        "service_boundary_above": boundary_fallback(above),
        "low_min_job_probability": sorted(
            candidates, key=lambda item: item["min_job_probability"]
        ),
        "pareto_tradeoff": sorted(
            candidates,
            key=lambda item: (
                item["pareto_rank"],
                max(item["objective_norm"], item["risk_norm"]),
                item["balanced_score"],
            ),
        ),
    }
    for category, ranked in rankings.items():
        for item in ranked:
            if sum(x["category"] == category for x in selected) >= requested[category]:
                break
            if item["candidate_id"] in used:
                continue
            selected.append({"candidate": item, "category": category})
            used.add(item["candidate_id"])

    while len(selected) < limit:
        remaining = [
            item for item in candidates if item["candidate_id"] not in used
        ]
        if not remaining:
            break
        chosen_items = [entry["candidate"] for entry in selected]
        next_item = max(
            remaining,
            key=lambda item: (
                min(
                    (_structure_distance(item, chosen) for chosen in chosen_items),
                    default=1.0,
                ),
                item["balanced_score"],
            ),
        )
        selected.append({
            "candidate": next_item,
            "category": "structurally_diverse",
        })
        used.add(next_item["candidate_id"])
    return selected


def _select_candidates(
    candidates,
    limit,
    ratios,
    service_level=None,
    boundary_width=0.10,
    quota_offset=0,
    hybrid_selection=None,
):
    if hybrid_selection and hybrid_selection.get("enabled", True):
        return _select_hybrid_candidates(
            candidates,
            limit,
            service_level,
            boundary_width,
            hybrid_selection,
        )
    return _select_candidates_legacy(
        candidates,
        limit,
        ratios,
        service_level,
        boundary_width,
        quota_offset,
    )


def _neighborhood_combinations(fixed):
    raw_fix_ratios = fixed.get("fix_ratios")
    fix_ratios = (
        [_validate_ratio(value, "fix_ratio") for value in raw_fix_ratios]
        if raw_fix_ratios else [None]
    )
    raw_sequence_ratios = fixed.get("sequence_fix_ratios")
    sequence_ratios = (
        [
            _validate_ratio(value, "sequence_fix_ratio")
            for value in raw_sequence_ratios
        ]
        if raw_sequence_ratios else [0.0]
    )
    if len(fix_ratios) != len(set(fix_ratios)):
        raise ValueError("fixed_y.fix_ratios must not contain duplicates.")
    if len(sequence_ratios) != len(set(sequence_ratios)):
        raise ValueError(
            "fixed_y.sequence_fix_ratios must not contain duplicates."
        )
    return [
        (fix_ratio, sequence_ratio)
        for fix_ratio in fix_ratios
        for sequence_ratio in sequence_ratios
    ]


def _run_neighborhood(
    instance,
    instance_name,
    run_index,
    generation,
    graph_config,
    simulation_config,
    fix_ratio,
    sequence_ratio,
    candidate_generation_mode="unconstrained",
    candidate_probability_band=None,
):
    candidate_generation_mode = _validate_candidate_generation_mode(
        candidate_generation_mode
    )
    rng = random.Random(_sample_seed(
        instance_name, run_index, generation["random_seed"]
    ))
    sample_instance = copy.deepcopy(instance)
    _sample_reliability(sample_instance, rng, generation)
    service_probability_band = None
    if candidate_probability_band is not None:
        if candidate_generation_mode != "nonlinear":
            raise ValueError(
                "Probability-band generation requires "
                "candidate_generation_mode='nonlinear'."
            )
        service_probability_band = _probability_band_limits(
            candidate_probability_band,
            graph_config.service_level,
            generation["fixed_y"].get(
                "service_boundary_width", 0.10
            ),
        )
    model, variables = _build_candidate_model(
        sample_instance,
        graph_config,
        candidate_generation_mode,
        service_probability_band,
    )
    if candidate_generation_mode == "nominal_ontime":
        _add_nominal_ontime_constraints(model, variables, sample_instance)
    fixed = generation["fixed_y"]
    model.Params.OutputFlag = int(fixed.get("output_flag", 0))
    model.Params.TimeLimit = float(fixed.get("time_limit_seconds", 5.0))
    model.Params.MIPGap = float(fixed.get("mip_gap", 0.2))
    model.Params.MIPFocus = int(fixed.get("mip_focus", 1))
    model.Params.Seed = int(
        fixed.get("nonlinear_solver_seed", 0)
        if candidate_generation_mode == "nonlinear"
        else _sample_seed(
            instance_name,
            run_index,
            generation["random_seed"],
            "gurobi",
        ) % 2_000_000_000
    )
    if fix_ratio is None:
        fix_ratio = rng.uniform(
            float(fixed.get("fix_ratio_min", 0.2)),
            float(fixed.get("fix_ratio_max", 0.4)),
        )
    fix_ratio = _validate_ratio(fix_ratio, "fix_ratio")
    sequence_ratio = _validate_ratio(sequence_ratio, "sequence_fix_ratio")
    reference_runtime = 0.0
    use_incumbent = candidate_generation_mode == "nonlinear"
    incumbent_starts = None
    if use_incumbent:
        model.Params.PoolSearchMode = 0
        model.Params.PoolSolutions = 1
        model.optimize()
        reference_runtime = float(model.Runtime)
        if model.SolCount == 0:
            model.dispose()
            return []
        incumbent_starts = {
            variable: float(variable.X) for variable in model.getVars()
        }
    fix_stats = _fix_neighborhood(
        model,
        variables,
        sample_instance,
        rng,
        fix_ratio,
        sequence_ratio,
        use_incumbent=use_incumbent,
    )
    if incumbent_starts is not None:
        for variable, value in incumbent_starts.items():
            variable.Start = value
    model.Params.PoolSearchMode = int(fixed.get("pool_search_mode", 2))
    model.Params.PoolSolutions = int(fixed.get("pool_candidates", 50))
    model.Params.PoolGap = float(fixed.get("pool_gap", 1.0))
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
            model,
            variables,
            sample_instance,
            graph_config,
            solution_number,
            run_index,
            fix_stats,
            candidate_generation_mode,
            candidate_probability_band,
        )
        candidate["row"]["solver_runtime_seconds"] = (
            reference_runtime + float(model.Runtime)
        )
        if candidate["structure"] in local_structures:
            continue
        local_structures.add(candidate["structure"])
        candidate["row"]["instance_name"] = instance_name
        if candidate_generation_mode == "nonlinear_evaluated":
            _evaluate_fixed_schedule_nonlinear(candidate, graph_config)
        _simulate_candidate(
            candidate,
            instance_name,
            simulation_config,
            replications=simulation_config.pilot_replications,
            purpose="pilot",
        )
        candidates.append(candidate)
    print(
        f"[Simulation:pilot] {instance_name} | run={run_index + 1} | "
        f"mode={candidate_generation_mode} | "
        f"band={candidate_probability_band or 'service_feasible'} | "
        f"candidates={len(candidates)} | "
        f"replications={simulation_config.pilot_replications}",
        flush=True,
    )
    model.dispose()
    return candidates


def _hybrid_selection_for_split(fixed, split=None):
    config = dict(fixed.get("hybrid_selection") or {})
    if not config or not bool(config.get("enabled", True)):
        return None
    apply_to = config.get("apply_to_splits")
    if apply_to is not None:
        apply_to = {str(value) for value in apply_to}
        unknown = apply_to - {"training", "valid", "test"}
        if unknown:
            raise ValueError(
                "hybrid_selection.apply_to_splits contains unknown splits: "
                f"{sorted(unknown)}"
            )
        if split is not None and split not in apply_to:
            return None
    return config


def _mixed_candidate_generation_for_split(fixed, split=None):
    config = dict(fixed.get("mixed_candidate_generation") or {})
    if not config or not bool(config.get("enabled", True)):
        return None
    apply_to = config.get("apply_to_splits")
    if apply_to is not None:
        apply_to = {str(value) for value in apply_to}
        unknown = apply_to - {"training", "valid", "test"}
        if unknown:
            raise ValueError(
                "mixed_candidate_generation.apply_to_splits contains "
                f"unknown splits: {sorted(unknown)}"
            )
        if split is not None and split not in apply_to:
            return None
    config["nominal_selected_fraction"] = _validate_ratio(
        config.get("nominal_selected_fraction", 0.50),
        "mixed_candidate_generation.nominal_selected_fraction",
    )
    minimum_runs = int(config.get("nominal_minimum_pool_runs", 2))
    maximum_runs = int(config.get("nominal_maximum_pool_runs", 4))
    if minimum_runs <= 0:
        raise ValueError(
            "mixed_candidate_generation.nominal_minimum_pool_runs must "
            "be positive."
        )
    if maximum_runs < minimum_runs:
        raise ValueError(
            "mixed_candidate_generation.nominal_maximum_pool_runs must "
            "be at least nominal_minimum_pool_runs."
        )
    config["nominal_minimum_pool_runs"] = minimum_runs
    config["nominal_maximum_pool_runs"] = maximum_runs
    run_offset = int(config.get(
        "nominal_run_index_offset", DEFAULT_NOMINAL_RUN_INDEX_OFFSET
    ))
    if run_offset <= 0:
        raise ValueError(
            "mixed_candidate_generation.nominal_run_index_offset must "
            "be positive."
        )
    config["nominal_run_index_offset"] = run_offset
    return config


def _nonlinear_probability_band_generation(fixed, candidate_mode):
    config = dict(fixed.get("nonlinear_probability_bands") or {})
    if (
        candidate_mode != "nonlinear"
        or not config
        or not bool(config.get("enabled", True))
    ):
        return None
    config["target_ratios"] = _validated_ratios(
        config.get("target_ratios"),
        DEFAULT_JOB_PROBABILITY_TARGET_RATIOS,
        name="nonlinear_probability_bands.target_ratios",
    )
    return config


def _hybrid_boundary_targets_met(
    candidates,
    limit,
    service_level,
    boundary_width,
    hybrid_selection,
):
    if len(candidates) < int(limit):
        return False
    selected = _select_hybrid_candidates(
        candidates,
        limit,
        service_level,
        boundary_width,
        hybrid_selection,
    )
    ratios = _validated_ratios(
        hybrid_selection.get("job_probability_target_ratios"),
        DEFAULT_JOB_PROBABILITY_TARGET_RATIOS,
        name="hybrid_selection.job_probability_target_ratios",
    )
    target_basis = str(hybrid_selection.get(
        "probability_target_basis", "job_probabilities"
    ))
    counts = _selection_probability_counts(
        selected,
        service_level,
        boundary_width,
        target_basis,
    )
    total = sum(counts.values())
    tolerance = 1.0
    return all(
        counts[name] + tolerance >= total * ratios[name]
        for name in ("boundary_below", "boundary_above")
    )


def _collect_instance_candidates(
    instance,
    instance_name,
    generation,
    graph_config,
    simulation_config,
    minimum_count,
    *,
    start_run=0,
    progress=None,
    hybrid_selection=None,
    candidate_generation_mode="unconstrained",
    minimum_runs_override=None,
    maximum_runs_override=None,
    probability_band_generation=None,
):
    candidate_generation_mode = _validate_candidate_generation_mode(
        candidate_generation_mode
    )
    fixed = generation["fixed_y"]
    neighborhoods = _neighborhood_combinations(fixed)
    pool_candidates = int(fixed.get("pool_candidates", 50))
    if pool_candidates <= 0:
        raise ValueError("fixed_y.pool_candidates must be positive.")
    planned_runs = math.ceil(int(minimum_count) / pool_candidates)
    configured_minimum_runs = int(
        minimum_runs_override
        if minimum_runs_override is not None
        else fixed.get("minimum_candidate_pool_runs", len(neighborhoods))
    )
    if configured_minimum_runs <= 0:
        raise ValueError(
            "fixed_y.minimum_candidate_pool_runs must be positive."
        )
    minimum_runs = (
        configured_minimum_runs
        if minimum_runs_override is not None
        else max(len(neighborhoods), configured_minimum_runs)
    )
    if maximum_runs_override is not None:
        maximum_runs = int(maximum_runs_override)
        if maximum_runs < minimum_runs:
            raise ValueError(
                "maximum_runs_override must be at least the minimum runs."
            )
    elif hybrid_selection:
        maximum_runs = int(hybrid_selection.get(
            "maximum_candidate_pool_runs",
            max(minimum_runs * 3, planned_runs * 3, minimum_runs + 2),
        ))
        if maximum_runs < minimum_runs:
            raise ValueError(
                "hybrid_selection.maximum_candidate_pool_runs must be at "
                f"least {minimum_runs}."
            )
    else:
        maximum_runs = max(
            minimum_runs * 3,
            planned_runs * 3,
            minimum_runs + 2,
        )
    probability_band_sequence = None
    if probability_band_generation:
        probability_band_sequence = _quota_sequence(
            maximum_runs,
            probability_band_generation["target_ratios"],
            list(PROBABILITY_BINS),
        )
    candidates = []
    for run_offset in range(maximum_runs):
        run_index = int(start_run) + run_offset
        fix_ratio, sequence_ratio = neighborhoods[
            run_offset % len(neighborhoods)
        ]
        candidates.extend(_run_neighborhood(
            instance,
            instance_name,
            run_index,
            generation,
            graph_config,
            simulation_config,
            fix_ratio,
            sequence_ratio,
            candidate_generation_mode,
            (
                probability_band_sequence[run_offset]
                if probability_band_sequence else None
            ),
        ))
        if progress is not None:
            progress(
                run_index,
                fix_ratio,
                sequence_ratio,
                len(candidates),
            )
        if run_offset + 1 >= minimum_runs and len(candidates) >= minimum_count:
            if not hybrid_selection or _hybrid_boundary_targets_met(
                candidates,
                minimum_count,
                graph_config.service_level,
                fixed.get("service_boundary_width", 0.10),
                hybrid_selection,
            ):
                break
    return candidates


def _select_mixed_candidates(
    unconstrained_candidates,
    nominal_candidates,
    limit,
    ratios,
    service_level,
    boundary_width,
    hybrid_selection,
    nominal_selected_fraction,
):
    """Select a mode-balanced, structurally unique candidate sample."""
    limit = int(limit)
    nominal_target = int(round(
        limit * _validate_ratio(
            nominal_selected_fraction,
            "mixed_candidate_generation.nominal_selected_fraction",
        )
    ))
    nominal_target = min(nominal_target, len(nominal_candidates))

    def select(pool, count):
        return _select_candidates(
            pool,
            count,
            ratios,
            service_level,
            boundary_width,
            hybrid_selection=hybrid_selection,
        )

    selected = select(nominal_candidates, nominal_target)
    used_ids = {
        entry["candidate"]["candidate_id"] for entry in selected
    }
    used_structures = {
        entry["candidate"]["structure"] for entry in selected
    }
    unconstrained_available = [
        candidate for candidate in unconstrained_candidates
        if candidate["candidate_id"] not in used_ids
        and candidate["structure"] not in used_structures
    ]
    selected.extend(select(
        unconstrained_available,
        min(limit - len(selected), len(unconstrained_available)),
    ))
    used_ids = {
        entry["candidate"]["candidate_id"] for entry in selected
    }
    used_structures = {
        entry["candidate"]["structure"] for entry in selected
    }
    if len(selected) < limit:
        nominal_available = [
            candidate for candidate in nominal_candidates
            if candidate["candidate_id"] not in used_ids
            and candidate["structure"] not in used_structures
        ]
        selected.extend(select(
            nominal_available,
            min(limit - len(selected), len(nominal_available)),
        ))
    for entry in selected:
        mode = entry["candidate"]["candidate_generation_mode"]
        entry["category"] = f"{mode}:{entry['category']}"
    return selected


def _collect_and_select_instance_candidates(
    instance,
    instance_name,
    generation,
    graph_config,
    simulation_config,
    count,
    *,
    split=None,
    start_run=0,
    progress=None,
):
    fixed = generation["fixed_y"]
    hybrid_selection = _hybrid_selection_for_split(fixed, split)
    mixed_generation = _mixed_candidate_generation_for_split(fixed, split)
    if not mixed_generation:
        candidate_generation_mode = _validate_candidate_generation_mode(
            fixed.get("candidate_generation_mode", "unconstrained")
        )
        probability_band_generation = (
            _nonlinear_probability_band_generation(
                fixed, candidate_generation_mode
            )
        )
        candidates = _collect_instance_candidates(
            instance,
            instance_name,
            generation,
            graph_config,
            simulation_config,
            int(count),
            start_run=start_run,
            progress=progress,
            hybrid_selection=hybrid_selection,
            candidate_generation_mode=candidate_generation_mode,
            probability_band_generation=probability_band_generation,
        )
        selected = _select_candidates(
            candidates,
            count,
            fixed.get("pool_selection_ratios"),
            graph_config.service_level,
            fixed.get("service_boundary_width", 0.10),
            hybrid_selection=hybrid_selection,
        )
        return candidates, selected

    nominal_target = (
        int(round(
            int(count) * mixed_generation["nominal_selected_fraction"]
        ))
        if mixed_generation else 0
    )
    unconstrained_target = int(count) - nominal_target
    collection_hybrid_selection = (
        None if mixed_generation else hybrid_selection
    )
    unconstrained = []
    if unconstrained_target > 0:
        unconstrained = _collect_instance_candidates(
            instance,
            instance_name,
            generation,
            graph_config,
            simulation_config,
            unconstrained_target,
            start_run=start_run,
            progress=progress,
            hybrid_selection=collection_hybrid_selection,
            candidate_generation_mode="unconstrained",
        )
    nominal = []
    if mixed_generation and nominal_target > 0:
        nominal = _collect_instance_candidates(
            instance,
            instance_name,
            generation,
            graph_config,
            simulation_config,
            nominal_target,
            start_run=(
                int(start_run)
                + mixed_generation["nominal_run_index_offset"]
            ),
            progress=progress,
            hybrid_selection=collection_hybrid_selection,
            candidate_generation_mode="nominal_ontime",
            minimum_runs_override=mixed_generation[
                "nominal_minimum_pool_runs"
            ],
            maximum_runs_override=mixed_generation[
                "nominal_maximum_pool_runs"
            ],
        )
    selected = _select_mixed_candidates(
        unconstrained,
        nominal,
        count,
        fixed.get("pool_selection_ratios"),
        graph_config.service_level,
        fixed.get("service_boundary_width", 0.10),
        hybrid_selection,
        mixed_generation["nominal_selected_fraction"],
    )
    return unconstrained + nominal, selected


def generate_from_config(generation):
    """Generate current-schema GNN graphs from fixed optimization neighborhoods."""
    generation = dict(generation)
    if generation.get("method") != "fix_and_optimize":
        raise ValueError("Fix-and-optimize generation requires method='fix_and_optimize'.")
    graph_config = normalize_reliability_graph_config(
        generation.get("reliability_graph")
    )
    simulation_config = normalize_simulation_config(
        generation.get("simulation")
    )
    samples_per_instance = int(generation["samples_per_instance"])
    if samples_per_instance <= 0:
        raise ValueError("samples_per_instance must be positive.")
    fixed = dict(generation.get("fixed_y") or {})
    _neighborhood_combinations(fixed)
    generation["fixed_y"] = fixed
    generation["random_seed"] = int(generation.get("random_seed", 42))
    output_root = Path(generation["output_directory"])
    if not output_root.is_absolute():
        output_root = ROOT_DIR / output_root
    splits = selected_instance_splits(
        generation["instance_splits"], generation.get("generate_splits")
    )
    for split, instance_names in splits.items():
        output_path = (
            output_root / SPLIT_DIRECTORIES[split]
            / SPLIT_CSV_FILENAMES[split]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            split_probability_counts = Counter()
            for instance_name in instance_names:
                instance = load_generated_instance(instance_name)
                def report(run_index, fix_ratio, sequence_ratio, candidate_count):
                    print(
                        f"[Fix-and-optimize] {split} | {instance_name} | "
                        f"run={run_index + 1} | graphs="
                        f"{candidate_count} candidates | fix_ratio="
                        f"{fix_ratio if fix_ratio is not None else 'random'} | "
                        f"sequence_fix_ratio={sequence_ratio}",
                        flush=True,
                    )
                candidates, selected = _collect_and_select_instance_candidates(
                    instance,
                    instance_name,
                    generation,
                    graph_config,
                    simulation_config,
                    samples_per_instance,
                    split=split,
                    progress=report,
                )
                if len(selected) < samples_per_instance:
                    raise RuntimeError(
                        f"Only {len(selected)}/{samples_per_instance} "
                        f"structurally unique candidate graphs selected for "
                        f"{instance_name} from {len(candidates)} candidates."
                    )
                rows = []
                instance_probability_counts = Counter()
                print(
                    f"[Simulation:labels] {split} | {instance_name} | "
                    f"graphs={len(selected)} | "
                    f"replications={simulation_config.label_replications}",
                    flush=True,
                )
                for entry in selected:
                    candidate = _simulate_candidate(
                        entry["candidate"],
                        instance_name,
                        simulation_config,
                        replications=simulation_config.label_replications,
                        purpose="label",
                    )
                    row = candidate["row"]
                    row["pool_selection_category"] = entry["category"]
                    rows.append(row)
                    instance_probability_counts.update(
                        _probability_bin_counts(
                            candidate["job_probabilities"],
                            graph_config.service_level,
                            fixed.get("service_boundary_width", 0.10),
                        )
                    )
                writer.writerows(rows)
                file.flush()
                written += len(rows)
                split_probability_counts.update(instance_probability_counts)
                print(
                    f"[Fix-and-optimize] {split} | {instance_name} | "
                    f"selected={len(rows)}/{len(candidates)} candidates",
                    flush=True,
                )
                print(
                    f"[Job-label distribution] {split} | {instance_name} | "
                    f"{_distribution_text(instance_probability_counts)}",
                    flush=True,
                )
        print(
            f"[Fix-and-optimize] finished {split}: {written} graphs -> "
            f"{output_path}",
            flush=True,
        )
        print(
            f"[Job-label distribution] finished {split} | "
            f"{_distribution_text(split_probability_counts)}",
            flush=True,
        )


def generate_rows_for_instance(
    instance_name,
    generation,
    *,
    count,
    start_run=0,
):
    """Small in-memory generator used by smoke tests and numerical studies."""
    generation = dict(generation)
    graph_config = normalize_reliability_graph_config(
        generation.get("reliability_graph")
    )
    simulation_config = normalize_simulation_config(
        generation.get("simulation")
    )
    generation["fixed_y"] = dict(generation.get("fixed_y") or {})
    generation["random_seed"] = int(generation.get("random_seed", 42))
    instance = load_generated_instance(instance_name)
    candidates, selected = _collect_and_select_instance_candidates(
        instance,
        instance_name,
        generation,
        graph_config,
        simulation_config,
        int(count),
        start_run=start_run,
    )
    if len(selected) < int(count):
        raise RuntimeError(
            f"Could not select {count} structurally unique candidate rows "
            f"for {instance_name} from {len(candidates)} candidates."
        )
    rows = []
    for entry in selected:
        candidate = _simulate_candidate(
            entry["candidate"],
            instance_name,
            simulation_config,
            replications=simulation_config.label_replications,
            purpose="label",
        )
        row = candidate["row"]
        row["pool_selection_category"] = entry["category"]
        rows.append(row)
    return rows
