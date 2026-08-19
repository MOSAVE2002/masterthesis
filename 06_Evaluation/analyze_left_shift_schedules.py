"""Compare stored Gurobi timings with canonical earliest-start timings."""

from __future__ import annotations

import copy
import csv
import importlib
import json
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config.json"
OUTPUT_PATH = (
    ROOT_DIR / "06_Evaluation" / "results" / "left_shift_analysis.csv"
)
INSTANCE_NAMES = (
    "i3_k3_o3-5_1",
    "i3_k5_o3-5_1",
    "i5_k3_o3-5_1",
    "i5_k5_o3-5_1",
    "i7_k3_o3-5_1",
    "i7_k5_o3-5_1",
)

ENTRY = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_weibull_training_data"
)
GENERATOR = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_fix_and_optimize_training_data"
)


def _left_shift_schedule(schedule):
    operations = tuple(schedule.operations)
    predecessors = {
        operation: set(schedule.job_predecessors.get(operation, ()))
        for operation in operations
    }
    successors = {operation: set() for operation in operations}
    for source, target, _machine in schedule.machine_edges:
        predecessors[target].add(source)
    for target, required in predecessors.items():
        for source in required:
            successors[source].add(target)

    indegree = {
        operation: len(predecessors[operation]) for operation in operations
    }
    available = deque(
        operation for operation in operations if indegree[operation] == 0
    )
    starts = {}
    completions = {}
    while available:
        operation = available.popleft()
        starts[operation] = max(
            (completions[source] for source in predecessors[operation]),
            default=0.0,
        )
        completions[operation] = (
            starts[operation]
            + float(schedule.processing_times[operation])
        )
        for target in successors[operation]:
            indegree[target] -= 1
            if indegree[target] == 0:
                available.append(target)
    if len(starts) != len(operations):
        raise ValueError("Candidate schedule contains a precedence cycle.")
    return replace(schedule, planned_starts=starts)


def _prepare_candidates(
    candidates,
    instance_name,
    graph_config,
    simulation_config,
    *,
    left_shift,
):
    prepared = copy.deepcopy(candidates)
    for candidate in prepared:
        if left_shift:
            candidate["simulation_schedule"] = _left_shift_schedule(
                candidate["simulation_schedule"]
            )
        GENERATOR._evaluate_fixed_schedule_nonlinear(
            candidate, graph_config
        )
        GENERATOR._simulate_candidate(
            candidate,
            instance_name,
            simulation_config,
            replications=simulation_config.pilot_replications,
            purpose="left-shift-common-pilot",
        )
    return prepared


def _evaluate_method(
    candidates,
    instance_name,
    generation,
    graph_config,
    simulation_config,
    method,
):
    fixed = generation["fixed_y"]
    prepared = _prepare_candidates(
        candidates,
        instance_name,
        graph_config,
        simulation_config,
        left_shift=method == "earliest_start",
    )
    selected = GENERATOR._select_candidates(
        prepared,
        generation["samples_per_instance"],
        fixed.get("pool_selection_ratios"),
        graph_config.service_level,
        fixed["service_boundary_width"],
        hybrid_selection=fixed["hybrid_selection"],
    )
    job_probabilities = []
    graph_minima = []
    nominal_tardy = 0
    job_labels = 0
    for entry in selected:
        candidate = GENERATOR._simulate_candidate(
            entry["candidate"],
            instance_name,
            simulation_config,
            replications=simulation_config.label_replications,
            purpose="left-shift-common-label",
        )
        schedule = candidate["simulation_schedule"]
        job_probabilities.extend(candidate["job_probabilities"])
        graph_minima.append(candidate["min_job_probability"])
        for job in sorted(schedule.jobs):
            end = schedule.job_end_operations[job]
            completion = (
                float(schedule.planned_starts[end])
                + float(schedule.processing_times[end])
            )
            nominal_tardy += completion > schedule.due_dates[job] + 1e-9
            job_labels += 1
    job_bins = Counter(
        GENERATOR._probability_bin(
            value,
            graph_config.service_level,
            fixed["service_boundary_width"],
        )
        for value in job_probabilities
    )
    graph_bins = Counter(
        GENERATOR._probability_bin(
            value,
            graph_config.service_level,
            fixed["service_boundary_width"],
        )
        for value in graph_minima
    )
    return {
        "method": method,
        "instance_name": instance_name,
        "candidate_pool": len(candidates),
        "selected_graphs": len(selected),
        "job_labels": job_labels,
        "job_exact_zero": sum(
            value == 0.0 for value in job_probabilities
        ),
        "job_exact_one": sum(
            value == 1.0 for value in job_probabilities
        ),
        "job_nominal_tardy": nominal_tardy,
        "job_low": job_bins["low"],
        "job_boundary_below": job_bins["boundary_below"],
        "job_boundary_above": job_bins["boundary_above"],
        "job_high": job_bins["high"],
        "graph_low": graph_bins["low"],
        "graph_boundary_below": graph_bins["boundary_below"],
        "graph_boundary_above": graph_bins["boundary_above"],
        "graph_high": graph_bins["high"],
        "graph_service_feasible": sum(
            value >= graph_config.service_level for value in graph_minima
        ),
        "mean_job_probability": statistics.fmean(job_probabilities),
    }


def _with_rates(row):
    row = dict(row)
    jobs = row["job_labels"]
    graphs = row["selected_graphs"]
    for field in (
        "job_exact_zero",
        "job_exact_one",
        "job_nominal_tardy",
        "job_low",
        "job_boundary_below",
        "job_boundary_above",
        "job_high",
    ):
        row[f"{field}_rate"] = row[field] / jobs
    row["job_boundary_rate"] = (
        row["job_boundary_below"] + row["job_boundary_above"]
    ) / jobs
    for field in (
        "graph_low",
        "graph_boundary_below",
        "graph_boundary_above",
        "graph_high",
        "graph_service_feasible",
    ):
        row[f"{field}_rate"] = row[field] / graphs
    row["graph_boundary_rate"] = (
        row["graph_boundary_below"] + row["graph_boundary_above"]
    ) / graphs
    return row


def _aggregate(rows, method):
    selected = [row for row in rows if row["method"] == method]
    count_fields = (
        "candidate_pool",
        "selected_graphs",
        "job_labels",
        "job_exact_zero",
        "job_exact_one",
        "job_nominal_tardy",
        "job_low",
        "job_boundary_below",
        "job_boundary_above",
        "job_high",
        "graph_low",
        "graph_boundary_below",
        "graph_boundary_above",
        "graph_high",
        "graph_service_feasible",
    )
    result = {
        "method": method,
        "instance_name": "overall",
        **{
            field: sum(row[field] for row in selected)
            for field in count_fields
        },
        "mean_job_probability": sum(
            row["mean_job_probability"] * row["job_labels"]
            for row in selected
        ) / sum(row["job_labels"] for row in selected),
    }
    return _with_rates(result)


def run_analysis():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    generation = ENTRY._generation_from_project_config(config)
    generation["simulation"] = {
        **generation["simulation"],
        "pilot_replications": 256,
        "label_replications": 1000,
    }
    graph_config = GENERATOR.normalize_reliability_graph_config(
        generation["reliability_graph"]
    )
    simulation_config = GENERATOR.normalize_simulation_config(
        generation["simulation"]
    )
    rows = []
    started = time.perf_counter()
    for position, instance_name in enumerate(INSTANCE_NAMES, start=1):
        print(
            f"[Left-shift] pool {position}/{len(INSTANCE_NAMES)} "
            f"{instance_name}",
            flush=True,
        )
        instance = GENERATOR.load_generated_instance(instance_name)
        candidates = GENERATOR._collect_instance_candidates(
            instance,
            instance_name,
            generation,
            graph_config,
            simulation_config,
            generation["samples_per_instance"],
            hybrid_selection=None,
            candidate_generation_mode="nonlinear_evaluated",
            minimum_runs_override=4,
            maximum_runs_override=4,
        )
        for method in ("stored_gurobi", "earliest_start"):
            print(
                f"[Left-shift] {instance_name} {method}", flush=True
            )
            rows.append(_with_rates(_evaluate_method(
                candidates,
                instance_name,
                generation,
                graph_config,
                simulation_config,
                method,
            )))
    summaries = [
        _aggregate(rows, "stored_gurobi"),
        _aggregate(rows, "earliest_start"),
    ]
    output_rows = rows + summaries
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({
        "wall_seconds": time.perf_counter() - started,
        "output": str(OUTPUT_PATH),
        "overall": summaries,
    }, indent=2), flush=True)
    return rows, summaries


if __name__ == "__main__":
    run_analysis()
