"""Compare constant due-date factors on identical candidate schedules."""

from __future__ import annotations

import copy
import csv
import importlib
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config.json"
OUTPUT_PATH = (
    ROOT_DIR / "06_Evaluation" / "results" / "due_date_factor_analysis.csv"
)
INSTANCE_NAMES = (
    "i3_k3_o3-5_1",
    "i3_k5_o3-5_1",
    "i5_k3_o3-5_1",
    "i5_k5_o3-5_1",
    "i7_k3_o3-5_1",
    "i7_k5_o3-5_1",
)
FACTORS = (0.55, 0.575, 0.60, 0.625, 0.65)

ENTRY = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_weibull_training_data"
)
GENERATOR = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_fix_and_optimize_training_data"
)
SIMULATION = importlib.import_module("05_Simulation.preempt_resume")


def _simulation_seed(candidate, instance_name, simulation_config, purpose):
    run_index, solution_number = candidate["candidate_id"]
    return GENERATOR._sample_seed(
        instance_name,
        run_index,
        simulation_config.random_seed,
        f"simulation:{purpose}:solution={solution_number}",
    )


def _simulate_candidate(
    candidate,
    instance_name,
    simulation_config,
    *,
    replications,
    purpose,
):
    result = SIMULATION.simulate_fixed_schedule(
        candidate["simulation_schedule"],
        replications=int(replications),
        seed=_simulation_seed(
            candidate, instance_name, simulation_config, purpose
        ),
        config=simulation_config,
    )
    probabilities = list(result.job_ontime_probabilities)
    candidate.update({
        "job_probabilities": probabilities,
        "total_failure_delay": float(result.mean_total_repair_delay),
        "min_job_probability": min(probabilities),
        "service_risk": 1.0 - min(probabilities),
        "effective_objective": candidate["pool_objective"],
    })
    return candidate


def _serial_horizon(instance):
    return sum(
        max(
            float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        for operation in instance.real_operations
    )


def _bin(value, service_level, boundary_width):
    return GENERATOR._probability_bin(
        value, service_level, boundary_width
    )


def _evaluate_factor(
    candidates,
    instance_name,
    due_dates,
    generation,
    graph_config,
    simulation_config,
    factor,
):
    fixed = generation["fixed_y"]
    evaluated = copy.deepcopy(candidates)
    for candidate in evaluated:
        candidate["simulation_schedule"] = replace(
            candidate["simulation_schedule"],
            due_dates=dict(due_dates),
        )
        GENERATOR._evaluate_fixed_schedule_nonlinear(
            candidate, graph_config
        )
        _simulate_candidate(
            candidate,
            instance_name,
            simulation_config,
            replications=simulation_config.pilot_replications,
            purpose="due-factor-common-pilot",
        )
    selected = GENERATOR._select_candidates(
        evaluated,
        generation["samples_per_instance"],
        fixed.get("pool_selection_ratios"),
        graph_config.service_level,
        fixed["service_boundary_width"],
        hybrid_selection=fixed["hybrid_selection"],
    )
    job_probabilities = []
    graph_minima = []
    for entry in selected:
        candidate = _simulate_candidate(
            entry["candidate"],
            instance_name,
            simulation_config,
            replications=simulation_config.label_replications,
            purpose="due-factor-common-label",
        )
        job_probabilities.extend(candidate["job_probabilities"])
        graph_minima.append(candidate["min_job_probability"])
    job_bins = Counter(
        _bin(
            value,
            graph_config.service_level,
            fixed["service_boundary_width"],
        )
        for value in job_probabilities
    )
    graph_bins = Counter(
        _bin(
            value,
            graph_config.service_level,
            fixed["service_boundary_width"],
        )
        for value in graph_minima
    )
    return {
        "factor": factor,
        "instance_name": instance_name,
        "candidate_pool": len(candidates),
        "selected_graphs": len(selected),
        "job_labels": len(job_probabilities),
        "mean_job_probability": statistics.fmean(job_probabilities),
        "job_exact_zero_rate": (
            sum(value == 0.0 for value in job_probabilities)
            / len(job_probabilities)
        ),
        "job_exact_one_rate": (
            sum(value == 1.0 for value in job_probabilities)
            / len(job_probabilities)
        ),
        "job_low_rate": job_bins["low"] / len(job_probabilities),
        "job_boundary_below_rate": (
            job_bins["boundary_below"] / len(job_probabilities)
        ),
        "job_boundary_above_rate": (
            job_bins["boundary_above"] / len(job_probabilities)
        ),
        "job_high_rate": job_bins["high"] / len(job_probabilities),
        "job_boundary_rate": (
            job_bins["boundary_below"]
            + job_bins["boundary_above"]
        ) / len(job_probabilities),
        "graph_low_rate": graph_bins["low"] / len(graph_minima),
        "graph_boundary_below_rate": (
            graph_bins["boundary_below"] / len(graph_minima)
        ),
        "graph_boundary_above_rate": (
            graph_bins["boundary_above"] / len(graph_minima)
        ),
        "graph_high_rate": graph_bins["high"] / len(graph_minima),
        "graph_boundary_rate": (
            graph_bins["boundary_below"]
            + graph_bins["boundary_above"]
        ) / len(graph_minima),
        "graph_service_feasible_rate": (
            sum(
                value >= graph_config.service_level
                for value in graph_minima
            ) / len(graph_minima)
        ),
    }


def _aggregate(rows):
    result = []
    rate_fields = [
        name for name in rows[0]
        if name.endswith("_rate") or name == "mean_job_probability"
    ]
    for factor in FACTORS:
        selected = [row for row in rows if row["factor"] == factor]
        summary = {
            "factor": factor,
            "instance_name": "overall",
            "candidate_pool": sum(row["candidate_pool"] for row in selected),
            "selected_graphs": sum(
                row["selected_graphs"] for row in selected
            ),
            "job_labels": sum(row["job_labels"] for row in selected),
        }
        for field in rate_fields:
            summary[field] = statistics.fmean(
                row[field] for row in selected
            )
        result.append(summary)
    return result


def run_analysis():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    generation = ENTRY._generation_from_project_config(config)
    simulation_values = {
        **dict(generation.get("simulation") or {}),
        "pilot_replications": 256,
        "label_replications": 1000,
    }
    graph_config = GENERATOR.normalize_reliability_graph_config(
        generation["reliability_graph"]
    )
    simulation_config = SIMULATION.normalize_simulation_config(
        simulation_values
    )
    rows = []
    started = time.perf_counter()
    for position, instance_name in enumerate(INSTANCE_NAMES, start=1):
        print(
            f"[Due-factor] candidate pool {position}/{len(INSTANCE_NAMES)} "
            f"{instance_name}",
            flush=True,
        )
        instance = GENERATOR.load_generated_instance(instance_name)
        candidates = GENERATOR._collect_instance_candidates(
            instance,
            instance_name,
            generation,
            graph_config,
            generation["samples_per_instance"],
            hybrid_selection=None,
            candidate_generation_mode="nonlinear_evaluated",
            minimum_runs_override=4,
            maximum_runs_override=4,
        )
        horizon = _serial_horizon(instance)
        for factor in FACTORS:
            print(
                f"[Due-factor] {instance_name} factor={factor:.3f}",
                flush=True,
            )
            rows.append(_evaluate_factor(
                candidates,
                instance_name,
                {job: factor * horizon for job in instance.jobs},
                generation,
                graph_config,
                simulation_config,
                factor,
            ))
    summaries = _aggregate(rows)
    output_rows = rows + summaries
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        json.dumps({
            "wall_seconds": time.perf_counter() - started,
            "output": str(OUTPUT_PATH),
            "overall": summaries,
        }, indent=2),
        flush=True,
    )
    return rows, summaries


if __name__ == "__main__":
    run_analysis()
