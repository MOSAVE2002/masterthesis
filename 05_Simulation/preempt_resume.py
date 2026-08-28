"""Constraint-consistent Monte-Carlo simulation for fixed FJSP schedules.

For every operation, the simulator evaluates the same nonlinear machine-down
probability at the nominal operation midpoint as the reference MINLP. A single
Bernoulli snapshot decides whether the operation is disrupted. Conditional on
a disruption, the remaining exponential repair duration is added to the
operation and propagated over all fixed job and machine arcs by right shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np

from helper.stochastic_fjsp import weibull_down_probability


LABEL_METHOD = (
    "monte_carlo_weibull_midpoint_snapshot_exponential_repair_"
    "preempt_resume_right_shift_v1"
)


@dataclass(frozen=True)
class SimulationConfig:
    pilot_replications: int = 256
    label_replications: int = 10_000
    random_seed: int = 42


@dataclass(frozen=True)
class FixedSchedule:
    operations: tuple[Hashable, ...]
    selected_machines: Mapping[Hashable, int]
    processing_times: Mapping[Hashable, float]
    planned_starts: Mapping[Hashable, float]
    job_predecessors: Mapping[Hashable, tuple[Hashable, ...]]
    machine_edges: tuple[tuple[Hashable, Hashable, int], ...]
    jobs: Mapping[Hashable, tuple[Hashable, ...]]
    job_end_operations: Mapping[Hashable, Hashable]
    due_dates: Mapping[Hashable, float]
    weibull_scale: Mapping[int, float]
    weibull_shape: Mapping[int, float]
    repair_rate: Mapping[int, float]


@dataclass(frozen=True)
class SimulationResult:
    replications: int
    job_ids: tuple[Hashable, ...]
    job_ontime_probabilities: tuple[float, ...]
    job_probability_standard_errors: tuple[float, ...]
    job_mean_completion_times: tuple[float, ...]
    operation_failure_probabilities: tuple[float, ...]
    operation_mean_repair_delays: tuple[float, ...]
    operation_mean_repair_durations: tuple[float, ...]
    mean_total_repair_delay: float
    mean_total_repair_duration: float
    mean_failures: float
    label_method: str = LABEL_METHOD


def normalize_simulation_config(config=None, **overrides) -> SimulationConfig:
    if isinstance(config, SimulationConfig):
        values = dict(config.__dict__)
    else:
        values = dict(config or {})
    values.update(overrides)
    allowed = set(SimulationConfig.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown simulation parameters: {sorted(unknown)}")
    result = SimulationConfig(**values)
    if int(result.pilot_replications) <= 0:
        raise ValueError("pilot_replications must be positive.")
    if int(result.label_replications) <= 0:
        raise ValueError("label_replications must be positive.")
    return SimulationConfig(
        pilot_replications=int(result.pilot_replications),
        label_replications=int(result.label_replications),
        random_seed=int(result.random_seed),
    )


def simulation_config_dict(config=None) -> dict:
    return dict(normalize_simulation_config(config).__dict__)


def _validated_topology(schedule: FixedSchedule):
    operations = list(schedule.operations)
    operation_set = set(operations)
    order = {operation: index for index, operation in enumerate(operations)}
    predecessors = {
        operation: set(schedule.job_predecessors.get(operation, ()))
        for operation in operations
    }
    machine_predecessor = {}
    machine_successor = {}
    for source, target, machine in schedule.machine_edges:
        if source not in operation_set or target not in operation_set:
            raise ValueError("Machine edge references an unknown operation.")
        if schedule.selected_machines[source] != machine:
            raise ValueError("Machine edge source is assigned to another machine.")
        if schedule.selected_machines[target] != machine:
            raise ValueError("Machine edge target is assigned to another machine.")
        if target in machine_predecessor:
            raise ValueError("An operation has multiple immediate machine predecessors.")
        if source in machine_successor:
            raise ValueError("An operation has multiple immediate machine successors.")
        machine_predecessor[target] = source
        machine_successor[source] = target
        predecessors[target].add(source)

    for operation, required in predecessors.items():
        if not required <= operation_set:
            raise ValueError(f"Unknown predecessor for operation {operation!r}.")
    successors = {operation: [] for operation in operations}
    indegree = {}
    for operation, required in predecessors.items():
        indegree[operation] = len(required)
        for source in required:
            successors[source].append(operation)
    available = sorted(
        (operation for operation in operations if indegree[operation] == 0),
        key=order.__getitem__,
    )
    topology = []
    while available:
        operation = available.pop(0)
        topology.append(operation)
        for target in sorted(successors[operation], key=order.__getitem__):
            indegree[target] -= 1
            if indegree[target] == 0:
                available.append(target)
                available.sort(key=order.__getitem__)
    if len(topology) != len(operations):
        raise ValueError("Combined job/machine predecessor graph contains a cycle.")
    return tuple(topology), {
        operation: tuple(sorted(required, key=order.__getitem__))
        for operation, required in predecessors.items()
    }


def _midpoint_disruption(rng, probability, repair_rate):
    """Draw one midpoint disruption and its remaining repair duration."""
    if float(rng.random()) >= float(probability):
        return 0, 0.0
    return 1, float(rng.exponential(1.0 / float(repair_rate)))


def simulate_fixed_schedule(
    schedule: FixedSchedule,
    *,
    replications: int,
    seed: int,
    config=None,
) -> SimulationResult:
    """Evaluate one fixed schedule under midpoint-snapshot disruptions."""
    normalize_simulation_config(config)
    replications = int(replications)
    if replications <= 0:
        raise ValueError("replications must be positive.")
    topology, predecessors = _validated_topology(schedule)
    operations = tuple(schedule.operations)
    operation_index = {
        operation: index for index, operation in enumerate(operations)
    }
    stable_operation_order = sorted(operations, key=repr)
    operation_stream = {
        operation: index
        for index, operation in enumerate(stable_operation_order)
    }
    job_ids = tuple(sorted(schedule.jobs))
    job_index = {job: index for index, job in enumerate(job_ids)}

    disruption_probabilities = {}
    for operation in operations:
        machine = schedule.selected_machines[operation]
        processing_time = float(schedule.processing_times[operation])
        planned_start = float(schedule.planned_starts[operation])
        if processing_time <= 0.0:
            raise ValueError("Processing times must be positive.")
        if planned_start < 0.0:
            raise ValueError("Planned start times must be nonnegative.")
        if float(schedule.weibull_scale[machine]) <= 0.0:
            raise ValueError("Weibull scales must be positive.")
        if float(schedule.weibull_shape[machine]) <= 1.0:
            raise ValueError("Weibull shapes must exceed one.")
        if float(schedule.repair_rate[machine]) <= 0.0:
            raise ValueError("Repair rates must be positive.")
        midpoint = planned_start + 0.5 * processing_time
        disruption_probabilities[operation] = weibull_down_probability(
            midpoint,
            schedule.weibull_scale[machine],
            schedule.weibull_shape[machine],
            schedule.repair_rate[machine],
        )

    ontime = np.zeros(len(job_ids), dtype=np.int64)
    completion_sum = np.zeros(len(job_ids), dtype=float)
    operation_failure_runs = np.zeros(len(operations), dtype=np.int64)
    operation_failure_counts = np.zeros(len(operations), dtype=np.int64)
    operation_repair_delay = np.zeros(len(operations), dtype=float)
    operation_repair_sum = np.zeros(len(operations), dtype=float)
    total_failures = 0
    total_repair_delay = 0.0

    for replication_index in range(replications):
        completion = {}
        for operation in topology:
            machine = schedule.selected_machines[operation]
            index = operation_index[operation]
            rng = np.random.default_rng(np.random.SeedSequence([
                int(seed),
                int(replication_index),
                int(operation_stream[operation]),
            ]))
            failed, repair_delay = _midpoint_disruption(
                rng,
                disruption_probabilities[operation],
                schedule.repair_rate[machine],
            )
            start = (
                max(
                    float(schedule.planned_starts[operation]),
                    *(completion[source] for source in predecessors[operation]),
                )
                if predecessors[operation]
                else float(schedule.planned_starts[operation])
            )
            completion[operation] = (
                start
                + float(schedule.processing_times[operation])
                + repair_delay
            )
            operation_failure_runs[index] += failed
            operation_failure_counts[index] += failed
            operation_repair_delay[index] += repair_delay
            operation_repair_sum[index] += repair_delay
            total_failures += failed
            total_repair_delay += repair_delay

        for job in job_ids:
            value = completion[schedule.job_end_operations[job]]
            index = job_index[job]
            completion_sum[index] += value
            ontime[index] += int(
                value <= float(schedule.due_dates[job]) + 1e-12
            )

    probabilities = ontime.astype(float) / replications
    standard_errors = np.sqrt(
        probabilities * (1.0 - probabilities) / replications
    )
    failure_counts_safe = np.maximum(operation_failure_counts, 1)
    return SimulationResult(
        replications=replications,
        job_ids=job_ids,
        job_ontime_probabilities=tuple(float(value) for value in probabilities),
        job_probability_standard_errors=tuple(
            float(value) for value in standard_errors
        ),
        job_mean_completion_times=tuple(
            float(value / replications) for value in completion_sum
        ),
        operation_failure_probabilities=tuple(
            float(value / replications) for value in operation_failure_runs
        ),
        operation_mean_repair_delays=tuple(
            float(value / replications) for value in operation_repair_delay
        ),
        operation_mean_repair_durations=tuple(
            float(total / count)
            for total, count in zip(operation_repair_sum, failure_counts_safe)
        ),
        mean_total_repair_delay=float(total_repair_delay / replications),
        mean_total_repair_duration=float(total_repair_delay / replications),
        mean_failures=float(total_failures / replications),
        label_method=LABEL_METHOD,
    )
