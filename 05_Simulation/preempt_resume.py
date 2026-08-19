"""Event-based Weibull breakdown simulation for fixed FJSP schedules.

The optimizer supplies a fixed assignment and a fixed immediate-predecessor
graph.  This module executes that predictive schedule repeatedly.  Failures
interrupt an operation, an exponentially distributed repair is performed, and
the remaining processing time resumes on the same machine.  Planned idle time
is preserved and all disruption delays propagate through the fixed job and
machine arcs (right shift).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np


LABEL_METHOD = "monte_carlo_weibull_preempt_resume_right_shift_v1"


@dataclass(frozen=True)
class SimulationConfig:
    pilot_replications: int = 64
    label_replications: int = 512
    random_seed: int = 42
    failure_clock: str = "productive_time"
    repair_restoration: str = "as_good_as_new"
    interruption_policy: str = "preempt_resume"
    schedule_policy: str = "right_shift"
    initial_virtual_age_fraction: float = 0.0
    max_failures_per_operation: int = 10_000


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
    if str(result.failure_clock).lower() != "productive_time":
        raise ValueError("Only failure_clock='productive_time' is supported.")
    if str(result.repair_restoration).lower() != "as_good_as_new":
        raise ValueError("Only repair_restoration='as_good_as_new' is supported.")
    if str(result.interruption_policy).lower() != "preempt_resume":
        raise ValueError("Only interruption_policy='preempt_resume' is supported.")
    if str(result.schedule_policy).lower() != "right_shift":
        raise ValueError("Only schedule_policy='right_shift' is supported.")
    if float(result.initial_virtual_age_fraction) < 0.0:
        raise ValueError("initial_virtual_age_fraction must be nonnegative.")
    if int(result.max_failures_per_operation) <= 0:
        raise ValueError("max_failures_per_operation must be positive.")
    return SimulationConfig(
        pilot_replications=int(result.pilot_replications),
        label_replications=int(result.label_replications),
        random_seed=int(result.random_seed),
        failure_clock="productive_time",
        repair_restoration="as_good_as_new",
        interruption_policy="preempt_resume",
        schedule_policy="right_shift",
        initial_virtual_age_fraction=float(result.initial_virtual_age_fraction),
        max_failures_per_operation=int(result.max_failures_per_operation),
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


def _remaining_weibull_life(rng, age, scale, shape):
    uniform = max(float(rng.random()), np.finfo(float).tiny)
    accumulated_hazard = (age / scale) ** shape - math.log(uniform)
    return max(0.0, scale * accumulated_hazard ** (1.0 / shape) - age)


def _execute_operation(
    rng,
    processing_time,
    virtual_age,
    weibull_scale,
    weibull_shape,
    repair_rate,
    max_failures,
):
    remaining = float(processing_time)
    elapsed = 0.0
    repair_delay = 0.0
    repair_sum = 0.0
    failures = 0
    tolerance = 1e-12
    while remaining > tolerance:
        life = _remaining_weibull_life(
            rng, virtual_age, weibull_scale, weibull_shape
        )
        if life >= remaining - tolerance:
            elapsed += remaining
            virtual_age += remaining
            remaining = 0.0
            break
        productive_slice = max(0.0, life)
        elapsed += productive_slice
        remaining -= productive_slice
        virtual_age += productive_slice
        repair = float(rng.exponential(1.0 / repair_rate))
        elapsed += repair
        repair_delay += repair
        repair_sum += repair
        failures += 1
        virtual_age = 0.0
        if failures > max_failures:
            raise RuntimeError(
                "Simulation exceeded max_failures_per_operation; check the "
                "Weibull scale and processing-time units."
            )
    return elapsed, virtual_age, failures, repair_delay, repair_sum


def simulate_fixed_schedule(
    schedule: FixedSchedule,
    *,
    replications: int,
    seed: int,
    config=None,
) -> SimulationResult:
    cfg = normalize_simulation_config(config)
    replications = int(replications)
    if replications <= 0:
        raise ValueError("replications must be positive.")
    topology, predecessors = _validated_topology(schedule)
    operations = tuple(schedule.operations)
    operation_index = {operation: index for index, operation in enumerate(operations)}
    job_ids = tuple(sorted(schedule.jobs))
    job_index = {job: index for index, job in enumerate(job_ids)}
    machines = sorted(set(schedule.selected_machines.values()))
    for operation in operations:
        machine = schedule.selected_machines[operation]
        if float(schedule.processing_times[operation]) <= 0.0:
            raise ValueError("Processing times must be positive.")
        if float(schedule.weibull_scale[machine]) <= 0.0:
            raise ValueError("Weibull scales must be positive.")
        if float(schedule.weibull_shape[machine]) <= 1.0:
            raise ValueError("Weibull shapes must exceed one.")
        if float(schedule.repair_rate[machine]) <= 0.0:
            raise ValueError("Repair rates must be positive.")

    root = np.random.SeedSequence(int(seed))
    replication_seeds = root.spawn(replications)
    ontime = np.zeros(len(job_ids), dtype=np.int64)
    completion_sum = np.zeros(len(job_ids), dtype=float)
    operation_failure_runs = np.zeros(len(operations), dtype=np.int64)
    operation_failure_counts = np.zeros(len(operations), dtype=np.int64)
    operation_repair_delay = np.zeros(len(operations), dtype=float)
    operation_repair_sum = np.zeros(len(operations), dtype=float)
    total_failures = 0

    for replication_seed in replication_seeds:
        rng = np.random.default_rng(replication_seed)
        virtual_age = {
            machine: cfg.initial_virtual_age_fraction
            * float(schedule.weibull_scale[machine])
            for machine in machines
        }
        completion = {}
        for operation in topology:
            machine = schedule.selected_machines[operation]
            start = max(
                float(schedule.planned_starts[operation]),
                *(completion[source] for source in predecessors[operation]),
            ) if predecessors[operation] else float(
                schedule.planned_starts[operation]
            )
            (
                elapsed,
                virtual_age[machine],
                failures,
                repair_delay,
                repair_sum,
            ) = _execute_operation(
                rng,
                schedule.processing_times[operation],
                virtual_age[machine],
                float(schedule.weibull_scale[machine]),
                float(schedule.weibull_shape[machine]),
                float(schedule.repair_rate[machine]),
                cfg.max_failures_per_operation,
            )
            completion[operation] = start + elapsed
            index = operation_index[operation]
            operation_failure_runs[index] += int(failures > 0)
            operation_failure_counts[index] += failures
            operation_repair_delay[index] += repair_delay
            operation_repair_sum[index] += repair_sum
            total_failures += failures

        for job in job_ids:
            value = completion[schedule.job_end_operations[job]]
            index = job_index[job]
            completion_sum[index] += value
            ontime[index] += int(value <= float(schedule.due_dates[job]) + 1e-12)

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
        mean_total_repair_delay=float(operation_repair_delay.sum() / replications),
        mean_failures=float(total_failures / replications),
    )

