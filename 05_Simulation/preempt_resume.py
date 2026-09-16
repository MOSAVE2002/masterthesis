"""Machine-level single-failure simulation for fixed FJSP schedules.

Each replication draws one Weibull failure time and one exponential repair
duration per used machine.  The resulting common downtime interval affects all
operations assigned to that machine consistently.  Operations wait when they
would start during the downtime and use preempt-resume when the failure occurs
during processing.  Delays propagate over fixed job and machine arcs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Mapping

import numpy as np


LABEL_METHOD = (
    "monte_carlo_machine_single_weibull_failure_exponential_repair_"
    "preempt_resume_right_shift_v2"
)
EXPECTED_COMPLETION_DELAY_LABEL_METHOD = (
    f"{LABEL_METHOD}_job_mean_completion_delay_v2"
)


@dataclass(frozen=True)
class SimulationConfig:
    pilot_replications: int = 256
    label_replications: int = 10_000
    random_seed: int = 42
    model: str = "preempt_resume"


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
    job_mean_completion_delays: tuple[float, ...]
    job_completion_delay_p90: tuple[float, ...]
    job_completion_delay_p95: tuple[float, ...]
    job_mean_tardiness: tuple[float, ...]
    job_tardiness_p90: tuple[float, ...]
    job_tardiness_p95: tuple[float, ...]
    job_completion_delay_standard_errors: tuple[float, ...]
    operation_failure_probabilities: tuple[float, ...]
    operation_mean_repair_delays: tuple[float, ...]
    operation_mean_repair_durations: tuple[float, ...]
    operation_mean_start_times: tuple[float, ...]
    operation_mean_start_shifts: tuple[float, ...]
    operation_start_shift_p90: tuple[float, ...]
    operation_start_shift_p95: tuple[float, ...]
    operation_start_shift_probabilities: tuple[float, ...]
    operation_maximum_start_shifts: tuple[float, ...]
    all_jobs_ontime_probability: float
    all_jobs_ontime_standard_error: float
    mean_simulated_makespan: float
    simulated_makespan_p90: float
    simulated_makespan_p95: float
    replication_makespans: tuple[float, ...]
    mean_makespan_increase: float
    mean_total_tardiness: float
    total_tardiness_p90: float
    total_tardiness_p95: float
    replication_total_tardiness: tuple[float, ...]
    mean_operation_start_shift: float
    operation_start_shift_p90_overall: float
    operation_start_shift_p95_overall: float
    maximum_mean_operation_start_shift: float
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
    if result.model != "preempt_resume":
        raise ValueError("Execution evaluation requires model='preempt_resume'.")
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


def _draw_machine_downtime(rng, weibull_scale, weibull_shape, repair_rate):
    """Draw the common first-failure time and repair duration of a machine."""
    failure_time = float(
        weibull_scale * rng.weibull(float(weibull_shape))
    )
    repair_duration = float(rng.exponential(1.0 / float(repair_rate)))
    return failure_time, repair_duration


def _preempt_resume_completion(
    start,
    processing_time,
    failure_time,
    repair_duration,
):
    """Return completion and direct delay under one machine downtime."""
    start = float(start)
    processing_time = float(processing_time)
    failure_time = float(failure_time)
    repair_end = failure_time + float(repair_duration)
    nominal_completion = start + processing_time

    if repair_end <= start or failure_time >= nominal_completion:
        return nominal_completion, 0.0
    if failure_time <= start < repair_end:
        direct_delay = repair_end - start
    else:
        # The failure occurs during processing. Work completed before the
        # failure is retained and processing resumes after the full repair.
        direct_delay = float(repair_duration)
    return nominal_completion + direct_delay, direct_delay


def simulate_fixed_schedule(
    schedule: FixedSchedule,
    *,
    replications: int,
    seed: int,
    config=None,
) -> SimulationResult:
    """Evaluate one fixed schedule under common machine-level downtimes."""
    normalize_simulation_config(config)
    replications = int(replications)
    if replications <= 0:
        raise ValueError("replications must be positive.")
    topology, predecessors = _validated_topology(schedule)
    operations = tuple(schedule.operations)
    operation_index = {
        operation: index for index, operation in enumerate(operations)
    }
    machines = tuple(sorted(
        {schedule.selected_machines[operation] for operation in operations},
        key=repr,
    ))
    all_machines = tuple(sorted(schedule.weibull_scale, key=repr))
    machine_stream = {
        machine: index for index, machine in enumerate(all_machines)
    }
    job_ids = tuple(sorted(schedule.jobs))
    job_index = {job: index for index, job in enumerate(job_ids)}

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

    ontime = np.zeros(len(job_ids), dtype=np.int64)
    completion_sum = np.zeros(len(job_ids), dtype=float)
    completion_square_sum = np.zeros(len(job_ids), dtype=float)
    job_completion_samples = np.empty(
        (replications, len(job_ids)), dtype=float
    )
    operation_start_shift_samples = np.empty(
        (replications, len(operations)), dtype=float
    )
    operation_failure_runs = np.zeros(len(operations), dtype=np.int64)
    operation_failure_counts = np.zeros(len(operations), dtype=np.int64)
    operation_repair_delay = np.zeros(len(operations), dtype=float)
    operation_repair_sum = np.zeros(len(operations), dtype=float)
    total_failures = 0
    total_repair_delay = 0.0
    total_repair_duration = 0.0

    for replication_index in range(replications):
        downtime = {}
        for machine in machines:
            rng = np.random.default_rng(np.random.SeedSequence([
                int(seed),
                int(replication_index),
                int(machine_stream[machine]),
            ]))
            downtime[machine] = _draw_machine_downtime(
                rng,
                float(schedule.weibull_scale[machine]),
                float(schedule.weibull_shape[machine]),
                float(schedule.repair_rate[machine]),
            )
        completion = {}
        affected_machines = set()
        for operation in topology:
            machine = schedule.selected_machines[operation]
            index = operation_index[operation]
            start = (
                max(
                    float(schedule.planned_starts[operation]),
                    *(completion[source] for source in predecessors[operation]),
                )
                if predecessors[operation]
                else float(schedule.planned_starts[operation])
            )
            operation_start_shift_samples[replication_index, index] = max(
                0.0, start - float(schedule.planned_starts[operation])
            )
            failure_time, repair_duration = downtime[machine]
            operation_completion, repair_delay = _preempt_resume_completion(
                start,
                float(schedule.processing_times[operation]),
                failure_time,
                repair_duration,
            )
            completion[operation] = operation_completion
            affected = int(repair_delay > 0.0)
            operation_failure_runs[index] += affected
            operation_failure_counts[index] += affected
            operation_repair_delay[index] += repair_delay
            operation_repair_sum[index] += affected * repair_duration
            total_repair_delay += repair_delay
            if affected:
                affected_machines.add(machine)

        total_failures += len(affected_machines)
        total_repair_duration += sum(
            downtime[machine][1] for machine in affected_machines
        )

        for job in job_ids:
            value = completion[schedule.job_end_operations[job]]
            index = job_index[job]
            completion_sum[index] += value
            completion_square_sum[index] += value * value
            job_completion_samples[replication_index, index] = value
            ontime[index] += int(
                value <= float(schedule.due_dates[job]) + 1e-12
            )

    probabilities = ontime.astype(float) / replications
    standard_errors = np.sqrt(
        probabilities * (1.0 - probabilities) / replications
    )
    failure_counts_safe = np.maximum(operation_failure_counts, 1)
    mean_completions = completion_sum / replications
    if replications > 1:
        completion_variances = np.maximum(
            0.0,
            (
                completion_square_sum
                - replications * mean_completions * mean_completions
            ) / (replications - 1),
        )
        completion_standard_errors = np.sqrt(
            completion_variances / replications
        )
    else:
        completion_standard_errors = np.zeros(len(job_ids), dtype=float)
    nominal_completions = np.asarray([
        float(schedule.planned_starts[schedule.job_end_operations[job]])
        + float(schedule.processing_times[schedule.job_end_operations[job]])
        for job in job_ids
    ])
    completion_delay_samples = np.maximum(
        0.0, job_completion_samples - nominal_completions[None, :]
    )
    due_dates = np.asarray([
        float(schedule.due_dates[job]) for job in job_ids
    ])
    all_jobs_ontime = np.all(
        job_completion_samples <= due_dates[None, :] + 1e-12, axis=1
    )
    all_jobs_ontime_probability = float(np.mean(all_jobs_ontime))
    all_jobs_ontime_standard_error = math.sqrt(
        all_jobs_ontime_probability
        * (1.0 - all_jobs_ontime_probability)
        / replications
    )
    simulated_makespans = np.max(job_completion_samples, axis=1)
    nominal_makespan = float(np.max(nominal_completions))
    total_tardiness = np.maximum(
        0.0, job_completion_samples - due_dates[None, :]
    ).sum(axis=1)
    job_tardiness = np.maximum(
        0.0, job_completion_samples - due_dates[None, :]
    )
    mean_operation_start_shifts = operation_start_shift_samples.mean(axis=0)
    flat_start_shifts = operation_start_shift_samples.reshape(-1)
    return SimulationResult(
        replications=replications,
        job_ids=job_ids,
        job_ontime_probabilities=tuple(float(value) for value in probabilities),
        job_probability_standard_errors=tuple(
            float(value) for value in standard_errors
        ),
        job_mean_completion_times=tuple(float(value) for value in mean_completions),
        job_mean_completion_delays=tuple(
            float(value)
            for value in np.maximum(0.0, mean_completions - nominal_completions)
        ),
        job_completion_delay_p90=tuple(
            float(value)
            for value in np.quantile(completion_delay_samples, 0.90, axis=0)
        ),
        job_completion_delay_p95=tuple(
            float(value)
            for value in np.quantile(completion_delay_samples, 0.95, axis=0)
        ),
        job_mean_tardiness=tuple(
            float(value) for value in np.mean(job_tardiness, axis=0)
        ),
        job_tardiness_p90=tuple(
            float(value) for value in np.quantile(job_tardiness, 0.90, axis=0)
        ),
        job_tardiness_p95=tuple(
            float(value) for value in np.quantile(job_tardiness, 0.95, axis=0)
        ),
        job_completion_delay_standard_errors=tuple(
            float(value) for value in completion_standard_errors
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
        operation_mean_start_times=tuple(
            float(schedule.planned_starts[operation] + mean_operation_start_shifts[index])
            for index, operation in enumerate(operations)
        ),
        operation_mean_start_shifts=tuple(
            float(value) for value in mean_operation_start_shifts
        ),
        operation_start_shift_p90=tuple(
            float(value)
            for value in np.quantile(operation_start_shift_samples, 0.90, axis=0)
        ),
        operation_start_shift_p95=tuple(
            float(value)
            for value in np.quantile(operation_start_shift_samples, 0.95, axis=0)
        ),
        operation_start_shift_probabilities=tuple(
            float(value)
            for value in np.mean(operation_start_shift_samples > 1e-12, axis=0)
        ),
        operation_maximum_start_shifts=tuple(
            float(value) for value in np.max(operation_start_shift_samples, axis=0)
        ),
        all_jobs_ontime_probability=all_jobs_ontime_probability,
        all_jobs_ontime_standard_error=all_jobs_ontime_standard_error,
        mean_simulated_makespan=float(np.mean(simulated_makespans)),
        simulated_makespan_p90=float(np.quantile(simulated_makespans, 0.90)),
        simulated_makespan_p95=float(np.quantile(simulated_makespans, 0.95)),
        replication_makespans=tuple(
            float(value) for value in simulated_makespans
        ),
        mean_makespan_increase=float(
            np.mean(np.maximum(0.0, simulated_makespans - nominal_makespan))
        ),
        mean_total_tardiness=float(np.mean(total_tardiness)),
        total_tardiness_p90=float(np.quantile(total_tardiness, 0.90)),
        total_tardiness_p95=float(np.quantile(total_tardiness, 0.95)),
        replication_total_tardiness=tuple(
            float(value) for value in total_tardiness
        ),
        mean_operation_start_shift=float(np.mean(flat_start_shifts)),
        operation_start_shift_p90_overall=float(
            np.quantile(flat_start_shifts, 0.90)
        ),
        operation_start_shift_p95_overall=float(
            np.quantile(flat_start_shifts, 0.95)
        ),
        maximum_mean_operation_start_shift=float(
            np.max(mean_operation_start_shifts)
        ),
        mean_total_repair_delay=float(total_repair_delay / replications),
        mean_total_repair_duration=float(
            total_repair_duration / replications
        ),
        mean_failures=float(total_failures / replications),
        label_method=LABEL_METHOD,
    )
