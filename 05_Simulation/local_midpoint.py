"""Monte Carlo check of the local buffer surrogate, without delay propagation.

Completion fields contain nominal job completion plus summed midpoint residual
repairs. They do not represent a right-shifted, feasible execution schedule.
Training labels use deterministic quadrature in helper.local_buffer instead.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .preempt_resume import FixedSchedule, SimulationResult
from helper.local_buffer import validate_schedule

LABEL_METHOD = "monte_carlo_local_midpoint_residual_repair_v1"


@dataclass(frozen=True)
class SimulationConfig:
    pilot_replications: int = 256
    label_replications: int = 10_000
    random_seed: int = 42
    model: str = "local_midpoint"


def normalize_simulation_config(config=None, **overrides):
    values = asdict(config) if isinstance(config, SimulationConfig) else dict(config or {})
    values.update(overrides)
    result = SimulationConfig(**values)
    if result.model != "local_midpoint":
        raise ValueError("This pipeline evaluates model='local_midpoint'.")
    if int(result.pilot_replications) <= 0 or int(result.label_replications) <= 0:
        raise ValueError("Replication counts must be positive.")
    return result


def simulation_config_dict(config=None):
    return asdict(normalize_simulation_config(config))


def simulate_fixed_schedule(schedule, *, replications, seed, config=None):
    normalize_simulation_config(config)
    validate_schedule(schedule)
    n = int(replications)
    if n <= 0:
        raise ValueError("replications must be positive.")
    machines = sorted(schedule.weibull_scale, key=repr)
    streams = np.random.SeedSequence(int(seed)).spawn(len(machines))
    downtime = {}
    for machine, stream in zip(machines, streams):
        a, b, rate = (float(mapping[machine]) for mapping in
                      (schedule.weibull_scale, schedule.weibull_shape, schedule.repair_rate))
        if not all(map(math.isfinite, (a, b, rate))) or a <= 0 or b <= 1 or rate <= 0:
            raise ValueError("Invalid machine failure/repair parameters.")
        rng = np.random.default_rng(stream)
        downtime[machine] = (a * rng.weibull(b, n), rng.exponential(1 / rate, n))
    residuals, probabilities, durations = {}, [], []
    used, failures, repair_duration = set(), np.zeros(n), np.zeros(n)
    for operation in schedule.operations:
        machine = schedule.selected_machines[operation]
        failure, repair = downtime[machine]
        midpoint = max(0., schedule.planned_starts[operation]) + .5 * schedule.processing_times[operation]
        residual = np.where(failure <= midpoint, np.maximum(failure + repair - midpoint, 0.), 0.)
        residuals[operation] = residual
        probabilities.append(float(np.mean(residual > 0)))
        durations.append(float(np.mean(np.where(residual > 0, repair, 0.))))
        used.add(machine)
    # Count each machine failure once up to its final nominal midpoint.
    for machine in used:
        last = max(schedule.planned_starts[o] + .5 * schedule.processing_times[o]
                   for o in schedule.operations if schedule.selected_machines[o] == machine)
        failure, repair = downtime[machine]
        failures += failure <= last
        repair_duration += np.where(failure <= last, repair, 0.)
    jobs = tuple(sorted(schedule.jobs))
    buffers = np.array([sum((residuals[o] for o in schedule.jobs[j]), np.zeros(n)) for j in jobs])
    completion = np.array([schedule.planned_starts[schedule.job_end_operations[j]]
                           + schedule.processing_times[schedule.job_end_operations[j]] for j in jobs])
    probability = np.mean(completion[:, None] + buffers <= np.array([schedule.due_dates[j] for j in jobs])[:, None], axis=1)
    se = np.std(buffers, axis=1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(len(jobs))
    means = buffers.mean(axis=1)
    completion_samples = (completion[:, None] + buffers).T
    due_dates = np.array([schedule.due_dates[j] for j in jobs])
    completion_delays = completion_samples - completion[None, :]
    all_jobs_ontime = np.all(completion_samples <= due_dates[None, :], axis=1)
    joint_probability = float(np.mean(all_jobs_ontime))
    makespans = np.max(completion_samples, axis=1)
    nominal_makespan = float(np.max(completion))
    total_tardiness = np.maximum(
        0.0, completion_samples - due_dates[None, :]
    ).sum(axis=1)
    job_tardiness = np.maximum(
        0.0, completion_samples - due_dates[None, :]
    )
    zero_operation_values = tuple(0.0 for _ in schedule.operations)
    return SimulationResult(
        replications=n, job_ids=jobs,
        job_ontime_probabilities=tuple(probability),
        job_probability_standard_errors=tuple(np.sqrt(probability * (1 - probability) / n)),
        job_mean_completion_times=tuple(completion + means),
        job_mean_completion_delays=tuple(means),
        job_completion_delay_p90=tuple(
            float(value) for value in np.quantile(completion_delays, .90, axis=0)
        ),
        job_completion_delay_p95=tuple(
            float(value) for value in np.quantile(completion_delays, .95, axis=0)
        ),
        job_mean_tardiness=tuple(
            float(value) for value in np.mean(job_tardiness, axis=0)
        ),
        job_tardiness_p90=tuple(
            float(value) for value in np.quantile(job_tardiness, .90, axis=0)
        ),
        job_tardiness_p95=tuple(
            float(value) for value in np.quantile(job_tardiness, .95, axis=0)
        ),
        job_completion_delay_standard_errors=tuple(se),
        operation_failure_probabilities=tuple(probabilities),
        operation_mean_repair_delays=tuple(float(residuals[o].mean()) for o in schedule.operations),
        operation_mean_repair_durations=tuple(durations),
        operation_mean_start_times=tuple(
            float(schedule.planned_starts[o]) for o in schedule.operations
        ),
        operation_mean_start_shifts=zero_operation_values,
        operation_start_shift_p90=zero_operation_values,
        operation_start_shift_p95=zero_operation_values,
        operation_start_shift_probabilities=zero_operation_values,
        operation_maximum_start_shifts=zero_operation_values,
        all_jobs_ontime_probability=joint_probability,
        all_jobs_ontime_standard_error=float(
            np.sqrt(joint_probability * (1. - joint_probability) / n)
        ),
        mean_simulated_makespan=float(np.mean(makespans)),
        simulated_makespan_p90=float(np.quantile(makespans, .90)),
        simulated_makespan_p95=float(np.quantile(makespans, .95)),
        replication_makespans=tuple(float(value) for value in makespans),
        mean_makespan_increase=float(
            np.mean(np.maximum(0., makespans - nominal_makespan))
        ),
        mean_total_tardiness=float(np.mean(total_tardiness)),
        total_tardiness_p90=float(np.quantile(total_tardiness, .90)),
        total_tardiness_p95=float(np.quantile(total_tardiness, .95)),
        replication_total_tardiness=tuple(
            float(value) for value in total_tardiness
        ),
        mean_operation_start_shift=0.0,
        operation_start_shift_p90_overall=0.0,
        operation_start_shift_p95_overall=0.0,
        maximum_mean_operation_start_shift=0.0,
        mean_total_repair_delay=float(means.sum()),
        mean_total_repair_duration=float(repair_duration.mean()),
        mean_failures=float(failures.mean()), label_method=LABEL_METHOD,
    )
