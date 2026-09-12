from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np

TARGET_COLUMN = "expected_local_midpoint_repair_buffer"
JOB_TARGET = "job_local_midpoint_repair_buffer"
LABEL_METHOD = "local_midpoint_expectation_quadrature_checked_v1"
LABEL_SOURCE = "deterministic_local_midpoint_residual_repair_expectation"


@dataclass(frozen=True)
class LocalBufferConfig:
    method: str = LABEL_METHOD
    quadrature_points: int = 128
    verification_points: int = 256
    absolute_tolerance: float = 1e-6


def normalize_label_config(config=None):
    values = asdict(config) if isinstance(config, LocalBufferConfig) else dict(config or {})
    result = LocalBufferConfig(**values)
    if result.method != LABEL_METHOD:
        raise ValueError(f"Local buffer labels require method={LABEL_METHOD!r}.")
    if not (4 <= int(result.quadrature_points) < int(result.verification_points)):
        raise ValueError("Require 4 <= quadrature_points < verification_points.")
    if not math.isfinite(result.absolute_tolerance) or result.absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be finite and positive.")
    return result


def label_config_dict(config=None):
    return asdict(normalize_label_config(config))


class InvalidScheduleError(ValueError):
    """A candidate must not be labelled if its nominal schedule is infeasible."""


def validate_schedule(schedule, tolerance=1e-4):
    operations = set(schedule.operations)
    members = [o for chain in schedule.jobs.values() for o in chain]
    if len(members) != len(operations) or set(members) != operations:
        raise InvalidScheduleError("Jobs must partition the operation set.")
    ends = {}
    machines = {}
    for o in schedule.operations:
        start, duration = float(schedule.planned_starts[o]), float(schedule.processing_times[o])
        if not math.isfinite(start) or not math.isfinite(duration) or start < -tolerance or duration <= 0:
            raise InvalidScheduleError("Invalid nominal start or duration.")
        ends[o] = start + duration
        machines.setdefault(schedule.selected_machines[o], []).append(o)
    for o in operations:
        for previous in schedule.job_predecessors.get(o, ()):
            if previous not in operations or ends[previous] > schedule.planned_starts[o] + tolerance:
                raise InvalidScheduleError("Nominal job precedence is violated.")
    for chain in machines.values():
        chain.sort(key=lambda o: float(schedule.planned_starts[o]))
        if any(ends[a] > schedule.planned_starts[b] + tolerance for a, b in zip(chain, chain[1:])):
            raise InvalidScheduleError("Nominal machine operations overlap.")
    for a, b, m in schedule.machine_edges:
        if (a not in operations or b not in operations
                or schedule.selected_machines[a] != m or schedule.selected_machines[b] != m
                or ends[a] > schedule.planned_starts[b] + tolerance):
            raise InvalidScheduleError("Invalid machine predecessor edge.")


@lru_cache(maxsize=16)
def _rule(order):
    return np.polynomial.legendre.leggauss(int(order))


def operation_expectation(midpoint, alpha, beta, repair_rate, order=128):
    t, a, b, rate = map(float, (midpoint, alpha, beta, repair_rate))
    if not all(map(math.isfinite, (t, a, b, rate))) or t < 0 or a <= 0 or b <= 1 or rate <= 0:
        raise ValueError("Require finite t>=0, alpha>0, beta>1 and repair_rate>0.")
    nodes, weights = _rule(order)
    x = .5 * t * (nodes + 1)
    density = (b / a) * (x / a) ** (b - 1) * np.exp(-(x / a) ** b)
    down = .5 * t * float(np.dot(weights, density * np.exp(-rate * (t - x))))
    return float(np.clip(down, 0., 1.)) / rate


def expected_job_buffers(schedule, config=None):
    """Return job expectations and the checked quadrature differences [ZE]."""
    cfg = normalize_label_config(config)
    validate_schedule(schedule)
    values, checks = {}, {}
    for o in schedule.operations:
        m = schedule.selected_machines[o]
        args = (max(0., float(schedule.planned_starts[o])) + .5 * schedule.processing_times[o],
                schedule.weibull_scale[m], schedule.weibull_shape[m], schedule.repair_rate[m])
        values[o] = operation_expectation(*args, order=cfg.quadrature_points)
        checks[o] = operation_expectation(*args, order=cfg.verification_points)
    jobs = {j: math.fsum(values[o] for o in chain) for j, chain in schedule.jobs.items()}
    errors = {j: math.fsum(abs(values[o] - checks[o]) for o in chain)
              for j, chain in schedule.jobs.items()}
    if max(errors.values(), default=0.) > cfg.absolute_tolerance:
        raise ValueError("Local buffer quadrature did not meet absolute_tolerance; increase its orders.")
    return jobs, errors


def service_lower_bounds(schedule, buffers):
    """Markov lower bounds for the local buffer model, not empirical service rates."""
    result = {}
    for j, o in schedule.job_end_operations.items():
        slack = schedule.due_dates[j] - schedule.planned_starts[o] - schedule.processing_times[o]
        result[j] = max(0., min(1., 1. - buffers[j] / slack)) if slack > 0 else 0.
    return result
