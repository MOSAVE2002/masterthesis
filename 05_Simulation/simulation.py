"""Thesis-consistent evaluation of fixed FJSP production plans.

The deterministic part computes local expected repair buffers by quadrature for
GNN labels.  The separate Monte Carlo part post-evaluates stored solver plans
with one Weibull failure and one exponential repair per machine and scenario,
preempt-resume processing, and delay propagation over fixed precedence arcs.
"""

import ast
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from itertools import pairwise
import json
import math
from pathlib import Path
import re
import sys
from typing import Hashable, Mapping

import numpy as np


TARGET_COLUMN = "expected_local_midpoint_repair_buffer"
JOB_TARGET = "job_local_midpoint_repair_buffer"
LABEL_METHOD = "local_midpoint_expectation_quadrature_checked_v1"
LABEL_SOURCE = "deterministic_local_midpoint_residual_repair_expectation"
MONTE_CARLO_METHOD = (
    "machine_single_weibull_failure_exponential_repair_"
    "preempt_resume_right_shift_v1"
)
DEFAULT_REPLICATIONS = 1_000
DEFAULT_RANDOM_SEED = 900_042
SOLUTION_DURATION_TOLERANCE = 1e-3
SOLUTION_RIGHT_SHIFT_TOLERANCE = 1e-2
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass(frozen=True)
class FixedSchedule:
    """Immutable nominal production plan used by both evaluation methods.

    The structure combines planned operation timing, fixed machine assignments,
    technological and machine precedence, job due dates and all stochastic
    machine parameters required for analytical labels and Monte Carlo draws.
    """

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
class LocalBufferConfig:
    """Numerical settings of the deterministic repair-buffer label integral.

    The primary and verification quadrature orders are evaluated independently;
    their job-level difference must remain below ``absolute_tolerance``.
    """
    method: str = LABEL_METHOD
    quadrature_points: int = 128
    verification_points: int = 256
    absolute_tolerance: float = 1e-6


LABEL_CONFIG = LocalBufferConfig()


def label_config_dict():
    """Return the immutable label settings as serializable metadata.

    Returns:
        Dictionary written to dataset summaries and trained-model metadata.
    """
    return asdict(LABEL_CONFIG)


class InvalidScheduleError(ValueError):
    """Signal that a nominal schedule is structurally or temporally infeasible.

    The dedicated exception lets batch evaluation skip invalid exported plans
    without hiding unrelated programming or I/O errors.
    """


def validate_schedule(schedule, tolerance=1e-4):
    """Validate the partition, timing and precedence of a fixed schedule.

    Args:
        schedule: :class:`FixedSchedule` to validate before evaluation.
        tolerance: Accepted floating-point slack for nominal timing comparisons.

    Raises:
        InvalidScheduleError: If jobs do not partition operations, timings are
            invalid, precedence is violated or assigned operations overlap.
    """
    operations = set(schedule.operations)
    members = [operation for chain in schedule.jobs.values() for operation in chain]
    if len(members) != len(operations) or set(members) != operations:
        raise InvalidScheduleError("Jobs must partition the operation set.")

    ends = {}
    machines = {}
    for operation in schedule.operations:
        start = float(schedule.planned_starts[operation])
        duration = float(schedule.processing_times[operation])
        if (
            not math.isfinite(start)
            or not math.isfinite(duration)
            or start < -tolerance
            or duration <= 0
        ):
            raise InvalidScheduleError("Invalid nominal start or duration.")
        ends[operation] = start + duration
        machines.setdefault(schedule.selected_machines[operation], []).append(
            operation
        )

    for operation in operations:
        for previous in schedule.job_predecessors.get(operation, ()):
            if (
                previous not in operations
                or ends[previous]
                > schedule.planned_starts[operation] + tolerance
            ):
                raise InvalidScheduleError("Nominal job precedence is violated.")

    for chain in machines.values():
        chain.sort(key=lambda operation: float(schedule.planned_starts[operation]))
        if any(
            ends[source] > schedule.planned_starts[target] + tolerance
            for source, target in pairwise(chain)
        ):
            raise InvalidScheduleError("Nominal machine operations overlap.")

    for source, target, machine in schedule.machine_edges:
        if (
            source not in operations
            or target not in operations
            or schedule.selected_machines[source] != machine
            or schedule.selected_machines[target] != machine
            or ends[source] > schedule.planned_starts[target] + tolerance
        ):
            raise InvalidScheduleError("Invalid machine predecessor edge.")


@lru_cache(maxsize=16)
def _rule(order):
    """Return and cache Gauss-Legendre nodes and weights for one order.

    Args:
        order: Positive quadrature order passed to NumPy.

    Returns:
        Pair of NumPy arrays containing integration nodes and weights.
    """
    return np.polynomial.legendre.leggauss(int(order))


def operation_expectation(midpoint, alpha, beta, repair_rate, order=128):
    """Compute expected residual repair time at one operation midpoint.

    Gauss-Legendre quadrature integrates the probability that a Weibull failure
    has occurred but its exponential repair is unfinished at the nominal
    midpoint. Dividing that probability by the repair rate yields the expected
    residual repair duration.

    Returns:
        Expected residual repair time in the scheduling time unit.
    """

    time, scale, shape, rate = map(
        float, (midpoint, alpha, beta, repair_rate)
    )
    if (
        not all(map(math.isfinite, (time, scale, shape, rate)))
        or time < 0
        or scale <= 0
        or shape <= 1
        or rate <= 0
    ):
        raise ValueError(
            "Require finite t>=0, alpha>0, beta>1 and repair_rate>0."
        )

    nodes, weights = _rule(order)
    integration_points = 0.5 * time * (nodes + 1)
    density = (
        (shape / scale)
        * (integration_points / scale) ** (shape - 1)
        * np.exp(-(integration_points / scale) ** shape)
    )
    down_probability = 0.5 * time * float(
        np.dot(
            weights,
            density * np.exp(-rate * (time - integration_points)),
        )
    )
    return float(np.clip(down_probability, 0.0, 1.0)) / rate


def expected_job_buffers(schedule):
    """Aggregate checked operation expectations into job repair buffers.

    The analytical expectation is evaluated at every nominal operation
    midpoint with both configured quadrature orders. Operation values are
    summed along each job and rejected if the accumulated numerical difference
    exceeds the configured tolerance.

    Returns:
        Two dictionaries containing job buffers and quadrature differences.
    """

    config = LABEL_CONFIG
    validate_schedule(schedule)
    values = {}
    checks = {}
    for operation in schedule.operations:
        machine = schedule.selected_machines[operation]
        arguments = (
            max(0.0, float(schedule.planned_starts[operation]))
            + 0.5 * schedule.processing_times[operation],
            schedule.weibull_scale[machine],
            schedule.weibull_shape[machine],
            schedule.repair_rate[machine],
        )
        values[operation] = operation_expectation(
            *arguments, order=config.quadrature_points
        )
        checks[operation] = operation_expectation(
            *arguments, order=config.verification_points
        )

    jobs = {
        job: math.fsum(values[operation] for operation in chain)
        for job, chain in schedule.jobs.items()
    }
    errors = {
        job: math.fsum(
            abs(values[operation] - checks[operation]) for operation in chain
        )
        for job, chain in schedule.jobs.items()
    }
    if max(errors.values(), default=0.0) > config.absolute_tolerance:
        raise ValueError(
            "Local buffer quadrature did not meet absolute_tolerance; "
            "increase its orders."
        )
    return jobs, errors


@dataclass(frozen=True)
class SimulationResult:
    """Immutable collection of Monte Carlo schedule statistics.

    The result retains job reliability estimates, completion and tardiness
    distributions, operation-level disruption measures, makespan samples and
    aggregate repair effects. Replication-level values required for economic
    post-evaluation are preserved alongside summarized statistics.
    """

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
    simulation_method: str = MONTE_CARLO_METHOD

    @property
    def mean_job_ontime_probability(self):
        """Return the arithmetic mean of all job on-time probabilities.

        This aggregate treats every job equally regardless of its duration.
        """
        return float(np.mean(self.job_ontime_probabilities))

    @property
    def minimum_job_ontime_probability(self):
        """Return the lowest estimated on-time probability among all jobs.

        The minimum highlights the least robust job in the production plan.
        """
        return min(self.job_ontime_probabilities)


@dataclass(frozen=True)
class SolutionSimulationResult:
    """Combine a simulation result with its solution and cost metadata.

    The wrapper identifies the solver, physical instance and common-random-
    number seed and stores replication-level total costs for direct comparison
    of nominal, nonlinear and GNN-generated production plans.
    """

    solution_path: Path
    instance_path: Path
    instance_name: str
    physical_instance_id: str
    solver: str
    formulation: str
    seed: int
    processing_cost: float
    facility_cost_per_time: float
    tardiness_cost_per_time: float
    simulation: SimulationResult
    replication_total_costs: tuple[float, ...]

    @property
    def mean_total_cost(self):
        """Return the mean simulated economic cost across replications.

        Each replication combines fixed processing cost with realized makespan
        and tardiness costs.
        """
        return float(np.mean(self.replication_total_costs))

    @property
    def total_cost_p90(self):
        """Return the empirical 90th percentile of simulated total cost.

        The percentile provides an upper-tail economic robustness indicator.
        """
        return float(np.quantile(self.replication_total_costs, 0.90))

    @property
    def total_cost_p95(self):
        """Return the empirical 95th percentile of simulated total cost.

        The percentile summarizes a more conservative upper-tail cost level.
        """
        return float(np.quantile(self.replication_total_costs, 0.95))


def _validated_topology(schedule):
    """Validate and topologically order the combined precedence graph.

    Fixed job arcs and immediate selected-machine arcs are merged. The function
    verifies machine consistency, direct predecessor/successor uniqueness and
    acyclicity before returning the deterministic execution order.

    Returns:
        A pair containing the operation order and complete predecessor tuples.
    """

    operations = list(schedule.operations)
    operation_set = set(operations)
    order = {operation: index for index, operation in enumerate(operations)}
    predecessors = {
        operation: set(schedule.job_predecessors.get(operation, ()))
        for operation in operations
    }
    machine_predecessors = {}
    machine_successors = {}
    for source, target, machine in schedule.machine_edges:
        if source not in operation_set or target not in operation_set:
            raise InvalidScheduleError(
                "Machine edge references an unknown operation."
            )
        if (
            schedule.selected_machines[source] != machine
            or schedule.selected_machines[target] != machine
        ):
            raise InvalidScheduleError(
                "Machine edge does not match the selected machine."
            )
        if target in machine_predecessors:
            raise InvalidScheduleError(
                "An operation has multiple immediate machine predecessors."
            )
        if source in machine_successors:
            raise InvalidScheduleError(
                "An operation has multiple immediate machine successors."
            )
        machine_predecessors[target] = source
        machine_successors[source] = target
        predecessors[target].add(source)

    for operation, required in predecessors.items():
        if not required <= operation_set:
            raise InvalidScheduleError(
                f"Unknown predecessor for operation {operation!r}."
            )

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
        raise InvalidScheduleError(
            "Combined job/machine predecessor graph contains a cycle."
        )
    return tuple(topology), {
        operation: tuple(sorted(required, key=order.__getitem__))
        for operation, required in predecessors.items()
    }


def _draw_machine_downtime(rng, weibull_scale, weibull_shape, repair_rate):
    """Draw one machine failure time and its exponential repair duration.

    Args:
        rng: NumPy random generator dedicated to one machine and replication.
        weibull_scale: Scale of the first-failure distribution.
        weibull_shape: Shape of the first-failure distribution.
        repair_rate: Rate of the exponential repair distribution.

    Returns:
        ``(failure_time, repair_duration)`` in scheduling time units.
    """

    failure_time = float(
        float(weibull_scale) * rng.weibull(float(weibull_shape))
    )
    repair_duration = float(rng.exponential(1.0 / float(repair_rate)))
    return failure_time, repair_duration


def _preempt_resume_completion(
    start,
    processing_time,
    failure_time,
    repair_duration,
):
    """Calculate completion under one preempt-resume machine interruption.

    A failure is irrelevant if repair ends before processing starts or begins
    after nominal completion. Otherwise processing waits for an ongoing repair
    or is interrupted for the full sampled repair duration.

    Returns:
        The realized completion time and direct repair-induced delay.
    """

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
        direct_delay = float(repair_duration)
    return nominal_completion + direct_delay, direct_delay


def simulate_fixed_schedule(
    schedule,
    *,
    replications=DEFAULT_REPLICATIONS,
    seed=DEFAULT_RANDOM_SEED,
):
    """Monte Carlo-evaluate one fixed production plan under machine failures.

    Each replication draws one failure and repair per used machine with
    independent deterministic random streams. Operations are executed in the
    fixed precedence topology, and delays propagate by right-shifting affected
    successors. The function summarizes reliability, delay, tardiness,
    makespan and repair statistics.

    Args:
        schedule: Valid fixed nominal schedule to post-evaluate.
        replications: Positive number of Monte Carlo scenarios.
        seed: Base seed used to construct machine-specific random streams.

    Returns:
        A fully populated :class:`SimulationResult`.
    """

    replications = int(replications)
    seed = int(seed)
    if replications <= 0:
        raise ValueError("replications must be positive.")
    validate_schedule(schedule)
    topology, predecessors = _validated_topology(schedule)
    operations = tuple(schedule.operations)
    operation_index = {
        operation: index for index, operation in enumerate(operations)
    }
    used_machines = tuple(sorted({
        schedule.selected_machines[operation] for operation in operations
    }))
    all_machines = tuple(sorted(schedule.weibull_scale))
    machine_stream = {
        machine: index for index, machine in enumerate(all_machines)
    }
    job_ids = tuple(sorted(schedule.jobs))
    job_index = {job: index for index, job in enumerate(job_ids)}

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

    job_completion_samples = np.empty(
        (replications, len(job_ids)), dtype=float
    )
    operation_start_samples = np.empty(
        (replications, len(operations)), dtype=float
    )
    operation_direct_delays = np.zeros(len(operations), dtype=float)
    operation_affected = np.zeros(len(operations), dtype=np.int64)
    operation_repair_durations = np.zeros(len(operations), dtype=float)
    affected_machine_count = 0
    total_repair_duration = 0.0

    for replication_index in range(replications):
        downtime = {}
        for machine in used_machines:
            rng = np.random.default_rng(np.random.SeedSequence([
                seed,
                replication_index,
                machine_stream[machine],
            ]))
            downtime[machine] = _draw_machine_downtime(
                rng,
                schedule.weibull_scale[machine],
                schedule.weibull_shape[machine],
                schedule.repair_rate[machine],
            )

        completions = {}
        affected_machines = set()
        for operation in topology:
            index = operation_index[operation]
            predecessor_completion = max(
                (completions[source] for source in predecessors[operation]),
                default=0.0,
            )
            start = max(
                float(schedule.planned_starts[operation]),
                predecessor_completion,
            )
            operation_start_samples[replication_index, index] = start
            machine = schedule.selected_machines[operation]
            failure_time, repair_duration = downtime[machine]
            completion, direct_delay = _preempt_resume_completion(
                start,
                schedule.processing_times[operation],
                failure_time,
                repair_duration,
            )
            completions[operation] = completion
            if direct_delay > 0.0:
                affected_machines.add(machine)
                operation_affected[index] += 1
                operation_direct_delays[index] += direct_delay
                operation_repair_durations[index] += repair_duration

        affected_machine_count += len(affected_machines)
        total_repair_duration += sum(
            downtime[machine][1] for machine in affected_machines
        )
        for job in job_ids:
            job_completion_samples[
                replication_index, job_index[job]
            ] = completions[schedule.job_end_operations[job]]

    due_dates = np.asarray(
        [float(schedule.due_dates[job]) for job in job_ids]
    )
    nominal_completions = np.asarray([
        float(schedule.planned_starts[schedule.job_end_operations[job]])
        + float(schedule.processing_times[schedule.job_end_operations[job]])
        for job in job_ids
    ])
    ontime = job_completion_samples <= due_dates[None, :] + 1e-12
    job_probabilities = np.mean(ontime, axis=0)
    job_standard_errors = np.sqrt(
        job_probabilities * (1.0 - job_probabilities) / replications
    )
    joint_ontime = np.all(ontime, axis=1)
    joint_probability = float(np.mean(joint_ontime))
    completion_delays = np.maximum(
        0.0, job_completion_samples - nominal_completions[None, :]
    )
    job_tardiness = np.maximum(
        0.0, job_completion_samples - due_dates[None, :]
    )
    total_tardiness = np.sum(job_tardiness, axis=1)
    makespans = np.max(job_completion_samples, axis=1)
    nominal_makespan = float(np.max(nominal_completions))
    planned_starts = np.asarray([
        float(schedule.planned_starts[operation]) for operation in operations
    ])
    start_shifts = np.maximum(
        0.0, operation_start_samples - planned_starts[None, :]
    )
    mean_start_shifts = np.mean(start_shifts, axis=0)
    completion_standard_errors = (
        np.std(job_completion_samples, axis=0, ddof=1)
        / math.sqrt(replications)
        if replications > 1
        else np.zeros(len(job_ids), dtype=float)
    )
    affected_safe = np.maximum(operation_affected, 1)
    flat_start_shifts = start_shifts.reshape(-1)

    return SimulationResult(
        replications=replications,
        job_ids=job_ids,
        job_ontime_probabilities=tuple(map(float, job_probabilities)),
        job_probability_standard_errors=tuple(map(float, job_standard_errors)),
        job_mean_completion_times=tuple(
            map(float, np.mean(job_completion_samples, axis=0))
        ),
        job_mean_completion_delays=tuple(
            map(float, np.mean(completion_delays, axis=0))
        ),
        job_completion_delay_p90=tuple(
            map(float, np.quantile(completion_delays, 0.90, axis=0))
        ),
        job_completion_delay_p95=tuple(
            map(float, np.quantile(completion_delays, 0.95, axis=0))
        ),
        job_mean_tardiness=tuple(
            map(float, np.mean(job_tardiness, axis=0))
        ),
        job_tardiness_p90=tuple(
            map(float, np.quantile(job_tardiness, 0.90, axis=0))
        ),
        job_tardiness_p95=tuple(
            map(float, np.quantile(job_tardiness, 0.95, axis=0))
        ),
        job_completion_delay_standard_errors=tuple(
            map(float, completion_standard_errors)
        ),
        operation_failure_probabilities=tuple(
            map(float, operation_affected / replications)
        ),
        operation_mean_repair_delays=tuple(
            map(float, operation_direct_delays / replications)
        ),
        operation_mean_repair_durations=tuple(
            map(float, operation_repair_durations / affected_safe)
        ),
        operation_mean_start_times=tuple(
            map(float, np.mean(operation_start_samples, axis=0))
        ),
        operation_mean_start_shifts=tuple(map(float, mean_start_shifts)),
        operation_start_shift_p90=tuple(
            map(float, np.quantile(start_shifts, 0.90, axis=0))
        ),
        operation_start_shift_p95=tuple(
            map(float, np.quantile(start_shifts, 0.95, axis=0))
        ),
        operation_start_shift_probabilities=tuple(
            map(float, np.mean(start_shifts > 1e-12, axis=0))
        ),
        operation_maximum_start_shifts=tuple(
            map(float, np.max(start_shifts, axis=0))
        ),
        all_jobs_ontime_probability=joint_probability,
        all_jobs_ontime_standard_error=math.sqrt(
            joint_probability * (1.0 - joint_probability) / replications
        ),
        mean_simulated_makespan=float(np.mean(makespans)),
        simulated_makespan_p90=float(np.quantile(makespans, 0.90)),
        simulated_makespan_p95=float(np.quantile(makespans, 0.95)),
        replication_makespans=tuple(map(float, makespans)),
        mean_makespan_increase=float(
            np.mean(np.maximum(0.0, makespans - nominal_makespan))
        ),
        mean_total_tardiness=float(np.mean(total_tardiness)),
        total_tardiness_p90=float(np.quantile(total_tardiness, 0.90)),
        total_tardiness_p95=float(np.quantile(total_tardiness, 0.95)),
        replication_total_tardiness=tuple(map(float, total_tardiness)),
        mean_operation_start_shift=float(np.mean(flat_start_shifts)),
        operation_start_shift_p90_overall=float(
            np.quantile(flat_start_shifts, 0.90)
        ),
        operation_start_shift_p95_overall=float(
            np.quantile(flat_start_shifts, 0.95)
        ),
        maximum_mean_operation_start_shift=float(np.max(mean_start_shifts)),
        mean_total_repair_delay=float(
            np.sum(operation_direct_delays) / replications
        ),
        mean_total_repair_duration=float(
            total_repair_duration / replications
        ),
        mean_failures=float(affected_machine_count / replications),
    )


def _solution_field(lines, label, default=""):
    """Extract a labelled scalar field from exported solution lines.

    Returns:
        Stripped text following ``<label>:`` or ``default`` when absent.
    """
    prefix = f"{label}:"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return default


def _solution_float(lines, label, default=None):
    """Extract an optional labelled solution field as ``float``.

    Returns:
        The parsed number or ``default`` when the field is empty or absent.
    """
    value = _solution_field(lines, label, "")
    return default if value == "" else float(value)


def _solution_instance_name(solution_path):
    """Recover the generated instance name from a solution filename.

    Raises:
        ValueError: If the expected solution prefix or solver suffix is absent.
    """
    stem = Path(solution_path).stem
    if not stem.startswith("solution_"):
        raise ValueError(f"Unexpected solution filename: {Path(solution_path).name}")
    remainder = stem[len("solution_"):]
    for marker in ("_gurobi_nonlinear", "_gurobi_gnn", "_gurobi"):
        if marker in remainder:
            return remainder.split(marker, 1)[0]
    raise ValueError(
        f"Cannot identify the instance in {Path(solution_path).name}."
    )


def parse_solution_file(solution_path):
    """Parse a schedule exported by ``helper.gurobi_solution_writer``.

    Operation assignments and timings, job values, active direct machine edges,
    formulation metadata and economic rates are reconstructed from the common
    text format.

    Returns:
        Dictionary containing the parsed schedule and solver metadata.

    Raises:
        ValueError: If edges, formulation, filename or incumbent are invalid.
    """

    solution_path = Path(solution_path).resolve()
    lines = solution_path.read_text(encoding="utf-8").splitlines()
    operations = {}
    jobs = {}
    machine_edges = []
    for line in lines:
        if line.startswith("op "):
            match = re.match(
                r"^op (?P<operation>-?\d+): machine=(?P<machine>-?\d+), "
                r"S=(?P<start>[^,]+), C=(?P<completion>[^,]+),",
                line,
            )
            if match:
                values = match.groupdict()
                operations[int(values["operation"])] = {
                    "machine": int(values["machine"]),
                    "start": float(values["start"]),
                    "completion": float(values["completion"]),
                }
            continue
        if line.startswith("job "):
            match = re.match(r"^job (?P<job>-?\d+): (?P<values>.+)$", line)
            if match:
                fields = {}
                for item in match.group("values").split(", "):
                    key, separator, value = item.partition("=")
                    if separator and value != "":
                        fields[key] = float(value)
                jobs[int(match.group("job"))] = fields
            continue
        if line.startswith("U(") and line.endswith("=1"):
            edge = ast.literal_eval(line[1:-2])
            if not isinstance(edge, tuple) or len(edge) != 3:
                raise ValueError(f"Malformed machine edge: {line}")
            machine_edges.append(tuple(map(int, edge)))

    formulation = _solution_field(lines, "Formulation")
    if formulation.startswith("nominal_"):
        solver = "gurobi"
    elif formulation.startswith("nonlinear_"):
        solver = "gurobi_nonlinear"
    elif formulation.startswith("gnn_"):
        solver = "gurobi_gnn"
    else:
        raise ValueError(
            f"Unsupported or missing formulation in {solution_path}."
        )
    solution_count = int(float(_solution_field(lines, "Solution count", "0")))
    if solution_count <= 0 or not operations:
        raise ValueError(f"Solution has no incumbent schedule: {solution_path}")
    return {
        "solution_path": solution_path,
        "instance_name": _solution_instance_name(solution_path),
        "solver": solver,
        "formulation": formulation,
        "operations": operations,
        "jobs": jobs,
        "machine_edges": tuple(machine_edges),
        "processing_cost": _solution_float(lines, "Processing cost", 0.0),
        "facility_cost_per_time": _solution_float(
            lines, "Facility cost per time", 1.0
        ),
        "tardiness_cost_per_time": _solution_float(
            lines, "Tardiness cost per time", 1.0
        ),
    }


def _find_instance_path(instances_root, instance_name):
    """Locate exactly one generated pickle for a solution's instance name.

    Raises:
        FileNotFoundError: If no matching instance exists.
        ValueError: If multiple matching pickles make the instance ambiguous.
    """
    matches = sorted(Path(instances_root).rglob(f"{instance_name}.pkl"))
    if not matches:
        raise FileNotFoundError(
            f"Instance {instance_name!r} not found below {instances_root}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Instance {instance_name!r} is ambiguous: "
            + ", ".join(map(str, matches))
        )
    return matches[0]


def _repair_export_rounding(
    operations,
    planned_starts,
    processing_times,
    job_predecessors,
    machine_edges,
    tolerance=SOLUTION_RIGHT_SHIFT_TOLERANCE,
):
    """Right-shift only precedence violations caused by export rounding.

    The combined precedence graph is topologically traversed and every start is
    raised to the largest predecessor completion. A required shift above the
    tolerance is considered a genuinely invalid solution rather than rounding.

    Returns:
        Corrected planned starts preserving all fixed precedence relations.
    """

    order = {operation: index for index, operation in enumerate(operations)}
    predecessors = {
        operation: set(job_predecessors.get(operation, ()))
        for operation in operations
    }
    for source, target, _machine in machine_edges:
        predecessors[target].add(source)
    successors = {operation: [] for operation in operations}
    indegree = {operation: len(required) for operation, required in predecessors.items()}
    for operation, required in predecessors.items():
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
        raise InvalidScheduleError(
            "Combined job/machine predecessor graph contains a cycle."
        )
    repaired = {}
    for operation in topology:
        required_start = max(
            (
                repaired[source] + processing_times[source]
                for source in predecessors[operation]
            ),
            default=0.0,
        )
        shift = required_start - planned_starts[operation]
        if shift > tolerance:
            raise InvalidScheduleError(
                "Solution contains a material overlap before operation "
                f"{operation}: required right shift={shift}."
            )
        repaired[operation] = max(planned_starts[operation], required_start)
    return repaired


def fixed_schedule_from_solution(
    solution_path,
    *,
    instances_root=ROOT_DIR / "02_data" / "fjsp_instances",
):
    """Reconstruct a validated fixed schedule from one exported solution.

    The associated generated instance supplies processing times, precedence,
    due dates and stochastic parameters. Exported assignments and durations are
    cross-checked before small text-rounding violations are repaired.

    Returns:
        The fixed schedule, parsed solution dictionary, loaded instance and
        concrete instance path.
    """

    import importlib

    parsed = parse_solution_file(solution_path)
    instance_path = _find_instance_path(
        instances_root, parsed["instance_name"]
    )
    instances = importlib.import_module("01_generator.instance_generator")
    instance = instances.load_generated_instance(
        parsed["instance_name"], instance_directory=instance_path.parent
    )
    operations = tuple(instance.real_operations)
    if set(parsed["operations"]) != set(operations):
        raise InvalidScheduleError(
            "Solution and instance contain different operations."
        )
    selected = {
        operation: parsed["operations"][operation]["machine"]
        for operation in operations
    }
    processing_times = {}
    planned_starts = {}
    for operation, machine in selected.items():
        if machine not in instance.eligible_machines[operation]:
            raise InvalidScheduleError(
                f"Operation {operation} uses ineligible machine {machine}."
            )
        expected_duration = float(instance.processing_times[operation, machine])
        stored_duration = (
            parsed["operations"][operation]["completion"]
            - parsed["operations"][operation]["start"]
        )
        if not math.isclose(
            expected_duration,
            stored_duration,
            rel_tol=0.0,
            abs_tol=SOLUTION_DURATION_TOLERANCE,
        ):
            raise InvalidScheduleError(
                f"Duration mismatch for operation {operation}: "
                f"solution={stored_duration}, instance={expected_duration}."
            )
        processing_times[operation] = expected_duration
        planned_starts[operation] = max(
            0.0, float(parsed["operations"][operation]["start"])
        )
    if set(parsed["jobs"]) != set(instance.jobs):
        raise InvalidScheduleError(
            "Solution and instance contain different jobs."
        )
    for job, fields in parsed["jobs"].items():
        due_date = fields.get("due_date")
        if due_date is None or not math.isclose(
            due_date,
            float(instance.due_dates[job]),
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise InvalidScheduleError(
                f"Due-date mismatch for job {job}."
            )
    job_predecessors = {
        operation: tuple(instance.predecessors.get(operation, ()))
        for operation in operations
    }
    planned_starts = _repair_export_rounding(
        operations,
        planned_starts,
        processing_times,
        job_predecessors,
        parsed["machine_edges"],
    )
    schedule = FixedSchedule(
        operations=operations,
        selected_machines=selected,
        processing_times=processing_times,
        planned_starts=planned_starts,
        job_predecessors=job_predecessors,
        machine_edges=parsed["machine_edges"],
        jobs={job: tuple(chain) for job, chain in instance.jobs.items()},
        job_end_operations=dict(instance.job_end_operations),
        due_dates={job: float(value) for job, value in instance.due_dates.items()},
        weibull_scale={
            machine: float(value)
            for machine, value in instance.weibull_alpha.items()
        },
        weibull_shape={
            machine: float(value)
            for machine, value in instance.weibull_beta.items()
        },
        repair_rate={
            machine: float(value)
            for machine, value in instance.repair_rate.items()
        },
    )
    validate_schedule(schedule)
    _validated_topology(schedule)
    return schedule, parsed, instance, instance_path


_DUE_DATE_SUFFIX = re.compile(
    r"_(?:benchmark|extrapolation|stress)_"
    r"(?:twk_d(?:m?\d+p\d+)|df(?:m?\d+p\d+)|configured)$"
)


def _physical_instance_id(instance, instance_name):
    """Return the base physical-instance ID shared by due-date variants.

    Stored metadata is preferred; otherwise known benchmark and extrapolation
    due-date suffixes are removed from the generated name.
    """
    stored = getattr(instance, "physical_instance_id", None)
    if stored is not None and str(stored).strip():
        return str(stored).strip()
    return _DUE_DATE_SUFFIX.sub("", str(instance_name))


def _evaluation_seed(base_seed, physical_instance_id):
    """Derive a stable common-random-number seed for one physical instance.

    All solver variants of the same physical instance therefore receive the
    same Monte Carlo scenarios.
    """
    payload = (
        f"postsolve:{int(base_seed)}:{physical_instance_id}".encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def simulate_solution_file(
    solution_path,
    *,
    instances_root=ROOT_DIR / "02_data" / "fjsp_instances",
    replications=DEFAULT_REPLICATIONS,
    random_seed=DEFAULT_RANDOM_SEED,
):
    """Evaluate one stored solution and derive replication-level total costs.

    Args:
        solution_path: Comparable text solution to evaluate.
        instances_root: Root searched for its generated instance pickle.
        replications: Number of Monte Carlo scenarios.
        random_seed: Global base seed for common random numbers.

    Returns:
        A :class:`SolutionSimulationResult` with schedule and cost statistics.
    """

    schedule, parsed, instance, instance_path = fixed_schedule_from_solution(
        solution_path, instances_root=instances_root
    )
    physical_id = _physical_instance_id(instance, parsed["instance_name"])
    seed = _evaluation_seed(random_seed, physical_id)
    result = simulate_fixed_schedule(
        schedule, replications=replications, seed=seed
    )
    processing_cost = float(parsed["processing_cost"])
    facility_rate = float(parsed["facility_cost_per_time"])
    tardiness_rate = float(parsed["tardiness_cost_per_time"])
    costs = tuple(
        processing_cost + facility_rate * makespan + tardiness_rate * tardiness
        for makespan, tardiness in zip(
            result.replication_makespans,
            result.replication_total_tardiness,
        )
    )
    return SolutionSimulationResult(
        solution_path=Path(solution_path).resolve(),
        instance_path=instance_path.resolve(),
        instance_name=parsed["instance_name"],
        physical_instance_id=physical_id,
        solver=parsed["solver"],
        formulation=parsed["formulation"],
        seed=seed,
        processing_cost=processing_cost,
        facility_cost_per_time=facility_rate,
        tardiness_cost_per_time=tardiness_rate,
        simulation=result,
        replication_total_costs=costs,
    )


def _solution_result_row(evaluation):
    """Flatten one evaluated solution into a comparison-CSV row.

    Nested job-level statistics are serialized as compact JSON while aggregate
    economic and robustness indicators remain individual numeric columns.
    """
    simulation = evaluation.simulation
    job_probabilities = {
        str(job): probability
        for job, probability in zip(
            simulation.job_ids, simulation.job_ontime_probabilities
        )
    }
    job_standard_errors = {
        str(job): value
        for job, value in zip(
            simulation.job_ids, simulation.job_probability_standard_errors
        )
    }
    job_mean_completions = {
        str(job): value
        for job, value in zip(
            simulation.job_ids, simulation.job_mean_completion_times
        )
    }
    job_mean_tardiness = {
        str(job): value
        for job, value in zip(
            simulation.job_ids, simulation.job_mean_tardiness
        )
    }
    return {
        "evaluation_status": "evaluated",
        "evaluation_error": "",
        "instance_name": evaluation.instance_name,
        "physical_instance_id": evaluation.physical_instance_id,
        "solver": evaluation.solver,
        "formulation": evaluation.formulation,
        "replications": simulation.replications,
        "simulation_seed": evaluation.seed,
        "simulation_method": simulation.simulation_method,
        "mean_job_ontime_probability": simulation.mean_job_ontime_probability,
        "minimum_job_ontime_probability": (
            simulation.minimum_job_ontime_probability
        ),
        "joint_all_jobs_ontime_probability": (
            simulation.all_jobs_ontime_probability
        ),
        "joint_all_jobs_ontime_standard_error": (
            simulation.all_jobs_ontime_standard_error
        ),
        "job_ontime_probabilities": json.dumps(
            job_probabilities, sort_keys=True, separators=(",", ":")
        ),
        "job_ontime_standard_errors": json.dumps(
            job_standard_errors, sort_keys=True, separators=(",", ":")
        ),
        "job_mean_completion_times": json.dumps(
            job_mean_completions, sort_keys=True, separators=(",", ":")
        ),
        "job_mean_tardiness": json.dumps(
            job_mean_tardiness, sort_keys=True, separators=(",", ":")
        ),
        "mean_simulated_makespan": simulation.mean_simulated_makespan,
        "simulated_makespan_p90": simulation.simulated_makespan_p90,
        "simulated_makespan_p95": simulation.simulated_makespan_p95,
        "mean_total_tardiness": simulation.mean_total_tardiness,
        "total_tardiness_p90": simulation.total_tardiness_p90,
        "total_tardiness_p95": simulation.total_tardiness_p95,
        "mean_total_cost": evaluation.mean_total_cost,
        "total_cost_p90": evaluation.total_cost_p90,
        "total_cost_p95": evaluation.total_cost_p95,
        "mean_operation_start_shift": simulation.mean_operation_start_shift,
        "maximum_mean_operation_start_shift": (
            simulation.maximum_mean_operation_start_shift
        ),
        "mean_affected_machines": simulation.mean_failures,
        "solution_file": str(evaluation.solution_path),
        "instance_file": str(evaluation.instance_path),
    }


def _collect_solution_paths(paths):
    """Resolve files and recursively collect unique solution text paths.

    Raises:
        FileNotFoundError: If an input path is neither a file nor a directory.
    """
    result = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            result.extend(path.rglob("solution_*.txt"))
        elif path.is_file():
            result.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(set(path.resolve() for path in result))


def simulate_solution_files(
    paths,
    *,
    instances_root=ROOT_DIR / "02_data" / "fjsp_instances",
    output_file=ROOT_DIR / "05_Simulation" / "results" / "monte_carlo_results.csv",
    replications=DEFAULT_REPLICATIONS,
    random_seed=DEFAULT_RANDOM_SEED,
):
    """Evaluate solution files or directories and write a comparison CSV.

    Invalid or unmatched solutions are retained as ``not_evaluated`` rows so
    the output documents exclusions instead of silently dropping them.

    Returns:
        The written CSV path and all result/status row dictionaries.
    """

    solution_paths = _collect_solution_paths(paths)
    if not solution_paths:
        raise ValueError("No solution_*.txt files found.")
    rows = []
    for index, solution_path in enumerate(solution_paths, start=1):
        try:
            evaluation = simulate_solution_file(
                solution_path,
                instances_root=instances_root,
                replications=replications,
                random_seed=random_seed,
            )
        except (FileNotFoundError, InvalidScheduleError, ValueError) as error:
            rows.append({
                "evaluation_status": "not_evaluated",
                "evaluation_error": str(error),
                "solution_file": str(solution_path),
            })
            print(
                f"[Simulation] {index}/{len(solution_paths)} "
                f"{solution_path.name} | skipped: {error}"
            )
            continue
        rows.append(_solution_result_row(evaluation))
        print(
            f"[Simulation] {index}/{len(solution_paths)} "
            f"{solution_path.name} | joint="
            f"{evaluation.simulation.all_jobs_ontime_probability:.4f}"
        )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_file, rows
