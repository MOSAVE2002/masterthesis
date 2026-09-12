"""Fix-and-optimize graphs labelled by deterministic local repair buffers."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import math
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from helper.time_units import normalize_time_unit

import gurobipy as gp

from helper.sequence_setup import (
    RELIABILITY_GNN_GRAPH_SCHEMA,
    add_reliability_graph_variables,
    normalize_reliability_graph_config,
    reliability_graph_config_dict,
    reliability_node_feature_names,
    local_buffer_node_features,
)
from helper.stochastic_fjsp import (
    ensure_stochastic_parameters,
    stochastic_parameters,
    weibull_down_probability,
)
from helper.surrogate_constraint import (
    configured_constraint_type,
    target_column,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
_instances = importlib.import_module("01_generator.instance_generator")
_base = importlib.import_module("03_Gurobi.build_fjsp")
_nonlinear = importlib.import_module("03_Gurobi.build_fjsp_with_nonlinear")
_simulation = importlib.import_module("05_Simulation.preempt_resume")
FixedSchedule = _simulation.FixedSchedule
from helper.local_buffer import (
    LABEL_METHOD as SIMULATION_LABEL_METHOD, LABEL_SOURCE,
    expected_job_buffers, label_config_dict, service_lower_bounds,
    InvalidScheduleError,
)
from helper.buffer_candidate_selection import (
    input_signature, unique_input_candidates, select_buffer_candidates,
)
TARGET_COLUMN = target_column(configured_constraint_type())

SPLIT_DIRECTORIES = _instances.SPLIT_DIRECTORIES
SPLIT_CSV_FILENAMES = _instances.SPLIT_CSV_FILENAMES
load_generated_instance = _instances.load_generated_instance
selected_instance_splits = _instances.selected_instance_splits

FIELDNAMES = [
    "instance_name",
    "due_date_factor",
    "training_due_date_scale",
    "nominal_makespan_calibration",
    "nominal_twk_due_date_factor",
    "training_due_date_delta",
    "training_effective_due_date_factor",
    "training_mean_due_date",
    "training_due_dates",
    "training_weibull_scale_factor",
    "candidate_generation_mode",
    "schedule_timing_method",
    "candidate_probability_band",
    "job_ids",
    TARGET_COLUMN,
    "simulated_completion_delay_standard_errors",
    "simulation_job_ontime_probabilities",
    "simulation_replications",
    "simulation_seed",
    "local_buffer_numerical_errors",
    "local_buffer_service_lower_bounds",
    "operation_job_indices",
    "gnn_feature_names",
    "gnn_node_features",
    "gnn_active_edges",
    "reliability_graph_parameters",
    "fix_ratio",
    "fixed_operations",
    "optimization_run",
    "pool_solution_number",
    "pool_objective",
    "pool_selection_category",
    "solver_runtime_seconds",
]

DEFAULT_SELECTION_RATIOS = {
    "service_boundary_below": 0.15,
    "service_boundary_above": 0.15,
    "low_expected_completion_delay": 0.15,
    "medium_expected_completion_delay": 0.10,
    "high_expected_completion_delay": 0.15,
    "good_nominal_objective": 0.10,
    "pareto_tradeoff": 0.10,
    "structurally_diverse": 0.10,
}
ADAPTIVE_SELECTION_CATEGORIES = tuple(DEFAULT_SELECTION_RATIOS)
DEFAULT_ADAPTIVE_CANDIDATE_SELECTION = {
    "ensure_all_categories": False,
    "rotate_repeated_categories": True,
    "rotate_surplus_offsets": True,
    "prefer_unique_structures": True,
}

PROBABILITY_BINS = (
    "low",
    "boundary_below",
    "boundary_above",
    "high",
)
ANCHOR_CATEGORIES = (
    "good_nominal_objective",
    "low_expected_completion_delay",
    "high_expected_completion_delay",
    "pareto_tradeoff",
    "structurally_diverse",
)
DEFAULT_HYBRID_ANCHOR_RATIOS = {
    "good_nominal_objective": 0.25,
    "low_expected_completion_delay": 0.20,
    "high_expected_completion_delay": 0.20,
    "pareto_tradeoff": 0.15,
    "structurally_diverse": 0.20,
}
DEFAULT_JOB_PROBABILITY_TARGET_RATIOS = {
    "low": 0.20,
    "boundary_below": 0.30,
    "boundary_above": 0.30,
    "high": 0.20,
}


def _label_distribution_parameters(fixed):
    """Return the neutral four-bin label-selection center and half-width."""
    center = float(fixed.get("label_distribution_center", 0.50))
    half_width = float(fixed.get("label_distribution_half_width", 0.25))
    if not 0.0 < center < 1.0:
        raise ValueError("label_distribution_center must lie in (0, 1).")
    if half_width <= 0.0 or center - half_width < 0.0 or center + half_width > 1.0:
        raise ValueError(
            "label_distribution_half_width must define valid bins in [0, 1]."
        )
    return center, half_width

CANDIDATE_GENERATION_MODES = (
    "unconstrained",
    "nominal_ontime",
    "nonlinear_evaluated",
    "nonlinear",
)
DEFAULT_NOMINAL_RUN_INDEX_OFFSET = 1_000_000
DEFAULT_INSTANCE_FAILURE_HANDLING = {
    "summary_filename": "generation_summary.json",
}
_NOMINAL_MAKESPAN_CACHE = {}


def _normalize_instance_failure_handling(config=None):
    values = dict(DEFAULT_INSTANCE_FAILURE_HANDLING)
    raw = dict(config or {})
    unknown = set(raw) - set(values)
    if unknown:
        raise ValueError(
            "Unknown instance_failure_handling settings: "
            f"{sorted(unknown)}"
        )
    values.update(raw)
    values["summary_filename"] = str(values["summary_filename"])
    summary_path = Path(values["summary_filename"])
    if (
        summary_path.name != values["summary_filename"]
        or summary_path.suffix.lower() != ".json"
    ):
        raise ValueError(
            "instance_failure_handling.summary_filename must be a JSON "
            "filename without directories."
        )
    return values


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _quality_distribution_key(value):
    """Return stable JSON keys for numeric configuration levels."""
    text = format(float(value), ".12g")
    return f"{text}.0" if "." not in text and "e" not in text.lower() else text


def _new_quality_accumulator(expected_scale_factors):
    return {
        "expected_scale_factors": [
            _quality_distribution_key(value)
            for value in expected_scale_factors
        ],
        "weibull_scale_factors": Counter(),
        "candidate_categories": Counter(),
        "labels": [],
        "label_standard_errors": [],
        "label_numerical_errors": [],
    }


def _quality_numeric_values(raw):
    """Read a scalar or compact JSON list without making reporting fatal."""
    if raw is None or raw == "":
        return []
    try:
        values = raw if isinstance(raw, (list, tuple)) else json.loads(str(raw))
        if not isinstance(values, (list, tuple)):
            values = [values]
        result = [float(value) for value in values]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [value for value in result if math.isfinite(value)]


def _update_quality_accumulator(accumulator, rows):
    for row in rows:
        factor = row.get("training_weibull_scale_factor")
        try:
            factor_value = float(factor)
        except (TypeError, ValueError):
            factor_value = math.nan
        if math.isfinite(factor_value):
            accumulator["weibull_scale_factors"][
                _quality_distribution_key(factor_value)
            ] += 1

        category = str(row.get("pool_selection_category") or "").strip()
        if category:
            # Adaptive and mixed generation prefix the actual selection role
            # with due-date or generation-mode information.
            base_category = category.rsplit(":", 1)[-1]
            accumulator["candidate_categories"][base_category] += 1

        accumulator["labels"].extend(
            _quality_numeric_values(row.get(TARGET_COLUMN))
        )
        accumulator["label_numerical_errors"].extend(
            _quality_numeric_values(row.get("local_buffer_numerical_errors"))
        )
        accumulator["label_standard_errors"].extend(
            _quality_numeric_values(
                row.get("simulated_completion_delay_standard_errors")
            )
        )


def _descriptive_quality_statistics(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.pstdev(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _quality_report(accumulator):
    factor_keys = set(accumulator["expected_scale_factors"])
    factor_keys.update(accumulator["weibull_scale_factors"])
    category_keys = set(ADAPTIVE_SELECTION_CATEGORIES)
    category_keys.update(accumulator["candidate_categories"])
    return {
        "weibull_scale_factor_graph_counts": {
            key: int(accumulator["weibull_scale_factors"].get(key, 0))
            for key in sorted(factor_keys, key=float)
        },
        "candidate_category_graph_counts": {
            key: int(accumulator["candidate_categories"].get(key, 0))
            for key in sorted(category_keys)
        },
        "label_statistics": _descriptive_quality_statistics(
            accumulator["labels"]
        ),
        "label_numerical_error_statistics": _descriptive_quality_statistics(
            accumulator["label_numerical_errors"]
        ),
        "label_standard_error_statistics": (
            _descriptive_quality_statistics(
                accumulator["label_standard_errors"]
            )
        ),
    }


def _quality_number(value):
    return "n/a" if value is None else f"{float(value):.6g}"


def _quality_report_text(label, report):
    labels = report["label_statistics"]
    errors = report["label_numerical_error_statistics"]
    return (
        f"[Training-data quality] {label} | "
        f"instances={report['successful_instances']} successful/"
        f"{report['skipped_instances']} skipped | "
        f"graphs={report['written_graphs']} | "
        f"weibull={report['weibull_scale_factor_graph_counts']} | "
        f"categories={report['candidate_category_graph_counts']} | "
        f"labels(n={labels['count']}, mean={_quality_number(labels['mean'])}, "
        f"std={_quality_number(labels['standard_deviation'])}) | "
        f"quadrature_difference(max={_quality_number(errors['maximum'])})"
    )


def _write_generation_summary(path, summary):
    """Atomically persist progress so interrupted cluster runs stay auditable."""
    summary["updated_at_utc"] = _utc_timestamp()
    split_values = list(summary["splits"].values())
    successful = sum(
        len(values["successful_instances"]) for values in split_values
    )
    skipped = sum(
        len(values["skipped_instances"]) for values in split_values
    )
    jittered = sum(
        sum(
            bool(item.get("training_parameter_jitter_applied", False))
            for item in values["successful_instances"]
        )
        for values in split_values
    )
    summary["totals"] = {
        "configured_instances": sum(
            values["configured_count"] for values in split_values
        ),
        "processed_instances": successful + skipped,
        "successful_instances": successful,
        "jittered_successful_instances": jittered,
        "skipped_instances": skipped,
        "written_graphs": sum(
            values["written_graphs"] for values in split_values
        ),
    }
    for values in split_values:
        report = values.setdefault("quality_report", {})
        report.update({
            "successful_instances": len(values["successful_instances"]),
            "jittered_successful_instances": sum(
                bool(item.get("training_parameter_jitter_applied", False))
                for item in values["successful_instances"]
            ),
            "skipped_instances": len(values["skipped_instances"]),
            "written_graphs": values["written_graphs"],
        })
    report = summary.setdefault("quality_report", {})
    report.update({
        "successful_instances": successful,
        "jittered_successful_instances": jittered,
        "skipped_instances": skipped,
        "written_graphs": summary["totals"]["written_graphs"],
        "graphs_per_split": {
            split: values["written_graphs"]
            for split, values in summary["splits"].items()
        },
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


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


def _sample_reliability(instance, rng, generation, run_index):
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
    scale_factors = generation.get("weibull_scale_factors")
    if scale_factors is None:
        scale_factor = 1.0
    else:
        values = [float(value) for value in scale_factors]
        if not values or any(value <= 0.0 for value in values):
            raise ValueError("weibull_scale_factors must be positive.")
        rotation = _sample_seed(
            getattr(instance, "instance_name", "instance"),
            0,
            generation.get("random_seed", 42),
            "weibull_scale_rotation",
        ) % len(values)
        scale_factor = values[(int(run_index) + rotation) % len(values)]
        instance.weibull_alpha = {
            machine: scale_factor * float(value)
            for machine, value in instance.weibull_alpha.items()
        }
    instance.training_weibull_scale_factor = scale_factor


def _sample_training_due_dates(instance, rng, generation, run_index):
    """Vary calibrated TWK due-date slack on the training copy only."""
    adaptive = dict(generation.get("adaptive_due_dates") or {})
    if bool(adaptive.get("enabled", False)):
        if generation.get("due_date_scale_factors") is not None or generation.get(
            "due_date_scale_range"
        ) is not None:
            raise ValueError(
                "adaptive_due_dates cannot be combined with legacy due-date "
                "scale factors."
            )
        method = str(
            adaptive.get("method", "calibrated_total_work_content")
        )
        if method != "calibrated_total_work_content":
            raise ValueError(
                "adaptive_due_dates.method must be "
                "'calibrated_total_work_content'."
            )
        aggregation = str(adaptive.get("machine_aggregation", "mean"))
        if aggregation != "mean":
            raise ValueError(
                "adaptive_due_dates.machine_aggregation must be 'mean'."
            )
        offsets = [
            float(value)
            for value in adaptive.get(
                "relative_makespan_offsets", [0.00, 0.15, 0.30, 0.45]
            )
        ]
        if not offsets or any(
            not math.isfinite(value) or value <= -1.0 for value in offsets
        ):
            raise ValueError(
                "adaptive_due_dates.relative_makespan_offsets must "
                "contain finite values greater than -1."
            )
        rotation = _sample_seed(
            getattr(instance, "instance_name", "instance"),
            0,
            generation.get("random_seed", 42),
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
        nominal_factor = calibrated["nominal_factor"]
        effective_factor = calibrated["effective_factor"]
        instance.due_dates = dict(calibrated["due_dates"])
        instance.training_due_date_scale = 1.0
        instance.nominal_makespan_calibration = calibration["makespan"]
        instance.nominal_twk_due_date_factor = nominal_factor
        instance.training_due_date_delta = offset
        instance.training_effective_due_date_factor = effective_factor
        instance.training_mean_due_date = (
            sum(instance.due_dates.values()) / len(instance.due_dates)
        )
        instance.training_due_dates = dict(instance.due_dates)
        instance.due_date_calibration_status = calibration["status"]
        instance.due_date_calibration_gap = calibration["gap"]
        instance.nominal_calibration_assignment = dict(
            calibration.get("assignment") or {}
        )
        return effective_factor

    factors = generation.get("due_date_scale_factors")
    raw = generation.get("due_date_scale_range")
    if factors is not None:
        values = [float(value) for value in factors]
        if not values or any(value <= 0.0 for value in values):
            raise ValueError(
                "due_date_scale_factors must contain positive values."
            )
        scale = values[int(run_index) % len(values)]
    elif raw is None:
        scale = 1.0
    else:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("due_date_scale_range must contain [lower, upper].")
        lower, upper = map(float, raw)
        if lower <= 0.0 or lower > upper:
            raise ValueError(
                "due_date_scale_range must satisfy 0 < lower <= upper."
            )
        scale = rng.uniform(lower, upper)
    instance.due_dates = {
        job: scale * float(value)
        for job, value in instance.due_dates.items()
    }
    instance.training_due_date_scale = scale
    return scale


def calibrated_total_work_content_due_dates(instance, config, relative_offset):
    """Return job-specific TWK dates anchored to a nominal makespan."""
    relative_offset = float(relative_offset)
    if not math.isfinite(relative_offset) or relative_offset <= -1.0:
        raise ValueError("relative_offset must be finite and greater than -1.")
    calibration = _nominal_makespan_calibration(instance, config)
    work_content = _total_work_content_by_job(instance)
    mean_work_content = sum(work_content.values()) / len(work_content)
    nominal_factor = calibration["makespan"] / mean_work_content
    effective_factor = (1.0 + relative_offset) * nominal_factor
    due_dates = {
        job: float(math.ceil(
            effective_factor * work_content[job] - 1e-12
        ))
        for job in instance.jobs
    }
    return {
        "calibration": calibration,
        "work_content": work_content,
        "nominal_factor": nominal_factor,
        "effective_factor": effective_factor,
        "relative_offset": relative_offset,
        "due_dates": due_dates,
        "mean_due_date": sum(due_dates.values()) / len(due_dates),
    }


def _total_work_content_by_job(instance):
    """Return TWK using each operation's mean eligible-machine duration."""
    work_content = {
        job: sum(
            sum(
                float(instance.processing_times[operation, machine])
                for machine in instance.eligible_machines[operation]
            ) / len(instance.eligible_machines[operation])
            for operation in operations
        )
        for job, operations in instance.jobs.items()
    }
    if not work_content or any(
        not math.isfinite(value) or value <= 0.0
        for value in work_content.values()
    ):
        raise ValueError("Every job must have positive total work content.")
    return work_content


def _nominal_makespan_calibration(instance, config):
    """Return a cached feasible nominal makespan for adaptive due dates."""
    signature = (
        tuple(sorted(
            (job, tuple(operations))
            for job, operations in instance.jobs.items()
        )),
        tuple(sorted(
            (operation, tuple(sorted(machines)))
            for operation, machines in instance.eligible_machines.items()
            if operation in set(instance.real_operations)
        )),
        tuple(sorted(
            (operation, machine, float(value))
            for (operation, machine), value in instance.processing_times.items()
            if operation in set(instance.real_operations)
            and machine in instance.eligible_machines[operation]
        )),
    )
    if signature in _NOMINAL_MAKESPAN_CACHE:
        return dict(_NOMINAL_MAKESPAN_CACHE[signature])

    model = gp.Model("adaptive_due_date_nominal_makespan")
    model.Params.OutputFlag = int(config.get("output_flag", 0))
    model.Params.TimeLimit = float(config.get("time_limit_seconds", 5.0))
    model.Params.MIPGap = float(config.get("mip_gap", 0.01))
    model.Params.Seed = int(config.get("seed", 42))
    model, variables = _base.build_fjsp(
        model,
        instance,
        include_makespan=True,
        enforce_due_dates=False,
        economic_objective=False,
    )
    model.optimize()
    if model.SolCount == 0:
        status = int(model.Status)
        model.dispose()
        raise RuntimeError(
            "Nominal makespan calibration found no feasible schedule; "
            f"Gurobi status={status}."
        )
    result = {
        "makespan": float(variables["C_max"].X),
        "status": int(model.Status),
        "gap": float(model.MIPGap),
        "runtime_seconds": float(model.Runtime),
        "assignment": {
            operation: max(
                instance.eligible_machines[operation],
                key=lambda machine: float(
                    variables["Y"][operation, machine].X
                ),
            )
            for operation in instance.real_operations
        },
    }
    model.dispose()
    _NOMINAL_MAKESPAN_CACHE[signature] = dict(result)
    return result


def _fix_neighborhood(
    model,
    variables,
    instance,
    rng,
    fix_ratio,
    *,
    use_incumbent=False,
    reference_assignment=None,
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
    elif reference_assignment:
        assignment = {
            operation: reference_assignment[operation]
            for operation in instance.real_operations
        }
    else:
        assignment = {
            operation: rng.choice(instance.eligible_machines[operation])
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
    return {
        "fix_ratio": len(fixed_operations) / len(flexible) if flexible else 0.0,
        "fixed_operations": len(fixed_operations),
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
        include_makespan=True,
        # Due dates are soft. In particular, negative calibration offsets may
        # put every due date below the feasible makespan, so they must not cap
        # completion-variable bounds.
        horizon_upper_bound=None,
    )
    parameters = stochastic_parameters(instance)
    variables.update({
        "weibull_alpha": parameters["alpha"],
        "weibull_beta": parameters["beta"],
        "repair_rate": parameters["repair_rate"],
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
    node_features = []
    for operation in operations:
        machine = selected[operation]
        completion = completions[operation]
        processing_time = processing_times[operation]
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
    nominal_ends = {
        j: planned_starts[instance.job_end_operations[j]]
        + processing_times[instance.job_end_operations[j]] for j in job_ids
    }
    nominal_schedule_cost = (
        sum(float(getattr(instance, "machine_cost", {}).get(selected[o], 0.0)) * processing_times[o] for o in operations)
        + max(planned_starts[o] + processing_times[o] for o in operations)
        + sum(max(0.0, nominal_ends[j] - instance.due_dates[j]) for j in job_ids)
    )
    return {
        "nominal_schedule_cost": nominal_schedule_cost,
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
            "due_date_factor": getattr(instance, "due_date_factor", ""),
            "training_due_date_scale": getattr(
                instance, "training_due_date_scale", 1.0
            ),
            "nominal_makespan_calibration": getattr(
                instance, "nominal_makespan_calibration", ""
            ),
            "nominal_twk_due_date_factor": getattr(
                instance, "nominal_twk_due_date_factor", ""
            ),
            "training_due_date_delta": getattr(
                instance, "training_due_date_delta", ""
            ),
            "training_effective_due_date_factor": getattr(
                instance, "training_effective_due_date_factor", ""
            ),
            "training_mean_due_date": getattr(
                instance, "training_mean_due_date", ""
            ),
            "training_due_dates": _compact(getattr(
                instance, "training_due_dates", {}
            )) if hasattr(instance, "training_due_dates") else "",
            "training_weibull_scale_factor": getattr(
                instance, "training_weibull_scale_factor", 1.0
            ),
            "candidate_generation_mode": candidate_generation_mode,
            "schedule_timing_method": timing_method,
            "candidate_probability_band": (
                "" if candidate_probability_band is None
                else candidate_probability_band
            ),
            "job_ids": _compact(job_ids),
            TARGET_COLUMN: "",
            "simulated_completion_delay_standard_errors": "",
            "simulation_job_ontime_probabilities": "",
            "simulation_replications": "",
            "simulation_seed": "",
            "operation_job_indices": _compact([
                job_to_index[operation_job[operation]]
                for operation in operations
            ]),
            "gnn_feature_names": _compact(reliability_node_feature_names()),
            "gnn_node_features": _compact(node_features),
            "gnn_active_edges": _compact(graph_edges),
            "reliability_graph_parameters": _compact(
                reliability_graph_config_dict(graph_config)
            ),
            **fix_stats,
            "optimization_run": run_index,
            "pool_solution_number": solution_number,
            "pool_objective": pool_objective,
            "pool_selection_category": "",
            "solver_runtime_seconds": float(model.Runtime),
        },
    }


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
    job_expected_repair_buffers = []
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
        job_expected_repair_buffers.append(expected_disruption)
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
    candidate["nonlinear_job_expected_repair_buffers"] = (
        job_expected_repair_buffers
    )
    candidate["nonlinear_min_job_probability"] = min(job_probabilities)
    candidate.update({
        "job_probabilities": job_probabilities,
        "min_job_probability": min(job_probabilities),
        "service_risk": 1.0 - min(job_probabilities),
        "effective_objective": candidate.get("pool_objective", 0.0),
    })
    return candidate


def _evaluate_fixed_schedule_simulation(candidate, label_config=None):
    """Compatibility entry point: deterministic LOCAL labels, no MC draws."""
    schedule = candidate["simulation_schedule"]
    buffers, errors = expected_job_buffers(schedule, label_config)
    jobs = sorted(schedule.jobs)
    values = [buffers[j] for j in jobs]
    lower_bounds = [service_lower_bounds(schedule, buffers)[j] for j in jobs]
    candidate.update({
        "local_job_repair_buffers": values,
        "nonlinear_job_expected_repair_buffers": values,
        "nonlinear_job_probabilities": lower_bounds,
        "nonlinear_min_job_probability": min(lower_bounds),
        "local_buffer_numerical_errors": [errors[j] for j in jobs],
        "simulated_job_expected_completion_delays": values,  # legacy ranking alias
        "job_probabilities": lower_bounds,
        "min_job_probability": min(lower_bounds),
        "service_risk": 1.0 - min(lower_bounds),
    })
    row = candidate.setdefault("row", {})
    row.update({
        TARGET_COLUMN: _compact(values),
        "local_buffer_numerical_errors": _compact([errors[j] for j in jobs]),
        "local_buffer_service_lower_bounds": _compact(lower_bounds),
        "simulated_completion_delay_standard_errors": "",
        "simulation_job_ontime_probabilities": "",
        "simulation_replications": 0,
        "simulation_seed": "",
    })
    return candidate


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


def _adaptive_due_date_offsets(generation):
    config = dict(generation.get("adaptive_due_dates") or {})
    if not bool(config.get("enabled", False)):
        return []
    return [
        float(value)
        for value in config.get(
            "relative_makespan_offsets", [0.00, 0.15, 0.30, 0.45]
        )
    ]


def _adaptive_candidate_selection(generation):
    adaptive = dict(generation.get("adaptive_due_dates") or {})
    raw = adaptive.get("candidate_selection")
    if raw is None:
        return dict(DEFAULT_ADAPTIVE_CANDIDATE_SELECTION)
    raw = dict(raw)
    unknown = set(raw) - set(DEFAULT_ADAPTIVE_CANDIDATE_SELECTION)
    if unknown:
        raise ValueError(
            "Unknown adaptive_due_dates.candidate_selection settings: "
            f"{sorted(unknown)}"
        )
    result = dict(DEFAULT_ADAPTIVE_CANDIDATE_SELECTION)
    result.update({name: bool(value) for name, value in raw.items()})
    return result


def _rotating_targets(values, total, selection_index, *, stride=1):
    values = list(values)
    if not values:
        return []
    rotation = int(selection_index) % len(values)
    return [
        values[(rotation + int(stride) * position) % len(values)]
        for position in range(int(total))
    ]


def _adaptive_offset_targets(generation, limit, selection_index=0):
    offsets = _adaptive_due_date_offsets(generation)
    if not offsets:
        return []
    config = _adaptive_candidate_selection(generation)
    rotation_index = (
        2 * int(selection_index)
        if config["rotate_surplus_offsets"] else 0
    )
    # The doubled instance rotation and reverse stride break the common
    # factors between four offsets and eight selection categories. Thus each
    # category is paired with every offset across physical instances.
    return _rotating_targets(
        offsets, limit, rotation_index, stride=max(1, len(offsets) - 1)
    )


def _adaptive_category_targets(generation, limit, selection_index=0):
    config = _adaptive_candidate_selection(generation)
    if not config["ensure_all_categories"]:
        return []
    if int(limit) < len(ADAPTIVE_SELECTION_CATEGORIES):
        raise ValueError(
            "adaptive_due_dates.candidate_selection.ensure_all_categories "
            f"requires at least {len(ADAPTIVE_SELECTION_CATEGORIES)} "
            "samples_per_instance."
        )
    rotation_index = (
        int(selection_index)
        if config["rotate_repeated_categories"] else 0
    )
    return _rotating_targets(
        ADAPTIVE_SELECTION_CATEGORIES,
        limit,
        rotation_index,
    )


def _adaptive_due_date_coverage_met(
    candidates, limit, generation, selection_index=0
):
    offsets = _adaptive_due_date_offsets(generation)
    if not offsets:
        return True
    quotas = Counter(_adaptive_offset_targets(
        generation, limit, selection_index
    ))
    structures = {
        offset: {
            candidate["structure"]
            for candidate in candidates
            if math.isclose(
                float(candidate["row"]["training_due_date_delta"]),
                offset,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        }
        for offset in offsets
    }
    return all(
        len(structures[offset]) >= quotas[offset] for offset in offsets
    )


def _select_adaptive_due_date_candidates(
    candidates,
    limit,
    ratios,
    service_level,
    boundary_width,
    generation,
    selection_index=0,
):
    """Select balanced adaptive-offset graphs, optionally by category cycle."""
    offsets = _adaptive_due_date_offsets(generation)
    if not offsets:
        return _select_candidates(
            candidates,
            limit,
            ratios,
            service_level,
            boundary_width,
            hybrid_selection=_hybrid_selection_for_split(
                generation["fixed_y"]
            ),
        )
    category_targets = _adaptive_category_targets(
        generation, limit, selection_index
    )
    if category_targets:
        return _select_balanced_adaptive_candidates(
            candidates,
            limit,
            service_level,
            boundary_width,
            generation,
            selection_index,
            category_targets,
        )
    quotas = _quota_counts(
        int(limit),
        {offset: 1.0 / len(offsets) for offset in offsets},
        offsets,
    )
    selected, used_ids, used_graph_signatures = [], set(), set()
    for offset in offsets:
        group = [
            candidate for candidate in candidates
            if math.isclose(
                float(candidate["row"]["training_due_date_delta"]),
                offset,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        ranked = _select_candidates(
            group,
            len(group),
            ratios,
            service_level,
            boundary_width,
            hybrid_selection=None,
        )
        for entry in ranked:
            candidate = entry["candidate"]
            graph_signature = (offset, candidate["structure"])
            if (
                candidate["candidate_id"] in used_ids
                or graph_signature in used_graph_signatures
            ):
                continue
            entry["category"] = (
                f"adaptive_delta_{offset:.3f}:{entry['category']}"
            )
            selected.append(entry)
            used_ids.add(candidate["candidate_id"])
            used_graph_signatures.add(graph_signature)
            if sum(
                math.isclose(
                    float(item["candidate"]["row"][
                        "training_due_date_delta"
                    ]),
                    offset,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for item in selected
            ) >= quotas[offset]:
                break
    if len(selected) < int(limit):
        remaining = [
            candidate for candidate in candidates
            if candidate["candidate_id"] not in used_ids
            and (
                float(candidate["row"]["training_due_date_delta"]),
                candidate["structure"],
            ) not in used_graph_signatures
        ]
        selected.extend(_select_candidates(
            remaining,
            int(limit) - len(selected),
            ratios,
            service_level,
            boundary_width,
            hybrid_selection=None,
        ))
    return selected[:int(limit)]


def _category_ranking(
    candidates, category, service_level, boundary_width, selected
):
    if category == "structurally_diverse":
        chosen = [entry["candidate"] for entry in selected]
        return sorted(
            candidates,
            key=lambda item: (
                -min(
                    (
                        _structure_distance(item, existing)
                        for existing in chosen
                    ),
                    default=1.0,
                ),
                item.get("balanced_score", 0.0),
            ),
        )
    one_hot = {
        name: float(name == category)
        for name in ADAPTIVE_SELECTION_CATEGORIES
    }
    return [
        entry["candidate"]
        for entry in _select_candidates_legacy(
            candidates,
            len(candidates),
            one_hot,
            service_level,
            boundary_width,
        )
    ]


def _select_balanced_adaptive_candidates(
    candidates,
    limit,
    service_level,
    boundary_width,
    generation,
    selection_index,
    category_targets,
):
    offset_targets = _adaptive_offset_targets(
        generation, limit, selection_index
    )
    config = _adaptive_candidate_selection(generation)
    selected, used_ids, used_graph_signatures = [], set(), set()
    used_structures = set()

    for category, offset in zip(category_targets, offset_targets):
        group = [
            candidate for candidate in candidates
            if math.isclose(
                float(candidate["row"]["training_due_date_delta"]),
                offset,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and candidate["candidate_id"] not in used_ids
            and (offset, candidate["structure"])
            not in used_graph_signatures
        ]
        ranked = _category_ranking(
            group, category, service_level, boundary_width, selected
        )
        preferred = (
            [
                candidate for candidate in ranked
                if candidate["structure"] not in used_structures
            ]
            if config["prefer_unique_structures"] else ranked
        )
        available = preferred or ranked
        if not available:
            raise RuntimeError(
                "Adaptive candidate pool does not contain enough distinct "
                f"graphs for due-date offset {offset:.3f}."
            )
        candidate = available[0]
        selected.append({
            "candidate": candidate,
            "category": f"adaptive_delta_{offset:.3f}:{category}",
        })
        used_ids.add(candidate["candidate_id"])
        used_graph_signatures.add((offset, candidate["structure"]))
        used_structures.add(candidate["structure"])

    actual_categories = {
        entry["category"].split(":", 1)[1] for entry in selected
    }
    if set(ADAPTIVE_SELECTION_CATEGORIES) - actual_categories:
        raise RuntimeError(
            "Adaptive candidate selection failed to cover every configured "
            "selection category."
        )
    if Counter(
        float(entry["candidate"]["row"]["training_due_date_delta"])
        for entry in selected
    ) != Counter(offset_targets):
        raise RuntimeError(
            "Adaptive candidate selection failed to meet its offset quotas."
        )
    return selected


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


def _candidate_mean_completion_delay(candidate):
    values = candidate.get("simulated_job_expected_completion_delays")
    if not values:
        raise ValueError(
            "Candidate selection requires simulated completion-delay "
            "labels for every job."
        )
    values = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(
            "Simulated completion-delay labels must be finite and "
            "nonnegative."
        )
    return sum(values) / len(values)


def _completion_delay_rankings(candidates):
    delay_by_id = {
        candidate["candidate_id"]: _candidate_mean_completion_delay(candidate)
        for candidate in candidates
    }
    median_delay = statistics.median(delay_by_id.values())
    return {
        "low_expected_completion_delay": sorted(
            candidates,
            key=lambda item: (
                delay_by_id[item["candidate_id"]],
                item["pool_objective"],
            ),
        ),
        "medium_expected_completion_delay": sorted(
            candidates,
            key=lambda item: (
                abs(delay_by_id[item["candidate_id"]] - median_delay),
                item["pool_objective"],
            ),
        ),
        "high_expected_completion_delay": sorted(
            candidates,
            key=lambda item: (
                -delay_by_id[item["candidate_id"]],
                item["pool_objective"],
            ),
        ),
    }


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
        "pareto_tradeoff": sorted(
            candidates,
            key=lambda item: (
                item["pareto_rank"],
                max(item["objective_norm"], item["risk_norm"]),
                item["balanced_score"],
            ),
        ),
        **_completion_delay_rankings(candidates),
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
        "service_boundary_below": boundary_fallback(below),
        "service_boundary_above": boundary_fallback(above),
        "pareto_tradeoff": sorted(
            candidates,
            key=lambda item: (
                item["pareto_rank"],
                max(item["objective_norm"], item["risk_norm"]),
                item["balanced_score"],
            ),
        ),
        **_completion_delay_rankings(candidates),
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


def _configured_fix_ratios(fixed):
    raw_fix_ratios = fixed.get("fix_ratios")
    fix_ratios = (
        [_validate_ratio(value, "fix_ratio") for value in raw_fix_ratios]
        if raw_fix_ratios else [None]
    )
    if len(fix_ratios) != len(set(fix_ratios)):
        raise ValueError("fixed_y.fix_ratios must not contain duplicates.")
    return fix_ratios


def _run_neighborhood(
    instance,
    instance_name,
    run_index,
    generation,
    graph_config,
    fix_ratio,
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
    _sample_reliability(sample_instance, rng, generation, run_index)
    _sample_training_due_dates(sample_instance, rng, generation, run_index)
    fixed = generation["fixed_y"]
    label_center, label_half_width = _label_distribution_parameters(fixed)
    service_probability_band = None
    if candidate_probability_band is not None:
        if candidate_generation_mode != "nonlinear":
            raise ValueError(
                "Probability-band generation requires "
                "candidate_generation_mode='nonlinear'."
            )
        service_probability_band = _probability_band_limits(
            candidate_probability_band,
            label_center,
            label_half_width,
        )
    model, variables = _build_candidate_model(
        sample_instance,
        graph_config,
        candidate_generation_mode,
        service_probability_band,
    )
    if candidate_generation_mode == "nominal_ontime":
        _add_nominal_ontime_constraints(model, variables, sample_instance)
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
        use_incumbent=use_incumbent,
        reference_assignment=getattr(
            sample_instance, "nominal_calibration_assignment", None
        ),
    )
    calibration_assignment = getattr(
        sample_instance, "nominal_calibration_assignment", None
    )
    if calibration_assignment and not use_incumbent:
        for operation, machine in calibration_assignment.items():
            variables["Y"][operation, machine].Start = 1.0
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
        signature = input_signature(candidate)
        if signature in local_structures:
            continue
        candidate["row"]["instance_name"] = instance_name
        try:
            _evaluate_fixed_schedule_simulation(candidate, generation.get("labels"))
        except InvalidScheduleError as error:
            print(f"[Invalid candidate] {instance_name}: {error}", flush=True)
            continue
        local_structures.add(signature)
        candidates.append(candidate)
    print(
        f"[Local buffer labels] {instance_name} | run={run_index + 1} | "
        f"mode={candidate_generation_mode} | "
        f"band={candidate_probability_band or 'service_feasible'} | "
        f"candidates={len(candidates)}",
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


def _validate_final_job_probability_coverage(
    fixed,
    split,
    probability_counts,
):
    """Require final MC job labels on both sides of the service boundary."""
    hybrid = _hybrid_selection_for_split(fixed, split)
    coverage = dict(
        (hybrid or {}).get("final_job_probability_coverage") or {}
    )
    if not coverage:
        return None
    apply_to = {
        str(value)
        for value in coverage.get("apply_to_splits", ["training"])
    }
    unknown = apply_to - {"training", "valid", "test"}
    if unknown:
        raise ValueError(
            "final_job_probability_coverage.apply_to_splits contains "
            f"unknown splits: {sorted(unknown)}"
        )
    if split not in apply_to:
        return None
    minimum_ratios = dict(coverage.get("minimum_ratios") or {})
    required_names = {"boundary_below", "boundary_above"}
    unknown = set(minimum_ratios) - required_names
    if unknown:
        raise ValueError(
            "final_job_probability_coverage.minimum_ratios contains "
            f"unknown bins: {sorted(unknown)}"
        )
    total = sum(probability_counts.values())
    if total <= 0:
        raise RuntimeError(
            f"No final job-probability labels were generated for {split}."
        )
    failures = []
    observed = {}
    for name in sorted(required_names):
        minimum = _validate_ratio(
            minimum_ratios.get(name, 0.0),
            f"final_job_probability_coverage.minimum_ratios.{name}",
        )
        actual = probability_counts[name] / total
        observed[name] = actual
        if actual + 1e-12 < minimum:
            failures.append(
                f"{name}={actual:.3f} < required {minimum:.3f}"
            )
    if failures:
        raise RuntimeError(
            f"Final job-level boundary coverage failed for {split}: "
            + "; ".join(failures)
        )
    return {
        "labels": total,
        "observed_ratios": observed,
        "minimum_ratios": {
            name: float(minimum_ratios.get(name, 0.0))
            for name in sorted(required_names)
        },
    }


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
    minimum_count,
    *,
    start_run=0,
    progress=None,
    hybrid_selection=None,
    candidate_generation_mode="unconstrained",
    minimum_runs_override=None,
    maximum_runs_override=None,
    probability_band_generation=None,
    selection_index=0,
):
    candidate_generation_mode = _validate_candidate_generation_mode(
        candidate_generation_mode
    )
    fixed = generation["fixed_y"]
    fix_ratios = _configured_fix_ratios(fixed)
    pool_candidates = int(fixed.get("pool_candidates", 50))
    if pool_candidates <= 0:
        raise ValueError("fixed_y.pool_candidates must be positive.")
    planned_runs = math.ceil(int(minimum_count) / pool_candidates)
    configured_minimum_runs = int(
        minimum_runs_override
        if minimum_runs_override is not None
        else fixed.get("minimum_candidate_pool_runs", len(fix_ratios))
    )
    if configured_minimum_runs <= 0:
        raise ValueError(
            "fixed_y.minimum_candidate_pool_runs must be positive."
        )
    minimum_runs = (
        configured_minimum_runs
        if minimum_runs_override is not None
        else max(len(fix_ratios), configured_minimum_runs)
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
    adaptive_due_dates = dict(generation.get("adaptive_due_dates") or {})
    if bool(adaptive_due_dates.get("enabled", False)):
        adaptive_maximum_runs = int(adaptive_due_dates.get(
            "maximum_candidate_pool_runs",
            max(minimum_runs * 3, 12),
        ))
        if adaptive_maximum_runs < minimum_runs:
            raise ValueError(
                "adaptive_due_dates.maximum_candidate_pool_runs must be at "
                f"least {minimum_runs}."
            )
        maximum_runs = max(maximum_runs, adaptive_maximum_runs)
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
        fix_ratio = fix_ratios[run_offset % len(fix_ratios)]
        candidates.extend(_run_neighborhood(
            instance,
            instance_name,
            run_index,
            generation,
            graph_config,
            fix_ratio,
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
                len(candidates),
            )
        if fixed.get("selection_method") == "buffer_structure_cost":
            if (run_offset + 1 >= minimum_runs
                    and len(unique_input_candidates(candidates)) >= minimum_count):
                break
            continue
        if (
            run_offset + 1 >= minimum_runs
            and len(candidates) >= minimum_count
            and _adaptive_due_date_coverage_met(
                candidates, minimum_count, generation, selection_index
            )
        ):
            if not hybrid_selection or _hybrid_boundary_targets_met(
                candidates,
                minimum_count,
                *_label_distribution_parameters(fixed),
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
    count,
    *,
    split=None,
    start_run=0,
    progress=None,
    selection_index=0,
):
    fixed = generation["fixed_y"]
    if fixed.get("selection_method") == "buffer_structure_cost":
        if graph_config.service_scope != "job":
            raise ValueError("Local job-buffer training requires service_scope='job'.")
        candidates = _collect_instance_candidates(
            instance, instance_name, generation, graph_config, int(count),
            start_run=start_run, progress=progress,
            candidate_generation_mode="nonlinear_evaluated",
            maximum_runs_override=int(fixed.get("maximum_candidate_pool_runs", 4)),
            selection_index=selection_index,
        )
        return candidates, select_buffer_candidates(candidates, count)
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
            int(count),
            start_run=start_run,
            progress=progress,
            hybrid_selection=hybrid_selection,
            candidate_generation_mode=candidate_generation_mode,
            probability_band_generation=probability_band_generation,
            selection_index=selection_index,
        )
        if _adaptive_due_date_offsets(generation):
            selected = _select_adaptive_due_date_candidates(
                candidates,
                count,
                fixed.get("pool_selection_ratios"),
                *_label_distribution_parameters(fixed),
                generation,
                selection_index,
            )
        else:
            selected = _select_candidates(
                candidates,
                count,
                fixed.get("pool_selection_ratios"),
                *_label_distribution_parameters(fixed),
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
        *_label_distribution_parameters(fixed),
        hybrid_selection,
        mixed_generation["nominal_selected_fraction"],
    )
    return unconstrained + nominal, selected


def generate_from_config(generation):
    """Generate current-schema GNN graphs from fixed optimization neighborhoods."""
    generation = dict(generation)
    if generation.get("method") != "fix_and_optimize":
        raise ValueError(
            "Fix-and-optimize generation requires method='fix_and_optimize'."
        )
    graph_config = normalize_reliability_graph_config(
        generation.get("reliability_graph")
    )
    if graph_config.service_scope != "job":
        raise ValueError("Local buffer generation requires service_scope='job'.")
    samples_per_instance = int(generation["samples_per_instance"])
    if samples_per_instance <= 0:
        raise ValueError("samples_per_instance must be positive.")
    fixed = dict(generation.get("fixed_y") or {})
    _configured_fix_ratios(fixed)
    failure_handling = _normalize_instance_failure_handling(
        generation.get("instance_failure_handling")
    )
    generation["fixed_y"] = fixed
    generation["random_seed"] = int(generation.get("random_seed", 42))
    generation["labels"] = label_config_dict(generation.get("labels"))
    fixed.setdefault("selection_method", "buffer_structure_cost")
    output_root = Path(generation["output_directory"])
    if not output_root.is_absolute():
        output_root = ROOT_DIR / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    splits = selected_instance_splits(
        generation["instance_splits"], generation.get("generate_splits")
    )
    expected_scale_factors = generation.get("weibull_scale_factors", [1.0])
    split_quality = {
        split: _new_quality_accumulator(expected_scale_factors)
        for split in splits
    }
    total_quality = _new_quality_accumulator(expected_scale_factors)
    summary_path = output_root / failure_handling["summary_filename"]
    summary = {
        "schema_version": 15,
        "status": "running",
        "started_at_utc": _utc_timestamp(),
        "completed_at_utc": None,
        "output_directory": str(output_root),
        "samples_per_instance": samples_per_instance,
        "training_due_date_scale_factors": generation.get(
            "due_date_scale_factors", [1.0]
        ),
        "adaptive_due_dates": dict(generation.get("adaptive_due_dates") or {}),
        "training_weibull_scale_factors": generation.get(
            "weibull_scale_factors", [1.0]
        ),
        "machine_profile_config": generation.get("machine_profile_config"),
        "training_parameter_jitter": generation.get(
            "training_parameter_jitter"
        ),
        "time_unit": normalize_time_unit(generation),
        "label": {
            "target_column": TARGET_COLUMN,
            "label_method": SIMULATION_LABEL_METHOD,
            "source": LABEL_SOURCE,
            "parameters": generation["labels"],
            "semantics": "Expected sum of residual repairs at nominal operation midpoints; no propagation or gap absorption.",
        },
        "candidate_selection": {
            "method": fixed.get("selection_method"),
            "deduplication": "features_rounded7_job_membership_directed_multiedges",
            "nominal_cost": "processing_cost_plus_nominal_makespan_plus_nominal_tardiness",
            "probability_diagnostics": "Markov lower bounds for local buffers, not MC service rates",
        },
        "graph": {
            "graph_schema": RELIABILITY_GNN_GRAPH_SCHEMA,
            "feature_names": reliability_node_feature_names(graph_config),
            "service_scope": graph_config.service_scope,
            "include_machine_predecessor_edges": True,
            "machine_predecessor_edge_scope": "direct",
            "include_job_precedence_edges": True,
        },
        "failure_handling": dict(failure_handling),
        "splits": {
            split: {
                "configured_count": len(instance_names),
                "configured_instances": list(instance_names),
                "successful_instances": [],
                "skipped_instances": [],
                "written_graphs": 0,
                "quality_report": _quality_report(split_quality[split]),
            }
            for split, instance_names in splits.items()
        },
        "quality_report": _quality_report(total_quality),
    }
    # A partial regeneration must not relabel incompatible files left on disk.
    retained = [split for split in SPLIT_DIRECTORIES if split not in splits
                and (output_root / SPLIT_DIRECTORIES[split] / SPLIT_CSV_FILENAMES[split]).exists()]
    if retained:
        if not summary_path.exists():
            raise ValueError("Existing unselected splits have no label metadata; regenerate all splits or use a new directory.")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") != "completed" or any(
            previous.get(key) != summary[key] for key in (
                "label", "graph", "time_unit", "machine_profile_config",
                "training_parameter_jitter",
            )
        ):
            raise ValueError("Existing unselected splits are incompatible; regenerate all splits or use a new directory.")
        for split in retained:
            summary["splits"][split] = previous["splits"][split]
            with (output_root / SPLIT_DIRECTORIES[split] / SPLIT_CSV_FILENAMES[split]).open(newline="", encoding="utf-8") as handle:
                _update_quality_accumulator(total_quality, csv.DictReader(handle))
        summary["quality_report"] = _quality_report(total_quality)
    _write_generation_summary(summary_path, summary)

    try:
        for split, instance_names in splits.items():
            output_path = (
                output_root / SPLIT_DIRECTORIES[split]
                / SPLIT_CSV_FILENAMES[split]
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            split_summary = summary["splits"][split]
            split_probability_counts = Counter()
            with output_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                file.flush()
                for selection_index, instance_name in enumerate(instance_names):
                    def report(run_index, fix_ratio, candidate_count):
                        print(
                            f"[Fix-and-optimize] {split} | "
                            f"{instance_name} | run={run_index + 1} | "
                            f"graphs={candidate_count} candidates | "
                            f"fix_ratio="
                            f"{fix_ratio if fix_ratio is not None else 'random'}",
                            flush=True,
                        )

                    try:
                        instance = load_generated_instance(instance_name)
                        candidates, selected = (
                            _collect_and_select_instance_candidates(
                                instance,
                                instance_name,
                                generation,
                                graph_config,
                                samples_per_instance,
                                split=split,
                                progress=report,
                                selection_index=selection_index,
                            )
                        )
                        if len(selected) < samples_per_instance:
                            raise RuntimeError(
                                f"Only {len(selected)}/"
                                f"{samples_per_instance} structurally "
                                "unique candidate graphs selected for "
                                f"{instance_name} from "
                                f"{len(candidates)} candidates."
                            )
                        rows = []
                        instance_probability_counts = Counter()
                        for entry in selected:
                            candidate = entry["candidate"]
                            row = candidate["row"]
                            row["pool_selection_category"] = entry["category"]
                            rows.append(row)
                            instance_probability_counts.update(
                                _probability_bin_counts(
                                    candidate["job_probabilities"],
                                    *_label_distribution_parameters(fixed),
                                )
                            )
                    except Exception as error:
                        split_summary["skipped_instances"].append({
                            "instance_name": instance_name,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        })
                        _write_generation_summary(summary_path, summary)
                        print(
                            f"[Fix-and-optimize:skip] {split} | "
                            f"{instance_name} | {type(error).__name__}: "
                            f"{error}",
                            flush=True,
                        )
                        continue

                    writer.writerows(rows)
                    file.flush()
                    split_probability_counts.update(
                        instance_probability_counts
                    )
                    _update_quality_accumulator(split_quality[split], rows)
                    _update_quality_accumulator(total_quality, rows)
                    split_summary["quality_report"] = _quality_report(
                        split_quality[split]
                    )
                    summary["quality_report"] = _quality_report(total_quality)
                    split_summary["written_graphs"] += len(rows)
                    split_summary["successful_instances"].append({
                        "instance_name": instance_name,
                        "training_parameter_jitter_applied": bool(getattr(
                            instance,
                            "training_parameter_jitter_applied",
                            False,
                        )),
                        "candidate_count": len(candidates),
                        "written_graphs": len(rows),
                        "job_probability_distribution": dict(
                            instance_probability_counts
                        ),
                    })
                    _write_generation_summary(summary_path, summary)
                    print(
                        f"[Fix-and-optimize] {split} | {instance_name} | "
                        f"selected={len(rows)}/{len(candidates)} candidates",
                        flush=True,
                    )
                    print(
                        f"[Local Markov-bound distribution] {split} | "
                        f"{instance_name} | "
                        f"{_distribution_text(instance_probability_counts)}",
                        flush=True,
                    )

            print(
                f"[Fix-and-optimize] finished {split}: "
                f"{split_summary['written_graphs']} graphs, "
                f"{len(split_summary['skipped_instances'])} skipped -> "
                f"{output_path}",
                flush=True,
            )
            coverage = _validate_final_job_probability_coverage(
                fixed, split, split_probability_counts
            )
            split_summary["final_job_probability_coverage"] = coverage
            _write_generation_summary(summary_path, summary)
            print(
                f"[Local Markov-bound distribution] finished {split} | "
                f"{_distribution_text(split_probability_counts)}",
                flush=True,
            )
            print(
                _quality_report_text(
                    split, split_summary["quality_report"]
                ),
                flush=True,
            )
        summary["status"] = "completed"
        summary["completed_at_utc"] = _utc_timestamp()
        _write_generation_summary(summary_path, summary)
        print(
            _quality_report_text("overall", summary["quality_report"]),
            flush=True,
        )
        print(
            f"[Fix-and-optimize] generation summary -> {summary_path}",
            flush=True,
        )
        return summary
    except Exception as error:
        summary["status"] = "failed"
        summary["completed_at_utc"] = _utc_timestamp()
        summary["fatal_error"] = {
            "error_type": type(error).__name__,
            "message": str(error),
        }
        _write_generation_summary(summary_path, summary)
        raise


def generate_rows_for_instance(
    instance_name,
    generation,
    *,
    count,
    start_run=0,
    include_diagnostics=False,
):
    """Small in-memory generator used by smoke tests and numerical studies."""
    generation = dict(generation)
    graph_config = normalize_reliability_graph_config(
        generation.get("reliability_graph")
    )
    generation["fixed_y"] = dict(generation.get("fixed_y") or {})
    generation["random_seed"] = int(generation.get("random_seed", 42))
    generation["labels"] = label_config_dict(generation.get("labels"))
    generation["fixed_y"].setdefault("selection_method", "buffer_structure_cost")
    instance = load_generated_instance(instance_name)
    candidates, selected = _collect_and_select_instance_candidates(
        instance,
        instance_name,
        generation,
        graph_config,
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
        candidate = entry["candidate"]
        row = candidate["row"]
        if include_diagnostics:
            row = dict(row)
            row["_nonlinear_job_probabilities"] = _compact(
                candidate.get("nonlinear_job_probabilities", [])
            )
            row["_label_method"] = (
                SIMULATION_LABEL_METHOD
            )
        row["pool_selection_category"] = entry["category"]
        rows.append(row)
    return rows
