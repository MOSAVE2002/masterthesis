"""Create reproducible chapter-7 tables and plots from evaluation outputs.

The script never re-solves or re-simulates schedules.  It scopes rows through
the solve manifest when available, enriches them with instance dimensions,
checks experiment completeness, and derives paired solver, quality, surrogate,
and robustness summaries.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CONFIG = ROOT_DIR / "config.json"
DEFAULT_RESULTS = ROOT_DIR / "06_Evaluation" / "results"
DEFAULT_INSTANCES = ROOT_DIR / "02_data" / "fjsp_instances"
GENERATED_TIERS = ("benchmark", "extrapolation", "stress")
INSTANCE_PATTERN = re.compile(r"^i(?P<jobs>\d+)_k(?P<machines>\d+)_")
PROGRESS_GRID_POINTS = 201
PROGRESS_VALUE_FIELDS = (
    "runtime_seconds", "incumbent_objective", "best_bound",
    "relative_gap", "node_count", "solution_count", "event",
    "objective_sense",
)


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value):
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value):
    number = _float(value)
    return None if number is None else int(round(number))


def _bool(value):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "ja"}


def _quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values, prefix):
    """Return count, mean, median, quartiles, and IQR for finite values."""
    finite = [float(value) for value in values if _float(value) is not None]
    if not finite:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_q1": None,
            f"{prefix}_q3": None,
            f"{prefix}_iqr": None,
            f"{prefix}_minimum": None,
            f"{prefix}_maximum": None,
        }
    q1 = _quantile(finite, 0.25)
    q3 = _quantile(finite, 0.75)
    return {
        f"{prefix}_count": len(finite),
        f"{prefix}_mean": statistics.fmean(finite),
        f"{prefix}_median": statistics.median(finite),
        f"{prefix}_q1": q1,
        f"{prefix}_q3": q3,
        f"{prefix}_iqr": q3 - q1,
        f"{prefix}_minimum": min(finite),
        f"{prefix}_maximum": max(finite),
    }


def _canonical_path(value):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return str(path.resolve())


def _manifest_paths(path):
    if path is None or not Path(path).exists():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        canonical
        for run in payload.get("runs", [])
        if (canonical := _canonical_path(run.get("solution_path"))) is not None
    }


def _scope_to_manifest(rows, manifest_paths):
    if manifest_paths is None:
        return list(rows)
    return [
        row for row in rows
        if _canonical_path(row.get("solution_file")) in manifest_paths
    ]


def _condition(row):
    offset = _float(row.get("due_date_relative_makespan_offset"))
    if offset is not None:
        return f"offset:{offset:.12g}"
    factor = _float(row.get("due_date_factor"))
    if factor is not None:
        return f"factor:{factor:.12g}"
    return "configured"


def _case_key(row):
    return (
        row.get("physical_instance_id") or row.get("instance_name"),
        _condition(row),
    )


def _schedule_key(row):
    return (*_case_key(row), row.get("model"))


def _load_instance_metadata(instance_name, instances_root, cache):
    if instance_name in cache:
        return cache[instance_name]
    matches = sorted(Path(instances_root).rglob(f"{instance_name}.pkl"))
    if len(matches) != 1:
        cache[instance_name] = {}
        return cache[instance_name]
    module = importlib.import_module("01_generator.instance_generator")
    instance = module.load_generated_instance(
        matches[0].stem, instance_directory=matches[0].parent
    )
    eligible_assignments = sum(
        len(instance.eligible_machines[operation])
        for operation in instance.real_operations
    )
    cache[instance_name] = {
        "number_of_jobs": int(instance.num_jobs),
        "number_of_machines": int(instance.num_machines),
        "number_of_operations": len(instance.real_operations),
        "number_of_eligible_assignments": eligible_assignments,
    }
    return cache[instance_name]


def enrich_rows(schedule_rows, job_rows, instances_root):
    cache = {}
    enriched = []
    by_solution = {}
    for original in schedule_rows:
        row = dict(original)
        match = INSTANCE_PATTERN.match(str(row.get("instance_name", "")))
        if match:
            row.setdefault("number_of_jobs", match.group("jobs"))
            row.setdefault("number_of_machines", match.group("machines"))
        metadata = _load_instance_metadata(
            row.get("instance_name", ""), instances_root, cache
        )
        for key, value in metadata.items():
            if row.get(key) in (None, ""):
                row[key] = value
        row["due_date_condition"] = _condition(row)
        row["has_incumbent_derived"] = _bool(row.get("has_incumbent"))
        enriched.append(row)
        by_solution[_canonical_path(row.get("solution_file"))] = row

    enriched_jobs = []
    for original in job_rows:
        row = dict(original)
        schedule = by_solution.get(_canonical_path(row.get("solution_file")), {})
        for key in (
            "physical_instance_id", "evaluation_tier", "number_of_jobs",
            "number_of_machines", "number_of_operations",
            "number_of_eligible_assignments",
            "due_date_relative_makespan_offset", "due_date_condition",
        ):
            if row.get(key) in (None, "") and schedule.get(key) not in (None, ""):
                row[key] = schedule[key]
        row.setdefault("due_date_condition", _condition(row))
        enriched_jobs.append(row)
    return enriched, enriched_jobs


def _expanded_gnn_model_names(config):
    requested = [str(value).lower() for value in config["solve"]["solvers"]]
    names = []
    if "gurobi" in requested:
        names.append("nominal")
    if "gurobi_nonlinear" in requested:
        names.append("nonlinear_midpoint")
    if "gurobi_gnn" in requested:
        architecture = importlib.import_module(
            "04_GraphNeuralNetworks.models.gnn_architecture"
        )
        for raw in config["training"]["gnn"]["combinations"]:
            for item in architecture.expand_architecture_variants(raw):
                names.append(
                    f"gnn_{item['convolution']}_layers{item['layers']}_"
                    f"hidden{item['hidden_channels']}"
                )
    return names


def _tier_conditions(tier):
    due = tier.get("due_dates")
    if due is not None:
        raw = due.get("relative_makespan_offsets")
        if raw is None:
            raw = [due.get("relative_makespan_offset", 0.0)]
        return [f"offset:{float(value):.12g}" for value in raw]
    raw = tier.get("due_date_factors")
    if raw is None:
        raw = [tier.get("due_date_factor")]
    return [
        "configured" if value is None else f"factor:{float(value):.12g}"
        for value in raw
    ]


def _expected_keys(config):
    evaluation = config.get("solve", {}).get("evaluation", {})
    generation = config["instances"]["generation"]
    operations = generation["operations_per_job"]
    models = _expanded_gnn_model_names(config)
    expected = set()
    for tier_name in GENERATED_TIERS:
        tier = evaluation.get(tier_name, {})
        if not tier.get("enabled", False):
            continue
        tier_operations = tier.get("operations_per_job", operations)
        for jobs in tier["num_jobs"]:
            for machines in tier["num_machines"]:
                for index in range(1, int(tier["instances_per_size"]) + 1):
                    physical = (
                        f"{tier_name}/i{int(jobs)}_k{int(machines)}_"
                        f"o{min(tier_operations)}-{max(tier_operations)}_{index}"
                    )
                    for condition in _tier_conditions(tier):
                        for model in models:
                            expected.add((physical, condition, model))
    return expected


def completeness_checks(
    rows, job_rows, config, manifest_paths, expected_replications
):
    checks = []

    def add(name, passed, expected, observed, details=""):
        checks.append({
            "check": name,
            "status": "passed" if passed else "failed",
            "expected": expected,
            "observed": observed,
            "details": details,
        })

    keys = [_schedule_key(row) for row in rows]
    duplicate_count = len(keys) - len(set(keys))
    add("unique_schedule_keys", duplicate_count == 0, 0, duplicate_count)

    expected = _expected_keys(config)
    observed = {
        _schedule_key(row) for row in rows
        if row.get("evaluation_tier") in GENERATED_TIERS
    }
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    add(
        "configured_schedule_matrix",
        not missing and not unexpected,
        len(expected),
        len(observed),
        f"missing={len(missing)}; unexpected={len(unexpected)}",
    )

    if manifest_paths is not None:
        represented = {
            _canonical_path(row.get("solution_file")) for row in rows
        }
        missing_manifest = sorted(manifest_paths - represented)
        add(
            "manifest_coverage",
            not missing_manifest,
            len(manifest_paths),
            len(represented & manifest_paths),
            f"missing_manifest_rows={len(missing_manifest)}",
        )
    else:
        add("manifest_coverage", False, "manifest required", 0, "not provided")

    seeds = defaultdict(set)
    for row in rows:
        seed = row.get("simulation_seed")
        if seed not in (None, ""):
            seeds[_case_key(row)[0]].add(str(seed))
    invalid_seed_groups = sorted(key for key, values in seeds.items() if len(values) != 1)
    add(
        "common_random_numbers",
        not invalid_seed_groups,
        "one simulation seed per physical instance",
        len(invalid_seed_groups),
        "; ".join(invalid_seed_groups[:10]),
    )

    wrong_replications = sum(
        1 for row in rows
        if row.get("postsolve_evaluation_status") == "evaluated"
        and _int(row.get("simulation_replications")) != expected_replications
    )
    add(
        "simulation_replications",
        wrong_replications == 0,
        expected_replications,
        f"mismatching_rows={wrong_replications}",
    )

    schedule_solutions = {
        _canonical_path(row.get("solution_file")) for row in rows
    }
    orphan_jobs = sum(
        _canonical_path(row.get("solution_file")) not in schedule_solutions
        for row in job_rows
    )
    add("job_rows_link_to_schedules", orphan_jobs == 0, 0, orphan_jobs)

    progress_files = [
        _canonical_path(row.get("solver_progress_file")) for row in rows
    ]
    available_progress = sum(
        path is not None and Path(path).is_file() for path in progress_files
    )
    add(
        "solver_progress_coverage",
        available_progress == len(rows),
        len(rows),
        available_progress,
        "one callback trajectory per evaluated solver run",
    )
    valid_progress = 0
    invalid_progress = []
    for row, path_value in zip(rows, progress_files):
        if path_value is None or not Path(path_value).is_file():
            continue
        try:
            trace = _read_csv(path_value)
            required_columns = set(PROGRESS_VALUE_FIELDS)
            stored_points = _int(row.get("solver_progress_points"))
            integrity_ok = (
                bool(trace)
                and required_columns <= set(trace[0])
                and trace[-1].get("event") == "final"
                and (stored_points is None or stored_points == len(trace))
            )
        except (OSError, csv.Error, ValueError) as error:
            integrity_ok = False
            invalid_progress.append(
                f"{row.get('solution_file')}: {type(error).__name__}"
            )
        if integrity_ok:
            valid_progress += 1
        else:
            invalid_progress.append(str(row.get("solution_file")))
    add(
        "solver_progress_integrity",
        valid_progress == len(rows),
        len(rows),
        valid_progress,
        "; ".join(invalid_progress[:10]),
    )
    return checks


def _group(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in fields)].append(row)
    return grouped


def _base_group_row(fields, key):
    return dict(zip(fields, key))


def _target_reached(row, target_gap):
    if not _bool(row.get("has_incumbent")):
        return False
    if str(row.get("status", "")).upper() == "OPTIMAL":
        return True
    gap = _float(row.get("mip_gap"))
    return gap is not None and gap <= target_gap + 1e-12


def solver_summary(rows, fields, time_limit, target_gap):
    summaries = []
    for key, values in sorted(_group(rows, fields).items()):
        runtimes = [_float(row.get("runtime_seconds")) for row in values]
        runtimes = [value for value in runtimes if value is not None]
        incumbent = [_bool(row.get("has_incumbent")) for row in values]
        target = [_target_reached(row, target_gap) for row in values]
        optimal = [str(row.get("status", "")).upper() == "OPTIMAL" for row in values]
        timeouts = [str(row.get("status", "")).upper() == "TIME_LIMIT" for row in values]
        par2 = []
        for row, reached in zip(values, target):
            runtime = _float(row.get("runtime_seconds"))
            par2.append(
                runtime if reached and runtime is not None
                else (time_limit if reached else 2.0 * time_limit)
            )
        summary = _base_group_row(fields, key)
        summary.update({
            "runs": len(values),
            "incumbent_count": sum(incumbent),
            "incumbent_rate": sum(incumbent) / len(values),
            "optimal_count": sum(optimal),
            "optimal_rate": sum(optimal) / len(values),
            "target_gap_count": sum(target),
            "target_gap_rate": sum(target) / len(values),
            "time_limit_count": sum(timeouts),
            "time_limit_rate": sum(timeouts) / len(values),
            **describe(runtimes, "runtime_seconds"),
            **describe(par2, "par2_target_gap_seconds"),
        })
        summaries.append(summary)
    return summaries


def model_size_summary(rows):
    fields = (
        "evaluation_tier", "number_of_jobs", "number_of_machines", "model"
    )
    metrics = (
        "number_of_variables", "number_of_continuous_variables",
        "number_of_binary_variables", "number_of_integer_variables",
        "number_of_linear_matrix_nonzeros", "number_of_linear_constraints",
        "number_of_quadratic_constraints", "number_of_general_constraints",
        "number_of_nonlinear_constraints",
    )
    output = []
    for key, values in sorted(_group(rows, fields).items()):
        row = _base_group_row(fields, key)
        row["runs"] = len(values)
        for metric in metrics:
            observed = [_float(item.get(metric)) for item in values]
            row.update(describe(observed, metric))
        output.append(row)
    return output


def _add_reference_cost_gaps(rows):
    best = {}
    for row in rows:
        value = _float(row.get("reference_total_cost"))
        if value is not None:
            key = _case_key(row)
            best[key] = value if key not in best else min(best[key], value)
    output = []
    for original in rows:
        row = dict(original)
        value = _float(row.get("reference_total_cost"))
        baseline = best.get(_case_key(row))
        row["reference_cost_gap_percent"] = (
            100.0 * (value - baseline) / baseline
            if value is not None and baseline is not None and abs(baseline) > 1e-12
            else None
        )
        output.append(row)
    return output


def quality_summary(rows):
    fields = (
        "evaluation_tier", "number_of_jobs", "number_of_machines",
        "due_date_condition", "model",
    )
    metrics = (
        "reference_total_cost", "reference_cost_gap_percent",
        "reference_due_date_violation", "maximum_repair_buffer_underestimation",
        "total_tardiness", "nominal_makespan",
    )
    output = []
    for key, values in sorted(_group(rows, fields).items()):
        row = _base_group_row(fields, key)
        row["evaluated_schedules"] = sum(
            item.get("postsolve_evaluation_status") == "evaluated"
            for item in values
        )
        for metric in metrics:
            row.update(describe(
                [_float(item.get(metric)) for item in values], metric
            ))
        output.append(row)
    return output


def paired_model_comparisons(rows):
    """Compare every model with nominal and nonlinear on identical cases."""
    pair_rows = []
    for values in _group(
        rows, ("physical_instance_id", "due_date_condition")
    ).values():
        by_model = {row.get("model"): row for row in values}
        for reference in ("nominal", "nonlinear_midpoint"):
            if reference not in by_model:
                continue
            reference_row = by_model[reference]
            for model, candidate in by_model.items():
                if model == reference:
                    continue
                pair = {
                    "evaluation_tier": candidate.get("evaluation_tier"),
                    "number_of_jobs": candidate.get("number_of_jobs"),
                    "number_of_machines": candidate.get("number_of_machines"),
                    "due_date_condition": candidate.get("due_date_condition"),
                    "model": model,
                    "reference_model": reference,
                }
                for metric in (
                    "reference_total_cost", "reference_cost_gap_percent",
                    "runtime_seconds", "minimum_mc_ontime_probability",
                    "maximum_repair_buffer_underestimation",
                ):
                    candidate_value = _float(candidate.get(metric))
                    reference_value = _float(reference_row.get(metric))
                    pair[f"{metric}_difference"] = (
                        candidate_value - reference_value
                        if candidate_value is not None
                        and reference_value is not None else None
                    )
                pair_rows.append(pair)

    fields = (
        "evaluation_tier", "number_of_jobs", "number_of_machines",
        "due_date_condition", "model", "reference_model",
    )
    metrics = (
        "reference_total_cost_difference",
        "reference_cost_gap_percent_difference", "runtime_seconds_difference",
        "minimum_mc_ontime_probability_difference",
        "maximum_repair_buffer_underestimation_difference",
    )
    output = []
    for key, values in sorted(_group(pair_rows, fields).items()):
        row = _base_group_row(fields, key)
        row["paired_cases"] = len(values)
        for metric in metrics:
            row.update(describe(
                [_float(item.get(metric)) for item in values], metric
            ))
        cost_differences = [
            _float(item.get("reference_total_cost_difference"))
            for item in values
        ]
        cost_differences = [
            value for value in cost_differences if value is not None
        ]
        row["reference_cost_win_rate"] = (
            sum(value < -1e-12 for value in cost_differences)
            / len(cost_differences) if cost_differences else None
        )
        output.append(row)
    return output


def due_date_effects(rows):
    """Return paired changes from the tightest to every looser due-date tier."""
    paired = []
    grouped = _group(rows, ("physical_instance_id", "model"))
    for values in grouped.values():
        by_offset = {
            _float(row.get("due_date_relative_makespan_offset")): row
            for row in values
            if _float(row.get("due_date_relative_makespan_offset")) is not None
        }
        if len(by_offset) < 2:
            continue
        baseline_offset = min(by_offset)
        baseline = by_offset[baseline_offset]
        for offset, candidate in sorted(by_offset.items()):
            if offset == baseline_offset:
                continue
            pair = {
                "evaluation_tier": candidate.get("evaluation_tier"),
                "number_of_jobs": candidate.get("number_of_jobs"),
                "number_of_machines": candidate.get("number_of_machines"),
                "model": candidate.get("model"),
                "baseline_due_date_offset": baseline_offset,
                "comparison_due_date_offset": offset,
            }
            for metric in (
                "reference_total_cost", "runtime_seconds",
                "minimum_mc_ontime_probability",
                "reference_due_date_violation", "total_tardiness",
            ):
                candidate_value = _float(candidate.get(metric))
                baseline_value = _float(baseline.get(metric))
                pair[f"{metric}_change"] = (
                    candidate_value - baseline_value
                    if candidate_value is not None
                    and baseline_value is not None else None
                )
            paired.append(pair)

    fields = (
        "evaluation_tier", "number_of_jobs", "number_of_machines", "model",
        "baseline_due_date_offset", "comparison_due_date_offset",
    )
    metrics = (
        "reference_total_cost_change", "runtime_seconds_change",
        "minimum_mc_ontime_probability_change",
        "reference_due_date_violation_change", "total_tardiness_change",
    )
    output = []
    for key, values in sorted(_group(paired, fields).items()):
        row = _base_group_row(fields, key)
        row["paired_physical_instances"] = len(values)
        for metric in metrics:
            row.update(describe(
                [_float(item.get(metric)) for item in values], metric
            ))
        output.append(row)
    return output


def surrogate_summary(job_rows):
    fields = (
        "evaluation_tier", "number_of_jobs", "number_of_machines",
        "due_date_condition", "model",
    )
    output = []
    for key, values in sorted(_group(job_rows, fields).items()):
        if not str(key[-1]).startswith("gnn_"):
            continue
        errors = [
            _float(item.get("repair_buffer_error")) for item in values
        ]
        errors = [value for value in errors if value is not None]
        if not errors:
            continue
        under = [max(0.0, -value) for value in errors]
        output.append({
            **_base_group_row(fields, key),
            "job_predictions": len(errors),
            "mae": statistics.fmean(abs(value) for value in errors),
            "rmse": math.sqrt(statistics.fmean(value * value for value in errors)),
            "mean_error_bias": statistics.fmean(errors),
            "absolute_error_p95": _quantile(
                [abs(value) for value in errors], 0.95
            ),
            "maximum_underestimation": max(under),
            "underestimation_rate": sum(value > 0.0 for value in under) / len(under),
        })
    return output


def _schedule_job_metrics(job_rows):
    output = {}
    grouped = _group(job_rows, ("solution_file",))
    for (solution_file,), values in grouped.items():
        probabilities = [
            _float(row.get("mc_ontime_probability")) for row in values
        ]
        probabilities = [value for value in probabilities if value is not None]
        delays = [_float(row.get("mc_mean_completion_delay")) for row in values]
        delays = [value for value in delays if value is not None]
        output[_canonical_path(solution_file)] = {
            "mean_mc_ontime_probability": (
                statistics.fmean(probabilities) if probabilities else None
            ),
            "mean_mc_completion_delay": (
                statistics.fmean(delays) if delays else None
            ),
            "jobs_meeting_service_threshold_rate": (
                statistics.fmean(
                    _bool(row.get("mc_meets_service_threshold")) for row in values
                ) if values else None
            ),
        }
    return output


def robustness_summary(rows, job_rows):
    job_metrics = _schedule_job_metrics(job_rows)
    merged = []
    for original in rows:
        row = dict(original)
        row.update(job_metrics.get(_canonical_path(row.get("solution_file")), {}))
        merged.append(row)
    fields = (
        "evaluation_tier", "number_of_jobs", "number_of_machines",
        "due_date_condition", "model",
    )
    metrics = (
        "minimum_mc_ontime_probability", "mean_mc_ontime_probability",
        "minimum_wilson_lower_bound",
        "minimum_bonferroni_wilson_lower_bound",
        "mean_mc_completion_delay", "jobs_meeting_service_threshold_rate",
        "simulation_mean_failures", "simulation_mean_total_repair_delay",
    )
    output = []
    for key, values in sorted(_group(merged, fields).items()):
        row = _base_group_row(fields, key)
        row["evaluated_schedules"] = sum(
            item.get("postsolve_evaluation_status") == "evaluated"
            for item in values
        )
        for metric in metrics:
            row.update(describe(
                [_float(item.get(metric)) for item in values], metric
            ))
        output.append(row)
    return output


def ecdf_rows(rows, time_limit, target_gap):
    output = []
    for (model,), values in sorted(_group(rows, ("model",)).items()):
        solved_times = sorted(
            _float(row.get("runtime_seconds"))
            for row in values if _target_reached(row, target_gap)
            and _float(row.get("runtime_seconds")) is not None
        )
        thresholds = sorted({0.0, time_limit, *solved_times})
        for threshold in thresholds:
            output.append({
                "model": model,
                "threshold_seconds": threshold,
                "fraction_reaching_target_gap": sum(
                    value <= threshold + 1e-12 for value in solved_times
                ) / len(values),
                "case_count": len(values),
            })
    return output


def performance_profile_rows(rows, target_gap):
    models = sorted({row.get("model") for row in rows})
    cases = _group(rows, ("physical_instance_id", "due_date_condition"))
    ratios = {model: [] for model in models}
    for values in cases.values():
        by_model = {row.get("model"): row for row in values}
        times = {
            model: _float(by_model[model].get("runtime_seconds"))
            for model in models
            if model in by_model and _target_reached(by_model[model], target_gap)
        }
        finite = [value for value in times.values() if value is not None]
        best = min(finite) if finite else None
        for model in models:
            value = times.get(model)
            ratios[model].append(
                value / best if best is not None and value is not None else math.inf
            )
    finite_ratios = sorted({
        value for values in ratios.values() for value in values
        if math.isfinite(value)
    })
    thresholds = sorted({1.0, *finite_ratios})
    output = []
    for model in models:
        values = ratios[model]
        for threshold in thresholds:
            output.append({
                "model": model,
                "performance_ratio": threshold,
                "fraction_within_ratio": sum(
                    value <= threshold + 1e-12 for value in values
                ) / len(values) if values else None,
                "case_count": len(values),
            })
    return output


def collect_solver_progress(schedule_rows):
    """Load trajectory sidecars and attach the schedule comparison fields."""
    output = []
    metadata_fields = (
        "evaluation_tier", "physical_instance_id", "instance_name",
        "number_of_jobs", "number_of_machines", "due_date_condition",
        "solver", "model", "formulation", "solution_file",
    )
    for schedule in schedule_rows:
        path_value = schedule.get("solver_progress_file")
        path = Path(_canonical_path(path_value)) if path_value else None
        if path is None or not path.is_file():
            continue
        trace = _read_csv(path)
        for index, original in enumerate(trace):
            missing = set(PROGRESS_VALUE_FIELDS) - set(original)
            if missing:
                raise ValueError(
                    f"Malformed solver progress file {path}: "
                    f"missing columns {sorted(missing)}"
                )
            row = {
                field: schedule.get(field) for field in metadata_fields
            }
            row.update({field: original.get(field) for field in PROGRESS_VALUE_FIELDS})
            row["final_status"] = schedule.get("status")
            row["final_runtime_seconds"] = schedule.get("runtime_seconds")
            row["final_mip_gap"] = schedule.get("mip_gap")
            row["final_has_incumbent"] = schedule.get("has_incumbent")
            row["trace_row_index"] = index
            row["solver_progress_file"] = schedule.get("solver_progress_file")
            output.append(row)
    output.sort(key=lambda row: (
        str(row.get("evaluation_tier", "")),
        str(row.get("instance_name", "")),
        str(row.get("model", "")),
        _float(row.get("runtime_seconds")) or 0.0,
        int(row["trace_row_index"]),
    ))
    return output


def _progress_grid(time_limit, points=PROGRESS_GRID_POINTS):
    regular = {
        float(time_limit) * index / (int(points) - 1)
        for index in range(int(points))
    }
    early = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}
    return sorted({
        min(float(time_limit), value)
        for value in regular | early
        if 0.0 <= value <= float(time_limit)
    })


def _incumbent_improvement(first, current, objective_sense):
    if first is None or current is None:
        return None
    denominator = abs(first)
    if denominator <= 1e-10:
        return 0.0 if abs(current - first) <= 1e-10 else None
    if str(objective_sense).lower() == "maximize":
        return 100.0 * (current - first) / denominator
    return 100.0 * (first - current) / denominator


def solver_progress_summary(
    schedule_rows, progress_rows, fields, time_limit, target_gap
):
    """Aggregate stepwise MIP trajectories on a common time grid."""
    traces = defaultdict(list)
    for row in progress_rows:
        traces[_canonical_path(row.get("solution_file"))].append(row)
    for values in traces.values():
        values.sort(key=lambda row: (
            _float(row.get("runtime_seconds")) or 0.0,
            _int(row.get("trace_row_index")) or 0,
        ))

    output = []
    for key, schedules in sorted(_group(schedule_rows, fields).items()):
        available = []
        for schedule in schedules:
            trace = traces.get(_canonical_path(schedule.get("solution_file")), [])
            if not trace:
                continue
            runtimes = [_float(row.get("runtime_seconds")) for row in trace]
            if any(value is None for value in runtimes):
                raise ValueError(
                    "Solver progress contains an invalid runtime for "
                    f"{schedule.get('solution_file')}"
                )
            first_incumbent = next(
                (_float(row.get("incumbent_objective")) for row in trace
                 if _float(row.get("incumbent_objective")) is not None),
                None,
            )
            available.append((schedule, trace, runtimes, first_incumbent))

        if not available:
            continue

        for threshold in _progress_grid(time_limit):
            states = []
            improvements = []
            for _schedule, trace, runtimes, first_incumbent in available:
                position = bisect.bisect_right(runtimes, threshold) - 1
                state = trace[position] if position >= 0 else None
                states.append(state)
                if state is not None:
                    improvement = _incumbent_improvement(
                        first_incumbent,
                        _float(state.get("incumbent_objective")),
                        state.get("objective_sense"),
                    )
                    if improvement is not None:
                        improvements.append(improvement)

            incumbents = [
                state for state in states
                if state is not None
                and _float(state.get("incumbent_objective")) is not None
            ]
            finite_gaps = [
                _float(state.get("relative_gap")) for state in incumbents
                if _float(state.get("relative_gap")) is not None
            ]
            finite_gaps = [value for value in finite_gaps if value is not None]
            denominator = len(available)
            row = {
                **_base_group_row(fields, key),
                "threshold_seconds": threshold,
                "configured_case_count": len(schedules),
                "trace_case_count": denominator,
                "trace_coverage_rate": (
                    denominator / len(schedules) if schedules else None
                ),
                "incumbent_count": len(incumbents),
                "incumbent_rate": (
                    len(incumbents) / denominator if denominator else None
                ),
                "finite_gap_count": len(finite_gaps),
                "finite_gap_rate": (
                    len(finite_gaps) / denominator if denominator else None
                ),
                "target_gap_count": sum(
                    gap <= float(target_gap) + 1e-12 for gap in finite_gaps
                ),
                "target_gap_rate": (
                    sum(gap <= float(target_gap) + 1e-12 for gap in finite_gaps)
                    / denominator if denominator else None
                ),
                **describe(finite_gaps, "relative_gap"),
                **describe(improvements, "incumbent_improvement_percent"),
            }
            output.append(row)
    return output


def heldout_gnn_metrics(config):
    root = Path(config["training"]["gnn"]["model_directory"])
    if not root.is_absolute():
        root = ROOT_DIR / root
    output = []
    for path in sorted(root.rglob("*_meta.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        validation = metadata.get("validation_metrics") or {}
        test = metadata.get("test_metrics") or {}
        output.append({
            "model": (
                f"gnn_{metadata.get('convolution')}_"
                f"layers{metadata.get('num_graphsage_layers')}_"
                f"hidden{metadata.get('hidden_channels')}"
            ),
            "seed": metadata.get("seed"),
            "validation_mae": validation.get("mae"),
            "test_mae": test.get("mae"),
            "test_rmse": test.get("rmse"),
            "test_mean_error_bias": test.get("mean_error_bias"),
            "test_r_squared": test.get("r_squared"),
            "test_absolute_error_p95": test.get("absolute_error_p95"),
            "test_maximum_underestimation": test.get("job_underestimation_max"),
            "test_underestimation_rate": test.get("underestimation_rate"),
            "metadata_file": str(path.relative_to(ROOT_DIR)),
        })
    return output


def _plot_runtime(output_directory, ecdf, profile):
    cache = Path(tempfile.gettempdir()) / "fjsp-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for filename, rows, x_key, y_key, xlabel, title in (
        (
            "runtime_ecdf", ecdf, "threshold_seconds",
            "fraction_reaching_target_gap", "Zeit [s]",
            "Anteil der Fälle innerhalb des Ziel-Gaps",
        ),
        (
            "performance_profile", profile, "performance_ratio",
            "fraction_within_ratio", "Performance-Verhältnis τ",
            "Solver-Performance-Profile",
        ),
    ):
        if not rows:
            continue
        figure, axis = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
        for model in sorted({row["model"] for row in rows}):
            values = [row for row in rows if row["model"] == model]
            axis.step(
                [float(row[x_key]) for row in values],
                [float(row[y_key]) for row in values],
                where="post", label=model,
            )
        axis.set(xlabel=xlabel, ylabel="Anteil", title=title, ylim=(0.0, 1.02))
        if filename == "performance_profile":
            axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        figure.savefig(output_directory / f"{filename}.png", dpi=180)
        figure.savefig(output_directory / f"{filename}.pdf")
        plt.close(figure)


def _plot_solver_progress(output_directory, rows, target_gap):
    """Plot architecture/model comparisons from aggregated MIP trajectories."""
    if not rows:
        return {}
    cache = Path(tempfile.gettempdir()) / "fjsp-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = {
        model: sorted(values, key=lambda row: float(row["threshold_seconds"]))
        for (model,), values in _group(rows, ("model",)).items()
    }
    paths = {}

    figure, axis = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    for model, values in sorted(grouped.items()):
        finite = [
            row for row in values
            if _float(row.get("relative_gap_median")) is not None
        ]
        if not finite:
            continue
        x_values = [float(row["threshold_seconds"]) for row in finite]
        medians = [
            max(float(row["relative_gap_median"]), 1e-8) for row in finite
        ]
        q1 = [max(float(row["relative_gap_q1"]), 1e-8) for row in finite]
        q3 = [max(float(row["relative_gap_q3"]), 1e-8) for row in finite]
        line = axis.step(x_values, medians, where="post", label=model)[0]
        axis.fill_between(
            x_values, q1, q3, step="post", alpha=0.12,
            color=line.get_color(),
        )
    axis.axhline(
        max(float(target_gap), 1e-8), color="black", linestyle="--",
        linewidth=1.0, label=f"Ziel-Gap {100.0 * float(target_gap):g}%",
    )
    axis.set(
        xlabel="Solverzeit [s]",
        ylabel="relatives MIP-Gap (Median; Band: IQR)",
        title="Gap-over-Time nach Modell und GNN-Architektur",
    )
    axis.set_yscale("log")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=7)
    for suffix in ("png", "pdf"):
        path = output_directory / f"solver_gap_over_time.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        paths[f"solver_gap_over_time_{suffix}"] = path
    plt.close(figure)

    figure, axes = plt.subplots(
        2, 1, figsize=(8.5, 7.0), sharex=True, layout="constrained"
    )
    for model, values in sorted(grouped.items()):
        x_values = [float(row["threshold_seconds"]) for row in values]
        axes[0].step(
            x_values,
            [float(row["incumbent_rate"]) for row in values],
            where="post", label=model,
        )
        axes[1].step(
            x_values,
            [float(row["target_gap_rate"]) for row in values],
            where="post", label=model,
        )
    axes[0].set(
        ylabel="Anteil",
        title="Anteil der Läufe mit zulässiger Lösung",
        ylim=(0.0, 1.02),
    )
    axes[1].set(
        xlabel="Solverzeit [s]", ylabel="Anteil",
        title=f"Anteil der Läufe innerhalb {100.0 * float(target_gap):g}% Gap",
        ylim=(0.0, 1.02),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    for suffix in ("png", "pdf"):
        path = output_directory / f"solver_incumbent_over_time.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        paths[f"solver_incumbent_over_time_{suffix}"] = path
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    for model, values in sorted(grouped.items()):
        finite = [
            row for row in values
            if _float(row.get("incumbent_improvement_percent_median"))
            is not None
        ]
        if not finite:
            continue
        x_values = [float(row["threshold_seconds"]) for row in finite]
        medians = [
            float(row["incumbent_improvement_percent_median"])
            for row in finite
        ]
        q1 = [
            float(row["incumbent_improvement_percent_q1"]) for row in finite
        ]
        q3 = [
            float(row["incumbent_improvement_percent_q3"]) for row in finite
        ]
        line = axis.step(x_values, medians, where="post", label=model)[0]
        axis.fill_between(
            x_values, q1, q3, step="post", alpha=0.12,
            color=line.get_color(),
        )
    axis.set(
        xlabel="Solverzeit [s]",
        ylabel="Verbesserung ggü. erstem Incumbent [%]",
        title="Incumbent-Verbesserung nach Modell und GNN-Architektur",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    for suffix in ("png", "pdf"):
        path = output_directory / f"solver_incumbent_improvement_over_time.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        paths[f"solver_incumbent_improvement_over_time_{suffix}"] = path
    plt.close(figure)
    return paths


def _environment_metadata():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "matplotlib", "torch", "torch_geometric", "gurobipy"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "versions": versions,
    }


def run_analysis(
    *, config_path=DEFAULT_CONFIG, config=None, result_table=None, job_table=None,
    manifest=None, instances_root=DEFAULT_INSTANCES, output_directory=None,
    strict=False,
):
    if config is None:
        config_path = Path(config_path)
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes.decode("utf-8"))
        config_file = str(config_path.resolve())
    else:
        config = dict(config)
        config_bytes = json.dumps(
            config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        config_file = None
    result_table = Path(result_table or DEFAULT_RESULTS / "result_table.csv")
    job_table = Path(job_table or DEFAULT_RESULTS / "job_comparison.csv")
    manifest = Path(manifest or DEFAULT_RESULTS / "solve_manifest.json")
    output_directory = Path(
        output_directory or DEFAULT_RESULTS / "numerical_analysis"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    manifest_paths = _manifest_paths(manifest)
    schedule_rows = _scope_to_manifest(_read_csv(result_table), manifest_paths)
    job_rows = _scope_to_manifest(_read_csv(job_table), manifest_paths)
    if not schedule_rows:
        raise ValueError("No schedule rows remain after manifest scoping.")
    schedule_rows, job_rows = enrich_rows(
        schedule_rows, job_rows, instances_root
    )
    schedule_rows = _add_reference_cost_gaps(schedule_rows)

    common = config["solvers"]["gurobi"]["common"]
    time_limit = float(common["TimeLimit"])
    target_gap = float(common["MIPGap"])
    replications = int(config["evaluation"]["replications"])
    checks = completeness_checks(
        schedule_rows, job_rows, config, manifest_paths, replications
    )

    group_fields = ("evaluation_tier", "model")
    detailed_fields = (
        "evaluation_tier", "number_of_jobs", "number_of_machines",
        "due_date_condition", "model",
    )
    overall_solver = solver_summary(
        schedule_rows, group_fields, time_limit, target_gap
    )
    detailed_solver = solver_summary(
        schedule_rows, detailed_fields, time_limit, target_gap
    )
    sizes = model_size_summary(schedule_rows)
    quality = quality_summary(schedule_rows)
    paired = paired_model_comparisons(schedule_rows)
    due_effect = due_date_effects(schedule_rows)
    surrogate = surrogate_summary(job_rows)
    robustness = robustness_summary(schedule_rows, job_rows)
    ecdf = ecdf_rows(schedule_rows, time_limit, target_gap)
    profile = performance_profile_rows(schedule_rows, target_gap)
    heldout = heldout_gnn_metrics(config)
    progress = collect_solver_progress(schedule_rows)
    progress_summary = solver_progress_summary(
        schedule_rows, progress, ("model",), time_limit, target_gap
    )
    progress_summary_detailed = solver_progress_summary(
        schedule_rows,
        progress,
        (
            "evaluation_tier", "number_of_jobs", "number_of_machines",
            "due_date_condition", "model",
        ),
        time_limit,
        target_gap,
    )

    outputs = {
        "enriched_result_table": output_directory / "enriched_result_table.csv",
        "enriched_job_comparison": output_directory / "enriched_job_comparison.csv",
        "completeness_checks": output_directory / "completeness_checks.csv",
        "solver_summary": output_directory / "solver_summary.csv",
        "solver_summary_by_size_due": output_directory / "solver_summary_by_size_due.csv",
        "model_size_summary": output_directory / "model_size_summary.csv",
        "quality_summary": output_directory / "quality_summary.csv",
        "paired_model_comparisons": output_directory / "paired_model_comparisons.csv",
        "due_date_effects": output_directory / "due_date_effects.csv",
        "surrogate_summary": output_directory / "surrogate_summary.csv",
        "robustness_summary": output_directory / "robustness_summary.csv",
        "runtime_ecdf": output_directory / "runtime_ecdf.csv",
        "performance_profile": output_directory / "performance_profile.csv",
        "heldout_gnn_metrics": output_directory / "heldout_gnn_metrics.csv",
        "solver_progress": output_directory / "solver_progress.csv",
        "solver_progress_summary": (
            output_directory / "solver_progress_summary.csv"
        ),
        "solver_progress_summary_by_size_due": (
            output_directory / "solver_progress_summary_by_size_due.csv"
        ),
    }
    for path, values in (
        (outputs["enriched_result_table"], schedule_rows),
        (outputs["enriched_job_comparison"], job_rows),
        (outputs["completeness_checks"], checks),
        (outputs["solver_summary"], overall_solver),
        (outputs["solver_summary_by_size_due"], detailed_solver),
        (outputs["model_size_summary"], sizes),
        (outputs["quality_summary"], quality),
        (outputs["paired_model_comparisons"], paired),
        (outputs["due_date_effects"], due_effect),
        (outputs["surrogate_summary"], surrogate),
        (outputs["robustness_summary"], robustness),
        (outputs["runtime_ecdf"], ecdf),
        (outputs["performance_profile"], profile),
        (outputs["heldout_gnn_metrics"], heldout),
    ):
        _write_csv(path, values)
    progress_metadata_fields = [
        "evaluation_tier", "physical_instance_id", "instance_name",
        "number_of_jobs", "number_of_machines", "due_date_condition",
        "solver", "model", "formulation", "solution_file",
    ]
    progress_stat_fields = [
        "threshold_seconds", "configured_case_count", "trace_case_count",
        "trace_coverage_rate", "incumbent_count", "incumbent_rate",
        "finite_gap_count", "finite_gap_rate", "target_gap_count",
        "target_gap_rate",
        *describe([], "relative_gap"),
        *describe([], "incumbent_improvement_percent"),
    ]
    _write_csv(
        outputs["solver_progress"],
        progress,
        fieldnames=[
            *progress_metadata_fields, *PROGRESS_VALUE_FIELDS,
            "final_status", "final_runtime_seconds", "final_mip_gap",
            "final_has_incumbent",
            "trace_row_index", "solver_progress_file",
        ],
    )
    _write_csv(
        outputs["solver_progress_summary"],
        progress_summary,
        fieldnames=["model", *progress_stat_fields],
    )
    _write_csv(
        outputs["solver_progress_summary_by_size_due"],
        progress_summary_detailed,
        fieldnames=[*detailed_fields, *progress_stat_fields],
    )
    _plot_runtime(output_directory, ecdf, profile)
    outputs.update(
        _plot_solver_progress(output_directory, progress_summary, target_gap)
    )

    failed = [row for row in checks if row["status"] == "failed"]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": config_file,
        "config_source": "file" if config_file is not None else "evaluation_workflow",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "result_table": str(result_table.resolve()),
        "job_table": str(job_table.resolve()),
        "manifest": str(manifest.resolve()) if manifest.exists() else None,
        "manifest_scoping": manifest_paths is not None,
        "schedule_rows": len(schedule_rows),
        "job_rows": len(job_rows),
        "time_limit_seconds": time_limit,
        "target_mip_gap": target_gap,
        "solver_progress_grid_points": PROGRESS_GRID_POINTS,
        "solver_progress_trace_rows": len(progress),
        "solver_progress_run_count": len({
            _canonical_path(row.get("solution_file")) for row in progress
        }),
        "solver_progress_definition": (
            "callback-observed changes of incumbent, best bound or solution "
            "count, the first subsequent MIP or MIPNODE callback after at "
            "least 0.25 seconds without a change, and the final state"
        ),
        "incumbent_improvement_definition": (
            "percentage improvement relative to the first incumbent within "
            "each run; summarized only across runs with an incumbent"
        ),
        "simulation_replications": replications,
        "par2_definition": (
            "runtime for cases reaching the configured target MIP gap; "
            "twice the time limit otherwise"
        ),
        "iqr_definition": "75th percentile minus 25th percentile",
        "strict": bool(strict),
        "failed_checks": [row["check"] for row in failed],
        "environment": _environment_metadata(),
        "outputs": {key: str(value.resolve()) for key, value in outputs.items()},
    }
    metadata_path = output_directory / "analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"WROTE {output_directory}")
    if strict and failed:
        raise ValueError(
            "Numerical-analysis completeness checks failed: "
            + ", ".join(row["check"] for row in failed)
        )
    return {**outputs, "metadata": metadata_path}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-table", type=Path)
    parser.add_argument("--job-table", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--instances-root", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail after writing diagnostics if the final run is incomplete.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_analysis(
        config_path=args.config,
        result_table=args.result_table,
        job_table=args.job_table,
        manifest=args.manifest,
        instances_root=args.instances_root,
        output_directory=args.output_directory,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
