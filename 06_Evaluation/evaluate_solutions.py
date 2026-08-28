"""Independently simulate and compare schedules from solution text files.

The evaluator is intentionally not imported or called by ``main.py``. It
parses already written nonlinear/GNN solutions, reconstructs their fixed
schedules, performs fresh Monte-Carlo replications, and writes CSV, LaTeX and
PDF result tables plus one detailed CSV table per job.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from statistics import NormalDist


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


_instances = importlib.import_module("01_generator.instance_generator")
_simulation = importlib.import_module("05_Simulation.preempt_resume")
from helper.stochastic_fjsp import (
    stochastic_parameters,
    weibull_down_probability,
)


CONFIG_PATH = ROOT_DIR / "config.json"
DEFAULT_SOLUTIONS_ROOT = ROOT_DIR / "02_data" / "fjsp_solutions"
DEFAULT_INSTANCES_ROOT = ROOT_DIR / "02_data" / "fjsp_instances"
DEFAULT_OUTPUT_DIRECTORY = ROOT_DIR / "06_Evaluation" / "results"

FixedSchedule = _simulation.FixedSchedule
normalize_simulation_config = _simulation.normalize_simulation_config
simulate_fixed_schedule = _simulation.simulate_fixed_schedule
simulation_config_dict = _simulation.simulation_config_dict


_JOB_PATTERN = re.compile(
    r"^job (?P<job>-?\d+): "
    r"completion=(?P<completion>[^,]+), "
    r"due_date=(?P<due_date>[^,]+), "
    r"expected_repair_buffer=(?P<buffer>[^,]+), "
    r"protected_completion=(?P<protected>[^,]+), "
    r"robust_slack=(?P<slack>[^,]+)$"
)
_OPERATION_PATTERN = re.compile(
    r"^op (?P<operation>-?\d+): "
    r"machine=(?P<machine>-?\d+), "
    r"S=(?P<start>[^,]+), "
    r"C=(?P<completion>[^,]+),"
)


def wilson_lower_bound(successes, trials, confidence=0.95):
    """Return the one-sided Wilson lower confidence bound."""
    successes = int(successes)
    trials = int(trials)
    confidence = float(confidence)
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials.")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0.5 and 1.")
    probability = successes / trials
    z_value = NormalDist().inv_cdf(confidence)
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / trials
    center = probability + z_squared / (2.0 * trials)
    radius = z_value * math.sqrt(
        probability * (1.0 - probability) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return max(0.0, (center - radius) / denominator)


def bonferroni_confidence(confidence, comparisons):
    """Return the per-comparison confidence for a joint confidence level."""
    confidence = float(confidence)
    comparisons = int(comparisons)
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0.5 and 1.")
    if comparisons <= 0:
        raise ValueError("comparisons must be positive.")
    return 1.0 - (1.0 - confidence) / comparisons


def _field(lines, label, default=""):
    prefix = f"{label}:"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return default


def _optional_float(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value == "" else float(value)


def _instance_name(solution_path, formulation):
    stem = solution_path.stem
    prefix = "solution_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected solution filename: {solution_path.name}")
    remainder = stem[len(prefix):]
    marker = (
        "_gurobi_nonlinear"
        if formulation.startswith("nonlinear_")
        else "_gurobi_gnn"
    )
    if marker not in remainder:
        raise ValueError(
            f"Cannot identify the instance in {solution_path.name}."
        )
    return remainder.split(marker, 1)[0]


def parse_solution(solution_path):
    """Parse the schedule and comparison metadata from one text solution."""
    solution_path = Path(solution_path)
    lines = solution_path.read_text(encoding="utf-8").splitlines()
    formulation = _field(lines, "Formulation")
    if formulation.startswith("nonlinear_"):
        solver = "gurobi_nonlinear"
        model_name = "nonlinear_midpoint"
    elif formulation.startswith("gnn_"):
        solver = "gurobi_gnn"
        convolution = _field(lines, "GNN convolution")
        layers = _field(lines, "GNN layers")
        hidden = _field(lines, "GNN hidden channels")
        model_name = f"gnn_{convolution}_layers{layers}_hidden{hidden}"
    else:
        raise ValueError(
            f"Unsupported or missing formulation in {solution_path}."
        )

    jobs = {}
    operations = {}
    machine_edges = []
    for line in lines:
        job_match = _JOB_PATTERN.match(line)
        if job_match:
            values = job_match.groupdict()
            job = int(values["job"])
            jobs[job] = {
                "nominal_completion": float(values["completion"]),
                "due_date_from_solution": float(values["due_date"]),
                "internal_repair_buffer": float(values["buffer"]),
                "protected_completion": float(values["protected"]),
                "robust_slack": float(values["slack"]),
            }
            continue
        operation_match = _OPERATION_PATTERN.match(line)
        if operation_match:
            values = operation_match.groupdict()
            operation = int(values["operation"])
            operations[operation] = {
                "machine": int(values["machine"]),
                "start": float(values["start"]),
                "completion": float(values["completion"]),
            }
            continue
        if line.startswith("U(") and line.endswith("=1"):
            edge = ast.literal_eval(line[1:-2])
            if not isinstance(edge, tuple) or len(edge) != 3:
                raise ValueError(f"Malformed machine edge: {line}")
            machine_edges.append(tuple(int(value) for value in edge))

    return {
        "solution_path": solution_path,
        "instance_name": _instance_name(solution_path, formulation),
        "solver": solver,
        "model_name": model_name,
        "formulation": formulation,
        "status": _field(lines, "Status"),
        "solution_count": int(_field(lines, "Solution count", "0")),
        "objective": _optional_float(_field(lines, "Objective")),
        "makespan": _optional_float(_field(lines, "Makespan")),
        "processing_cost": _optional_float(_field(lines, "Processing cost")),
        "operating_cost": _optional_float(_field(lines, "Operating cost")),
        "total_cost": _optional_float(_field(lines, "Total cost")),
        "best_bound": _optional_float(_field(lines, "Best bound")),
        "mip_gap": _optional_float(_field(lines, "MIP gap")),
        "runtime_seconds": _optional_float(_field(lines, "Runtime [s]")),
        "model_build_seconds": _optional_float(
            _field(lines, "Model build runtime [s]")
        ),
        "optimizer_wall_seconds": _optional_float(
            _field(lines, "Optimizer wall runtime [s]")
        ),
        "repair_buffer_label_method": _field(
            lines, "Repair buffer label method"
        ),
        "jobs": jobs,
        "operations": operations,
        "machine_edges": tuple(machine_edges),
    }


def _find_instance_path(instances_root, instance_name):
    matches = sorted(Path(instances_root).rglob(f"{instance_name}.pkl"))
    if not matches:
        raise FileNotFoundError(
            f"Instance {instance_name!r} not found below {instances_root}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Instance {instance_name!r} is ambiguous: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _evaluation_tier(instance_path, instances_root):
    relative = instance_path.relative_to(instances_root)
    directory = relative.parts[0] if len(relative.parts) > 1 else "flat"
    return "in_distribution" if directory == "test" else directory


def _load_instance(instances_root, instance_name):
    path = _find_instance_path(instances_root, instance_name)
    instance = _instances.load_generated_instance(
        instance_name, instance_directory=path.parent
    )
    return instance, path


def _fixed_schedule(parsed, instance):
    operations = tuple(instance.real_operations)
    missing = set(operations) - set(parsed["operations"])
    extra = set(parsed["operations"]) - set(operations)
    if missing or extra:
        raise ValueError(
            f"Operation mismatch for {parsed['solution_path']}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    parameters = stochastic_parameters(instance)
    selected = {
        operation: parsed["operations"][operation]["machine"]
        for operation in operations
    }
    for operation, machine in selected.items():
        if machine not in instance.eligible_machines[operation]:
            raise ValueError(
                f"Operation {operation} is assigned to ineligible machine "
                f"{machine}."
            )
        stored_duration = (
            parsed["operations"][operation]["completion"]
            - parsed["operations"][operation]["start"]
        )
        expected_duration = float(
            instance.processing_times[operation, machine]
        )
        if not math.isclose(
            stored_duration, expected_duration, rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError(
                f"Solution/instance duration mismatch for operation "
                f"{operation}: stored={stored_duration}, "
                f"expected={expected_duration}."
            )
    if set(parsed["jobs"]) != set(instance.jobs):
        raise ValueError(
            f"Job mismatch for {parsed['solution_path']}: "
            f"solution={sorted(parsed['jobs'])}, "
            f"instance={sorted(instance.jobs)}"
        )
    for job, values in parsed["jobs"].items():
        if not math.isclose(
            values["due_date_from_solution"],
            float(instance.due_dates[job]),
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise ValueError(
                f"Solution/instance due-date mismatch for job {job}."
            )
    return FixedSchedule(
        operations=operations,
        selected_machines=selected,
        processing_times={
            operation: float(instance.processing_times[operation, machine])
            for operation, machine in selected.items()
        },
        planned_starts={
            operation: parsed["operations"][operation]["start"]
            for operation in operations
        },
        job_predecessors={
            operation: tuple(
                predecessor
                for predecessor in instance.predecessors.get(operation, [])
                if predecessor in operations
            )
            for operation in operations
        },
        machine_edges=parsed["machine_edges"],
        jobs={
            job: tuple(instance.jobs[job]) for job in sorted(instance.jobs)
        },
        job_end_operations=dict(instance.job_end_operations),
        due_dates={
            job: float(instance.due_dates[job])
            for job in sorted(instance.jobs)
        },
        weibull_scale=dict(parameters["alpha"]),
        weibull_shape=dict(parameters["beta"]),
        repair_rate=dict(parameters["repair_rate"]),
    )


def _evaluation_seed(base_seed, instance_name):
    payload = f"postsolve:{int(base_seed)}:{instance_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _reference_expected_repair_buffers(schedule, service_scope="job"):
    """Recompute the nonlinear buffer for a fixed optimized schedule."""
    operation_buffers = {}
    for operation in schedule.operations:
        machine = schedule.selected_machines[operation]
        midpoint = (
            float(schedule.planned_starts[operation])
            + 0.5 * float(schedule.processing_times[operation])
        )
        probability = weibull_down_probability(
            midpoint,
            schedule.weibull_scale[machine],
            schedule.weibull_shape[machine],
            schedule.repair_rate[machine],
            order=64,
        )
        operation_buffers[operation] = (
            probability / float(schedule.repair_rate[machine])
        )
    return {
        job: sum(
            operation_buffers[operation]
            for operation in (
                schedule.jobs[job]
                if service_scope == "job" else schedule.operations
            )
        )
        for job in schedule.jobs
    }


def evaluate_solution(
    solution_path,
    *,
    instances_root,
    replications,
    base_seed,
    confidence,
    simulation_config,
):
    parsed = parse_solution(solution_path)
    instance, instance_path = _load_instance(
        instances_root, parsed["instance_name"]
    )
    tier = _evaluation_tier(instance_path, Path(instances_root))
    relative_solution_path = _portable_path(parsed["solution_path"])
    joint_confidence = bonferroni_confidence(
        confidence, len(instance.jobs)
    )
    if parsed["solution_count"] <= 0 or not parsed["operations"]:
        return {
            "instance_name": parsed["instance_name"],
            "evaluation_tier": tier,
            "solver": parsed["solver"],
            "model": parsed["model_name"],
            "formulation": parsed["formulation"],
            "status": parsed["status"],
            "postsolve_evaluation_status": "not_evaluated_no_incumbent",
            "objective": parsed["objective"],
            "processing_cost": parsed["processing_cost"],
            "operating_cost": parsed["operating_cost"],
            "total_cost": parsed["total_cost"],
            "best_bound": parsed["best_bound"],
            "mip_gap": parsed["mip_gap"],
            "runtime_seconds": parsed["runtime_seconds"],
            "model_build_seconds": parsed["model_build_seconds"],
            "optimizer_wall_seconds": parsed["optimizer_wall_seconds"],
            "number_of_jobs": len(instance.jobs),
            "due_date_factor": getattr(instance, "due_date_factor", None),
            "nominal_makespan": parsed["makespan"],
            "maximum_internal_repair_buffer": None,
            "repair_buffer_mae": None,
            "repair_buffer_rmse": None,
            "repair_buffer_mean_error": None,
            "minimum_mc_ontime_probability": None,
            "minimum_wilson_lower_bound": None,
            "minimum_bonferroni_wilson_lower_bound": None,
            "simulation_replications": 0,
            "simulation_seed": None,
            "simulation_mean_failures": None,
            "simulation_mean_total_repair_delay": None,
            "solution_file": relative_solution_path,
        }, []
    if not parsed["jobs"]:
        raise ValueError(
            f"Incumbent has no per-job values in {parsed['solution_path']}."
        )
    schedule = _fixed_schedule(parsed, instance)
    reference_buffers = _reference_expected_repair_buffers(schedule)
    seed = _evaluation_seed(base_seed, parsed["instance_name"])
    result = simulate_fixed_schedule(
        schedule,
        replications=replications,
        seed=seed,
        config=simulation_config,
    )

    job_rows = []
    for index, job in enumerate(result.job_ids):
        if job not in parsed["jobs"]:
            raise ValueError(
                f"Job {job!r} is missing in {parsed['solution_path']}."
            )
        probability = float(result.job_ontime_probabilities[index])
        successes = int(round(probability * result.replications))
        lower_bound = wilson_lower_bound(
            successes, result.replications, confidence
        )
        bonferroni_lower_bound = wilson_lower_bound(
            successes, result.replications, joint_confidence
        )
        internal = parsed["jobs"][job]
        job_rows.append({
            "instance_name": parsed["instance_name"],
            "evaluation_tier": _evaluation_tier(
                instance_path, Path(instances_root)
            ),
            "solver": parsed["solver"],
            "model": parsed["model_name"],
            "formulation": parsed["formulation"],
            "status": parsed["status"],
            "postsolve_evaluation_status": "evaluated",
            "job_id": job,
            "due_date_factor": getattr(instance, "due_date_factor", None),
            "nominal_completion": internal["nominal_completion"],
            "due_date": float(instance.due_dates[job]),
            "internal_expected_repair_buffer": internal[
                "internal_repair_buffer"
            ],
            "reference_expected_repair_buffer": reference_buffers[job],
            "repair_buffer_error": (
                internal["internal_repair_buffer"] - reference_buffers[job]
            ),
            "protected_completion": internal["protected_completion"],
            "robust_slack": internal["robust_slack"],
            "repair_buffer_method": parsed["repair_buffer_label_method"],
            "mc_ontime_probability": probability,
            "mc_standard_error": float(
                result.job_probability_standard_errors[index]
            ),
            "wilson_lower_bound": lower_bound,
            "wilson_confidence": float(confidence),
            "bonferroni_wilson_lower_bound": bonferroni_lower_bound,
            "bonferroni_wilson_confidence": joint_confidence,
            "mc_mean_completion": float(
                result.job_mean_completion_times[index]
            ),
            "simulation_replications": int(result.replications),
            "simulation_seed": seed,
            "objective": parsed["objective"],
            "processing_cost": parsed["processing_cost"],
            "operating_cost": parsed["operating_cost"],
            "total_cost": parsed["total_cost"],
            "nominal_makespan": parsed["makespan"],
            "runtime_seconds": parsed["runtime_seconds"],
            "mip_gap": parsed["mip_gap"],
            "solution_file": relative_solution_path,
        })

    maximum_internal_buffer = max(
        row["internal_expected_repair_buffer"] for row in job_rows
    )
    minimum_mc = min(row["mc_ontime_probability"] for row in job_rows)
    minimum_wilson = min(row["wilson_lower_bound"] for row in job_rows)
    minimum_bonferroni_wilson = min(
        row["bonferroni_wilson_lower_bound"] for row in job_rows
    )
    schedule_row = {
        "instance_name": parsed["instance_name"],
        "evaluation_tier": job_rows[0]["evaluation_tier"],
        "solver": parsed["solver"],
        "model": parsed["model_name"],
        "formulation": parsed["formulation"],
        "status": parsed["status"],
        "postsolve_evaluation_status": "evaluated",
        "objective": parsed["objective"],
        "processing_cost": parsed["processing_cost"],
        "operating_cost": parsed["operating_cost"],
        "total_cost": parsed["total_cost"],
        "best_bound": parsed["best_bound"],
        "mip_gap": parsed["mip_gap"],
        "runtime_seconds": parsed["runtime_seconds"],
        "model_build_seconds": parsed["model_build_seconds"],
        "optimizer_wall_seconds": parsed["optimizer_wall_seconds"],
        "number_of_jobs": len(job_rows),
        "due_date_factor": getattr(instance, "due_date_factor", None),
        "nominal_makespan": parsed["makespan"],
        "maximum_internal_repair_buffer": maximum_internal_buffer,
        "repair_buffer_mae": sum(
            abs(row["repair_buffer_error"]) for row in job_rows
        ) / len(job_rows),
        "repair_buffer_rmse": math.sqrt(
            sum(row["repair_buffer_error"] ** 2 for row in job_rows)
            / len(job_rows)
        ),
        "repair_buffer_mean_error": sum(
            row["repair_buffer_error"] for row in job_rows
        ) / len(job_rows),
        "minimum_mc_ontime_probability": minimum_mc,
        "minimum_wilson_lower_bound": minimum_wilson,
        "minimum_bonferroni_wilson_lower_bound": (
            minimum_bonferroni_wilson
        ),
        "simulation_replications": int(result.replications),
        "simulation_seed": seed,
        "simulation_mean_failures": float(result.mean_failures),
        "simulation_mean_total_repair_delay": float(
            result.mean_total_repair_delay
        ),
        "solution_file": relative_solution_path,
    }
    return schedule_row, job_rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty comparison table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


_PRESENTATION_COLUMNS = (
    ("instance_name", "Instanz"),
    ("evaluation_tier", "Tier"),
    ("model", "Modell"),
    ("status", "Status"),
    ("due_date_factor", "Due-Date-Faktor"),
    ("total_cost", "Gesamtkosten"),
    ("processing_cost", "Bearbeitungskosten"),
    ("operating_cost", "Betriebskosten"),
    ("nominal_makespan", "Makespan"),
    ("runtime_seconds", "Laufzeit [s]"),
    ("mip_gap", "MIP-Gap"),
    ("maximum_internal_repair_buffer", "max. Reparaturpuffer"),
    ("repair_buffer_mae", "Puffer-MAE"),
    ("minimum_mc_ontime_probability", "min. MC"),
    ("minimum_wilson_lower_bound", "Wilson-LB"),
    (
        "minimum_bonferroni_wilson_lower_bound",
        "Bonf.-Wilson-LB",
    ),
)


def _presentation_value(key, value):
    if value is None or value == "":
        return "--"
    if isinstance(value, bool):
        return "ja" if value else "nein"
    if key == "evaluation_tier":
        return {
            "in_distribution": "ID",
            "benchmark": "Benchmark",
            "extrapolation": "Extrap.",
            "stress": "Stress",
        }.get(str(value), str(value))
    if key == "model":
        match = re.fullmatch(
            r"gnn_([^_]+)_layers([^_]+)_hidden([^_]+)", str(value)
        )
        if match:
            convolution, layers, hidden = match.groups()
            return f"GNN-{convolution}-L{layers}-H{hidden}"
        if value == "nonlinear_midpoint":
            return "MINLP-Mittelpunkt"
    if key == "mip_gap":
        return f"{100.0 * float(value):.2f}%"
    if key == "runtime_seconds":
        return f"{float(value):.3f}"
    if key in {
        "objective",
        "due_date_factor",
        "total_cost",
        "processing_cost",
        "operating_cost",
        "nominal_makespan",
        "maximum_internal_repair_buffer",
        "repair_buffer_mae",
        "minimum_mc_ontime_probability",
        "minimum_wilson_lower_bound",
        "minimum_bonferroni_wilson_lower_bound",
    }:
        return f"{float(value):.4f}"
    return str(value)


def _presentation_table(rows):
    headers = [label for _key, label in _PRESENTATION_COLUMNS]
    values = [
        [
            _presentation_value(key, row.get(key))
            for key, _label in _PRESENTATION_COLUMNS
        ]
        for row in rows
    ]
    return headers, values


def _latex_escape(value):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _write_latex_table(path, rows, confidence):
    headers, values = _presentation_table(rows)
    columns = "l" * len(headers)
    line_break = r" \\"
    body = [
        r"\documentclass[a4paper,landscape]{article}",
        r"\usepackage[margin=1cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\begin{document}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{" + columns + "}",
        r"\caption{Monte-Carlo-Auswertung der optimierten FJSP-Schedules. "
        + f"Wilson-Konfidenzniveau: {100.0 * float(confidence):.1f}\\%."
        + r"}\label{tab:fjsp-results}" + line_break,
        r"\toprule",
        " & ".join(_latex_escape(header) for header in headers) + line_break,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(_latex_escape(header) for header in headers) + line_break,
        r"\midrule",
        r"\endhead",
    ]
    body.extend(
        " & ".join(_latex_escape(value) for value in row) + line_break
        for row in values
    )
    body.extend([
        r"\bottomrule",
        r"\end{longtable}",
        r"\end{document}",
        "",
    ])
    Path(path).write_text("\n".join(body), encoding="utf-8")


def _write_pdf_table(path, rows, confidence, rows_per_page=12):
    cache_directory = Path(tempfile.gettempdir()) / "fjsp-matplotlib-cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_directory))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    headers, values = _presentation_table(rows)
    chunks = [
        values[index:index + int(rows_per_page)]
        for index in range(0, len(values), int(rows_per_page))
    ]
    with PdfPages(path) as pdf:
        for page, chunk in enumerate(chunks, start=1):
            figure, axis = plt.subplots(figsize=(16.5, 11.7))
            axis.axis("off")
            axis.set_title(
                "Monte-Carlo-Auswertung der optimierten FJSP-Schedules\n"
                f"Wilson-Konfidenzniveau: {100.0 * float(confidence):.1f}% | "
                f"Seite {page}/{len(chunks)}",
                fontsize=14,
                pad=18,
            )
            table = axis.table(
                cellText=chunk,
                colLabels=headers,
                cellLoc="center",
                colLoc="center",
                loc="upper center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6.5)
            table.scale(1.0, 1.55)
            for (row_index, _column_index), cell in table.get_celld().items():
                if row_index == 0:
                    cell.set_facecolor("#D9EAF7")
                    cell.set_text_props(weight="bold")
                elif row_index % 2 == 0:
                    cell.set_facecolor("#F3F6F8")
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def _solution_paths(arguments, solutions_root):
    if arguments:
        paths = [Path(argument).resolve() for argument in arguments]
    else:
        paths = sorted(Path(solutions_root).rglob("solution_*.txt"))
    supported = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        if "Formulation: nonlinear_" in text or "Formulation: gnn_" in text:
            supported.append(path)
    if not supported:
        raise ValueError("No supported nonlinear/GNN solution files found.")
    return supported


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Post-evaluate solved schedules with independent Monte Carlo."
    )
    parser.add_argument(
        "solutions", nargs="*", help="Optional solution text files."
    )
    parser.add_argument(
        "--solutions-root", type=Path, default=DEFAULT_SOLUTIONS_ROOT
    )
    parser.add_argument(
        "--instances-root", type=Path, default=DEFAULT_INSTANCES_ROOT
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--replications", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=900_042)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def _project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def _portable_path(path, root=ROOT_DIR):
    """Prefer a project-relative path but retain valid external paths."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def run_evaluation(
    *,
    config,
    solutions=None,
    solutions_root=DEFAULT_SOLUTIONS_ROOT,
    instances_root=DEFAULT_INSTANCES_ROOT,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    replications=10_000,
    random_seed=900_042,
    confidence=0.95,
    dataset_diagnostics=None,
    prediction_diagnostics=None,
):
    """Run post-solve evaluation and return the generated table paths."""
    replications = int(replications)
    random_seed = int(random_seed)
    confidence = float(confidence)
    if replications <= 0:
        raise ValueError("--replications must be positive.")
    if not 0.5 < confidence < 1.0:
        raise ValueError("--confidence must lie strictly between 0.5 and 1.")

    evaluation_config = config.get("evaluation", {})
    simulation_config = normalize_simulation_config(
        evaluation_config.get("simulation")
    )
    dataset_diagnostics = dict(
        evaluation_config.get("dataset_diagnostics", {})
        if dataset_diagnostics is None else dataset_diagnostics
    )
    prediction_diagnostics = dict(
        evaluation_config.get("prediction_diagnostics", {})
        if prediction_diagnostics is None else prediction_diagnostics
    )
    solutions_root = _project_path(solutions_root).resolve()
    instances_root = _project_path(instances_root).resolve()
    output_directory = _project_path(output_directory).resolve()
    paths = _solution_paths(solutions or [], solutions_root)
    schedule_rows = []
    job_rows = []
    for position, path in enumerate(paths, start=1):
        schedule_row, rows = evaluate_solution(
            path,
            instances_root=instances_root,
            replications=replications,
            base_seed=random_seed,
            confidence=confidence,
            simulation_config=simulation_config,
        )
        schedule_rows.append(schedule_row)
        job_rows.extend(rows)
        if schedule_row["postsolve_evaluation_status"] == "evaluated":
            print(
                f"EVALUATED {position}/{len(paths)} | "
                f"instance={schedule_row['instance_name']} | "
                f"model={schedule_row['model']} | "
                "min_mc="
                f"{schedule_row['minimum_mc_ontime_probability']:.4f} | "
                "min_wilson="
                f"{schedule_row['minimum_wilson_lower_bound']:.4f} | "
                "min_wilson_bonferroni="
                f"{schedule_row['minimum_bonferroni_wilson_lower_bound']:.4f}",
                flush=True,
            )
        else:
            print(
                f"SKIPPED {position}/{len(paths)} | "
                f"instance={schedule_row['instance_name']} | "
                f"model={schedule_row['model']} | "
                f"status={schedule_row['status']} | reason=no_incumbent",
                flush=True,
            )

    schedule_rows.sort(key=lambda row: (
        row["evaluation_tier"], row["instance_name"], row["model"]
    ))
    job_rows.sort(key=lambda row: (
        row["evaluation_tier"], row["instance_name"], row["model"],
        int(row["job_id"]),
    ))
    result_path = output_directory / "result_table.csv"
    latex_path = output_directory / "result_table.tex"
    pdf_path = output_directory / "result_table.pdf"
    job_path = output_directory / "job_comparison.csv"
    _write_csv(result_path, schedule_rows)
    _write_csv(job_path, job_rows)
    _write_latex_table(latex_path, schedule_rows, confidence)
    _write_pdf_table(pdf_path, schedule_rows, confidence)
    diagnostic_paths = {}
    if bool(dataset_diagnostics.get("enabled", False)):
        dataset_module = importlib.import_module(
            "06_Evaluation.evaluate_gnn_dataset"
        )
        data_config = config["training"]["data_generation"]
        fixed = data_config["fixed_y"]
        dataset_result = dataset_module.run_dataset_evaluation(
            dataset_directory=dataset_diagnostics.get(
                "dataset_directory", data_config["output_directory"]
            ),
            output_directory=dataset_diagnostics.get(
                "output_directory", output_directory
            ),
            expected_service_level=fixed.get(
                "label_distribution_center", 0.50
            ),
            boundary_width=dataset_diagnostics.get(
                "boundary_width",
                fixed.get("label_distribution_half_width", 0.25),
            ),
            histogram_bins=dataset_diagnostics.get("histogram_bins", 20),
        )
        diagnostic_paths["dataset"] = {
            key: str(value)
            for key, value in dataset_result.items()
            if key not in {"distribution_rows", "baseline_rows"}
        }
    if bool(prediction_diagnostics.get("enabled", False)):
        prediction_module = importlib.import_module(
            "06_Evaluation.evaluate_gnn_predictions"
        )
        prediction_result = prediction_module.run_prediction_evaluation(
            job_rows=job_rows,
            output_directory=prediction_diagnostics.get(
                "output_directory", output_directory
            ),
            boundary_width=prediction_diagnostics.get(
                "boundary_width",
                config["training"]["data_generation"]["fixed_y"][
                    "label_distribution_half_width"
                ],
            ),
            calibration_bins=prediction_diagnostics.get(
                "calibration_bins", 10
            ),
        )
        diagnostic_paths["predictions"] = {
            key: str(value)
            for key, value in prediction_result.items()
            if key not in {"metric_rows", "calibration_rows"}
        }
    metadata_path = output_directory / "evaluation_metadata.json"
    metadata_path.write_text(json.dumps({
        "solution_count": len(schedule_rows),
        "evaluated_schedule_count": sum(
            row["postsolve_evaluation_status"] == "evaluated"
            for row in schedule_rows
        ),
        "no_incumbent_count": sum(
            row["postsolve_evaluation_status"]
            == "not_evaluated_no_incumbent"
            for row in schedule_rows
        ),
        "job_row_count": len(job_rows),
        "replications": replications,
        "random_seed": random_seed,
        "seed_scope": "same seed for every solver schedule of one instance",
        "wilson_confidence": confidence,
        "wilson_interval": "one_sided_lower",
        "bonferroni_scope": "all jobs within one schedule",
        "simulation_parameters": simulation_config_dict(simulation_config),
        "result_table_csv": str(result_path),
        "result_table_latex": str(latex_path),
        "result_table_pdf": str(pdf_path),
        "job_comparison": str(job_path),
        "gnn_diagnostics": diagnostic_paths,
    }, indent=2), encoding="utf-8")
    print(f"WROTE {result_path}")
    print(f"WROTE {latex_path}")
    print(f"WROTE {pdf_path}")
    print(f"WROTE {job_path}")
    print(f"WROTE {metadata_path}")
    return {
        "result_table_csv": result_path,
        "result_table_latex": latex_path,
        "result_table_pdf": pdf_path,
        "job_comparison": job_path,
        "metadata": metadata_path,
        "schedule_rows": schedule_rows,
        "job_rows": job_rows,
        "gnn_diagnostics": diagnostic_paths,
    }


def evaluate_from_config(config=None):
    """Run the optional workflow phase using top-level evaluation settings."""
    if config is None:
        with CONFIG_PATH.open(encoding="utf-8") as file:
            config = json.load(file)
    if not bool(config.get("workflow", {}).get("evaluate", False)):
        print("[Evaluate] skipped: workflow.evaluate is false", flush=True)
        return None
    settings = dict(config.get("evaluation", {}))
    allowed = {
        "solutions_directory",
        "instances_directory",
        "output_directory",
        "replications",
        "random_seed",
        "wilson_confidence",
        "simulation",
        "dataset_diagnostics",
        "prediction_diagnostics",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(
            f"Unknown post-solve evaluation settings: {sorted(unknown)}"
        )
    return run_evaluation(
        config=config,
        solutions_root=settings.get(
            "solutions_directory", DEFAULT_SOLUTIONS_ROOT
        ),
        instances_root=settings.get(
            "instances_directory", DEFAULT_INSTANCES_ROOT
        ),
        output_directory=settings.get(
            "output_directory", DEFAULT_OUTPUT_DIRECTORY
        ),
        replications=settings.get("replications", 10_000),
        random_seed=settings.get("random_seed", 900_042),
        confidence=settings.get("wilson_confidence", 0.95),
        dataset_diagnostics=settings.get("dataset_diagnostics"),
        prediction_diagnostics=settings.get("prediction_diagnostics"),
    )


def main():
    args = _parse_args()
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if not bool(config.get("workflow", {}).get("evaluate", False)):
        print("[Evaluate] skipped: workflow.evaluate is false", flush=True)
        return None
    return run_evaluation(
        config=config,
        solutions=args.solutions,
        solutions_root=args.solutions_root,
        instances_root=args.instances_root,
        output_directory=args.output_directory,
        replications=args.replications,
        random_seed=args.random_seed,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
