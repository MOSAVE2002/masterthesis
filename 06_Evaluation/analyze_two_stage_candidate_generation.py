"""Evaluate two-stage Fix-and-Optimize candidate generation numerically."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_CACHE_ROOT = Path(tempfile.gettempdir()) / "fjsp_two_stage_analysis_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_PATH = ROOT_DIR / "config.json"
DEFAULT_OUTPUT_DIRECTORY = ROOT_DIR / "06_Evaluation" / "results"
ENTRY = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_weibull_training_data"
)
GENERATOR = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_fix_and_optimize_training_data"
)
BINS = tuple(GENERATOR.PROBABILITY_BINS)
COLORS = {
    "low": "#4C78A8",
    "boundary_below": "#F58518",
    "boundary_above": "#54A24B",
    "high": "#E45756",
}


def _natural_instance_key(name):
    suffix = name.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else name


def _instance_names(jobs, machines, count):
    directory = ROOT_DIR / "02_data" / "fjsp_instances" / "training"
    names = sorted(
        (
            path.stem
            for path in directory.glob(
                f"i{int(jobs)}_k{int(machines)}_*.pkl"
            )
        ),
        key=_natural_instance_key,
    )
    if len(names) < int(count):
        raise ValueError(
            f"Only {len(names)} matching instances are available; "
            f"requested {count}."
        )
    return names[:int(count)]


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    return statistics.fmean(values) if values else None


def _pearson(left, right):
    if len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0.0 else None


def _rounded(value):
    return "" if value is None else round(float(value), 6)


def _summary(name, rows, service_level, boundary_width, wall_seconds):
    mc = [row["mc_p_min"] for row in rows]
    nonlinear = [row["nonlinear_p_min"] for row in rows]
    counts = Counter(row["mc_probability_bin"] for row in rows)
    target_ratios = {
        "low": 0.10,
        "boundary_below": 0.40,
        "boundary_above": 0.40,
        "high": 0.10,
    }
    total = len(rows)
    return {
        "scope": name,
        "graphs": total,
        "mc_low": counts["low"],
        "mc_boundary_below": counts["boundary_below"],
        "mc_boundary_above": counts["boundary_above"],
        "mc_high": counts["high"],
        "mc_near_boundary": sum(
            service_level - boundary_width <= value
            <= service_level + boundary_width
            for value in mc
        ),
        "mc_service_feasible": sum(
            value >= service_level for value in mc
        ),
        "mean_mc_p_min": _rounded(_mean(mc)),
        "min_mc_p_min": _rounded(min(mc) if mc else None),
        "max_mc_p_min": _rounded(max(mc) if mc else None),
        "mean_nonlinear_p_min": _rounded(_mean(nonlinear)),
        "mean_absolute_gap": _rounded(_mean([
            abs(mc_value - nonlinear_value)
            for mc_value, nonlinear_value in zip(mc, nonlinear)
        ])),
        "mean_mc_minus_nonlinear": _rounded(_mean([
            mc_value - nonlinear_value
            for mc_value, nonlinear_value in zip(mc, nonlinear)
        ])),
        "pearson_nonlinear_mc": _rounded(_pearson(nonlinear, mc)),
        "bin_target_absolute_error": _rounded(sum(
            abs(counts[band] / total - target_ratios[band])
            for band in BINS
        ) if total else None),
        "wall_seconds": _rounded(wall_seconds),
    }


def _plot(rows, summaries, output_directory, service_level, boundary_width):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    mc = [row["mc_p_min"] for row in rows]
    nonlinear = [row["nonlinear_p_min"] for row in rows]

    axes[0].hist(mc, bins=20, range=(0.0, 1.0), color="#4C78A8")
    for value in (
        service_level - boundary_width,
        service_level,
        service_level + boundary_width,
    ):
        axes[0].axvline(value, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel=r"Simuliertes $p_{min}^{MC}$",
        ylabel="Graphen",
        title="Verteilung der finalen Labels",
    )

    axes[1].scatter(nonlinear, mc, s=32, alpha=0.75, color="#F58518")
    axes[1].plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    axes[1].axhline(service_level, color="grey", linestyle=":")
    axes[1].axvline(service_level, color="grey", linestyle=":")
    axes[1].set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel=r"Nichtlineare Markov-LB $p_{min}^{NL}$",
        ylabel=r"Monte Carlo $p_{min}^{MC}$",
        title="Interne Auswertung gegen Label",
    )

    counts = Counter(row["mc_probability_bin"] for row in rows)
    axes[2].bar(
        range(len(BINS)),
        [counts[band] for band in BINS],
        color=[COLORS[band] for band in BINS],
    )
    axes[2].set_xticks(range(len(BINS)), BINS, rotation=25, ha="right")
    axes[2].set(
        ylabel="Graphen",
        title="Ausgewählte Wahrscheinlichkeitsbereiche",
    )

    overall = summaries[-1]
    figure.suptitle(
        "Zweistufige Kandidatenerzeugung: "
        f"n={overall['graphs']}, MAE={overall['mean_absolute_gap']}, "
        f"r={overall['pearson_nonlinear_mc']}"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    pdf_path = output_directory / "two_stage_candidate_analysis.pdf"
    png_path = output_directory / "two_stage_candidate_analysis.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def run_analysis(
    *,
    instance_count=5,
    jobs=3,
    machines=3,
    samples_per_instance=15,
    pilot_replications=None,
    label_replications=None,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    instance_names=None,
):
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    generation = ENTRY._generation_from_project_config(config)
    if pilot_replications is not None:
        generation["simulation"] = {
            **generation["simulation"],
            "pilot_replications": int(pilot_replications),
        }
    if label_replications is not None:
        generation["simulation"] = {
            **generation["simulation"],
            "label_replications": int(label_replications),
        }
    fixed = generation["fixed_y"]
    if fixed.get("candidate_generation_mode") != "nonlinear_evaluated":
        raise ValueError(
            "config.json must use candidate_generation_mode="
            "'nonlinear_evaluated'."
        )
    graph_config = GENERATOR.normalize_reliability_graph_config(
        generation["reliability_graph"]
    )
    names = (
        [str(name) for name in instance_names]
        if instance_names
        else _instance_names(jobs, machines, instance_count)
    )
    rows = []
    instance_summaries = []
    failures = []
    total_started = time.perf_counter()
    for index, instance_name in enumerate(names, start=1):
        print(
            f"[Two-stage analysis] {index}/{len(names)} {instance_name}",
            flush=True,
        )
        started = time.perf_counter()
        try:
            generated = GENERATOR.generate_rows_for_instance(
                instance_name,
                generation,
                count=int(samples_per_instance),
            )
        except Exception as exc:
            failures.append({
                "instance_name": instance_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        wall_seconds = time.perf_counter() - started
        instance_rows = []
        for graph_number, row in enumerate(generated, start=1):
            mc_probabilities = json.loads(row["job_ontime_probabilities"])
            nonlinear_probabilities = json.loads(
                row["nonlinear_job_ontime_probability_lbs"]
            )
            mc_p_min = min(mc_probabilities)
            nonlinear_p_min = min(nonlinear_probabilities)
            item = {
                "instance_name": instance_name,
                "graph_number": graph_number,
                "candidate_generation_mode": row[
                    "candidate_generation_mode"
                ],
                "mc_p_min": mc_p_min,
                "nonlinear_p_min": nonlinear_p_min,
                "mc_probability_bin": GENERATOR._probability_bin(
                    mc_p_min,
                    graph_config.service_level,
                    fixed["service_boundary_width"],
                ),
                "nonlinear_probability_bin": GENERATOR._probability_bin(
                    nonlinear_p_min,
                    graph_config.service_level,
                    fixed["service_boundary_width"],
                ),
                "mc_service_feasible": mc_p_min >= graph_config.service_level,
                "nonlinear_service_feasible": (
                    nonlinear_p_min >= graph_config.service_level
                ),
                "absolute_gap": abs(mc_p_min - nonlinear_p_min),
                "mc_minus_nonlinear": mc_p_min - nonlinear_p_min,
                "mc_job_probabilities": row["job_ontime_probabilities"],
                "nonlinear_job_probability_lbs": row[
                    "nonlinear_job_ontime_probability_lbs"
                ],
                "selection_category": row["pool_selection_category"],
                "optimization_run": row["optimization_run"],
                "pool_solution_number": row["pool_solution_number"],
                "pool_objective": row["pool_objective"],
                "fix_ratio": row["fix_ratio"],
                "sequence_fix_ratio": row["sequence_fix_ratio"],
                "solver_runtime_seconds": row["solver_runtime_seconds"],
                "instance_wall_seconds": wall_seconds,
                "label_replications": row["simulation_replications"],
            }
            instance_rows.append(item)
            rows.append(item)
        instance_summaries.append(_summary(
            instance_name,
            instance_rows,
            graph_config.service_level,
            fixed["service_boundary_width"],
            wall_seconds,
        ))

    total_wall_seconds = time.perf_counter() - total_started
    summaries = instance_summaries + [_summary(
        "overall",
        rows,
        graph_config.service_level,
        fixed["service_boundary_width"],
        total_wall_seconds,
    )]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = output_directory / "two_stage_candidate_analysis.csv"
    summary_path = output_directory / "two_stage_candidate_summary.csv"
    failure_path = output_directory / "two_stage_candidate_failures.csv"
    raw_fields = list(rows[0]) if rows else [
        "instance_name", "graph_number", "mc_p_min", "nonlinear_p_min"
    ]
    _write_csv(raw_path, rows, raw_fields)
    _write_csv(summary_path, summaries, list(summaries[0]))
    _write_csv(
        failure_path,
        failures,
        ["instance_name", "error"],
    )
    pdf_path, png_path = _plot(
        rows,
        summaries,
        output_directory,
        graph_config.service_level,
        fixed["service_boundary_width"],
    )
    metadata_path = output_directory / "two_stage_candidate_analysis.json"
    metadata_path.write_text(json.dumps({
        "instances_requested": names,
        "instances_completed": len(instance_summaries),
        "failures": failures,
        "samples_per_instance": int(samples_per_instance),
        "pilot_replications": generation["simulation"][
            "pilot_replications"
        ],
        "label_replications": generation["simulation"][
            "label_replications"
        ],
        "service_level": graph_config.service_level,
        "service_boundary_width": fixed["service_boundary_width"],
        "fixed_y": fixed,
        "overall": summaries[-1],
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "overall": summaries[-1],
        "failures": failures,
        "csv": str(raw_path),
        "summary": str(summary_path),
        "pdf": str(pdf_path),
        "png": str(png_path),
    }, indent=2), flush=True)
    return rows, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--machines", type=int, default=3)
    parser.add_argument("--samples-per-instance", type=int, default=15)
    parser.add_argument("--pilot-replications", type=int)
    parser.add_argument("--label-replications", type=int)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    run_analysis(
        instance_count=args.instances,
        jobs=args.jobs,
        machines=args.machines,
        samples_per_instance=args.samples_per_instance,
        pilot_replications=args.pilot_replications,
        label_replications=args.label_replications,
        output_directory=(
            args.output_directory or DEFAULT_OUTPUT_DIRECTORY
        ),
    )


if __name__ == "__main__":
    main()
