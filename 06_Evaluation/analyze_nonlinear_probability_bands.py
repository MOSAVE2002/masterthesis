"""Numerically compare nonlinear probability bands with Monte Carlo labels."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "fjsp_band_analysis_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config.json"
DEFAULT_OUTPUT_DIRECTORY = ROOT_DIR / "06_Evaluation" / "results"
GENERATOR = importlib.import_module(
    "04_GraphNeuralNetworks.models."
    "generate_fix_and_optimize_training_data"
)
BANDS = tuple(GENERATOR.PROBABILITY_BINS)
BAND_COLORS = {
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
    pattern = f"i{int(jobs)}_k{int(machines)}_*.pkl"
    names = sorted(
        (path.stem for path in directory.glob(pattern)),
        key=_natural_instance_key,
    )
    if len(names) < int(count):
        raise ValueError(
            f"Requested {count} instances matching {pattern}, but only "
            f"{len(names)} are available in {directory}."
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


def _median(values):
    return statistics.median(values) if values else None


def _rounded(value):
    return None if value is None else round(float(value), 6)


def _summaries(rows):
    summaries = []
    for target_band in (*BANDS, "overall"):
        selected = (
            rows if target_band == "overall"
            else [row for row in rows if row["target_band"] == target_band]
        )
        solved = [row for row in selected if row["candidate_found"]]
        internal = [row["nonlinear_p_min"] for row in solved]
        monte_carlo = [row["mc_p_min"] for row in solved]
        gaps = [
            abs(row["nonlinear_p_min"] - row["mc_p_min"])
            for row in solved
        ]
        actual = Counter(row["mc_probability_bin"] for row in solved)
        summaries.append({
            "target_band": target_band,
            "attempts": len(selected),
            "solved": len(solved),
            "solve_rate": (
                len(solved) / len(selected) if selected else 0.0
            ),
            "target_hits": sum(row["target_hit"] for row in solved),
            "target_hit_rate": (
                sum(row["target_hit"] for row in solved) / len(solved)
                if solved else 0.0
            ),
            "mc_low": actual["low"],
            "mc_boundary_below": actual["boundary_below"],
            "mc_boundary_above": actual["boundary_above"],
            "mc_high": actual["high"],
            "mean_nonlinear_p_min": _rounded(_mean(internal)),
            "mean_mc_p_min": _rounded(_mean(monte_carlo)),
            "median_mc_p_min": _rounded(_median(monte_carlo)),
            "mean_absolute_gap": _rounded(_mean(gaps)),
            "mean_wall_seconds": _rounded(_mean([
                row["wall_seconds"] for row in selected
            ])),
        })
    return summaries


def _plot(rows, output_directory, service_level, boundary_width):
    solved = [row for row in rows if row["candidate_found"]]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    for band in BANDS:
        selected = [row for row in solved if row["target_band"] == band]
        axes[0].scatter(
            [row["nonlinear_p_min"] for row in selected],
            [row["mc_p_min"] for row in selected],
            label=band,
            color=BAND_COLORS[band],
            s=45,
            alpha=0.85,
        )
    axes[0].plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    axes[0].axvline(service_level, color="grey", linestyle=":")
    axes[0].axhline(service_level, color="grey", linestyle=":")
    axes[0].set(
        xlim=(0, 1.01),
        ylim=(0, 1.01),
        xlabel=r"Nichtlinear $p_{min}^{NL}$",
        ylabel=r"Monte Carlo $p_{min}^{MC}$",
        title="Interne Näherung gegen Simulation",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    positions = list(range(len(BANDS)))
    bottoms = [0] * len(BANDS)
    for actual_band in BANDS:
        counts = [
            sum(
                row["target_band"] == target
                and row["mc_probability_bin"] == actual_band
                for row in solved
            )
            for target in BANDS
        ]
        axes[1].bar(
            positions,
            counts,
            bottom=bottoms,
            label=actual_band,
            color=BAND_COLORS[actual_band],
        )
        bottoms = [left + value for left, value in zip(bottoms, counts)]
    axes[1].set_xticks(positions, BANDS, rotation=20, ha="right")
    axes[1].set(
        ylabel="Anzahl gelöster Kandidaten",
        title="MC-Bin je internem Zielband",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)

    lower = service_level - boundary_width
    upper = service_level + boundary_width
    figure.suptitle(
        f"Nichtlineare Bänder vs. Monte Carlo "
        f"(Grenzen {lower:.2f}, {service_level:.2f}, {upper:.2f})"
    )
    figure.tight_layout()
    pdf_path = output_directory / "nonlinear_probability_band_analysis.pdf"
    png_path = output_directory / "nonlinear_probability_band_analysis.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def run_analysis(
    *,
    instance_count=5,
    jobs=3,
    machines=3,
    replications=1000,
    time_limit_seconds=10.0,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    instance_names=None,
):
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    source = config["training"]["data_generation"]
    fixed = dict(source["fixed_y"])
    fixed.update({
        "time_limit_seconds": float(time_limit_seconds),
        "pool_candidates": 1,
        "pool_search_mode": 0,
    })
    ranges = source.get("reliability_ranges") or {}
    generation = {
        "random_seed": int(source.get("random_seed", 42)),
        "simulation": source.get("simulation"),
        "alpha_range": ranges.get("alpha"),
        "beta_range": ranges.get("beta"),
        "repair_rate_range": ranges.get("repair_rate"),
        "fixed_y": fixed,
    }
    graph_config = GENERATOR.normalize_reliability_graph_config(
        config["constraint"]["weibull"]["reliability_graph"]
    )
    names = (
        [str(name) for name in instance_names]
        if instance_names
        else _instance_names(jobs, machines, instance_count)
    )
    fix_ratio = float(fixed["fix_ratios"][0])
    rows = []
    total = len(names) * len(BANDS)
    position = 0
    for instance_name in names:
        instance = GENERATOR.load_generated_instance(instance_name)
        for target_band in BANDS:
            position += 1
            started = time.perf_counter()
            error = ""
            try:
                candidates = GENERATOR._run_neighborhood(
                    instance,
                    instance_name,
                    0,
                    generation,
                    graph_config,
                    fix_ratio,
                    "nonlinear",
                    target_band,
                )
            except Exception as exc:  # preserve the remaining numerical run
                candidates = []
                error = f"{type(exc).__name__}: {exc}"
            wall_seconds = time.perf_counter() - started
            candidate = candidates[0] if candidates else None
            nonlinear_p_min = (
                candidate["nonlinear_min_job_probability"]
                if candidate else None
            )
            mc_p_min = candidate["min_job_probability"] if candidate else None
            actual_bin = (
                GENERATOR._probability_bin(
                    mc_p_min,
                    graph_config.service_level,
                    fixed["service_boundary_width"],
                ) if candidate else ""
            )
            row = {
                "instance_name": instance_name,
                "target_band": target_band,
                "candidate_found": bool(candidate),
                "nonlinear_p_min": nonlinear_p_min,
                "mc_p_min": mc_p_min,
                "mc_probability_bin": actual_bin,
                "target_hit": bool(candidate and actual_bin == target_band),
                "mc_service_feasible": bool(
                    candidate and mc_p_min >= graph_config.service_level
                ),
                "nonlinear_job_probabilities": (
                    json.dumps(candidate["nonlinear_job_probabilities"])
                    if candidate else ""
                ),
                "mc_job_probabilities": (
                    json.dumps(candidate["job_probabilities"])
                    if candidate else ""
                ),
                "wall_seconds": wall_seconds,
                "solver_runtime_seconds": (
                    candidate["row"]["solver_runtime_seconds"]
                    if candidate else None
                ),
                "replications": int(replications),
                "error": error,
            }
            rows.append(row)
            print(
                f"[Band analysis] {position}/{total} | {instance_name} | "
                f"target={target_band} | found={bool(candidate)} | "
                f"p_nl={_rounded(nonlinear_p_min)} | "
                f"p_mc={_rounded(mc_p_min)} | wall={wall_seconds:.2f}s",
                flush=True,
            )

    output_directory = Path(output_directory)
    if not output_directory.is_absolute():
        output_directory = ROOT_DIR / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = output_directory / "nonlinear_probability_band_analysis.csv"
    summary_path = (
        output_directory / "nonlinear_probability_band_summary.csv"
    )
    metadata_path = (
        output_directory / "nonlinear_probability_band_analysis.json"
    )
    raw_fields = list(rows[0]) if rows else []
    _write_csv(raw_path, rows, raw_fields)
    summaries = _summaries(rows)
    _write_csv(summary_path, summaries, list(summaries[0]))
    pdf_path, png_path = _plot(
        rows,
        output_directory,
        graph_config.service_level,
        fixed["service_boundary_width"],
    )
    metadata_path.write_text(json.dumps({
        "instances": names,
        "jobs": int(jobs),
        "machines": int(machines),
        "replications": int(replications),
        "time_limit_seconds": float(time_limit_seconds),
        "service_level": graph_config.service_level,
        "boundary_width": fixed["service_boundary_width"],
        "fix_ratio": fix_ratio,
        "summary": summaries,
    }, indent=2), encoding="utf-8")
    for path in (raw_path, summary_path, pdf_path, png_path, metadata_path):
        print(f"WROTE {path}", flush=True)
    return {
        "rows": rows,
        "summaries": summaries,
        "raw_csv": raw_path,
        "summary_csv": summary_path,
        "pdf": pdf_path,
        "png": png_path,
        "metadata": metadata_path,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--machines", type=int, default=3)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--instance-name",
        dest="instance_names",
        action="append",
        help="Explicit instance name; repeat the option for several names.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    return run_analysis(
        instance_count=args.instances,
        jobs=args.jobs,
        machines=args.machines,
        replications=args.replications,
        time_limit_seconds=args.time_limit,
        instance_names=args.instance_names,
        output_directory=(
            args.output_directory
            if args.output_directory is not None
            else DEFAULT_OUTPUT_DIRECTORY
        ),
    )


if __name__ == "__main__":
    main()
