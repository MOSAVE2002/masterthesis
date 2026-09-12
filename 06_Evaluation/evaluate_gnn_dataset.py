"""Evaluate expected-repair-buffer labels and a constant baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config.json"
from helper.surrogate_constraint import configured_constraint_type, target_column

TARGET_COLUMN = target_column(configured_constraint_type())
DEFAULT_DATASET_DIRECTORY = ROOT_DIR / "02_data" / "gnn_dataset"
DEFAULT_OUTPUT_DIRECTORY = ROOT_DIR / "06_Evaluation" / "results"
SPLIT_FILES = {
    "training": Path("training") / "graphs_training.csv",
    "valid": Path("valid") / "graphs_valid.csv",
    "test": Path("test") / "graphs_test.csv",
}
PROBABILITY_BINS = (
    "low",
    "boundary_below",
    "boundary_above",
    "high",
)
_INSTANCE_PATTERN = re.compile(r"^i(?P<jobs>\d+)_k(?P<machines>\d+)_")


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _probability_bin(value, service_level, boundary_width):
    value = float(value)
    lower = max(0.0, float(service_level) - float(boundary_width))
    upper = min(1.0, float(service_level) + float(boundary_width))
    if value < lower - 1e-12:
        return "low"
    if value < float(service_level) - 1e-12:
        return "boundary_below"
    if value <= upper + 1e-12:
        return "boundary_above"
    return "high"


def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_dataset(dataset_directory, _expected_service_level):
    dataset_directory = Path(dataset_directory)
    labels_by_split = defaultdict(list)
    graph_rows = []
    source_paths = {}
    for split, relative_path in SPLIT_FILES.items():
        path = dataset_directory / relative_path
        if not path.exists():
            continue
        source_paths[split] = path
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "instance_name",
                TARGET_COLUMN,
                "reliability_graph_parameters",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} is missing columns: {sorted(missing)}"
                )
            for row_number, row in enumerate(reader, start=2):
                instance_name = row["instance_name"]
                match = _INSTANCE_PATTERN.match(instance_name)
                if not match:
                    raise ValueError(
                        f"Cannot parse instance size {instance_name!r} "
                        f"in {path}:{row_number}."
                    )
                probabilities = [
                    float(value) for value in json.loads(
                        row[TARGET_COLUMN]
                    )
                ]
                if not probabilities or any(
                    value < 0.0 for value in probabilities
                ):
                    raise ValueError(
                        f"Invalid repair buffers in {path}:{row_number}."
                    )
                generation_mode = (
                    row.get("candidate_generation_mode")
                    or "legacy_unspecified"
                )
                graph_record = {
                    "split": split,
                    "instance_name": instance_name,
                    "jobs": int(match.group("jobs")),
                    "machines": int(match.group("machines")),
                    "generation_mode": generation_mode,
                    "due_date_factor": (
                        float(row["training_effective_due_date_factor"])
                        if row.get("training_effective_due_date_factor")
                        not in (None, "")
                        else float(row["due_date_factor"])
                        if row.get("due_date_factor") not in (None, "")
                        else None
                    ),
                    "probabilities": probabilities,
                }
                graph_rows.append(graph_record)
                labels_by_split[split].extend(probabilities)
    if "training" not in source_paths:
        raise FileNotFoundError(
            f"Training dataset not found below {dataset_directory}."
        )
    return graph_rows, dict(labels_by_split), source_paths


def _distribution_row(
    split,
    group_type,
    group,
    graphs,
    probabilities,
    service_level,
    boundary_width,
):
    del service_level, boundary_width
    labels = len(probabilities)
    mean = sum(probabilities) / labels
    variance = sum((value - mean) ** 2 for value in probabilities) / labels
    exact_zero = sum(value <= 1e-12 for value in probabilities)
    ordered = sorted(probabilities)
    quantile = lambda fraction: ordered[round(fraction * (labels - 1))]
    return {
        "split": split,
        "group_type": group_type,
        "group": group,
        "graphs": int(graphs),
        "labels": labels,
        "mean_repair_buffer": mean,
        "standard_deviation": math.sqrt(variance),
        "minimum": ordered[0],
        "p10": quantile(0.10),
        "median": quantile(0.50),
        "p90": quantile(0.90),
        "maximum": ordered[-1],
        "exact_zero_rate": _safe_divide(exact_zero, labels),
    }


def _distribution_rows(
    graph_rows,
    service_level,
    boundary_width,
):
    result = []
    for split in SPLIT_FILES:
        split_graphs = [row for row in graph_rows if row["split"] == split]
        if not split_graphs:
            continue
        groups = [("overall", "all", split_graphs)]
        for jobs in sorted({row["jobs"] for row in split_graphs}):
            groups.append((
                "job_count",
                str(jobs),
                [row for row in split_graphs if row["jobs"] == jobs],
            ))
        for mode in sorted({row["generation_mode"] for row in split_graphs}):
            groups.append((
                "generation_mode",
                mode,
                [
                    row for row in split_graphs
                    if row["generation_mode"] == mode
                ],
            ))
        for factor in sorted({
            row["due_date_factor"]
            for row in split_graphs
            if row["due_date_factor"] is not None
        }):
            groups.append((
                "effective_due_date_factor",
                f"{factor:.2f}",
                [
                    row for row in split_graphs
                    if row["due_date_factor"] == factor
                ],
            ))
        for group_type, group, rows in groups:
            probabilities = [
                value for row in rows for value in row["probabilities"]
            ]
            result.append(_distribution_row(
                split,
                group_type,
                group,
                len(rows),
                probabilities,
                service_level,
                boundary_width,
            ))
    return result


def _constant_baseline_rows(
    labels_by_split,
    service_level,
    boundary_width,
):
    del service_level, boundary_width
    constant = sum(labels_by_split["training"]) / len(
        labels_by_split["training"]
    )
    rows = []
    for split in SPLIT_FILES:
        labels = labels_by_split.get(split)
        if not labels:
            continue
        predictions = [constant] * len(labels)
        errors = [prediction - label for prediction, label in zip(
            predictions, labels
        )]
        rows.append({
            "split": split,
            "baseline": "training_mean_repair_buffer",
            "constant_repair_buffer": constant,
            "labels": len(labels),
            "mae": sum(abs(error) for error in errors) / len(errors),
            "rmse": math.sqrt(
                sum(error * error for error in errors) / len(errors)
            ),
            "mean_error_bias": sum(errors) / len(errors),
        })
    return rows


def _write_histogram(
    pdf_path,
    png_path,
    labels_by_split,
    service_level,
    boundary_width,
    histogram_bins,
):
    cache_directory = Path(tempfile.gettempdir()) / "fjsp-matplotlib-cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_directory))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [split for split in SPLIT_FILES if labels_by_split.get(split)]
    figure, axes = plt.subplots(
        len(available),
        1,
        figsize=(10, max(3.2, 3.0 * len(available))),
        squeeze=False,
    )
    for axis, split in zip(axes[:, 0], available):
        values = labels_by_split[split]
        axis.hist(
            values,
            bins=int(histogram_bins),
            color="#4C78A8",
            edgecolor="white",
        )
        axis.set_title(f"{split}: {len(values)} Job-Labels")
        axis.set_ylabel("Anzahl")
    axes[-1, 0].set_xlabel("Erwarteter Reparaturpuffer [ZE]")
    figure.suptitle("Verteilung der GNN-Reparaturpuffer-Labels", fontsize=14)
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _due_factor_plot_groups(graph_rows, maximum_groups=12):
    """Bound figure height for calibrated, almost-continuous due-date factors."""
    values_by_factor = defaultdict(list)
    for row in graph_rows:
        factor = row["due_date_factor"]
        if factor is not None:
            values_by_factor[factor].extend(row["probabilities"])
    factors = sorted(values_by_factor)
    if not factors:
        return []
    group_size = max(1, math.ceil(len(factors) / maximum_groups))
    return [(part[0], part[-1], [value for factor in part for value in values_by_factor[factor]])
            for start in range(0, len(factors), group_size)
            if (part := factors[start:start + group_size])]


def _write_due_factor_histogram(
    pdf_path,
    png_path,
    graph_rows,
    service_level,
    boundary_width,
    histogram_bins,
):
    """Plot individual job labels by due-date factor, not graph minima."""
    groups = _due_factor_plot_groups(graph_rows)
    if not groups:
        return False

    cache_directory = Path(tempfile.gettempdir()) / "fjsp-matplotlib-cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_directory))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(groups),
        1,
        figsize=(10, max(3.2, 3.0 * len(groups))),
        squeeze=False,
    )
    for axis, (lower, upper, values) in zip(axes[:, 0], groups):
        axis.hist(
            values,
            bins=int(histogram_bins),
            color="#4C78A8",
            edgecolor="white",
        )
        label = (f"Due-Date-Faktor {lower:.3f}" if lower == upper
                 else f"Due-Date-Faktorbereich {lower:.3f}–{upper:.3f}")
        axis.set_title(f"{label}: {len(values)} Job-Labels")
        axis.set_ylabel("Anzahl")
    axes[-1, 0].set_xlabel("Erwarteter Reparaturpuffer [ZE]")
    figure.suptitle(
        "Job-Level-Trainingslabels nach Due-Date-Faktor", fontsize=14
    )
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def run_dataset_evaluation(
    *,
    dataset_directory=DEFAULT_DATASET_DIRECTORY,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    expected_service_level=0.50,
    boundary_width=0.25,
    histogram_bins=20,
):
    dataset_directory = _project_path(dataset_directory).resolve()
    output_directory = _project_path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    graph_rows, labels_by_split, source_paths = _load_dataset(
        dataset_directory, expected_service_level
    )
    distributions = _distribution_rows(
        graph_rows, expected_service_level, boundary_width
    )
    baselines = _constant_baseline_rows(
        labels_by_split, expected_service_level, boundary_width
    )
    distribution_path = output_directory / "gnn_dataset_distribution.csv"
    baseline_path = output_directory / "gnn_constant_baseline.csv"
    histogram_pdf = output_directory / "gnn_label_histograms.pdf"
    histogram_png = output_directory / "gnn_label_histograms.png"
    factor_histogram_pdf = (
        output_directory / "gnn_job_label_histograms_by_due_factor.pdf"
    )
    factor_histogram_png = (
        output_directory / "gnn_job_label_histograms_by_due_factor.png"
    )
    metadata_path = output_directory / "gnn_dataset_diagnostics.json"
    _write_csv(distribution_path, distributions, list(distributions[0]))
    _write_csv(baseline_path, baselines, list(baselines[0]))
    _write_histogram(
        histogram_pdf,
        histogram_png,
        labels_by_split,
        expected_service_level,
        boundary_width,
        histogram_bins,
    )
    has_factor_histogram = _write_due_factor_histogram(
        factor_histogram_pdf,
        factor_histogram_png,
        graph_rows,
        expected_service_level,
        boundary_width,
        histogram_bins,
    )
    metadata = {
        "time_unit": "ZE",
        "status": "evaluated",
        "dataset_directory": str(dataset_directory),
        "available_splits": list(source_paths),
        "source_files": {
            split: str(path) for split, path in source_paths.items()
        },
        "graphs": len(graph_rows),
        "labels": sum(len(values) for values in labels_by_split.values()),
        "label_distribution_center": float(expected_service_level),
        "label_distribution_half_width": float(boundary_width),
        "histogram_bins": int(histogram_bins),
        "outputs": {
            "distribution_csv": str(distribution_path),
            "constant_baseline_csv": str(baseline_path),
            "histogram_pdf": str(histogram_pdf),
            "histogram_png": str(histogram_png),
            **({
                "due_factor_histogram_pdf": str(factor_histogram_pdf),
                "due_factor_histogram_png": str(factor_histogram_png),
            } if has_factor_histogram else {}),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"WROTE {distribution_path}")
    print(f"WROTE {baseline_path}")
    print(f"WROTE {histogram_pdf}")
    print(f"WROTE {histogram_png}")
    if has_factor_histogram:
        print(f"WROTE {factor_histogram_pdf}")
        print(f"WROTE {factor_histogram_png}")
    print(f"WROTE {metadata_path}")
    return {
        "distribution_csv": distribution_path,
        "constant_baseline_csv": baseline_path,
        "histogram_pdf": histogram_pdf,
        "histogram_png": histogram_png,
        "due_factor_histogram_pdf": (
            factor_histogram_pdf if has_factor_histogram else None
        ),
        "due_factor_histogram_png": (
            factor_histogram_png if has_factor_histogram else None
        ),
        "metadata": metadata_path,
        "distribution_rows": distributions,
        "baseline_rows": baselines,
    }


def evaluate_from_config(config=None):
    if config is None:
        with CONFIG_PATH.open(encoding="utf-8") as handle:
            config = json.load(handle)
    if not bool(config.get("workflow", {}).get("evaluate", False)):
        print("[Evaluate:GNN dataset] skipped: workflow.evaluate is false")
        return None
    settings = config.get("evaluation", {}).get("dataset_diagnostics", {})
    if settings and not bool(settings.get("enabled", True)):
        return None
    data_config = config["training"]["data_generation"]
    fixed = data_config["fixed_y"]
    return run_dataset_evaluation(
        dataset_directory=settings.get(
            "dataset_directory", data_config["output_directory"]
        ),
        output_directory=settings.get(
            "output_directory",
            config.get("evaluation", {}).get(
                "output_directory", DEFAULT_OUTPUT_DIRECTORY
            ),
        ),
        expected_service_level=fixed.get("label_distribution_center", 0.50),
        boundary_width=settings.get(
            "boundary_width", fixed.get("label_distribution_half_width", 0.25)
        ),
        histogram_bins=settings.get("histogram_bins", 20),
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GNN dataset labels and constant baseline."
    )
    parser.add_argument("--dataset-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    settings = config.get("evaluation", {}).get("dataset_diagnostics", {})
    data_config = config["training"]["data_generation"]
    fixed = data_config["fixed_y"]
    return run_dataset_evaluation(
        dataset_directory=(
            args.dataset_directory
            if args.dataset_directory is not None
            else settings.get(
                "dataset_directory", data_config["output_directory"]
            )
        ),
        output_directory=(
            args.output_directory
            if args.output_directory is not None
            else settings.get(
                "output_directory",
                config.get("evaluation", {}).get(
                    "output_directory", DEFAULT_OUTPUT_DIRECTORY
                ),
            )
        ),
        expected_service_level=fixed.get("label_distribution_center", 0.50),
        boundary_width=settings.get(
            "boundary_width", fixed.get("label_distribution_half_width", 0.25)
        ),
        histogram_bins=settings.get("histogram_bins", 20),
    )


if __name__ == "__main__":
    main()
