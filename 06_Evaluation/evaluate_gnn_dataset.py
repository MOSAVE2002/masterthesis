"""Evaluate GNN label coverage and a constant-probability baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config.json"
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


def _load_dataset(dataset_directory, expected_service_level):
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
                "job_ontime_probabilities",
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
                        row["job_ontime_probabilities"]
                    )
                ]
                if not probabilities or any(
                    not 0.0 <= value <= 1.0 for value in probabilities
                ):
                    raise ValueError(
                        f"Invalid job probabilities in {path}:{row_number}."
                    )
                graph_config = json.loads(
                    row["reliability_graph_parameters"]
                )
                service_level = float(graph_config["service_level"])
                if not math.isclose(
                    service_level,
                    float(expected_service_level),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"{path}:{row_number} uses service_level="
                        f"{service_level}, expected {expected_service_level}. "
                        "Regenerate the GNN dataset before evaluation."
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
    counts = Counter(
        _probability_bin(value, service_level, boundary_width)
        for value in probabilities
    )
    labels = len(probabilities)
    mean = sum(probabilities) / labels
    variance = sum((value - mean) ** 2 for value in probabilities) / labels
    exact_zero = sum(value <= 1e-12 for value in probabilities)
    exact_one = sum(value >= 1.0 - 1e-12 for value in probabilities)
    boundary = counts["boundary_below"] + counts["boundary_above"]
    entropy = -sum(
        _safe_divide(counts[name], labels)
        * math.log2(_safe_divide(counts[name], labels))
        for name in PROBABILITY_BINS
        if counts[name]
    )
    return {
        "split": split,
        "group_type": group_type,
        "group": group,
        "graphs": int(graphs),
        "labels": labels,
        "mean_probability": mean,
        "standard_deviation": math.sqrt(variance),
        "exact_zero_rate": _safe_divide(exact_zero, labels),
        "exact_one_rate": _safe_divide(exact_one, labels),
        "endpoint_rate": _safe_divide(exact_zero + exact_one, labels),
        "low_rate": _safe_divide(counts["low"], labels),
        "boundary_below_rate": _safe_divide(
            counts["boundary_below"], labels
        ),
        "boundary_above_rate": _safe_divide(
            counts["boundary_above"], labels
        ),
        "high_rate": _safe_divide(counts["high"], labels),
        "boundary_rate": _safe_divide(boundary, labels),
        "four_bin_entropy_bits": entropy,
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


def _classification_metrics(labels, predictions, threshold):
    true = [value >= threshold for value in labels]
    predicted = [value >= threshold for value in predictions]
    tp = sum(actual and estimate for actual, estimate in zip(true, predicted))
    tn = sum(not actual and not estimate for actual, estimate in zip(true, predicted))
    fp = sum(not actual and estimate for actual, estimate in zip(true, predicted))
    fn = sum(actual and not estimate for actual, estimate in zip(true, predicted))
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    return {
        "accuracy": _safe_divide(tp + tn, len(labels)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _safe_divide(2.0 * precision * recall, precision + recall),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _constant_baseline_rows(
    labels_by_split,
    service_level,
    boundary_width,
):
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
        boundary_errors = [
            abs(error)
            for label, error in zip(labels, errors)
            if abs(label - service_level) <= boundary_width + 1e-12
        ]
        metrics = _classification_metrics(
            labels, predictions, service_level
        )
        rows.append({
            "split": split,
            "baseline": "training_mean_probability",
            "constant_probability": constant,
            "labels": len(labels),
            "mae": sum(abs(error) for error in errors) / len(errors),
            "rmse": math.sqrt(
                sum(error * error for error in errors) / len(errors)
            ),
            "soft_brier_score": (
                sum(error * error for error in errors) / len(errors)
            ),
            "mean_error_bias": sum(errors) / len(errors),
            "boundary_labels": len(boundary_errors),
            "boundary_mae": (
                sum(boundary_errors) / len(boundary_errors)
                if boundary_errors else None
            ),
            **metrics,
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
            range=(0.0, 1.0),
            color="#4C78A8",
            edgecolor="white",
        )
        axis.axvspan(
            service_level - boundary_width,
            service_level + boundary_width,
            color="#F58518",
            alpha=0.18,
            label="Boundary",
        )
        axis.axvline(
            service_level,
            color="#E45756",
            linestyle="--",
            linewidth=1.5,
            label=f"alpha={service_level:.2f}",
        )
        axis.set_title(f"{split}: {len(values)} Job-Labels")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylabel("Anzahl")
        axis.legend(loc="upper center")
    axes[-1, 0].set_xlabel("Monte-Carlo-Pünktlichkeitswahrscheinlichkeit")
    figure.suptitle("Verteilung der GNN-Trainingslabels", fontsize=14)
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_dataset_evaluation(
    *,
    dataset_directory=DEFAULT_DATASET_DIRECTORY,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    expected_service_level=0.95,
    boundary_width=0.03,
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
    metadata = {
        "status": "evaluated",
        "dataset_directory": str(dataset_directory),
        "available_splits": list(source_paths),
        "source_files": {
            split: str(path) for split, path in source_paths.items()
        },
        "graphs": len(graph_rows),
        "labels": sum(len(values) for values in labels_by_split.values()),
        "service_level": float(expected_service_level),
        "boundary_width": float(boundary_width),
        "histogram_bins": int(histogram_bins),
        "outputs": {
            "distribution_csv": str(distribution_path),
            "constant_baseline_csv": str(baseline_path),
            "histogram_pdf": str(histogram_pdf),
            "histogram_png": str(histogram_png),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"WROTE {distribution_path}")
    print(f"WROTE {baseline_path}")
    print(f"WROTE {histogram_pdf}")
    print(f"WROTE {histogram_png}")
    print(f"WROTE {metadata_path}")
    return {
        "distribution_csv": distribution_path,
        "constant_baseline_csv": baseline_path,
        "histogram_pdf": histogram_pdf,
        "histogram_png": histogram_png,
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
    graph_config = config["constraint"]["weibull"]["reliability_graph"]
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
        expected_service_level=graph_config["service_level"],
        boundary_width=settings.get(
            "boundary_width", fixed["service_boundary_width"]
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
    graph_config = config["constraint"]["weibull"]["reliability_graph"]
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
        expected_service_level=graph_config["service_level"],
        boundary_width=settings.get(
            "boundary_width", fixed["service_boundary_width"]
        ),
        histogram_bins=settings.get("histogram_bins", 20),
    )


if __name__ == "__main__":
    main()
