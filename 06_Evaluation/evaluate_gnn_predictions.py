"""Evaluate GNN probability accuracy and calibration against Monte Carlo."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config.json"
DEFAULT_INPUT_PATH = ROOT_DIR / "06_Evaluation" / "results" / "job_comparison.csv"
DEFAULT_OUTPUT_DIRECTORY = ROOT_DIR / "06_Evaluation" / "results"


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_job_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _gnn_probability_rows(job_rows):
    result = []
    for row in job_rows:
        if row.get("solver") != "gurobi_gnn":
            continue
        if row.get("postsolve_evaluation_status", "evaluated") != "evaluated":
            continue
        prediction = float(row["internal_probability"])
        observed = float(row["mc_ontime_probability"])
        target = float(row["target_service_level"])
        if not 0.0 <= prediction <= 1.0:
            raise ValueError(
                f"GNN prediction outside [0,1]: {prediction}."
            )
        if not 0.0 <= observed <= 1.0:
            raise ValueError(
                f"Monte-Carlo probability outside [0,1]: {observed}."
            )
        result.append({
            "model": row["model"],
            "instance_name": row["instance_name"],
            "job_id": row["job_id"],
            "prediction": prediction,
            "observed": observed,
            "target": target,
        })
    return result


def _classification_metrics(rows):
    pairs = [
        (
            row["prediction"] >= row["target"],
            row["observed"] >= row["target"],
        )
        for row in rows
    ]
    tp = sum(predicted and actual for predicted, actual in pairs)
    tn = sum(not predicted and not actual for predicted, actual in pairs)
    fp = sum(predicted and not actual for predicted, actual in pairs)
    fn = sum(not predicted and actual for predicted, actual in pairs)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    return {
        "threshold_accuracy": _safe_divide(tp + tn, len(rows)),
        "threshold_balanced_accuracy": 0.5 * (recall + specificity),
        "threshold_precision": precision,
        "threshold_recall": recall,
        "threshold_specificity": specificity,
        "threshold_f1": _safe_divide(
            2.0 * precision * recall, precision + recall
        ),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _calibration_rows(rows, calibration_bins, model):
    grouped = defaultdict(list)
    for row in rows:
        index = min(
            int(calibration_bins) - 1,
            int(row["prediction"] * int(calibration_bins)),
        )
        grouped[index].append(row)
    result = []
    for index in range(int(calibration_bins)):
        values = grouped.get(index, [])
        if not values:
            continue
        mean_prediction = sum(
            row["prediction"] for row in values
        ) / len(values)
        mean_observed = sum(row["observed"] for row in values) / len(values)
        result.append({
            "model": model,
            "bin_index": index,
            "bin_lower": index / int(calibration_bins),
            "bin_upper": (index + 1) / int(calibration_bins),
            "count": len(values),
            "mean_prediction": mean_prediction,
            "mean_mc_probability": mean_observed,
            "calibration_gap": mean_prediction - mean_observed,
            "absolute_calibration_gap": abs(
                mean_prediction - mean_observed
            ),
        })
    return result


def _metric_row(rows, model, boundary_width, calibration_rows):
    errors = [row["prediction"] - row["observed"] for row in rows]
    boundary_errors = [
        abs(row["prediction"] - row["observed"])
        for row in rows
        if abs(row["observed"] - row["target"])
        <= float(boundary_width) + 1e-12
    ]
    ece = sum(
        row["count"] * row["absolute_calibration_gap"]
        for row in calibration_rows
    ) / len(rows)
    maximum_calibration_error = max(
        (row["absolute_calibration_gap"] for row in calibration_rows),
        default=0.0,
    )
    return {
        "model": model,
        "job_predictions": len(rows),
        "instances": len({row["instance_name"] for row in rows}),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "soft_brier_score": (
            sum(error * error for error in errors) / len(errors)
        ),
        "mean_error_bias": sum(errors) / len(errors),
        "maximum_absolute_error": max(abs(error) for error in errors),
        "boundary_predictions": len(boundary_errors),
        "boundary_mae": (
            sum(boundary_errors) / len(boundary_errors)
            if boundary_errors else None
        ),
        "expected_calibration_error": ece,
        "maximum_calibration_error": maximum_calibration_error,
        **_classification_metrics(rows),
    }


def _write_calibration_plot(pdf_path, png_path, calibration_rows):
    cache_directory = Path(tempfile.gettempdir()) / "fjsp-matplotlib-cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_directory))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    models = sorted({row["model"] for row in calibration_rows})
    for model in models:
        rows = [row for row in calibration_rows if row["model"] == model]
        axis.plot(
            [row["mean_prediction"] for row in rows],
            [row["mean_mc_probability"] for row in rows],
            marker="o",
            linewidth=1.5,
            label=model,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mittlere GNN-Wahrscheinlichkeit")
    axis.set_ylabel("Mittlere Monte-Carlo-Wahrscheinlichkeit")
    axis.set_title("Kalibrierung der GNN-Wahrscheinlichkeiten")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_prediction_evaluation(
    *,
    job_rows=None,
    input_path=DEFAULT_INPUT_PATH,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    boundary_width=0.03,
    calibration_bins=10,
):
    output_directory = _project_path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if job_rows is None:
        input_path = _project_path(input_path).resolve()
        job_rows = _load_job_rows(input_path)
    else:
        input_path = None
    probability_rows = _gnn_probability_rows(job_rows)
    metric_path = output_directory / "gnn_prediction_metrics.csv"
    calibration_path = output_directory / "gnn_calibration.csv"
    calibration_pdf = output_directory / "gnn_calibration.pdf"
    calibration_png = output_directory / "gnn_calibration.png"
    metadata_path = output_directory / "gnn_prediction_diagnostics.json"
    metric_fields = [
        "model", "job_predictions", "instances", "mae", "rmse",
        "soft_brier_score", "mean_error_bias", "maximum_absolute_error",
        "boundary_predictions", "boundary_mae",
        "expected_calibration_error", "maximum_calibration_error",
        "threshold_accuracy", "threshold_balanced_accuracy",
        "threshold_precision", "threshold_recall",
        "threshold_specificity", "threshold_f1", "true_positive",
        "true_negative", "false_positive", "false_negative",
    ]
    calibration_fields = [
        "model", "bin_index", "bin_lower", "bin_upper", "count",
        "mean_prediction", "mean_mc_probability", "calibration_gap",
        "absolute_calibration_gap",
    ]
    if not probability_rows:
        _write_csv(metric_path, [], metric_fields)
        _write_csv(calibration_path, [], calibration_fields)
        metadata = {
            "status": "skipped_no_gnn_predictions",
            "input_path": str(input_path) if input_path else None,
            "outputs": {
                "metrics_csv": str(metric_path),
                "calibration_csv": str(calibration_path),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print("[Evaluate:GNN predictions] skipped: no GNN rows")
        return {
            "metrics_csv": metric_path,
            "calibration_csv": calibration_path,
            "metadata": metadata_path,
            "metric_rows": [],
            "calibration_rows": [],
        }

    all_calibration = []
    metric_rows = []
    by_model = defaultdict(list)
    for row in probability_rows:
        by_model[row["model"]].append(row)
    for model in sorted(by_model):
        rows = by_model[model]
        calibration = _calibration_rows(rows, calibration_bins, model)
        all_calibration.extend(calibration)
        metric_rows.append(_metric_row(
            rows, model, boundary_width, calibration
        ))
    _write_csv(metric_path, metric_rows, metric_fields)
    _write_csv(calibration_path, all_calibration, calibration_fields)
    _write_calibration_plot(
        calibration_pdf, calibration_png, all_calibration
    )
    metadata = {
        "status": "evaluated",
        "input_path": str(input_path) if input_path else "in_memory_job_rows",
        "models": sorted(by_model),
        "job_predictions": len(probability_rows),
        "boundary_width": float(boundary_width),
        "calibration_bins": int(calibration_bins),
        "calibration_weighting": "equal-width bins, weighted ECE",
        "observed_probability": "independent Monte Carlo point estimate",
        "outputs": {
            "metrics_csv": str(metric_path),
            "calibration_csv": str(calibration_path),
            "calibration_pdf": str(calibration_pdf),
            "calibration_png": str(calibration_png),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"WROTE {metric_path}")
    print(f"WROTE {calibration_path}")
    print(f"WROTE {calibration_pdf}")
    print(f"WROTE {calibration_png}")
    print(f"WROTE {metadata_path}")
    return {
        "metrics_csv": metric_path,
        "calibration_csv": calibration_path,
        "calibration_pdf": calibration_pdf,
        "calibration_png": calibration_png,
        "metadata": metadata_path,
        "metric_rows": metric_rows,
        "calibration_rows": all_calibration,
    }


def evaluate_from_config(config=None):
    if config is None:
        with CONFIG_PATH.open(encoding="utf-8") as handle:
            config = json.load(handle)
    if not bool(config.get("workflow", {}).get("evaluate", False)):
        print("[Evaluate:GNN predictions] skipped: workflow.evaluate is false")
        return None
    settings = config.get("evaluation", {}).get(
        "prediction_diagnostics", {}
    )
    if settings and not bool(settings.get("enabled", True)):
        return None
    evaluation = config.get("evaluation", {})
    output_directory = settings.get(
        "output_directory",
        evaluation.get("output_directory", DEFAULT_OUTPUT_DIRECTORY),
    )
    return run_prediction_evaluation(
        input_path=settings.get(
            "input_path", Path(output_directory) / "job_comparison.csv"
        ),
        output_directory=output_directory,
        boundary_width=settings.get(
            "boundary_width",
            config["training"]["data_generation"]["fixed_y"][
                "service_boundary_width"
            ],
        ),
        calibration_bins=settings.get("calibration_bins", 10),
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate embedded GNN probabilities against Monte Carlo."
    )
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    evaluation = config.get("evaluation", {})
    settings = evaluation.get("prediction_diagnostics", {})
    output_directory = (
        args.output_directory
        if args.output_directory is not None
        else settings.get(
            "output_directory",
            evaluation.get("output_directory", DEFAULT_OUTPUT_DIRECTORY),
        )
    )
    input_path = (
        args.input_path
        if args.input_path is not None
        else settings.get(
            "input_path", Path(output_directory) / "job_comparison.csv"
        )
    )
    return run_prediction_evaluation(
        input_path=input_path,
        output_directory=output_directory,
        boundary_width=settings.get(
            "boundary_width",
            config["training"]["data_generation"]["fixed_y"][
                "service_boundary_width"
            ],
        ),
        calibration_bins=settings.get("calibration_bins", 10),
    )


if __name__ == "__main__":
    main()
