"""Compare probability losses on an instance-disjoint training holdout."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

training = importlib.import_module(
    "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
)

DEFAULT_DATASET = (
    ROOT_DIR / "02_data/gnn_dataset/training/graphs_training.csv"
)
DEFAULT_OUTPUT_DIRECTORY = ROOT_DIR / "06_Evaluation/results/gnn_loss_experiment"
DEFAULT_LOSSES = [
    training.LOSS_MSE,
    training.LOSS_ASYMMETRIC_MSE,
    training.LOSS_HUBER,
    training.LOSS_BOUNDARY_WEIGHTED_MSE,
    training.LOSS_BOUNDARY_WEIGHTED_ASYMMETRIC_MSE,
]
INSTANCE_SIZE_PATTERN = re.compile(r"^i(?P<jobs>\d+)_k(?P<machines>\d+)_")


def _instance_names(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return [row["instance_name"] for row in csv.DictReader(handle)]


def _stratum(instance_name):
    match = INSTANCE_SIZE_PATTERN.match(instance_name)
    if not match:
        raise ValueError(f"Cannot parse instance size from {instance_name!r}.")
    return int(match.group("jobs")), int(match.group("machines"))


def _holdout_instances(instance_names, fraction, seed):
    grouped = {}
    for name in sorted(set(instance_names)):
        grouped.setdefault(_stratum(name), []).append(name)
    rng = random.Random(int(seed))
    holdout = set()
    for names in grouped.values():
        rng.shuffle(names)
        count = max(1, round(len(names) * float(fraction)))
        holdout.update(names[:count])
    return holdout


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, rows):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    labels = [row["loss"] for row in rows]
    x = range(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, [row["mae"] for row in rows], label="MAE")
    axes[0].bar(
        x,
        [row["boundary_mae"] for row in rows],
        alpha=0.75,
        label="Boundary-MAE",
    )
    axes[0].set_ylabel("Absoluter Wahrscheinlichkeitsfehler")
    axes[0].legend()
    axes[1].bar(
        x,
        [row["threshold_balanced_accuracy"] for row in rows],
        label="Balanced Accuracy",
    )
    axes[1].bar(
        x,
        [row["unsafe_acceptance_rate"] for row in rows],
        alpha=0.75,
        label="Unsafe Acceptance Rate",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(list(x), labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("GNN-Loss-Vergleich auf instanzgetrenntem Holdout")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return True


def compare_losses(
    *,
    dataset=DEFAULT_DATASET,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    losses=DEFAULT_LOSSES,
    epochs=250,
    batch_size=64,
    learning_rate=0.001,
    hidden_channels=4,
    seed=42,
    holdout_fraction=0.2,
    validation_interval=10,
    patience=60,
    boundary_width=0.03,
    boundary_weight=4.0,
    overestimation_weight=2.0,
    huber_delta=0.05,
):
    unknown = set(losses) - training.SUPPORTED_LOSSES
    if unknown:
        raise ValueError(f"Unknown losses: {sorted(unknown)}")
    graphs, feature_names, graph_config, label_method = training.load_graphs(
        Path(dataset)
    )
    names = _instance_names(dataset)
    if len(names) != len(graphs):
        raise ValueError("Graph and instance-name counts differ.")
    holdout_instances = _holdout_instances(
        names, holdout_fraction, seed
    )
    train_graphs = [
        graph for graph, name in zip(graphs, names)
        if name not in holdout_instances
    ]
    holdout_graphs = [
        graph for graph, name in zip(graphs, names)
        if name in holdout_instances
    ]
    train_instances = set(names) - holdout_instances
    if not train_graphs or not holdout_graphs:
        raise ValueError("The instance-level holdout split is empty.")

    device = training._device()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for loss_name in losses:
        torch.manual_seed(int(seed))
        random.seed(int(seed))
        generator = torch.Generator().manual_seed(int(seed))
        train_loader = DataLoader(
            train_graphs,
            batch_size=min(int(batch_size), len(train_graphs)),
            shuffle=True,
            generator=generator,
        )
        holdout_loader = DataLoader(
            holdout_graphs,
            batch_size=min(int(batch_size), len(holdout_graphs)),
        )
        model = training.FJSPGraphSAGE(
            input_size=len(feature_names),
            hidden_channels=int(hidden_channels),
            num_graphsage_layers=1,
            convolution=training.CONV_SAGE,
            initial_probability=graph_config.service_level,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(learning_rate)
        )
        loss_parameters = {
            "loss_name": loss_name,
            "service_level": graph_config.service_level,
            "boundary_width": float(boundary_width),
            "boundary_weight": float(boundary_weight),
            "overestimation_weight": float(overestimation_weight),
            "huber_delta": float(huber_delta),
        }
        best_state = training._cpu_state(model)
        best_mae, best_epoch = math.inf, 0
        started = time.perf_counter()
        for epoch in range(1, int(epochs) + 1):
            train_loss = training._train_epoch(
                model, train_loader, optimizer, **loss_parameters
            )
            if (
                epoch == 1
                or epoch % int(validation_interval) == 0
                or epoch == int(epochs)
            ):
                metrics = training._evaluate(
                    model,
                    holdout_loader,
                    service_level=graph_config.service_level,
                    boundary_width=boundary_width,
                )
                print(
                    f"loss={loss_name} epoch={epoch:04d} "
                    f"train_loss={train_loss:.6g} "
                    f"holdout_mae={metrics['mae']:.6g} "
                    f"boundary_mae={metrics['boundary_mae']:.6g}",
                    flush=True,
                )
                if metrics["mae"] < best_mae - 1e-10:
                    best_state = training._cpu_state(model)
                    best_mae, best_epoch = metrics["mae"], epoch
                elif epoch - best_epoch >= int(patience):
                    break
        seconds = time.perf_counter() - started
        model.load_state_dict(best_state)
        metrics = training._evaluate(
            model,
            holdout_loader,
            service_level=graph_config.service_level,
            boundary_width=boundary_width,
        )
        model_path = output_directory / f"{loss_name}.pt"
        torch.save(training._cpu_state(model), model_path)
        rows.append({
            "loss": loss_name,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "training_seconds": seconds,
            "train_graphs": len(train_graphs),
            "holdout_graphs": len(holdout_graphs),
            "train_instances": len(train_instances),
            "holdout_instances": len(holdout_instances),
            **metrics,
        })
        training._release_memory()

    rows.sort(key=lambda row: row["mae"])
    csv_path = output_directory / "gnn_loss_comparison.csv"
    json_path = output_directory / "gnn_loss_experiment.json"
    pdf_path = output_directory / "gnn_loss_comparison.pdf"
    _write_csv(csv_path, rows)
    metadata = {
        "dataset": str(Path(dataset).resolve()),
        "split": "stratified instance-disjoint holdout",
        "seed": int(seed),
        "service_level": graph_config.service_level,
        "boundary_width": float(boundary_width),
        "boundary_weight": float(boundary_weight),
        "overestimation_weight": float(overestimation_weight),
        "huber_delta": float(huber_delta),
        "label_method": label_method,
        "selection_metric": "holdout_mae",
        "results": rows,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_written = _plot(pdf_path, rows)
    print(f"WROTE {csv_path}")
    print(f"WROTE {json_path}")
    if plot_written:
        print(f"WROTE {pdf_path}")
    else:
        print("SKIPPED PDF: matplotlib is not installed")
    return rows


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--boundary-width", type=float, default=0.03)
    parser.add_argument("--boundary-weight", type=float, default=4.0)
    parser.add_argument("--overestimation-weight", type=float, default=2.0)
    parser.add_argument("--huber-delta", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    compare_losses(**vars(arguments))
