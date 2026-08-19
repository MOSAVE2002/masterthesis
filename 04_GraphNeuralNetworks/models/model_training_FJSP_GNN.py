"""Train one jobspecific on-time probability for every job in each graph."""

from __future__ import annotations

import csv
import gc
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helper.sequence_setup import (
    RELIABILITY_GNN_GRAPH_SCHEMA,
    RELIABILITY_GNN_OUTPUT_HEAD,
    normalize_reliability_graph_config,
    reliability_graph_config_dict,
    reliability_node_feature_names,
)
from helper.surrogate_constraint import configured_constraint_type, target_column


_instances = importlib.import_module("01_generator.instance_generator")
_architectures = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)
MONTE_CARLO_LABEL_METHOD = importlib.import_module(
    "05_Simulation.preempt_resume"
).LABEL_METHOD

SPLIT_DIRECTORIES = _instances.SPLIT_DIRECTORIES
SPLIT_CSV_FILENAMES = _instances.SPLIT_CSV_FILENAMES
load_generated_instance = _instances.load_generated_instance

CONV_LINEAR = _architectures.CONV_LINEAR
CONV_SAGE = _architectures.CONV_SAGE
GRAPH_FIXED = _architectures.GRAPH_FIXED
POOL_ADD = _architectures.POOL_ADD
VALID_LAYER_COUNTS = _architectures.VALID_LAYER_COUNTS
architecture_model_dir = _architectures.architecture_model_dir
architecture_stem = _architectures.architecture_stem
expand_architecture_variants = _architectures.expand_architecture_variants
validate_architecture = _architectures.validate_architecture

CONFIG_PATH = ROOT_DIR / "config.json"
DEFAULT_DATASET_DIR = ROOT_DIR / "02_data" / "gnn_dataset"
MODEL_DIR = ROOT_DIR / "04_GraphNeuralNetworks" / "trained_gnn_models"
CONSTRAINT_TYPE = configured_constraint_type()
TARGET_COLUMN = target_column(CONSTRAINT_TYPE)

LOSS_MSE = "mse"
LOSS_ASYMMETRIC_MSE = "asymmetric_mse"
LOSS_HUBER = "huber"
LOSS_BOUNDARY_WEIGHTED_MSE = "boundary_weighted_mse"
LOSS_BOUNDARY_WEIGHTED_ASYMMETRIC_MSE = (
    "boundary_weighted_asymmetric_mse"
)
SUPPORTED_LOSSES = {
    LOSS_MSE,
    LOSS_ASYMMETRIC_MSE,
    LOSS_HUBER,
    LOSS_BOUNDARY_WEIGHTED_MSE,
    LOSS_BOUNDARY_WEIGHTED_ASYMMETRIC_MSE,
}


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_graphs(csv_path: Path, target_name: str = TARGET_COLUMN):
    """Load node-feature graphs written by the data generator."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    required = {
        "instance_name",
        target_name,
        "gnn_feature_names",
        "gnn_node_features",
        "gnn_active_edges",
        "job_ids",
        "operation_job_indices",
        "job_probability_label_method",
        "job_probability_standard_errors",
        "simulation_replications",
        "simulation_parameters",
        "reliability_graph_parameters",
    }
    graphs, feature_names = [], None
    graph_config = None
    label_method = None
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            row_config_values = json.loads(
                row["reliability_graph_parameters"]
            )
            row_config_values.pop("makespan_weight", None)
            row_config = normalize_reliability_graph_config(row_config_values)
            if graph_config is None:
                graph_config = row_config
            elif graph_config != row_config:
                raise ValueError("Reliability parameters differ inside one CSV.")
            row_label_method = row["job_probability_label_method"]
            if row_label_method != MONTE_CARLO_LABEL_METHOD:
                raise ValueError(
                    "The active GNN pipeline requires Monte-Carlo "
                    f"Preempt-Resume labels, got {row_label_method!r}."
                )
            if int(row["simulation_replications"]) <= 0:
                raise ValueError("Simulation replications must be positive.")
            if label_method is None:
                label_method = row_label_method
            elif label_method != row_label_method:
                raise ValueError("Job probability label methods differ.")

            names = json.loads(row["gnn_feature_names"])
            if names != reliability_node_feature_names():
                raise ValueError(f"Unexpected node feature order: {names}")
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("Node feature order differs between rows.")

            active_edges = json.loads(row["gnn_active_edges"])
            edge_index = torch.tensor(
                [
                    [source for source, _target in active_edges],
                    [target for _source, target in active_edges],
                ],
                dtype=torch.long,
            ).reshape(2, -1)
            x = torch.tensor(
                json.loads(row["gnn_node_features"]), dtype=torch.float32
            )
            graphs.append(Data(
                x=x,
                edge_index=edge_index,
                job_y=torch.tensor(
                    json.loads(row[target_name]),
                    dtype=torch.float32,
                ),
                job_membership=torch.tensor(
                    json.loads(row["operation_job_indices"]),
                    dtype=torch.long,
                ),
                num_jobs_tensor=torch.tensor(
                    [len(json.loads(row["job_ids"]))], dtype=torch.long
                ),
            ))
    if not graphs:
        raise ValueError(f"Dataset contains no graphs: {csv_path}")
    return graphs, feature_names, graph_config, label_method


class NodeGraphSAGEConv(nn.Module):
    """Root transform plus summed predecessor-node messages."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.lin_root = nn.Linear(input_size, output_size)
        self.lin_message = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x, edge_index):
        source, target = edge_index
        predecessor_sum = x.new_zeros(x.shape)
        if source.numel():
            predecessor_sum.index_add_(0, target, x[source])
        return self.lin_root(x) + self.lin_message(predecessor_sum)


class FJSPGraphSAGE(nn.Module):
    """Linear baseline or relational SAGE predictor with one output per job."""

    def __init__(
        self,
        input_size,
        hidden_channels,
        num_graphsage_layers,
        convolution,
        initial_probability=0.95,
    ):
        super().__init__()
        if convolution not in {CONV_LINEAR, CONV_SAGE}:
            raise ValueError("Only linear and sage are supported.")
        if num_graphsage_layers not in VALID_LAYER_COUNTS:
            choices = ", ".join(map(str, sorted(VALID_LAYER_COUNTS)))
            raise ValueError(
                f"The active pipeline supports {choices} layers."
            )
        initial_probability = float(initial_probability)
        if not 0.0 < initial_probability < 1.0:
            raise ValueError(
                "initial_probability must lie strictly between 0 and 1."
            )
        self.convolution = convolution

        def layer(input_channels):
            if convolution == CONV_LINEAR:
                return nn.Linear(input_channels, hidden_channels)
            return NodeGraphSAGEConv(input_channels, hidden_channels)

        self.conv1 = layer(input_size)
        self.conv2 = (
            layer(hidden_channels) if num_graphsage_layers >= 2 else None
        )
        self.conv3 = (
            layer(hidden_channels) if num_graphsage_layers >= 3 else None
        )
        self.out = nn.Linear(hidden_channels, 1)
        self.out_input = nn.Linear(input_size, 1)
        nn.init.constant_(self.out.weight, 0.001)
        nn.init.constant_(self.out.bias, initial_probability)
        nn.init.zeros_(self.out_input.weight)
        nn.init.zeros_(self.out_input.bias)

    def _layer(self, layer, x, edge_index):
        return layer(x) if self.convolution == CONV_LINEAR else (
            layer(x, edge_index)
        )

    def forward(self, data, return_nodes=False):
        x_input = data.x
        x = F.relu(self._layer(self.conv1, x_input, data.edge_index))
        if self.conv2 is not None:
            x = F.relu(self._layer(self.conv2, x, data.edge_index))
        if self.conv3 is not None:
            x = F.relu(self._layer(self.conv3, x, data.edge_index))
        job_counts = data.num_jobs_tensor.view(-1).long()
        job_offsets = torch.cat((
            job_counts.new_zeros(1), job_counts.cumsum(0)[:-1]
        ))
        node_job = data.job_membership.view(-1).long() + job_offsets[data.batch]
        total_jobs = int(job_counts.sum().item())
        pooled = global_add_pool(x, node_job, size=total_jobs)
        pooled_input = global_add_pool(
            x_input, node_job, size=total_jobs
        )
        raw_probability = F.relu(
            self.out(pooled).view(-1)
            + self.out_input(pooled_input).view(-1)
        )
        clipped = raw_probability.clamp(0.0, 1.0)
        probability = (
            raw_probability + (clipped - raw_probability).detach()
            if self.training else clipped
        )
        return (probability, probability) if return_nodes else probability


def _loss(
    graph_prediction,
    _job_prediction,
    batch,
    *,
    loss_name=LOSS_ASYMMETRIC_MSE,
    service_level=0.95,
    boundary_width=0.03,
    boundary_weight=4.0,
    overestimation_weight=2.0,
    huber_delta=0.05,
):
    """Return a job-level probability loss for one mini-batch."""
    if loss_name not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unknown loss {loss_name!r}; expected one of "
            f"{sorted(SUPPORTED_LOSSES)}."
        )
    if float(boundary_width) <= 0.0:
        raise ValueError("boundary_width must be positive.")
    if float(boundary_weight) < 1.0:
        raise ValueError("boundary_weight must be at least one.")
    if float(overestimation_weight) < 1.0:
        raise ValueError("overestimation_weight must be at least one.")
    if float(huber_delta) <= 0.0:
        raise ValueError("huber_delta must be positive.")

    target = batch.job_y.view(-1)
    error = graph_prediction - target
    if loss_name == LOSS_HUBER:
        per_job = F.huber_loss(
            graph_prediction,
            target,
            reduction="none",
            delta=float(huber_delta),
        )
    else:
        per_job = error.square()

    if loss_name in {
        LOSS_ASYMMETRIC_MSE,
        LOSS_BOUNDARY_WEIGHTED_ASYMMETRIC_MSE,
    }:
        per_job = torch.where(
            error > 0.0,
            float(overestimation_weight) * per_job,
            per_job,
        )
    if loss_name in {
        LOSS_BOUNDARY_WEIGHTED_MSE,
        LOSS_BOUNDARY_WEIGHTED_ASYMMETRIC_MSE,
    }:
        boundary = (
            (target - float(service_level)).abs()
            <= float(boundary_width) + 1e-12
        )
        per_job = torch.where(
            boundary,
            float(boundary_weight) * per_job,
            per_job,
        )
    return per_job.mean()


def _train_epoch(model, loader, optimizer, **loss_kwargs):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        optimizer.zero_grad()
        graph_prediction, node_prediction = model(batch, return_nodes=True)
        loss = _loss(
            graph_prediction, node_prediction, batch, **loss_kwargs
        )
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
    return total / len(loader.dataset)


def _evaluate(
    model,
    loader,
    zero_edges=False,
    *,
    service_level=0.95,
    boundary_width=0.03,
    calibration_bins=10,
):
    model.eval()
    job_errors, predictions, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            if zero_edges:
                batch.edge_index = batch.edge_index.new_empty((2, 0))
            job_prediction, _ = model(batch, return_nodes=True)
            target = batch.job_y.view(-1)
            job_errors.append(job_prediction - target)
            predictions.append(job_prediction)
            targets.append(target)
    job_error = torch.cat(job_errors)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    boundary = (
        (target - float(service_level)).abs()
        <= float(boundary_width) + 1e-12
    )
    predicted_feasible = prediction >= float(service_level)
    actual_feasible = target >= float(service_level)
    true_positive = int((predicted_feasible & actual_feasible).sum())
    true_negative = int((~predicted_feasible & ~actual_feasible).sum())
    false_positive = int((predicted_feasible & ~actual_feasible).sum())
    false_negative = int((~predicted_feasible & actual_feasible).sum())

    def divide(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    recall = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)
    ece = 0.0
    for index in range(int(calibration_bins)):
        lower = index / int(calibration_bins)
        upper = (index + 1) / int(calibration_bins)
        mask = (prediction >= lower) & (
            prediction <= upper if index == int(calibration_bins) - 1
            else prediction < upper
        )
        count = int(mask.sum())
        if count:
            ece += count * abs(
                float(prediction[mask].mean()) - float(target[mask].mean())
            )
    ece /= int(target.numel())
    return {
        "mae": float(job_error.abs().mean()),
        "rmse": math.sqrt(float(job_error.square().mean())),
        "soft_brier_score": float(job_error.square().mean()),
        "mean_error_bias": float(job_error.mean()),
        "boundary_labels": int(boundary.sum()),
        "boundary_mae": (
            float(job_error[boundary].abs().mean())
            if bool(boundary.any()) else None
        ),
        "expected_calibration_error": ece,
        "job_overestimation_max": float(F.relu(job_error).max()),
        "overestimation_rate": float((job_error > 0.0).float().mean()),
        "threshold_accuracy": divide(
            true_positive + true_negative, int(target.numel())
        ),
        "threshold_balanced_accuracy": 0.5 * (recall + specificity),
        "threshold_recall": recall,
        "threshold_specificity": specificity,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "unsafe_acceptance_rate": divide(
            false_positive, true_negative + false_positive
        ),
    }


def _cpu_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_from_file(
    csv_path,
    validation_csv_path,
    test_csv_path,
    *,
    seed=42,
    epochs=500,
    hidden_channels=8,
    batch_size=32,
    learning_rate=0.001,
    graph_mode=GRAPH_FIXED,
    convolution=CONV_SAGE,
    aggregation="sum",
    pooling=POOL_ADD,
    num_graphsage_layers=1,
    validation_interval=10,
    early_stopping_patience=100,
    enforce_graph_influence=False,
    expected_service_level=None,
    loss_name=LOSS_ASYMMETRIC_MSE,
    boundary_width=0.03,
    boundary_weight=4.0,
    overestimation_weight=2.0,
    huber_delta=0.05,
    output_stem=None,
    model_dir=MODEL_DIR,
):
    architecture = validate_architecture(
        graph_mode, convolution, aggregation, pooling
    )
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    train_graphs, feature_names, graph_config, label_method = load_graphs(
        Path(csv_path)
    )
    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        training_columns = set(csv.DictReader(file).fieldnames or [])
    data_generation_method = (
        "fix_and_optimize"
        if "optimization_run" in training_columns
        else "random_feasible"
    )
    valid_graphs, valid_names, valid_config, valid_label_method = load_graphs(
        Path(validation_csv_path)
    )
    test_graphs, test_names, test_config, test_label_method = load_graphs(
        Path(test_csv_path)
    )
    if feature_names != valid_names or feature_names != test_names:
        raise ValueError("Feature names differ between data splits.")
    if graph_config != valid_config or graph_config != test_config:
        raise ValueError("Reliability parameters differ between data splits.")
    if (
        expected_service_level is not None
        and not math.isclose(
            graph_config.service_level,
            float(expected_service_level),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "Dataset service_level="
            f"{graph_config.service_level} differs from configured "
            f"service_level={float(expected_service_level)}. Regenerate the "
            "GNN dataset before training."
        )
    if label_method != valid_label_method or label_method != test_label_method:
        raise ValueError("Job probability label methods differ between splits.")

    device = _device()
    train_loader = DataLoader(
        train_graphs, batch_size=min(batch_size, len(train_graphs)), shuffle=True
    )
    valid_loader = DataLoader(
        valid_graphs, batch_size=min(batch_size, len(valid_graphs))
    )
    test_loader = DataLoader(
        test_graphs, batch_size=min(batch_size, len(test_graphs))
    )
    model = FJSPGraphSAGE(
        input_size=len(feature_names),
        hidden_channels=int(hidden_channels),
        num_graphsage_layers=int(num_graphsage_layers),
        convolution=architecture["convolution"],
        initial_probability=graph_config.service_level,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_parameters = {
        "loss_name": loss_name,
        "service_level": graph_config.service_level,
        "boundary_width": float(boundary_width),
        "boundary_weight": float(boundary_weight),
        "overestimation_weight": float(overestimation_weight),
        "huber_delta": float(huber_delta),
    }
    best_state, best_loss, best_epoch = _cpu_state(model), math.inf, 0
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        training_loss = _train_epoch(
            model, train_loader, optimizer, **loss_parameters
        )
        if epoch == 1 or epoch % int(validation_interval) == 0 or epoch == epochs:
            valid_metrics = _evaluate(
                model,
                valid_loader,
                service_level=graph_config.service_level,
                boundary_width=boundary_width,
            )
            selection_loss = valid_metrics["mae"]
            print(
                f"epoch={epoch:04d} train_loss={training_loss:.6g} "
                f"valid_mae={valid_metrics['mae']:.6g}",
                flush=True,
            )
            if selection_loss < best_loss - 1e-10:
                best_state, best_loss, best_epoch = (
                    _cpu_state(model), selection_loss, epoch
                )
            elif epoch - best_epoch >= int(early_stopping_patience):
                print(
                    "EARLY_STOPPING | "
                    f"epoch={epoch} | best_epoch={best_epoch} | "
                    f"patience_epochs={int(early_stopping_patience)} | "
                    f"best_valid_mae={best_loss:.6g} | "
                    f"current_valid_mae={selection_loss:.6g}",
                    flush=True,
                )
                break
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    valid_metrics = _evaluate(
        model,
        valid_loader,
        service_level=graph_config.service_level,
        boundary_width=boundary_width,
    )
    test_metrics = _evaluate(
        model,
        test_loader,
        service_level=graph_config.service_level,
        boundary_width=boundary_width,
    )
    zero_edge_metrics = (
        _evaluate(
            model,
            test_loader,
            zero_edges=True,
            service_level=graph_config.service_level,
            boundary_width=boundary_width,
        )
        if architecture["convolution"] == CONV_SAGE else None
    )
    if (
        enforce_graph_influence
        and zero_edge_metrics is not None
        and test_metrics["mae"] >= zero_edge_metrics["mae"]
    ):
        raise RuntimeError("SAGE did not outperform its zero-edge ablation.")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or architecture_stem(
        **architecture,
        target=TARGET_COLUMN,
        seed=seed,
        layers=num_graphsage_layers,
        hidden_channels=hidden_channels,
    )
    model_path = model_dir / f"{stem}.pt"
    metadata_path = model_dir / f"{stem}_meta.json"
    torch.save(_cpu_state(model), model_path)
    metadata = {
        "target_column": TARGET_COLUMN,
        "constraint_type": CONSTRAINT_TYPE,
        "input_size": len(feature_names),
        "feature_names": feature_names,
        "graph_mode": GRAPH_FIXED,
        "convolution": architecture["convolution"],
        "aggregation": architecture["aggregation"],
        "pooling": architecture["pooling"],
        "num_graphsage_layers": int(num_graphsage_layers),
        "hidden_channels": int(hidden_channels),
        "output_head": RELIABILITY_GNN_OUTPUT_HEAD,
        "job_target": "job_ontime_probability",
        "job_probability_label_method": label_method,
        "graph_schema": RELIABILITY_GNN_GRAPH_SCHEMA,
        "message_passing": "source_node_states_only",
        "include_machine_predecessor_edges": True,
        "include_job_precedence_edges": True,
        "reliability_graph_config": reliability_graph_config_dict(graph_config),
        "seed": int(seed),
        "loss": loss_parameters,
        "epochs_completed": int(epoch),
        "best_epoch": int(best_epoch),
        "training_seconds": elapsed,
        "num_train_graphs": len(train_graphs),
        "num_valid_graphs": len(valid_graphs),
        "num_test_graphs": len(test_graphs),
        "data_generation_method": data_generation_method,
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "zero_edge_metrics": zero_edge_metrics,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"saved model={model_path}", flush=True)
    return model_path, metadata_path


def train_from_config(seed=42, csv_path=None):
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    training = config["training"]
    gnn = training["gnn"]
    optimizer = gnn.get("optimizer", {})
    loss_config = gnn.get("loss", {})
    validation = gnn.get("validation", {})
    dataset = Path(csv_path or training["data_generation"]["output_directory"])
    model_root = Path(gnn["model_directory"])
    if not dataset.is_absolute():
        dataset = ROOT_DIR / dataset
    if not model_root.is_absolute():
        model_root = ROOT_DIR / model_root
    paths = {
        split: dataset / SPLIT_DIRECTORIES[split] / SPLIT_CSV_FILENAMES[split]
        for split in ("train", "valid", "test")
    }
    results = []
    seen = set()
    for raw in gnn["combinations"]:
        for architecture in expand_architecture_variants(raw):
            key = tuple(architecture.values())
            if key in seen:
                continue
            seen.add(key)
            output_dir = architecture_model_dir(
                model_root,
                architecture["graph_mode"],
                architecture["convolution"],
                architecture["aggregation"],
                architecture["pooling"],
                architecture["layers"],
                architecture["hidden_channels"],
            )
            stem = architecture_stem(
                architecture["graph_mode"],
                architecture["convolution"],
                architecture["aggregation"],
                architecture["pooling"],
                TARGET_COLUMN,
                seed,
                architecture["layers"],
                architecture["hidden_channels"],
            )
            try:
                results.append(train_from_file(
                    paths["train"],
                    paths["valid"],
                    paths["test"],
                    seed=seed,
                    epochs=int(optimizer.get("epochs", 500)),
                    hidden_channels=architecture["hidden_channels"],
                    batch_size=int(optimizer.get("batch_size", 32)),
                    learning_rate=float(optimizer.get("learning_rate", 0.001)),
                    graph_mode=architecture["graph_mode"],
                    convolution=architecture["convolution"],
                    aggregation=architecture["aggregation"],
                    pooling=architecture["pooling"],
                    num_graphsage_layers=architecture["layers"],
                    validation_interval=int(
                        validation.get("interval_epochs", 10)
                    ),
                    early_stopping_patience=int(
                        validation.get("patience_epochs", 100)
                    ),
                    enforce_graph_influence=bool(
                        validation.get("enforce_graph_influence", False)
                    ),
                    expected_service_level=config["constraint"]["weibull"][
                        "reliability_graph"
                    ]["service_level"],
                    loss_name=loss_config.get(
                        "name", LOSS_ASYMMETRIC_MSE
                    ),
                    boundary_width=float(loss_config.get(
                        "boundary_width",
                        config["training"]["data_generation"]["fixed_y"][
                            "service_boundary_width"
                        ],
                    )),
                    boundary_weight=float(
                        loss_config.get("boundary_weight", 4.0)
                    ),
                    overestimation_weight=float(
                        loss_config.get("overestimation_weight", 2.0)
                    ),
                    huber_delta=float(
                        loss_config.get("huber_delta", 0.05)
                    ),
                    output_stem=stem,
                    model_dir=output_dir,
                ))
            finally:
                _release_memory()
    return results


if __name__ == "__main__":
    train_from_config(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 42)
