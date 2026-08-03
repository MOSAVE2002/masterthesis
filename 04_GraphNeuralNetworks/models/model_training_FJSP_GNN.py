import csv
import gc
import importlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from functools import lru_cache
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

from helper.surrogate_constraint import (
    configured_constraint_type,
    target_column,
)
from helper.sequence_setup import (
    RELIABILITY_EDGE_FEATURE_NAMES,
    RELIABILITY_GNN_GRAPH_SCHEMA,
    RELIABILITY_GNN_GRAPH_SCHEMA_JOB_ONLY,
    RELIABILITY_GNN_GRAPH_SCHEMA_WITH_JOB_EDGES,
    RELIABILITY_GNN_OUTPUT_HEAD,
    normalize_reliability_graph_config,
    reliability_graph_config_dict,
    reliability_edge_feature_names,
    reliability_node_feature_names,
)
_gnn_architecture = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)

# load constanv Values from gnn_architecture
# irgendwie dumm, anders machen
CONV_GCN = _gnn_architecture.CONV_GCN
CONV_GINE = _gnn_architecture.CONV_GINE
CONV_LINEAR = _gnn_architecture.CONV_LINEAR
CONV_MPNN = _gnn_architecture.CONV_MPNN
CONV_SAGE = _gnn_architecture.CONV_SAGE
MESSAGE_PASSING_CONVOLUTIONS = (
    _gnn_architecture.MESSAGE_PASSING_CONVOLUTIONS
)
POOL_ADD = _gnn_architecture.POOL_ADD
architecture_model_dir = _gnn_architecture.architecture_model_dir
architecture_stem = _gnn_architecture.architecture_stem
expand_architecture_variants = (
    _gnn_architecture.expand_architecture_variants
)
validate_architecture = _gnn_architecture.validate_architecture

_instance_generator = importlib.import_module("01_generator.instance_generator")
SPLIT_DIRECTORIES = _instance_generator.SPLIT_DIRECTORIES
SPLIT_CSV_FILENAMES = _instance_generator.SPLIT_CSV_FILENAMES
FIXED_SOLUTION_CSV_FILENAMES = (
    _instance_generator.FIXED_SOLUTION_CSV_FILENAMES
)
load_generated_instance = _instance_generator.load_generated_instance
_training_data_generator = importlib.import_module(
    "04_GraphNeuralNetworks.models.generate_weibull_training_data"
)
fixed_machine_multiedges = _training_data_generator.fixed_machine_multiedges

# Paths
DEFAULT_DATASET_DIR = ROOT_DIR / "02_data" / "gnn_dataset"
MODEL_DIR = ROOT_DIR / "04_GraphNeuralNetworks" / "trained_gnn_models"
CONFIG_PATH = ROOT_DIR / "config.json"

# Choose Constraint
CONSTRAINT_TYPE = configured_constraint_type()
TARGET_COLUMN = target_column(CONSTRAINT_TYPE)
UNDERESTIMATION_LOSS_FACTOR = 2.0

# Chose GraphMode
GRAPH_MODE_FIXED_CANDIDATE = "fixed_candidate"
VALID_GRAPH_MODES = {GRAPH_MODE_FIXED_CANDIDATE}

def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _synchronize_device(device: torch.device) -> None:
    """Wait until asynchronous accelerator work has completed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _release_training_memory() -> None:
    """Release CPU objects and accelerator caches between model variants."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()


@lru_cache(maxsize=1)
def _training_hardware() -> dict:
    """Return reproducibility-relevant hardware data without unique IDs."""
    hardware = {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "pytorch_version": torch.__version__,
    }
    if platform.system() != "Darwin":
        return hardware

    try:
        completed = subprocess.run(
            [
                "system_profiler",
                "SPHardwareDataType",
                "SPDisplaysDataType",
                "-json",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        profiler = json.loads(completed.stdout)
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        return hardware

    overview = (profiler.get("SPHardwareDataType") or [{}])[0]
    display = (profiler.get("SPDisplaysDataType") or [{}])[0]
    hardware.update(
        {
            "computer_model": overview.get("machine_name"),
            "model_identifier": overview.get("machine_model"),
            "chip": overview.get("chip_type"),
            "unified_memory": overview.get("physical_memory"),
            "gpu_model": display.get("sppci_model") or display.get("_name"),
            "gpu_cores": display.get("sppci_cores"),
        }
    )
    return {
        key: value for key, value in hardware.items() if value not in (None, "")
    }


def _training_time_summary(
    *,
    epoch_seconds: list[float],
    total_seconds: float,
    best_epoch: int,
    time_to_best_epoch_seconds: float | None,
    device: torch.device,
) -> dict:
    """Build wall-clock and occupied-accelerator time metrics."""
    completed_epochs = len(epoch_seconds)
    uses_gpu = device.type in {"cuda", "mps"}
    accelerator_count = 1 if uses_gpu else 0
    gpu_hours = (
        total_seconds * accelerator_count / 3600.0
        if uses_gpu else None
    )
    return {
        "scope": (
            "training_loop_including_periodic_training_and_validation_"
            "evaluation"
        ),
        "completed_epochs": completed_epochs,
        "wall_clock_seconds": total_seconds,
        "wall_clock_hours": total_seconds / 3600.0,
        "mean_seconds_per_epoch": (
            sum(epoch_seconds) / completed_epochs
            if completed_epochs else None
        ),
        "min_seconds_per_epoch": (
            min(epoch_seconds) if completed_epochs else None
        ),
        "max_seconds_per_epoch": (
            max(epoch_seconds) if completed_epochs else None
        ),
        "epoch_seconds": epoch_seconds,
        "best_epoch": best_epoch,
        "time_to_best_epoch_seconds": time_to_best_epoch_seconds,
        "accelerator_type": device.type if uses_gpu else None,
        "accelerator_count": accelerator_count,
        # One integrated Apple GPU counts as one occupied GPU. Its GPU cores
        # are hardware details and are deliberately not used as a multiplier.
        "gpu_hours": gpu_hours,
        "gpu_days": gpu_hours / 24.0 if gpu_hours is not None else None,
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _load_reliability_graphs(
    csv_path: Path,
    graph_mode: str | None,
    target_name: str,
    include_job_precedence_edges: bool = False,
    include_machine_predecessor_edges: bool = True,
):
    if not include_machine_predecessor_edges and not include_job_precedence_edges:
        raise ValueError("At least one predecessor edge type must be enabled.")
    if graph_mode not in (None, "", GRAPH_MODE_FIXED_CANDIDATE):
        raise ValueError(
            "Reliability-graph training requires "
            "graph_mode='fixed_candidate'."
        )
    if not csv_path.exists():
        raise FileNotFoundError(f"Trainings-CSV nicht gefunden: {csv_path}")

    required = {
        "instance_name",
        "total_failure_delay",
        "gnn_feature_names",
        "gnn_edge_feature_names",
        "gnn_node_features",
        "gnn_active_edge_indices",
        "gnn_active_edge_features",
        "operation_failure_delays",
        "operation_failure_probabilities",
        "operation_repair_durations",
        "reliability_graph_parameters",
    }
    graphs = []
    feature_names = None
    graph_config = None
    fixed_edges_by_instance = {}
    job_edges_by_instance = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} fehlt das relationale Setup-Schema: "
                f"{sorted(missing)}. Trainingsdaten neu erzeugen."
            )
        for row in reader:
            row_config = normalize_reliability_graph_config(
                json.loads(row["reliability_graph_parameters"])
            )
            if graph_config is None:
                graph_config = row_config
            elif graph_config != row_config:
                raise ValueError(
                    "reliability_graph_parameters must be identical in one CSV."
                )
            current_names = json.loads(row["gnn_feature_names"])
            expected_names = reliability_node_feature_names(row_config)
            if current_names != expected_names:
                raise ValueError(
                    f"Expected node features {expected_names}, got "
                    f"{current_names}."
                )
            edge_names = json.loads(row["gnn_edge_feature_names"])
            if edge_names != RELIABILITY_EDGE_FEATURE_NAMES:
                raise ValueError(
                    "Relational edge feature order does not match: "
                    f"{edge_names}."
                )
            if feature_names is None:
                feature_names = current_names
            elif feature_names != current_names:
                raise ValueError(
                    "gnn_feature_names ist nicht in allen Zeilen gleich."
                )

            x = torch.tensor(
                json.loads(row["gnn_node_features"]),
                dtype=torch.float32,
            )
            instance_name = row["instance_name"]
            if instance_name not in fixed_edges_by_instance:
                instance = load_generated_instance(instance_name)
                operations = list(instance.real_operations)
                fixed_edges_by_instance[instance_name] = (
                    fixed_machine_multiedges(instance, operations)
                )
                operation_to_index = {
                    operation: index
                    for index, operation in enumerate(operations)
                }
                job_edges_by_instance[instance_name] = [
                    (
                        operation_to_index[predecessor],
                        operation_to_index[operation],
                    )
                    for operation in operations
                    for predecessor in instance.predecessors.get(operation, [])
                    if predecessor in operation_to_index
                ]
            fixed_edges = fixed_edges_by_instance[instance_name]
            active_indices = torch.tensor(
                json.loads(row["gnn_active_edge_indices"]),
                dtype=torch.long,
            ).view(-1)
            edge_attr = torch.tensor(
                json.loads(row["gnn_active_edge_features"]),
                dtype=torch.float32,
            ).reshape(-1, len(RELIABILITY_EDGE_FEATURE_NAMES))
            if len(active_indices) != edge_attr.size(0):
                raise ValueError(
                    "Active edge indices and edge features must have "
                    "the same length."
                )
            if len(active_indices) and (
                int(active_indices.min()) < 0
                or int(active_indices.max()) >= len(fixed_edges)
            ):
                raise ValueError(
                    "gnn_active_edge_indices contains an invalid direct "
                    "predecessor supergraph index."
                )
            machine_edges = [
                fixed_edges[index] for index in active_indices.tolist()
            ]
            selected_edges = list(machine_edges) if include_machine_predecessor_edges else []
            if not include_machine_predecessor_edges:
                edge_attr = torch.empty((0, 1), dtype=torch.float32)
            if include_job_precedence_edges:
                if include_machine_predecessor_edges:
                    edge_attr = torch.cat(
                        (
                            edge_attr,
                            torch.zeros((edge_attr.size(0), 1), dtype=torch.float32),
                        ),
                        dim=1,
                    )
                job_edges = job_edges_by_instance[instance_name]
                selected_edges.extend(
                    (source, target, "job")
                    for source, target in job_edges
                )
                if job_edges:
                    edge_attr = torch.cat(
                        (
                            edge_attr,
                            torch.tensor(
                                [
                                    ([0.0, 0.0, 1.0] if include_machine_predecessor_edges else [1.0])
                                    for _edge in job_edges
                                ],
                                dtype=torch.float32,
                            ),
                        ),
                        dim=0,
                    )
            edge_index = torch.tensor(
                [
                    [source for source, _target, _machine in selected_edges],
                    [target for _source, target, _machine in selected_edges],
                ],
                dtype=torch.long,
            ).reshape(2, -1)
            incoming_counts = torch.zeros(x.size(0), dtype=torch.long)
            if edge_index.numel():
                incoming_counts.index_add_(
                    0,
                    edge_index[1],
                    torch.ones(edge_index.size(1), dtype=torch.long),
                )
            maximum_incoming = int(include_machine_predecessor_edges) + int(
                include_job_precedence_edges
            )
            if torch.any(incoming_counts > maximum_incoming):
                raise ValueError(
                    "A reliability graph has more incoming predecessor "
                    f"edges than expected ({maximum_incoming})."
                )

            repair_duration = torch.tensor(
                json.loads(row["operation_repair_durations"]),
                dtype=torch.float32,
            ).view(-1)
            node_probability_y = torch.tensor(
                json.loads(row["operation_failure_probabilities"]),
                dtype=torch.float32,
            ).view(-1)
            node_delay_y = torch.tensor(
                json.loads(row["operation_failure_delays"]),
                dtype=torch.float32,
            ).view(-1)
            if not (
                x.size(0)
                == len(repair_duration)
                == len(node_probability_y)
                == len(node_delay_y)
            ):
                raise ValueError(
                    "Node features, probabilities, delays and repair "
                    "durations must contain the same number of operations."
                )
            if torch.any(
                (node_probability_y < -1e-6)
                | (node_probability_y > 1.0 + 1e-6)
            ):
                raise ValueError(
                    "Failure-probability targets must lie in [0, 1]."
                )
            target_value = float(row[target_name])
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([target_value], dtype=torch.float32),
                node_y=node_probability_y.clamp(0.0, 1.0),
                node_delay_y=node_delay_y,
                repair_duration=repair_duration,
            )
            graphs.append(graph)
    if not graphs:
        raise ValueError(f"{csv_path} enthaelt keine Graphen.")
    return graphs, feature_names


def load_graphs(
    csv_path: Path,
    graph_mode: str | None = None,
    target_name: str = TARGET_COLUMN,
    include_job_precedence_edges: bool = False,
    include_machine_predecessor_edges: bool = True,
):
    """Laedt FJSP-Graphen aus einer CSV."""
    return _load_reliability_graphs(
        csv_path,
        graph_mode,
        target_name,
        include_job_precedence_edges=include_job_precedence_edges,
        include_machine_predecessor_edges=include_machine_predecessor_edges,
    )

class ConfigurableGCNConv(nn.Module):
    """Edge-conditioned GCN transform for direct predecessor messages."""

    def __init__(
        self,
        input_size,
        output_size,
        edge_size=2,
        aggregation="sum",
    ):
        super().__init__()
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.lin_edge = nn.Linear(edge_size, output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.aggregation = aggregation

    def forward(self, x, edge_index, edge_attr):
        source, target = edge_index
        transformed = self.lin(x)
        aggregated = transformed.new_zeros(
            (x.size(0), transformed.size(1))
        )
        if source.numel():
            messages = transformed[source] + self.lin_edge(edge_attr)
            aggregated.index_add_(0, target, messages)
            if self.aggregation == "mean":
                degree = transformed.new_zeros(x.size(0))
                degree.index_add_(
                    0, target, transformed.new_ones(len(target))
                )
                aggregated = (
                    aggregated
                    / degree.clamp(min=1.0).unsqueeze(-1)
                )
        return transformed + aggregated + self.bias


class EdgeGraphSAGEConv(nn.Module):
    """Root transform plus edge-conditioned predecessor messages."""

    def __init__(
        self,
        input_size,
        output_size,
        edge_size=2,
        aggregation="sum",
    ):
        super().__init__()
        self.lin_root = nn.Linear(input_size, output_size)
        self.lin_message = nn.Linear(
            input_size, output_size, bias=False
        )
        self.lin_edge = nn.Linear(edge_size, output_size, bias=False)
        self.aggregation = aggregation

    def forward(self, x, edge_index, edge_attr):
        source, target = edge_index
        aggregated = x.new_zeros(x.shape)
        if source.numel():
            aggregated.index_add_(0, target, x[source])
            if self.aggregation == "mean":
                degree = x.new_zeros(x.size(0))
                degree.index_add_(0, target, x.new_ones(len(target)))
                aggregated = (
                    aggregated
                    / degree.clamp(min=1.0).unsqueeze(-1)
                )
        edge_aggregated = x.new_zeros(
            (x.size(0), edge_attr.size(1))
        )
        if source.numel():
            edge_aggregated.index_add_(0, target, edge_attr)
            if self.aggregation == "mean":
                degree = x.new_zeros(x.size(0))
                degree.index_add_(0, target, x.new_ones(len(target)))
                edge_aggregated = (
                    edge_aggregated
                    / degree.clamp(min=1.0).unsqueeze(-1)
                )
        return (
            self.lin_root(x)
            + self.lin_message(aggregated)
            + self.lin_edge(edge_aggregated)
        )


class EdgeMPNNConv(nn.Module):
    """ReLU edge messages followed by an affine root update."""

    def __init__(self, input_size, output_size, edge_size=2):
        super().__init__()
        self.lin_root = nn.Linear(input_size, output_size)
        self.lin_message = nn.Linear(input_size, output_size, bias=False)
        self.lin_edge = nn.Linear(edge_size, output_size, bias=False)

    def forward(self, x, edge_index, edge_attr):
        source, target = edge_index
        aggregated = x.new_zeros((x.size(0), self.lin_root.out_features))
        if source.numel():
            messages = F.relu(
                self.lin_message(x[source]) + self.lin_edge(edge_attr)
            )
            aggregated.index_add_(0, target, messages)
        return self.lin_root(x) + aggregated


class DirectPredecessorGINEConv(nn.Module):
    """GINE update specialized to at most one direct predecessor."""

    def __init__(self, input_size, output_size, edge_size=2):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.lin_edge = nn.Linear(edge_size, input_size, bias=False)
        self.lin1 = nn.Linear(input_size, output_size)
        self.lin2 = nn.Linear(output_size, output_size)

    def forward(self, x, edge_index, edge_attr):
        source, target = edge_index
        aggregated = x.new_zeros(x.shape)
        if source.numel():
            messages = F.relu(x[source] + self.lin_edge(edge_attr))
            aggregated.index_add_(0, target, messages)
        combined = (1.0 + self.eps) * x + aggregated
        return self.lin2(F.relu(self.lin1(combined)))


class FJSPGraphSAGE(nn.Module):
    """Configurable reliability GNN kept under the legacy class name."""

    def __init__(
        self,
        input_size: int,
        hidden_channels: int = 16,
        aggregation: str = "sum",
        output_mode: str = "node_sum",
        num_graphsage_layers: int = 2,
        convolution: str = "sage",
        pooling: str | None = None,
        edge_feature_size: int = 2,
    ):
        super().__init__()
        self.convolution = convolution
        self.base_convolution = convolution
        self.pooling = pooling or POOL_ADD
        if self.pooling != POOL_ADD or output_mode != "node_sum":
            raise ValueError(
                "Weibull failure probability requires node_sum/global_add output."
            )

        self.edge_feature_size = int(edge_feature_size)
        # Edge features enter messages only.  Consequently the linear
        # architecture is a genuine per-node NN baseline.
        self.local_input_size = input_size

        def make_layer(input_channels):
            if self.base_convolution == CONV_LINEAR:
                return nn.Linear(input_channels, hidden_channels)
            if self.base_convolution == CONV_GCN:
                layer = ConfigurableGCNConv(
                    input_channels,
                    hidden_channels,
                    edge_size=self.edge_feature_size,
                    aggregation=aggregation,
                )
                if aggregation == "sum":
                    with torch.no_grad():
                        layer.lin.weight.mul_(0.05)
                return layer
            if self.base_convolution == CONV_SAGE:
                layer = EdgeGraphSAGEConv(
                    input_channels,
                    hidden_channels,
                    edge_size=self.edge_feature_size,
                    aggregation=aggregation,
                )
                if aggregation == "sum":
                    with torch.no_grad():
                        layer.lin_message.weight.mul_(0.05)
                return layer
            if self.base_convolution == CONV_MPNN:
                return EdgeMPNNConv(
                    input_channels,
                    hidden_channels,
                    edge_size=self.edge_feature_size,
                )
            if self.base_convolution == CONV_GINE:
                return DirectPredecessorGINEConv(
                    input_channels,
                    hidden_channels,
                    edge_size=self.edge_feature_size,
                )
            raise ValueError(f"Unknown convolution: {convolution}")

        self.conv1 = make_layer(self.local_input_size)
        if int(num_graphsage_layers) >= 2:
            self.conv2 = make_layer(hidden_channels)
        else:
            self.conv2 = None
        if int(num_graphsage_layers) == 3:
            self.conv3 = make_layer(hidden_channels)
        else:
            self.conv3 = None
        if int(num_graphsage_layers) not in {1, 2, 3}:
            raise ValueError("num_graphsage_layers must be 1, 2 or 3.")
        self.out = nn.Linear(hidden_channels, 1)
        self.out_input = nn.Linear(self.local_input_size, 1)
        self.output_mode = "node_sum"
        # Both output biases start at zero. A small positive output weight
        # keeps the following ReLU trainable; zero weights and zero biases
        # together would produce a dead output head with zero gradients.
        nn.init.constant_(self.out.weight, 0.01)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.out_input.weight)
        nn.init.zeros_(self.out_input.bias)

    def forward(self, data_or_x, edge_index=None, batch=None, return_nodes=False):
        if edge_index is None:
            data = data_or_x
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            repair_duration = getattr(data, "repair_duration", None)
            edge_attr = getattr(data, "edge_attr", None)
        else:
            x = data_or_x
            repair_duration = None
            edge_attr = None

        if (
            self.base_convolution in MESSAGE_PASSING_CONVOLUTIONS
            and edge_attr is None
        ):
            raise ValueError(
                "Message-passing reliability models require edge_attr."
            )
        effective_input_x = x
        x = effective_input_x
        x = F.relu(
            self.conv1(x) if self.base_convolution == CONV_LINEAR
            else self.conv1(x, edge_index, edge_attr)
        )
        if self.conv2 is not None:
            x = F.relu(
                self.conv2(x)
                if self.base_convolution == CONV_LINEAR
                else self.conv2(x, edge_index, edge_attr)
            )
        if self.conv3 is not None:
            x = F.relu(
                self.conv3(x)
                if self.base_convolution == CONV_LINEAR
                else self.conv3(x, edge_index, edge_attr)
            )
        node_probability_unclipped = F.relu(
            self.out(x).view(-1)
            + self.out_input(effective_input_x).view(-1)
        )
        node_probability_clipped = node_probability_unclipped.clamp(
            min=0.0, max=1.0
        )
        if self.training:
            # Forward pass is exactly clipped to [0, 1], while the backward
            # pass follows the unclipped value so saturated outputs can recover.
            node_probability = (
                node_probability_unclipped
                + (
                    node_probability_clipped
                    - node_probability_unclipped
                ).detach()
            )
        else:
            node_probability = node_probability_clipped
        if repair_duration is None:
            raise ValueError(
                "repair_duration is required to convert predicted failure "
                "probabilities into expected delays."
            )
        node_delay = node_probability * repair_duration.view(-1)
        graph_prediction = global_add_pool(
            node_delay.unsqueeze(-1), batch
        ).view(-1)
        if return_nodes:
            return graph_prediction, node_probability
        return graph_prediction


def _asymmetric_mse(error):
    squared_error = error.square()
    return torch.where(
        error < 0.0,
        UNDERESTIMATION_LOSS_FACTOR * squared_error,
        squared_error,
    ).mean()


def _training_loss(graph_prediction, node_prediction, batch):
    node_error = node_prediction - batch.node_y.view(-1)
    if not hasattr(batch, "repair_duration"):
        raise ValueError(
            "repair_duration is required for graph-loss normalization."
        )
    graph_repair_duration = global_add_pool(
        batch.repair_duration.view(-1, 1),
        batch.batch,
    ).view(-1)
    normalized_graph_error = (
        graph_prediction - batch.y.view(-1)
    ) / graph_repair_duration.clamp_min(1e-12)
    return (
        _asymmetric_mse(node_error)
        + _asymmetric_mse(normalized_graph_error)
    )


def _project_monotone_nonnegative(model):
    """Project affine layers onto a solver-friendly monotone network.

    All reliability node and edge features are nonnegative.  Nonnegative
    affine weights and biases therefore give every hidden pre-activation a
    certified nonnegative lower bound.  The embedded ReLUs become linear and
    require no phase binaries.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.clamp_(min=0.0)
                if module.bias is not None:
                    module.bias.clamp_(min=0.0)


def _prune_small_parameters(model, threshold):
    threshold = float(threshold)
    total = 0
    already_zero = 0
    newly_zeroed = 0
    with torch.no_grad():
        for parameter in model.parameters():
            total += parameter.numel()
            zero_before = parameter == 0.0
            already_zero += int(zero_before.sum().item())
            if threshold > 0.0:
                prune = parameter.abs() < threshold
                newly_zeroed += int((prune & ~zero_before).sum().item())
                parameter[prune] = 0.0
    return {
        "threshold": threshold,
        "parameters": total,
        "already_zero": already_zero,
        "newly_zeroed": newly_zeroed,
        "zero_after": already_zero + newly_zeroed,
    }


def train_epoch(
    model,
    loader,
    optimizer,
    solver_aware_monotonic=False,
    l1_regularization=0.0,
):
    model.train()
    accumulated_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        optimizer.zero_grad()
        graph_prediction, node_prediction = model(batch, return_nodes=True)
        if node_prediction is None or not hasattr(batch, "node_y"):
            raise ValueError(
                "Operation-level targets are required for GNN training."
            )
        loss = _training_loss(
            graph_prediction,
            node_prediction,
            batch,
        )
        if l1_regularization > 0.0:
            weight_penalties = [
                parameter.abs().mean()
                for name, parameter in model.named_parameters()
                if "weight" in name
            ]
            loss = loss + float(l1_regularization) * torch.stack(
                weight_penalties
            ).sum()
        loss.backward()
        optimizer.step()
        if solver_aware_monotonic:
            _project_monotone_nonnegative(model)

        batch_graphs = int(batch.num_graphs)
        accumulated_loss += float(loss.item()) * batch_graphs
        total_graphs += batch_graphs

    return accumulated_loss / max(total_graphs, 1)


def evaluate(model, loader, zero_edge_contributions=False):
    model.eval()
    predictions = []
    targets = []
    node_predictions = []
    node_targets = []
    node_delay_predictions = []
    node_delay_targets = []
    per_node_underestimation = []
    graph_repair_durations = []
    metrics_by_num_nodes = {}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            if (
                zero_edge_contributions
                and hasattr(batch, "edge_attr")
            ):
                batch.edge_attr = torch.empty(
                    (0, batch.edge_attr.size(1)),
                    dtype=batch.edge_attr.dtype,
                    device=batch.edge_attr.device,
                )
                batch.edge_index = torch.empty(
                    (2, 0),
                    dtype=batch.edge_index.dtype,
                    device=batch.edge_index.device,
                )
            prediction, node_prediction = model(batch, return_nodes=True)
            predictions.append(prediction)
            targets.append(batch.y.view(-1))
            counts = (batch.ptr[1:] - batch.ptr[:-1]).to(prediction.dtype)
            batch_error = prediction - batch.y.view(-1)
            per_node_underestimation.append(
                F.relu(batch.y.view(-1) - prediction) / counts
            )
            if hasattr(batch, "repair_duration"):
                graph_repair_durations.append(
                    global_add_pool(
                        batch.repair_duration.view(-1, 1),
                        batch.batch,
                    ).view(-1)
                )
            if node_prediction is not None and hasattr(batch, "node_y"):
                node_predictions.append(node_prediction)
                node_targets.append(batch.node_y.view(-1))
                if (
                    hasattr(batch, "repair_duration")
                    and hasattr(batch, "node_delay_y")
                ):
                    node_delay_predictions.append(
                        node_prediction * batch.repair_duration.view(-1)
                    )
                    node_delay_targets.append(
                        batch.node_delay_y.view(-1)
                    )
            for graph_index, count in enumerate(counts):
                num_nodes = int(count.item())
                size_metrics = metrics_by_num_nodes.setdefault(
                    num_nodes,
                    {
                        "num_graphs": 0,
                        "total_delay_absolute_error": 0.0,
                        "total_delay_squared_error": 0.0,
                        "total_delay_error": 0.0,
                        "total_delay_absolute_error_per_operation": 0.0,
                        "underestimated_graphs": 0,
                        "node_probability_absolute_error": 0.0,
                        "node_probability_error": 0.0,
                        "node_delay_absolute_error": 0.0,
                        "node_observations": 0,
                        "machine_age_over_eta_sum": 0.0,
                        "machine_age_over_eta_max": -math.inf,
                    },
                )
                graph_error = float(batch_error[graph_index].item())
                size_metrics["num_graphs"] += 1
                size_metrics["total_delay_absolute_error"] += abs(graph_error)
                size_metrics["total_delay_squared_error"] += graph_error**2
                size_metrics["total_delay_error"] += graph_error
                size_metrics[
                    "total_delay_absolute_error_per_operation"
                ] += abs(graph_error) / num_nodes
                size_metrics["underestimated_graphs"] += int(graph_error < 0.0)
                if node_prediction is not None and hasattr(batch, "node_y"):
                    node_mask = batch.batch == graph_index
                    graph_machine_age = batch.x[node_mask, 1]
                    size_metrics["machine_age_over_eta_sum"] += float(
                        graph_machine_age.sum().item()
                    )
                    size_metrics["machine_age_over_eta_max"] = max(
                        size_metrics["machine_age_over_eta_max"],
                        float(graph_machine_age.max().item()),
                    )
                    graph_node_error = (
                        node_prediction[node_mask] - batch.node_y.view(-1)[node_mask]
                    )
                    size_metrics["node_probability_absolute_error"] += float(
                        graph_node_error.abs().sum().item()
                    )
                    size_metrics["node_probability_error"] += float(
                        graph_node_error.sum().item()
                    )
                    size_metrics["node_observations"] += num_nodes
                    if (
                        hasattr(batch, "repair_duration")
                        and hasattr(batch, "node_delay_y")
                    ):
                        graph_node_delay_error = (
                            node_prediction[node_mask]
                            * batch.repair_duration.view(-1)[node_mask]
                            - batch.node_delay_y.view(-1)[node_mask]
                        )
                        size_metrics["node_delay_absolute_error"] += float(
                            graph_node_delay_error.abs().sum().item()
                        )

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    error = prediction - target
    mae = float(error.abs().mean())
    rmse = math.sqrt(float((error**2).mean()))
    under = torch.cat(per_node_underestimation)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "underestimation_rate": float((error < 0).float().mean()),
        "underestimation_per_node_q95": float(torch.quantile(under, 0.95)),
    }
    if node_predictions:
        node_error = torch.cat(node_predictions) - torch.cat(node_targets)
        metrics["node_mae"] = float(node_error.abs().mean())
        metrics["node_rmse"] = math.sqrt(float((node_error**2).mean()))
        metrics["node_probability_mae"] = metrics["node_mae"]
        metrics["node_probability_rmse"] = metrics["node_rmse"]
        if graph_repair_durations:
            normalized_graph_error = error / torch.cat(
                graph_repair_durations
            ).clamp_min(1e-12)
            metrics["asymmetric_node_loss"] = float(
                _asymmetric_mse(node_error)
            )
            metrics["asymmetric_graph_loss"] = float(
                _asymmetric_mse(normalized_graph_error)
            )
            metrics["selection_loss"] = (
                metrics["asymmetric_node_loss"]
                + metrics["asymmetric_graph_loss"]
            )
    if node_delay_predictions:
        node_delay_error = (
            torch.cat(node_delay_predictions)
            - torch.cat(node_delay_targets)
        )
        metrics["node_delay_mae"] = float(
            node_delay_error.abs().mean()
        )
        metrics["node_delay_rmse"] = math.sqrt(
            float((node_delay_error**2).mean())
        )
    metrics["by_num_nodes"] = {}
    for num_nodes, accumulated in sorted(metrics_by_num_nodes.items()):
        num_graphs = accumulated["num_graphs"]
        node_observations = accumulated["node_observations"]
        metrics["by_num_nodes"][str(num_nodes)] = {
            "num_graphs": num_graphs,
            "total_delay_mae": (
                accumulated["total_delay_absolute_error"] / num_graphs
            ),
            "total_delay_rmse": math.sqrt(
                accumulated["total_delay_squared_error"] / num_graphs
            ),
            "total_delay_bias": accumulated["total_delay_error"] / num_graphs,
            "total_delay_mae_per_operation": (
                accumulated["total_delay_absolute_error_per_operation"]
                / num_graphs
            ),
            "underestimation_rate": (
                accumulated["underestimated_graphs"] / num_graphs
            ),
            "node_probability_mae": (
                accumulated["node_probability_absolute_error"]
                / node_observations
                if node_observations
                else None
            ),
            "node_probability_bias": (
                accumulated["node_probability_error"] / node_observations
                if node_observations
                else None
            ),
            "node_delay_mae": (
                accumulated["node_delay_absolute_error"] / node_observations
                if node_observations
                else None
            ),
            "machine_age_over_eta_mean": (
                accumulated["machine_age_over_eta_sum"] / node_observations
                if node_observations
                else None
            ),
            "machine_age_over_eta_max": (
                accumulated["machine_age_over_eta_max"]
                if node_observations
                else None
            ),
        }
    return metrics


def train_from_file(
    csv_path: Path = (
        DEFAULT_DATASET_DIR
        / SPLIT_DIRECTORIES["train"]
        / SPLIT_CSV_FILENAMES["train"]
    ),
    validation_csv_path: Path | None = None,
    test_csv_path: Path | None = None,
    seed: int = 42,
    epochs: int = 100,
    hidden_channels: int = 16,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    aggregation: str = "sum",
    graph_mode: str = GRAPH_MODE_FIXED_CANDIDATE,
    convolution: str = CONV_SAGE,
    pooling: str = "global_add",
    constraint_type: str = CONSTRAINT_TYPE,
    split_strategy: str = "random",
    num_graphsage_layers: int = 2,
    output_stem: str | None = None,
    model_dir: Path = MODEL_DIR,
    enforce_graph_influence: bool = True,
    validation_interval: int = 10,
    early_stopping_patience: int = 10,
    solver_aware_monotonic: bool = False,
    l1_regularization: float = 0.0,
    pruning_threshold: float = 0.0,
    include_job_precedence_edges: bool = False,
    include_machine_predecessor_edges: bool = True,
    max_training_graphs: int | None = None,
):
    validation_interval = int(validation_interval)
    if validation_interval <= 0:
        raise ValueError("validation_interval must be greater than zero.")
    early_stopping_patience = int(early_stopping_patience)
    if early_stopping_patience <= 0:
        raise ValueError(
            "early_stopping_patience must be greater than zero."
        )
    selected_target = target_column(constraint_type)
    architecture = validate_architecture(
        graph_mode, convolution, aggregation, pooling
    )
    graph_mode = architecture["graph_mode"]
    convolution = architecture["convolution"]
    aggregation = architecture["aggregation"]
    pooling = architecture["pooling"]
    l1_regularization = float(l1_regularization)
    pruning_threshold = float(pruning_threshold)
    if l1_regularization < 0.0:
        raise ValueError("l1_regularization must be nonnegative.")
    if pruning_threshold < 0.0:
        raise ValueError("pruning_threshold must be nonnegative.")
    if solver_aware_monotonic and convolution != CONV_SAGE:
        raise ValueError(
            "solver_aware_monotonic is currently validated only for SAGE."
        )
    torch.manual_seed(seed)
    device = _default_device()
    training_hardware = _training_hardware()
    if graph_mode != GRAPH_MODE_FIXED_CANDIDATE:
        raise ValueError(
            "Expected-delay training currently supports only "
            "graph_mode='fixed_candidate'."
        )
    if graph_mode not in VALID_GRAPH_MODES:
        raise ValueError(
            "gnn_params.graph_mode muss 'fixed_candidate' sein."
        )
    train_graphs, feature_names = load_graphs(
        csv_path,
        graph_mode=graph_mode,
        target_name=selected_target,
        include_job_precedence_edges=include_job_precedence_edges,
        include_machine_predecessor_edges=include_machine_predecessor_edges,
    )
    if validation_csv_path is None or test_csv_path is None:
        raise ValueError(
            "Training, Validation und Test müssen als getrennte CSV-Dateien "
            "übergeben werden."
        )
    valid_graphs, valid_feature_names = load_graphs(
        validation_csv_path,
        graph_mode=graph_mode,
        target_name=selected_target,
        include_job_precedence_edges=include_job_precedence_edges,
        include_machine_predecessor_edges=include_machine_predecessor_edges,
    )
    test_graphs, test_feature_names = load_graphs(
        test_csv_path,
        graph_mode=graph_mode,
        target_name=selected_target,
        include_job_precedence_edges=include_job_precedence_edges,
        include_machine_predecessor_edges=include_machine_predecessor_edges,
    )
    if feature_names != valid_feature_names or feature_names != test_feature_names:
        raise ValueError("Feature names differ between train, valid and test data.")
    if max_training_graphs is not None:
        max_training_graphs = int(max_training_graphs)
        if max_training_graphs <= 0:
            raise ValueError("max_training_graphs must be positive.")
        if len(train_graphs) > max_training_graphs:
            train_graphs = random.Random(seed).sample(
                train_graphs, max_training_graphs
            )
    with csv_path.open(newline="", encoding="utf-8") as file:
        first_row = next(csv.DictReader(file))
    reliability_graph_config = normalize_reliability_graph_config(
        json.loads(first_row["reliability_graph_parameters"])
    )
    graphs = train_graphs + valid_graphs + test_graphs
    split_strategy = "pre_split"

    train_loader = DataLoader(
        train_graphs,
        batch_size=min(batch_size, len(train_graphs)),
        shuffle=True,
    )
    test_loader = DataLoader(test_graphs, batch_size=min(batch_size, len(test_graphs)))
    valid_loader = DataLoader(
        valid_graphs, batch_size=min(batch_size, len(valid_graphs))
    )

    edge_feature_names = reliability_edge_feature_names(
        include_job_precedence_edges,
        include_machine_predecessor_edges,
    )
    model = FJSPGraphSAGE(
        input_size=train_graphs[0].num_node_features,
        hidden_channels=hidden_channels,
        aggregation=aggregation,
        output_mode="node_sum",
        num_graphsage_layers=num_graphsage_layers,
        convolution=convolution,
        pooling=pooling,
        edge_feature_size=len(edge_feature_names),
    ).to(device)
    if solver_aware_monotonic:
        _project_monotone_nonnegative(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = _cpu_state_dict(model)
    best_valid_selection_loss = math.inf
    best_epoch = 0
    time_to_best_epoch_seconds = None
    epochs_without_improvement = 0
    epoch_seconds = []

    print(f"Trainingsdaten: {csv_path}")
    print(f"Graphen: {len(graphs)}")
    print(f"Trainingsgraphen: {len(train_graphs)}")
    print(f"Knotenfeatures: {train_graphs[0].num_node_features}")
    print(f"Gerät: {device}")
    if training_hardware.get("chip"):
        print(
            "Hardware: "
            f"{training_hardware.get('computer_model', 'Mac')} mit "
            f"{training_hardware['chip']} "
            f"({training_hardware.get('gpu_cores', '?')} GPU-Kerne)"
        )
    print(f"Kantenfeatures: {edge_feature_names}")
    print(f"Graph-Modus: {graph_mode}")
    model_description = (
        "Node NN baseline (no edge inputs)"
        if convolution == CONV_LINEAR
        else "direct-predecessor edge-conditioned GNN"
    )
    print(
        f"Modell: {convolution}({num_graphsage_layers} Layer, "
        f"{aggregation} aggregation, {pooling}) | {model_description} "
        "+ per-node probability in [0,1] + repair duration + global sum"
    )
    print(
        f"Validierung: Epoche 1, danach alle {validation_interval} "
        "Epochen und am Trainingsende"
    )
    print(
        "Early Stopping: "
        f"{early_stopping_patience} Epochen ohne Verbesserung"
    )
    print(
        "Loss: asymmetrische Node- und normalisierte Graph-MSE "
        f"(Unterschätzungsfaktor {UNDERESTIMATION_LOSS_FACTOR:g})"
    )

    _synchronize_device(device)
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            solver_aware_monotonic=solver_aware_monotonic,
            l1_regularization=l1_regularization,
        )
        if (
            epoch == 1
            or (epoch - 1) % validation_interval == 0
            or epoch == epochs
        ):
            train_metrics = evaluate(model, train_loader)
            valid_metrics = evaluate(model, valid_loader)
            valid_probability_mae = valid_metrics.get(
                "node_mae", valid_metrics["mae"]
            )
            valid_selection_loss = valid_metrics["selection_loss"]
            if valid_selection_loss < best_valid_selection_loss - 1e-8:
                best_valid_selection_loss = valid_selection_loss
                best_state = _cpu_state_dict(model)
                best_epoch = epoch
                _synchronize_device(device)
                time_to_best_epoch_seconds = (
                    time.perf_counter() - training_started
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement = epoch - best_epoch
            print(
                f"Epoch {epoch:03d} | Loss: {train_loss:.3e} | "
                f"Train MAE: {train_metrics['mae']:.4f} | "
                f"Valid MAE: {valid_metrics['mae']:.4f} | "
                f"Valid probability MAE: {valid_probability_mae:.6f} | "
                f"Valid selection loss: {valid_selection_loss:.3e}"
            )
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}; restoring best checkpoint "
                    "with valid selection loss "
                    f"{best_valid_selection_loss:.3e}."
                )
                _synchronize_device(device)
                epoch_seconds.append(time.perf_counter() - epoch_started)
                break
        _synchronize_device(device)
        epoch_seconds.append(time.perf_counter() - epoch_started)

    _synchronize_device(device)
    training_wall_clock_seconds = time.perf_counter() - training_started
    training_time = _training_time_summary(
        epoch_seconds=epoch_seconds,
        total_seconds=training_wall_clock_seconds,
        best_epoch=best_epoch,
        time_to_best_epoch_seconds=time_to_best_epoch_seconds,
        device=device,
    )

    model.load_state_dict(best_state)
    pruning_summary = _prune_small_parameters(model, pruning_threshold)
    test_metrics = evaluate(model, test_loader)
    zero_edge_metrics = None
    graph_influence_relative_improvement = None
    graph_influence_passed = None
    if convolution in MESSAGE_PASSING_CONVOLUTIONS:
        zero_edge_metrics = evaluate(
            model,
            test_loader,
            zero_edge_contributions=True,
        )
        graph_node_mae = test_metrics["node_mae"]
        zero_edge_node_mae = zero_edge_metrics["node_mae"]
        graph_influence_relative_improvement = (
            zero_edge_node_mae - graph_node_mae
        ) / max(zero_edge_node_mae, 1e-12)
        graph_influence_passed = (
            graph_influence_relative_improvement >= 0.01
        )
        if enforce_graph_influence and not graph_influence_passed:
            raise RuntimeError(
                "The graph failed the influence ablation: test "
                f"probability MAE={graph_node_mae:.6f}, zero-edge probability "
                f"MAE={zero_edge_node_mae:.6f}. No model was saved."
            )

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or architecture_stem(
        graph_mode,
        convolution,
        aggregation,
        pooling,
        selected_target,
        seed,
        layers=num_graphsage_layers,
        hidden_channels=hidden_channels,
    )
    model_path = model_dir / f"{stem}.pt"
    metadata_path = model_dir / f"{stem}_meta.json"
    torch.save(_cpu_state_dict(model), model_path)

    training_features = torch.cat([graph.x for graph in train_graphs], dim=0)
    training_targets = torch.cat(
        [graph.y.view(-1) for graph in train_graphs]
    )
    training_probability_targets = torch.cat(
        [graph.node_y.view(-1) for graph in train_graphs]
    )
    feature_summary = {
        name: {
            "min": float(training_features[:, index].min()),
            "max": float(training_features[:, index].max()),
            "mean": float(training_features[:, index].mean()),
            "std": float(training_features[:, index].std(unbiased=False)),
        }
        for index, name in enumerate(feature_names)
    }

    metadata = {
        "target_column": selected_target,
        "constraint_type": constraint_type,
        "input_size": train_graphs[0].num_node_features,
        "local_input_size": train_graphs[0].num_node_features,
        "edge_feature_size": len(edge_feature_names),
        "feature_names": feature_names,
        "edge_feature_names": edge_feature_names,
        "architecture": convolution,
        "model_family": (
            "node_nn_baseline"
            if convolution == CONV_LINEAR
            else "graph_neural_network"
        ),
        "convolution": convolution,
        "num_graphsage_layers": int(num_graphsage_layers),
        "graph_mode": graph_mode,
        "aggregation": aggregation,
        "pooling": pooling,
        "hidden_channels": hidden_channels,
        "training_device": device.type,
        "training_hardware": training_hardware,
        "training_time": training_time,
        "output_head": RELIABILITY_GNN_OUTPUT_HEAD,
        "straight_through_probability_clamp_during_training": True,
        "training_loss": {
            "node": "asymmetric_probability_mse",
            "graph": (
                "asymmetric_total_delay_error_divided_by_"
                "total_repair_duration_mse"
            ),
            "underestimation_factor": UNDERESTIMATION_LOSS_FACTOR,
            "l1_regularization": float(l1_regularization),
        },
        "solver_aware_monotonic": bool(solver_aware_monotonic),
        "pruning": pruning_summary,
        "node_target": "operation_failure_probability",
        "graph_schema": (
            RELIABILITY_GNN_GRAPH_SCHEMA_WITH_JOB_EDGES
            if include_job_precedence_edges and include_machine_predecessor_edges
            else RELIABILITY_GNN_GRAPH_SCHEMA_JOB_ONLY
            if include_job_precedence_edges
            else RELIABILITY_GNN_GRAPH_SCHEMA
        ),
        "include_machine_predecessor_edges": bool(
            include_machine_predecessor_edges
        ),
        "include_job_precedence_edges": bool(
            include_job_precedence_edges
        ),
        "reliability_graph_config": reliability_graph_config_dict(
            reliability_graph_config
        ),
        "split_strategy": split_strategy,
        "train_node_counts": sorted({g.num_nodes for g in train_graphs}),
        "valid_node_counts": sorted({g.num_nodes for g in valid_graphs}),
        "test_node_counts": sorted({g.num_nodes for g in test_graphs}),
        "training_data_path": str(csv_path),
        "validation_data_path": (
            str(validation_csv_path) if validation_csv_path is not None else None
        ),
        "test_data_path": str(test_csv_path) if test_csv_path is not None else None,
        "num_graphs": len(graphs),
        "num_train_graphs": len(train_graphs),
        "num_valid_graphs": len(valid_graphs),
        "num_test_graphs": len(test_graphs),
        "epochs": epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_interval_epochs": validation_interval,
        "early_stopping_patience_epochs": early_stopping_patience,
        "seed": seed,
        "test_mae": test_metrics["mae"],
        "test_mse": test_metrics["rmse"] ** 2,
        "test_rmse": test_metrics["rmse"],
        "test_node_mae": test_metrics.get("node_mae"),
        "test_node_rmse": test_metrics.get("node_rmse"),
        "test_node_metric": "operation_failure_probability",
        "test_node_probability_mae": test_metrics.get(
            "node_probability_mae"
        ),
        "test_node_probability_rmse": test_metrics.get(
            "node_probability_rmse"
        ),
        "test_node_delay_mae": test_metrics.get("node_delay_mae"),
        "test_node_delay_rmse": test_metrics.get("node_delay_rmse"),
        "test_metrics_by_num_nodes": test_metrics.get("by_num_nodes"),
        "test_selection_loss": test_metrics.get("selection_loss"),
        "test_asymmetric_node_loss": test_metrics.get(
            "asymmetric_node_loss"
        ),
        "test_asymmetric_graph_loss": test_metrics.get(
            "asymmetric_graph_loss"
        ),
        "test_underestimation_rate": test_metrics["underestimation_rate"],
        "underestimation_per_node_q95": test_metrics[
            "underestimation_per_node_q95"
        ],
        "training_feature_summary": feature_summary,
        "training_target_summary": {
            "min": float(training_targets.min()),
            "max": float(training_targets.max()),
            "mean": float(training_targets.mean()),
            "std": float(training_targets.std(unbiased=False)),
        },
        "training_node_probability_summary": {
            "min": float(training_probability_targets.min()),
            "max": float(training_probability_targets.max()),
            "mean": float(training_probability_targets.mean()),
            "std": float(
                training_probability_targets.std(unbiased=False)
            ),
        },
        "graph_influence_ablation": (
            {
                "criterion": (
                    "test_probability_mae_at_least_1_percent_better_"
                    "than_zero_edges"
                ),
                "required": bool(enforce_graph_influence),
                "passed": graph_influence_passed,
                "zero_edge_test_mae": zero_edge_metrics["mae"],
                "zero_edge_test_probability_mae": zero_edge_metrics[
                    "node_probability_mae"
                ],
                "relative_probability_mae_improvement": (
                    graph_influence_relative_improvement
                ),
            }
            if zero_edge_metrics is not None else None
        ),
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(
        f"Test MAE: {test_metrics['mae']:.4f} | "
        f"Test RMSE: {test_metrics['rmse']:.4f} | "
        f"Probability MAE: {test_metrics['node_probability_mae']:.6f} | "
        f"Node delay MAE: {test_metrics['node_delay_mae']:.4f} | "
        "q95 underestimation/node: "
        f"{test_metrics['underestimation_per_node_q95']:.6f}"
    )
    if zero_edge_metrics is not None:
        print(
            "Graph influence "
            f"{'passed' if graph_influence_passed else 'failed'}: "
            "probability MAE "
            f"{test_metrics['node_probability_mae']:.6f} vs "
            f"{zero_edge_metrics['node_probability_mae']:.6f} "
            "with zero edges "
            f"({100.0 * graph_influence_relative_improvement:.1f}% better)."
        )
    best_time_seconds = training_time["time_to_best_epoch_seconds"]
    best_time_text = (
        f"Epoche {best_epoch} nach {best_time_seconds:.2f} s"
        if best_time_seconds is not None
        else "nicht bestimmt"
    )
    print(
        "Trainingszeit: "
        f"{training_time['wall_clock_seconds']:.2f} s "
        f"({training_time['wall_clock_hours']:.4f} h) | "
        f"Ø {training_time['mean_seconds_per_epoch']:.3f} s/Epoche | "
        f"beste Epoche: {best_time_text}"
    )
    if training_time["gpu_hours"] is not None:
        print(
            f"GPU-Zeit ({device.type}, 1 GPU): "
            f"{training_time['gpu_hours']:.6f} GPU-h | "
            f"{training_time['gpu_days']:.8f} GPU-Tage"
        )
    else:
        print(
            "GPU-Zeit: nicht berechnet, da das Training auf der CPU lief."
        )
    print(f"Modell gespeichert: {model_path}")
    print(f"Metadaten gespeichert: {metadata_path}")
    return model_path, metadata_path


def _parse_args():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    training_config = config.get("training", {})
    data_generation = training_config.get("data_generation", {})
    configured_path = data_generation.get(
        "output_directory", str(DEFAULT_DATASET_DIR)
    )
    dataset_directory = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path(configured_path)
    )
    if not dataset_directory.is_absolute():
        dataset_directory = ROOT_DIR / dataset_directory
    return seed, dataset_directory


def train_from_config(seed=42, csv_path=None):
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    training_config = config.get("training", {})
    gnn_config = training_config.get("gnn", {})
    data_generation = training_config.get("data_generation", {})
    optimizer_config = gnn_config.get("optimizer", {})
    validation_config = gnn_config.get("validation", {})
    configured_model_dir = Path(
        gnn_config.get("model_directory", MODEL_DIR)
    )
    if not configured_model_dir.is_absolute():
        configured_model_dir = ROOT_DIR / configured_model_dir
    data_generation_method = str(
        data_generation.get(
            "method", "random_feasible"
        )
    ).strip().lower()
    split_csv_filenames = (
        FIXED_SOLUTION_CSV_FILENAMES
        if data_generation_method in {
            "gurobi_fixed",
            "gurobi_linear_labeled",
        }
        else SPLIT_CSV_FILENAMES
    )
    if csv_path is None:
        dataset_directory = Path(
            data_generation.get(
                "output_directory", str(DEFAULT_DATASET_DIR)
            )
        )
    else:
        dataset_directory = Path(csv_path)
    if not dataset_directory.is_absolute():
        dataset_directory = ROOT_DIR / dataset_directory

    csv_path = (
        dataset_directory
        / SPLIT_DIRECTORIES["train"]
        / split_csv_filenames["train"]
    )
    validation_csv_path = (
        dataset_directory
        / SPLIT_DIRECTORIES["valid"]
        / split_csv_filenames["valid"]
    )
    test_csv_path = (
        dataset_directory
        / SPLIT_DIRECTORIES["test"]
        / split_csv_filenames["test"]
    )
    split_paths = {
        "training": csv_path,
        "valid": validation_csv_path,
        "test": test_csv_path,
    }
    split_instances = {}
    for split_name, split_path in split_paths.items():
        if not split_path.exists():
            raise FileNotFoundError(
                f"GNN-{split_name}-Datensatz nicht gefunden: {split_path}"
            )
        with split_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if "instance_name" not in (reader.fieldnames or []):
                raise ValueError(
                    f"{split_path} enthält keine instance_name-Spalte."
                )
            split_instances[split_name] = {
                row["instance_name"] for row in reader
            }
    split_names = list(split_instances)
    for index, split_name in enumerate(split_names):
        for other_split in split_names[index + 1:]:
            overlap = split_instances[split_name] & split_instances[other_split]
            if overlap:
                raise ValueError(
                    "Data Leakage: Instanzen kommen in mehreren GNN-Splits vor: "
                    f"{sorted(overlap)}"
                )

    results = []
    raw_combinations = gnn_config.get("combinations")
    if not raw_combinations:
        raise ValueError(
            "training.gnn.combinations must contain at least one "
            "architecture."
        )
    combinations = []
    seen = set()
    for raw in raw_combinations:
        for architecture in expand_architecture_variants(raw):
            key = (
                architecture["graph_mode"],
                architecture["convolution"],
                architecture["aggregation"],
                architecture["pooling"],
                architecture["layers"],
                architecture["hidden_channels"],
            )
            if key not in seen:
                combinations.append(architecture)
                seen.add(key)
    print(
        "GNN architecture variants to train: "
        f"{len(combinations)} "
        f"({len(raw_combinations)} base architectures)",
        flush=True,
    )
    for architecture in combinations:
        output_dir = architecture_model_dir(
            configured_model_dir,
            architecture["graph_mode"],
            architecture["convolution"],
            architecture["aggregation"],
            architecture["pooling"],
            layers=architecture["layers"],
            hidden_channels=architecture["hidden_channels"],
        )
        output_stem = architecture_stem(
            architecture["graph_mode"],
            architecture["convolution"],
            architecture["aggregation"],
            architecture["pooling"],
            TARGET_COLUMN,
            seed,
            layers=architecture["layers"],
            hidden_channels=architecture["hidden_channels"],
        )
        print(
            "\n=== Training architecture: "
            f"{architecture['graph_mode']} / {architecture['convolution']} / "
            f"{architecture['aggregation']} / {architecture['pooling']} / "
            f"layers={architecture['layers']} / "
            f"hidden={architecture['hidden_channels']} ==="
        )
        try:
            result = train_from_file(
                csv_path=csv_path,
                validation_csv_path=validation_csv_path,
                test_csv_path=test_csv_path,
                seed=int(seed),
                epochs=int(optimizer_config.get("epochs", 10000)),
                hidden_channels=architecture["hidden_channels"],
                batch_size=int(optimizer_config.get("batch_size", 64)),
                learning_rate=float(
                    optimizer_config.get("learning_rate", 0.01)
                ),
                aggregation=architecture["aggregation"],
                graph_mode=architecture["graph_mode"],
                convolution=architecture["convolution"],
                pooling=architecture["pooling"],
                split_strategy=validation_config.get(
                    "split_strategy", "random"
                ),
                validation_interval=int(
                    validation_config.get("interval_epochs", 10)
                ),
                early_stopping_patience=int(
                    validation_config.get("patience_epochs", 10)
                ),
                solver_aware_monotonic=bool(
                    optimizer_config.get("solver_aware_monotonic", False)
                ),
                l1_regularization=float(
                    optimizer_config.get("l1_regularization", 0.0)
                ),
                pruning_threshold=float(
                    optimizer_config.get("pruning_threshold", 0.0)
                ),
                num_graphsage_layers=architecture["layers"],
                enforce_graph_influence=bool(
                    validation_config.get(
                        "enforce_graph_influence", False
                    )
                ),
                output_stem=output_stem,
                model_dir=output_dir,
            )
            results.append(result)
        finally:
            _release_training_memory()
            print(
                "Training memory released "
                "(Python GC + accelerator cache).",
                flush=True,
            )
    return results


if __name__ == "__main__":
    seed_gnn, csv_file = _parse_args()
    train_from_config(seed=seed_gnn, csv_path=csv_file)
