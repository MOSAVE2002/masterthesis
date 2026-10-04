"""Train GNN surrogates for job-specific expected repair buffers.

The module validates generated graph datasets, converts their CSV rows into
PyTorch Geometric graphs and trains every architecture selected in the project
configuration. Each model predicts one nonnegative expected repair buffer per
job. The best validation checkpoint and the metadata required for the later
Gurobi embedding are stored together in the configured model directory.
"""

import csv
import gc
import importlib
from itertools import pairwise
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool


ROOT_DIR = Path(__file__).resolve().parents[2]

from helper.sequence_setup import (
    RELIABILITY_GNN_GRAPH_SCHEMA,
    RELIABILITY_GNN_OUTPUT_HEAD,
    reliability_graph_config_dict,
    reliability_node_feature_names,
)
from helper.stochastic_fjsp import (
    normalize_machine_profile_config,
    normalize_training_parameter_jitter,
)


instances = importlib.import_module("01_generator.instance_generator")
architectures = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)
from helper.local_buffer import (
    LABEL_METHOD as LOCAL_BUFFER_LABEL_METHOD,
    JOB_TARGET,
    LABEL_SOURCE,
    TARGET_COLUMN,
    label_config_dict,
)

SPLIT_DIRECTORIES = instances.SPLIT_DIRECTORIES
SPLIT_CSV_FILENAMES = instances.SPLIT_CSV_FILENAMES

CONV_LINEAR = architectures.CONV_LINEAR
CONV_SAGE = architectures.CONV_SAGE
CONV_JOB = architectures.CONV_JOB
VALID_LAYER_COUNTS = architectures.VALID_LAYER_COUNTS
architecture_model_dir = architectures.architecture_model_dir
architecture_stem = architectures.architecture_stem
architecture_from_config = architectures.architecture_from_config
validate_architecture = architectures.validate_architecture


def _device():
    """Return the fastest available PyTorch execution device.

    CUDA is preferred over Apple's Metal Performance Shaders backend. The CPU
    is used when neither accelerator is available.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def release_memory():
    """Release Python and accelerator memory after one training run.

    The explicit cleanup prevents successive architecture combinations from
    retaining unused tensors on CUDA or MPS devices.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def edge_index(edges) -> torch.Tensor:
    """Convert directed edge pairs into PyTorch Geometric COO format.

    Args:
        edges: Iterable of ``(source, target)`` node-index pairs.

    Returns:
        A long tensor with shape ``[2, number_of_edges]``. Empty input is
        represented by a tensor with shape ``[2, 0]``.
    """
    return torch.tensor(
        [
            [source for source, _target in edges],
            [target for _source, target in edges],
        ],
        dtype=torch.long,
    ).reshape(2, -1)


def job_precedence_edges(job_membership) -> list[tuple[int, int]]:
    """Build directed precedence edges between consecutive job operations.

    Args:
        job_membership: Job index of every operation node in dataset order.

    Returns:
        Pairs connecting each operation to the next operation of the same job.
    """
    nodes_by_job = {}
    for node_index, job_index in enumerate(job_membership):
        nodes_by_job.setdefault(int(job_index), []).append(node_index)
    return [
        (source, target)
        for nodes in nodes_by_job.values()
        for source, target in pairwise(nodes)
    ]


def validate_dataset_metadata(
    csv_path,
    machine_profile_config,
    training_parameter_jitter,
):
    """Validate that a generated dataset matches the active GNN pipeline.

    The generation summary is checked for completion, label definition, graph
    schema, machine profiles and train-only parameter jitter. This prevents a
    model from being trained on data produced with incompatible assumptions.

    Args:
        csv_path: Path to a split CSV inside the generated dataset directory.
        machine_profile_config: Machine profiles requested by ``config.json``.
        training_parameter_jitter: Training jitter requested by the current
            instance-generation configuration.

    Returns:
        The validated label-method identifier stored in the dataset metadata.

    Raises:
        FileNotFoundError: If the generation summary is missing.
        ValueError: If any dataset property differs from the active pipeline.
    """
    summary_path = Path(csv_path).parent.parent / "generation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Dataset label metadata not found: "
            f"{summary_path}. Regenerate the dataset with schema_version >= 3."
        )
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    if summary.get("status") != "completed":
        raise ValueError(
            "Dataset generation is incomplete; finish generation before training."
        )
    label = summary.get("label") or {}
    if label.get("target_column") != TARGET_COLUMN:
        raise ValueError(
            f"Dataset target is {label.get('target_column')!r}, expected "
            f"{TARGET_COLUMN!r}."
        )
    label_method = label.get("label_method")
    if label_method != LOCAL_BUFFER_LABEL_METHOD:
        raise ValueError(
            "The active GNN pipeline requires deterministic local buffer "
            f"labels, got {label_method!r}."
        )
    if label.get("source") != LABEL_SOURCE:
        raise ValueError(
            "Dataset label source does not match local buffer expectations."
        )
    if label.get("parameters") != label_config_dict():
        raise ValueError(
            "Dataset label parameters do not match the fixed local-buffer "
            "settings."
        )
    graph = summary.get("graph") or {}
    expected_graph = {
        "graph_schema": RELIABILITY_GNN_GRAPH_SCHEMA,
        "service_scope": "job",
        "feature_names": reliability_node_feature_names(),
        "include_machine_predecessor_edges": True,
        "machine_predecessor_edge_scope": "direct",
        "include_job_precedence_edges": True,
    }
    if any(graph.get(key) != value for key, value in expected_graph.items()):
        raise ValueError(
            "The active GNN pipeline requires direct U machine edges and "
            "fixed job edges. Regenerate the dataset with schema_version >= 3."
        )
    dataset_profiles = summary.get("machine_profile_config")
    if dataset_profiles is None:
        raise ValueError(
            "Dataset metadata does not contain machine_profile_config. "
            "Regenerate the GNN dataset before training."
        )
    if normalize_machine_profile_config(
        dataset_profiles
    ) != normalize_machine_profile_config(machine_profile_config):
        raise ValueError(
            "Dataset machine profiles differ from the training config. "
            "Regenerate the GNN dataset before training."
        )
    if normalize_training_parameter_jitter(
        summary.get("training_parameter_jitter")
    ) != normalize_training_parameter_jitter(training_parameter_jitter):
        raise ValueError(
            "Dataset training jitter differs from the training config. "
            "Regenerate the GNN dataset before training."
        )
    return label_method


def load_graphs(csv_path):
    """Load one dataset split as PyTorch Geometric graph objects.

    Each CSV row becomes one graph. Stored direct machine-predecessor edges are
    used by the SAGE variant, while fixed within-job edges are reconstructed
    from the operation-to-job membership. The function also verifies that all
    rows use the expected feature order and reliability configuration.

    Args:
        csv_path: CSV file produced for a train or validation split.

    Returns:
        A pair containing the list of :class:`torch_geometric.data.Data`
        graphs and their common ordered node-feature names.

    Raises:
        FileNotFoundError: If the split CSV does not exist.
        ValueError: If required columns, graph parameters, features or graph
            rows are missing or inconsistent.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    required = {
        "instance_name",
        TARGET_COLUMN,
        "gnn_feature_names",
        "gnn_node_features",
        "gnn_active_edges",
        "job_ids",
        "operation_job_indices",
        "reliability_graph_parameters",
    }
    graphs, feature_names = [], None
    expected_graph_config = reliability_graph_config_dict()
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            row_config_values = json.loads(
                row["reliability_graph_parameters"]
            )
            if row_config_values != expected_graph_config:
                raise ValueError(
                    "Reliability parameters do not match the fixed graph "
                    "configuration. Regenerate the dataset."
                )
            names = json.loads(row["gnn_feature_names"])
            if names != reliability_node_feature_names():
                raise ValueError(
                    f"Unexpected node feature order: {names}. Expected "
                    f"{reliability_node_feature_names()}. Regenerate the "
                    "dataset features before training."
                )
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("Node feature order differs between rows.")

            active_edges = json.loads(row["gnn_active_edges"])
            job_membership = json.loads(row["operation_job_indices"])
            job_edges = job_precedence_edges(job_membership)
            x = torch.tensor(
                json.loads(row["gnn_node_features"]), dtype=torch.float32
            )
            if x.ndim != 2 or x.shape[1] != len(names):
                raise ValueError("Node feature values do not match feature metadata.")
            graphs.append(Data(
                x=x,
                edge_index=edge_index(active_edges),
                job_edge_index=edge_index(job_edges),
                job_y=torch.tensor(
                    json.loads(row[TARGET_COLUMN]),
                    dtype=torch.float32,
                ),
                job_membership=torch.tensor(
                    job_membership,
                    dtype=torch.long,
                ),
                num_jobs_tensor=torch.tensor(
                    [len(json.loads(row["job_ids"]))], dtype=torch.long
                ),
            ))
    if not graphs:
        raise ValueError(f"Dataset contains no graphs: {csv_path}")
    return graphs, feature_names


class NodeGraphSAGEConv(nn.Module):
    """Apply one predecessor-sum GraphSAGE-style message-passing layer.

    A node receives a linear transformation of its own state and a second
    linear transformation of the sum of all incoming predecessor states. The
    layer deliberately omits degree normalization so that its operations can
    later be reproduced exactly inside the Gurobi model.

    Args:
        input_size: Number of input features per operation node.
        output_size: Number of generated hidden features per node.
    """

    def __init__(self, input_size, output_size):
        """Initialize separate root-node and predecessor-message transforms.

        Args:
            input_size: Number of features in each incoming node state.
            output_size: Number of channels produced for each node.
        """
        super().__init__()
        self.lin_root = nn.Linear(input_size, output_size)
        self.lin_message = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x, edge_index):
        """Return updated node states for the supplied directed graph.

        Args:
            x: Node-feature tensor with shape ``[num_nodes, input_size]``.
            edge_index: Directed COO edge tensor with shape ``[2, num_edges]``.

        Returns:
            Tensor with one ``output_size``-dimensional state per node.
        """
        source, target = edge_index
        predecessor_sum = x.new_zeros(x.shape)
        if source.numel():
            predecessor_sum.index_add_(0, target, x[source])
        return self.lin_root(x) + self.lin_message(predecessor_sum)


class FJSPGraphSAGE(nn.Module):
    """Predict one nonnegative expected repair buffer for every job.

    Depending on ``convolution``, the network either processes nodes
    independently, exchanges messages over direct machine-predecessor edges,
    or exchanges messages only over fixed job-precedence edges. After the
    configured hidden layers, operation states and original node features are
    summed by job and passed through a nonnegative output head.

    Args:
        input_size: Number of input features per operation node.
        hidden_channels: Width of every hidden layer.
        num_graphsage_layers: Number of hidden processing layers.
        convolution: Active linear, machine-edge or job-edge architecture.
        initial_repair_buffer: Initial output bias in time units.
    """

    def __init__(
        self,
        input_size,
        hidden_channels,
        num_graphsage_layers,
        convolution,
        initial_repair_buffer=0.50,
    ):
        """Construct one configured repair-buffer prediction architecture.

        The output bias starts at the mean training buffer supplied by the
        caller, while the skip branch initially contributes zero.
        """
        super().__init__()
        if convolution not in {CONV_LINEAR, CONV_SAGE, CONV_JOB}:
            raise ValueError("Only linear, sage and job are supported.")
        if num_graphsage_layers not in VALID_LAYER_COUNTS:
            choices = ", ".join(map(str, sorted(VALID_LAYER_COUNTS)))
            raise ValueError(
                f"The active pipeline supports {choices} layers."
            )
        initial_repair_buffer = float(initial_repair_buffer)
        if initial_repair_buffer < 0.0:
            raise ValueError("The initial repair buffer must be nonnegative.")
        self.convolution = convolution

        def layer(input_channels):
            """Create one hidden transformation for the active architecture.

            The linear baseline receives a dense layer; relational variants
            receive the custom predecessor-sum message-passing layer.
            """
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
        nn.init.constant_(self.out.bias, initial_repair_buffer)
        nn.init.zeros_(self.out_input.weight)
        nn.init.zeros_(self.out_input.bias)

    def _layer(self, layer, x, edge_index):
        """Apply the active hidden-layer interface.

        Linear layers consume only node states, whereas relational layers also
        receive the directed edge index selected by :meth:`forward`.
        """
        return layer(x) if self.convolution == CONV_LINEAR else (
            layer(x, edge_index)
        )

    def forward(self, data):
        """Predict flattened job-level repair buffers for a graph batch.

        Args:
            data: Batched PyTorch Geometric data containing node features,
                alternative edge sets, node-to-job memberships and batch IDs.

        Returns:
            A one-dimensional tensor containing one nonnegative prediction for
            every job in every graph of the batch.
        """
        x_input = data.x
        edge_index = (
            data.job_edge_index
            if self.convolution == CONV_JOB
            else data.edge_index
        )
        x = F.relu(self._layer(self.conv1, x_input, edge_index))
        if self.conv2 is not None:
            x = F.relu(self._layer(self.conv2, x, edge_index))
        if self.conv3 is not None:
            x = F.relu(self._layer(self.conv3, x, edge_index))
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
        return F.relu(
            self.out(pooled).view(-1)
            + self.out_input(pooled_input).view(-1)
        )


def train_epoch(model, loader, optimizer):
    """Run one optimization epoch and return its graph-weighted MSE.

    Args:
        model: Repair-buffer predictor being optimized.
        loader: Iterable yielding batches of training graphs.
        optimizer: Initialized PyTorch optimizer for the model parameters.

    Returns:
        Mean batch MSE weighted by the number of graphs in each batch.
    """
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(batch), batch.job_y.view(-1))
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
    return total / len(loader.dataset)


def validation_mae(model, loader):
    """Evaluate the job-level mean absolute error without gradient tracking.

    Args:
        model: Repair-buffer predictor to evaluate.
        loader: Iterable yielding validation graph batches.

    Returns:
        Mean absolute error over all job targets in time units.
    """
    model.eval()
    absolute_error = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            target = batch.job_y.view(-1)
            absolute_error += float((model(batch) - target).abs().sum())
            count += target.numel()
    return absolute_error / count


def cpu_state(model):
    """Copy all model parameters to independent CPU checkpoint tensors.

    Detaching and cloning prevents later optimizer steps from modifying the
    checkpoint retained for early stopping.
    """
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


class CachedGraphBatches:
    """Cache fixed graph batches and reproducibly reorder them each epoch.

    This lightweight loader avoids rebuilding PyTorch Geometric batches during
    long training runs. Graph membership inside each batch remains fixed; only
    the order of complete batches changes between epochs.

    Args:
        graphs: Sequence of graph samples to batch.
        batch_size: Maximum number of graphs per cached batch.
        shuffle: Whether to shuffle graphs initially and batches per epoch.
        seed: Random seed controlling both reproducible permutations.
    """

    def __init__(self, graphs, batch_size, *, shuffle=False, seed=42):
        """Materialize fixed graph batches and initialize epoch ordering.

        Args:
            graphs: Sequence of PyTorch Geometric graph samples.
            batch_size: Maximum number of graphs in one cached batch.
            shuffle: Whether to randomize initial membership and epoch order.
            seed: Base seed for reproducible permutations.
        """
        self.dataset = graphs
        indices = list(range(len(graphs)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        self.batches = [
            Batch.from_data_list([
                graphs[index]
                for index in indices[start:start + batch_size]
            ])
            for start in range(0, len(indices), batch_size)
        ]
        self.shuffle, self.seed, self.epoch = shuffle, int(seed), 0

    def __iter__(self):
        """Yield cached batches in a reproducible epoch-specific order.

        Returns:
            Generator over each cached batch exactly once.
        """
        self.epoch += 1
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.Random(self.seed * 100000 + self.epoch).shuffle(order)
        return (self.batches[i] for i in order)


def train_model(
    train_graphs,
    valid_graphs,
    feature_names,
    label_method,
    *,
    seed=42,
    epochs=500,
    hidden_channels=8,
    batch_size=32,
    learning_rate=0.001,
    convolution=CONV_SAGE,
    num_graphsage_layers=1,
    validation_interval=10,
    early_stopping_patience=100,
    machine_profile_config=None,
    training_parameter_jitter=None,
    output_stem,
    model_dir,
    cache_batches=False,
):
    """Train, select and persist one configured GNN architecture.

    Training minimizes mean squared error with Adam. Validation MAE selects the
    best checkpoint and triggers early stopping when it does not improve for
    the configured number of epochs. The resulting CPU state dictionary is
    stored alongside metadata needed to reconstruct and embed the architecture.

    Args:
        train_graphs: Graphs used for gradient-based optimization.
        valid_graphs: Graphs used for checkpoint selection and early stopping.
        feature_names: Ordered names of the node-feature columns.
        label_method: Identifier of the validated target-generation method.
        seed: Random seed for PyTorch, Python and optional cached batches.
        epochs: Maximum number of training epochs.
        hidden_channels: Width of each hidden layer.
        batch_size: Maximum number of graphs per training batch.
        learning_rate: Adam learning rate.
        convolution: Linear, direct-machine-edge or job-edge architecture.
        num_graphsage_layers: Number of hidden layers.
        validation_interval: Number of epochs between validation evaluations.
        early_stopping_patience: Epochs without improvement before stopping.
        machine_profile_config: Machine-profile metadata saved with the model.
        training_parameter_jitter: Training-jitter metadata saved with the
            model.
        output_stem: Shared filename stem for weights and metadata.
        model_dir: Destination directory for trained artifacts.
        cache_batches: Whether graph batches are materialized once and reused.

    Returns:
        Paths to the saved weight file and its JSON metadata file.
    """
    architecture = validate_architecture(convolution)
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    device = _device()
    loader_class = CachedGraphBatches if cache_batches else DataLoader
    train_loader = loader_class(
        train_graphs, batch_size=min(batch_size, len(train_graphs)), shuffle=True,
        **({"seed": seed} if cache_batches else {}),
    )
    valid_loader = loader_class(
        valid_graphs, batch_size=min(batch_size, len(valid_graphs))
    )
    initial_repair_buffer = float(torch.cat([
        graph.job_y.view(-1) for graph in train_graphs
    ]).mean())
    model = FJSPGraphSAGE(
        input_size=len(feature_names),
        hidden_channels=int(hidden_channels),
        num_graphsage_layers=int(num_graphsage_layers),
        convolution=architecture["convolution"],
        initial_repair_buffer=initial_repair_buffer,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best_state, best_loss, best_epoch = cpu_state(model), math.inf, 0
    for epoch in range(1, int(epochs) + 1):
        training_loss = train_epoch(model, train_loader, optimizer)
        if (
            epoch == 1
            or epoch % int(validation_interval) == 0
            or epoch == epochs
        ):
            selection_loss = validation_mae(model, valid_loader)
            print(
                f"epoch={epoch:04d} train_loss={training_loss:.6g} "
                f"valid_mae={selection_loss:.6g} ZE",
                flush=True,
            )
            if selection_loss < best_loss - 1e-10:
                best_state, best_loss, best_epoch = (
                    cpu_state(model), selection_loss, epoch
                )
            elif epoch - best_epoch >= int(early_stopping_patience):
                print(
                    "EARLY_STOPPING | "
                    f"epoch={epoch} | best_epoch={best_epoch} | "
                    f"patience_epochs={int(early_stopping_patience)} | "
                    f"best_valid_mae={best_loss:.6g} ZE | "
                    f"current_valid_mae={selection_loss:.6g} ZE",
                    flush=True,
                )
                break
    model.load_state_dict(best_state)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{output_stem}.pt"
    metadata_path = model_dir / f"{output_stem}_meta.json"
    torch.save(cpu_state(model), model_path)
    metadata = {
        "target_column": TARGET_COLUMN,
        "input_size": len(feature_names),
        "feature_names": feature_names,
        "graph_mode": architecture["graph_mode"],
        "convolution": architecture["convolution"],
        "aggregation": architecture["aggregation"],
        "pooling": architecture["pooling"],
        "num_graphsage_layers": int(num_graphsage_layers),
        "hidden_channels": int(hidden_channels),
        "output_head": RELIABILITY_GNN_OUTPUT_HEAD,
        "job_target": JOB_TARGET,
        "job_repair_buffer_label_method": label_method,
        "graph_schema": RELIABILITY_GNN_GRAPH_SCHEMA,
        "message_passing": (
            "none"
            if architecture["convolution"] == CONV_LINEAR
            else "source_node_states_only"
        ),
        "include_machine_predecessor_edges": (
            architecture["convolution"] == CONV_SAGE
        ),
        "machine_predecessor_edge_scope": (
            "direct" if architecture["convolution"] == CONV_SAGE else "none"
        ),
        "include_job_precedence_edges": (
            architecture["convolution"] in {CONV_SAGE, CONV_JOB}
        ),
        "reliability_graph_config": reliability_graph_config_dict(),
        "machine_profile_config": normalize_machine_profile_config(
            machine_profile_config
        ),
        "training_parameter_jitter": normalize_training_parameter_jitter(
            training_parameter_jitter
        ),
        "seed": int(seed),
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"saved model={model_path}", flush=True)
    return model_path, metadata_path


def train_from_config(config):
    """Train every unique GNN architecture selected in ``config.json``.

    The function resolves the generated train and validation splits, verifies
    their shared metadata, loads both graph datasets and expands the configured
    architecture combinations. Every unique combination is trained separately,
    stored in its architecture-specific directory and released from memory
    before the next run starts.

    Args:
        config: Complete project configuration containing instance-generation,
            data-generation and GNN-training settings.

    Returns:
        A list of ``(model_path, metadata_path)`` pairs, one for every unique
        trained architecture.

    Raises:
        ValueError: If dataset metadata or feature orders are inconsistent.
    """
    training = config["training"]
    gnn = training["gnn"]
    seed = int(gnn.get("seed", 42))
    optimizer = gnn.get("optimizer", {})
    validation = gnn.get("validation", {})
    dataset = Path(training["data_generation"]["output_directory"])
    model_root = Path(gnn["model_directory"])
    if not dataset.is_absolute():
        dataset = ROOT_DIR / dataset
    if not model_root.is_absolute():
        model_root = ROOT_DIR / model_root
    paths = {
        split: dataset / SPLIT_DIRECTORIES[split] / SPLIT_CSV_FILENAMES[split]
        for split in ("train", "valid")
    }
    machine_profile_config = config["instances"]["generation"][
        "machine_profiles"
    ]
    training_parameter_jitter = config["instances"]["generation"].get(
        "training_parameter_jitter"
    )
    label_method = validate_dataset_metadata(
        paths["train"],
        machine_profile_config,
        training_parameter_jitter,
    )
    train_graphs, feature_names = load_graphs(paths["train"])
    valid_graphs, valid_names = load_graphs(paths["valid"])
    if feature_names != valid_names:
        raise ValueError("Feature names differ between data splits.")

    results = []
    seen = set()
    for raw in gnn["combinations"]:
        architecture = architecture_from_config(raw)
        key = tuple(architecture.values())
        if key in seen:
            continue
        seen.add(key)
        output_dir = architecture_model_dir(
            model_root,
            architecture["convolution"],
            architecture["layers"],
            architecture["hidden_channels"],
        )
        stem = architecture_stem(
            architecture["convolution"],
            TARGET_COLUMN,
            seed,
            architecture["layers"],
            architecture["hidden_channels"],
        )
        try:
            results.append(train_model(
                train_graphs,
                valid_graphs,
                feature_names,
                label_method,
                seed=seed,
                epochs=int(optimizer.get("epochs", 500)),
                hidden_channels=architecture["hidden_channels"],
                batch_size=int(optimizer.get("batch_size", 32)),
                learning_rate=float(optimizer.get("learning_rate", 0.001)),
                convolution=architecture["convolution"],
                num_graphsage_layers=architecture["layers"],
                validation_interval=int(
                    validation.get("interval_epochs", 10)
                ),
                early_stopping_patience=int(
                    validation.get("patience_epochs", 100)
                ),
                machine_profile_config=machine_profile_config,
                training_parameter_jitter=training_parameter_jitter,
                output_stem=stem,
                model_dir=output_dir,
                cache_batches=bool(optimizer.get("cache_batches", False)),
            ))
        finally:
            release_memory()
    return results
