"""Locate the exact trained GNN artifacts requested by the configuration.

Artifact names are reconstructed from the shared architecture contract. Stored
machine-profile and training-jitter metadata are validated before model paths
are handed to the Gurobi solver.
"""

import importlib
import json
from pathlib import Path

from helper.local_buffer import TARGET_COLUMN
from helper.stochastic_fjsp import (
    normalize_machine_profile_config,
    normalize_training_parameter_jitter,
)
ROOT_DIR = Path(__file__).resolve().parents[1]
_architectures = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)


def _absolute(path):
    """Resolve an artifact path relative to the repository root.

    Already absolute model-directory paths are returned without modification.
    """
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def trained_models(config):
    """Resolve and validate all GNN models selected for the experiment.

    Returns:
        Dictionaries containing the weight and metadata paths of each model.

    Raises:
        FileNotFoundError: If a configured model artifact is absent.
        ValueError: If training metadata differs from the active instance
            generation configuration.
    """
    gnn = config["training"]["gnn"]
    model_root = _absolute(gnn["model_directory"])
    seed = int(gnn.get("seed", 42))
    target = TARGET_COLUMN
    configured_profiles = normalize_machine_profile_config(
        config["instances"]["generation"]["machine_profiles"]
    )
    configured_jitter = normalize_training_parameter_jitter(
        config["instances"]["generation"].get("training_parameter_jitter")
    )
    models = []

    for raw in gnn["combinations"]:
        architecture = _architectures.architecture_from_config(raw)
        directory = _architectures.architecture_model_dir(
            model_root,
            architecture["convolution"],
            architecture["layers"],
            architecture["hidden_channels"],
        )
        stem = _architectures.architecture_stem(
            architecture["convolution"],
            target,
            seed,
            architecture["layers"],
            architecture["hidden_channels"],
        )
        model_path = directory / f"{stem}.pt"
        metadata_path = directory / f"{stem}_meta.json"
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Trained GNN missing: {model_path} / {metadata_path}"
            )
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        metadata_jitter = normalize_training_parameter_jitter(
            metadata["training_parameter_jitter"]
        )
        if (
            normalize_machine_profile_config(
                metadata["machine_profile_config"]
            )
            != configured_profiles
            or metadata_jitter != configured_jitter
        ):
            raise ValueError(
                "Configured machine profiles differ from the trained GNN "
                f"metadata {metadata_path}. Regenerate the training data "
                "and retrain all GNN models before running gurobi_gnn."
            )
        models.append({
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        })
    return models
