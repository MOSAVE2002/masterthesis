"""
Trainiert ein einfaches Sequential-MLP (Linear + ReLU) zur Vorhersage des
Makespan aus den festen Maschinenzuweisungen einer Instanz.

Eingabe pro Sample (flach): [y_op_1, y_op_2, ..., y_op_n]
Ausgabe: makespan

Die Trainingsdaten werden aus `data/fixed_y_results` gelesen.
Das trainierte Modell wird als .pt-Datei gespeichert. Zusaetzlich werden
Metadaten in einer JSON-Datei abgelegt, damit Input-Reihenfolge und Bounds
spaeter einfach weiterverwendet werden koennen.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SequentialMLP(nn.Module):
    def __init__(self, n_inputs: int, layer_sizes: tuple[int, ...]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_features = n_inputs

        for size in layer_sizes:
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.ReLU())
            in_features = size

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    feature_columns = [column for column in df.columns if column.startswith("y_op_")]
    target_col = "makespan"

    if not feature_columns:
        raise ValueError("Keine Feature-Spalten gefunden. Erwartet werden Spalten wie 'y_op_1'.")

    if target_col not in df.columns:
        raise ValueError("Die CSV enthaelt keine Spalte 'makespan'.")

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X, y, feature_columns


def build_mlp(n_inputs: int, layer_sizes=(32, 32, 16)):
    return SequentialMLP(n_inputs=n_inputs, layer_sizes=layer_sizes)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_true - y_pred)
    mse = np.mean((y_true - y_pred) ** 2)

    non_zero_mask = np.abs(y_true) > 1e-8
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0
    else:
        mape = float("nan")

    return {
        "loss": float(mse),
        "mae": float(abs_err.mean()),
        "mape": float(mape),
        "max_error": float(abs_err.max()),
    }


def train_and_save(
    csv_path: str,
    model_path: str,
    meta_path: str,
    layer_sizes=(32, 32, 16),
    epochs: int = 100,
    batch_size: int = 32,
    test_size: float = 0.2,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, feature_columns = load_dataset(csv_path)

    n = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = build_mlp(n_inputs=X.shape[1], layer_sizes=layer_sizes)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)

        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_predictions = model(torch.tensor(X_test, dtype=torch.float32))
            val_loss = criterion(val_predictions, torch.tensor(y_test, dtype=torch.float32)).item()

        print(
            f"Epoch {epoch + 1:03d}/{epochs} - "
            f"loss: {epoch_loss:.6f} - val_loss: {val_loss:.6f}"
        )

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()

    results = evaluate_regression(y_test, y_pred)
    print("\n=== Validation Results ===")
    print(f"  loss: {results['loss']:.6f}")
    print(f"  mae:  {results['mae']:.6f}")
    print(f"  mape: {results['mape']:.6f}")
    print(f"  Max:  {results['max_error']:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_inputs": X.shape[1],
            "layer_sizes": tuple(layer_sizes),
            "feature_columns": feature_columns,
            "target_column": "makespan",
            "source_csv": csv_path,
        },
        model_path,
    )
    print(f"\nModell gespeichert unter: {model_path}")
    # Bounds for MIP, ATM not very big deal
    meta = {
        "input_order": feature_columns,
        "feature_min": X.min(axis=0).tolist(),
        "feature_max": X.max(axis=0).tolist(),
        "target_name": "makespan",
        "target_min": float(y.min()),
        "target_max": float(y.max()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadaten gespeichert unter: {meta_path}")

    return model, meta


if __name__ == "__main__":
    csv_path = os.path.join(ROOT_DIR, "data", "fixed_y_results", "i5_k5_1.csv")
    model_path = os.path.join(ROOT_DIR, "NN Modell", "i5_k5_1.pt")
    meta_path = os.path.join(ROOT_DIR, "NN Modell", "i5_k5_1_meta.json")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_and_save(
        csv_path=csv_path,
        model_path=model_path,
        meta_path=meta_path,
        layer_sizes=(32, 32, 16),
        epochs=100,
        batch_size=32,
    )
