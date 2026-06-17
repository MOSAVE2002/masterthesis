"""
Trainiert ein kleines PyTorch-Netz zur Makespan-Vorhersage.

Inputfeatures pro Sample werden aus den CSV-Spalten gelesen.
Alte, ausgeschlossene Feature-Spalten werden ignoriert.

Target:
- makespan
"""
import json
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from neuralnetwork.paths import METADATA_PATH, MODEL_DIR, MODEL_PATH, ROOT_DIR


DATA_DIR = ROOT_DIR / "data" / "fixed_y_results"
CONFIG_PATH = ROOT_DIR / "config.json"

TARGET_COLUMN = "makespan"
IGNORED_FEATURE_COLUMNS = {
    "max_machine_load_mean_setup",
    "max_job_path_mean_setup",
    "bottleneck_ratio_mean_setup",
    # Linearkombination der machine_*_load-Features (Gesamtlast / #Operationen),
    # daher kein Mehrwert (Ablation: Val MAE 1.393 -> 1.371). Entfernt.
    "mean_assigned_operation_time_mean_setup",
}
REQUIRED_FEATURE_COLUMNS = set()


class MakespanNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.network(x)


def _configured_training_data_path() -> Path:
    if not CONFIG_PATH.exists():
        return DATA_DIR

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = json.load(file)

    merged_csv_path = (
        config.get("random_fixed_y", {}).get("merged_csv_path")
    )
    if not merged_csv_path:
        return DATA_DIR

    merged_csv_path = Path(merged_csv_path)
    if merged_csv_path.is_absolute():
        return merged_csv_path

    return DATA_DIR / merged_csv_path


def load_dataset(data_path: Path):
    if data_path.is_file():
        csv_paths = [data_path]
    elif data_path.is_dir():
        csv_paths = sorted(data_path.glob("*.csv"))
    else:
        raise FileNotFoundError(f"Trainingsdaten nicht gefunden: {data_path}")

    if not csv_paths:
        raise FileNotFoundError(f"Keine CSV-Dateien in {data_path} gefunden.")

    dataframes = []
    feature_columns = None
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"{csv_path} enthaelt keine Spalte '{TARGET_COLUMN}'.")

        current_feature_columns = [
            column
            for column in df.columns
            if column != TARGET_COLUMN and column not in IGNORED_FEATURE_COLUMNS
        ]
        if not current_feature_columns:
            raise ValueError(f"{csv_path} enthaelt keine Input-Features.")
        missing_required_columns = sorted(
            REQUIRED_FEATURE_COLUMNS - set(current_feature_columns)
        )
        has_machine_count_columns = any(
            column.endswith("_assigned_operation_count")
            for column in current_feature_columns
        )
        if missing_required_columns or not has_machine_count_columns:
            missing_parts = missing_required_columns
            if not has_machine_count_columns:
                missing_parts.append("machine_*_assigned_operation_count")
            raise ValueError(
                f"{csv_path} enthaelt noch nicht das neue Feature-Schema. "
                f"Fehlend: {missing_parts}. Bitte Fixed-Y-Daten mit sample_idx=0 "
                "neu erzeugen und danach erneut trainieren."
            )

        if feature_columns is None:
            feature_columns = current_feature_columns
        elif current_feature_columns != feature_columns:
            raise ValueError(
                f"{csv_path} hat ein anderes Feature-Schema als die erste CSV. "
                "Bitte alte CSVs loeschen oder alle Daten mit demselben Schema "
                "neu erzeugen."
            )

        dataframes.append(df[[TARGET_COLUMN, *feature_columns]])

    dataset = pd.concat(dataframes, ignore_index=True)
    dataset = dataset.apply(pd.to_numeric, errors="coerce").dropna()

    x = dataset[feature_columns].to_numpy(dtype=np.float32)
    y = dataset[[TARGET_COLUMN]].to_numpy(dtype=np.float32)
    return x, y, feature_columns


def train_validation_test_split(
    x, y, validation_ratio=0.15, test_ratio=0.15, seed=42
):
    """Split numpy arrays x and y into train/val/test via a reproducible permutation."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(x), generator=generator).numpy()

    validation_size = max(1, round(len(x) * validation_ratio))
    test_size = max(1, round(len(x) * test_ratio))
    # Keep at least one training sample even on tiny datasets.
    validation_size = min(validation_size, len(x) - 2)
    test_size = min(test_size, len(x) - 1 - validation_size)

    val_idx = indices[:validation_size]
    test_idx = indices[validation_size : validation_size + test_size]
    train_idx = indices[validation_size + test_size :]

    return (
        x[train_idx],
        y[train_idx],
        x[val_idx],
        y[val_idx],
        x[test_idx],
        y[test_idx],
    )


def _standardization_stats(x_train, y_train):
    """Mean/std for inputs and target, computed on the training split only."""
    feature_mean = x_train.mean(axis=0)
    feature_std = np.maximum(x_train.std(axis=0), 1e-6)
    target_mean = float(y_train.mean())
    target_std = max(float(y_train.std()), 1e-6)
    return feature_mean, feature_std, target_mean, target_std


def _regression_metrics(prediction_raw, target_raw):
    """MSE, RMSE, MAE and R2 in raw makespan units (numpy arrays)."""
    error = prediction_raw - target_raw
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    target_variance = float(np.var(target_raw))
    r2 = 1.0 - mse / target_variance if target_variance > 0 else 0.0
    return mse, rmse, mae, r2


def _bake_normalization(model, feature_mean, feature_std, target_mean, target_std):
    """Fold input standardization and target de-standardization into the
    first/last Linear layers, so the saved model maps raw features -> raw makespan.

    First layer:  W1' = W1 / sigma (per input column), b1' = b1 - W1 @ (mu / sigma)
    Last layer:   WL' = WL * s_y,                       bL' = bL * s_y + m_y
    """
    linear_layers = [m for m in model.network if isinstance(m, nn.Linear)]
    first_layer = linear_layers[0]
    last_layer = linear_layers[-1]

    sigma = torch.tensor(feature_std, dtype=first_layer.weight.dtype)
    mu = torch.tensor(feature_mean, dtype=first_layer.weight.dtype)

    with torch.no_grad():
        # Input standardization folded into the first layer.
        w1 = first_layer.weight.data
        first_layer.bias.data = first_layer.bias.data - (w1 @ (mu / sigma))
        first_layer.weight.data = w1 / sigma  # broadcast over input columns

        # Target de-standardization folded into the last layer.
        last_layer.weight.data = last_layer.weight.data * target_std
        last_layer.bias.data = last_layer.bias.data * target_std + target_mean


def train_model(
    epochs=2000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=1e-5,
    early_stopping_patience=200,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    training_data_path = _configured_training_data_path()
    x_np, y_np, feature_columns = load_dataset(training_data_path)
    print(f"Trainingsdaten: {training_data_path}")
    print(f"Inputfeatures: {len(feature_columns)}")
    print(", ".join(feature_columns))

    (
        x_train_np,
        y_train_np,
        x_val_np,
        y_val_np,
        x_test_np,
        y_test_np,
    ) = train_validation_test_split(x_np, y_np, seed=seed)

    # Standardize using training-split statistics only (no leakage).
    feature_mean, feature_std, target_mean, target_std = _standardization_stats(
        x_train_np, y_train_np
    )

    def standardize_x(x):
        return ((x - feature_mean) / feature_std).astype(np.float32)

    def standardize_y(y):
        return ((y - target_mean) / target_std).astype(np.float32)

    x_train = torch.tensor(standardize_x(x_train_np), dtype=torch.float32)
    y_train = torch.tensor(standardize_y(y_train_np), dtype=torch.float32)
    x_val = torch.tensor(standardize_x(x_val_np), dtype=torch.float32)
    y_val = torch.tensor(standardize_y(y_val_np), dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
    )

    model = MakespanNet(input_size=len(feature_columns))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_val_mse_raw = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for batch_x, batch_y in train_loader:
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(batch_x)

        model.eval()
        with torch.no_grad():
            train_loss = train_loss_sum / len(train_dataset)
            val_prediction = model(x_val)
            # Validation loss in normalized space drives scheduling/early stop.
            val_loss = loss_fn(val_prediction, y_val).item()
            # Report MAE/MSE in raw makespan units for comparability.
            val_prediction_raw = (
                val_prediction.numpy() * target_std + target_mean
            )
            val_mse_raw, _, val_mae_raw, _ = _regression_metrics(
                val_prediction_raw, y_val_np
            )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_mae = val_mae_raw
            best_val_mse_raw = val_mse_raw
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"Train MSE (norm): {train_loss:.4f} | "
                f"Val MSE: {val_mse_raw:.4f} | "
                f"Val MAE Makespan: {val_mae_raw:.2f}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping bei Epoch {epoch} "
                f"(keine Verbesserung seit {early_stopping_patience} Epochen)."
            )
            break

    model.load_state_dict(best_state)

    # Fold the (linear) normalization into the first/last layers so the saved
    # model maps raw features -> raw makespan. This keeps the MILP embedding
    # (Gurobi/SCIP build_fjsp_with_ml) consistent without any changes there.
    model.eval()
    with torch.no_grad():
        normalized_val_raw = (
            model(x_val).numpy() * target_std + target_mean
        )
    _bake_normalization(model, feature_mean, feature_std, target_mean, target_std)
    model.eval()
    with torch.no_grad():
        baked_val_raw = model(
            torch.tensor(x_val_np, dtype=torch.float32)
        ).numpy()
    max_fold_diff = float(np.max(np.abs(baked_val_raw - normalized_val_raw)))
    print(f"Fold-Equivalenz max abs diff: {max_fold_diff:.6e}")
    assert max_fold_diff < 1e-3, (
        "Normalization folding changed the model output beyond tolerance "
        f"(max abs diff {max_fold_diff})."
    )

    # Final honest metrics on the held-out test set (raw units, baked model).
    with torch.no_grad():
        test_prediction_raw = model(
            torch.tensor(x_test_np, dtype=torch.float32)
        ).numpy()
    test_mse, test_rmse, test_mae, test_r2 = _regression_metrics(
        test_prediction_raw, y_test_np
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_PATH
    metadata_path = METADATA_PATH

    torch.save(model.state_dict(), model_path)
    metadata = {
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "input_size": len(feature_columns),
        # Standardization was applied during training and folded into the
        # saved weights, so the persisted model consumes raw features and
        # produces raw makespan.
        "input_normalization": "folded",
        "target_normalization": "folded",
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean,
        "target_std": target_std,
        "training_data_path": str(training_data_path),
        "num_samples": int(len(x_np)),
        "num_train_samples": int(len(x_train)),
        "num_validation_samples": int(len(x_val)),
        "num_test_samples": int(len(x_test_np)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse_raw,
        "best_val_mae_makespan": best_val_mae,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae_makespan": test_mae,
        "test_r2": test_r2,
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(
        f"Bestes Modell: Epoch {best_epoch} | "
        f"Val MSE: {best_val_mse_raw:.4f} | "
        f"Val MAE Makespan: {best_val_mae:.2f}"
    )
    print(
        f"Testset: MAE {test_mae:.2f} | RMSE {test_rmse:.2f} | R2 {test_r2:.4f}"
    )
    print(f"Modell gespeichert: {model_path}")
    print(f"Metadaten gespeichert: {metadata_path}")
    return model


if __name__ == "__main__":
    train_model()
