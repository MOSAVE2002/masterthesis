from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def load_dataset(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    return df["makespan"].dropna().astype(float)


def plot_makespan_histogram(makespan: pd.Series, instance_name: str) -> None:
    min_value = int(makespan.min())
    max_value = int(makespan.max())
    bins = np.arange(min_value - 0.5, max_value + 1.5, 1)

    plt.figure(figsize=(10, 6))
    plt.hist(makespan, bins=bins, edgecolor="black", alpha=0.8)
    plt.title(f"Histogramm der Makespan-Werte: {instance_name}")
    plt.xlabel("Makespan")
    plt.ylabel("Haeufigkeit")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    instance_name = "i5_k5_1"
    root_dir = Path(__file__).resolve().parents[1]
    csv_path = root_dir / "data" / "fixed_y_results" / f"{instance_name}.csv"

    makespan = load_dataset(csv_path)
    plot_makespan_histogram(makespan, instance_name)
