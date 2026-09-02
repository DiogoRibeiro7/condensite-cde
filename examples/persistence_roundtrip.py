"""Save/load example demonstrating evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


def make_dataset(n_samples: int = 180) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.15 + 0.1 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.3 * np.sin(X[:, 0]) - 0.2 * X[:, 1] + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    split = int(0.75 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    grid = make_y_grid(y_train, grid_size=96, mode="quantile")

    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=32,
        epochs=6,
        patience=2,
        sampler="sobol",
        monitor_metric="val_crps",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=0).fit(X_train, y_train)
    metrics = estimator.evaluate(X_val, y_val, y_grid=grid)
    print("Validation metrics:", metrics)

    save_dir = Path("artifacts") / "condensite_demo"
    estimator.save(save_dir)
    restored = CondensiteTorchCDE.load(save_dir)
    restored_metrics = restored.evaluate(X_val, y_val, y_grid=grid)
    print("Restored metrics:", restored_metrics)
    tolerance = 1e-5
    assert all(abs(metrics[key] - restored_metrics[key]) < tolerance for key in metrics), (
        "Round-trip should preserve predictions."
    )


if __name__ == "__main__":
    main()
