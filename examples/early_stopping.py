"""Demonstrate validation-driven early stopping with CondensiteTorchCDE."""

from __future__ import annotations

import numpy as np

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, nll_from_pdf


def make_dataset(n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.1 + 0.25 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.4 * np.sin(X[:, 0]) - 0.2 * X[:, 1] + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    split = int(0.75 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=32,
        epochs=20,
        patience=3,
        batch_size=64,
        sampler="sobol",
        monitor_metric="val_nll",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=5)
    estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    history = estimator.training_history
    best_epoch = estimator._best_epoch
    print(f"Training stopped after {len(history)} epochs (requested {config.epochs}).")
    print(f"Best epoch: {best_epoch} with val_nll={history[best_epoch]['val_nll']:.4f}")
    print(f"Restored checkpoint epoch: {estimator._restored_best_epoch}")

    grid = make_y_grid(y_train, grid_size=96, mode="quantile")
    pdf = estimator.predict_density(X_val, grid)
    val_nll = nll_from_pdf(y_val, grid, pdf)
    print(f"NLL of restored model on validation set: {val_nll:.4f}")


if __name__ == "__main__":
    main()
