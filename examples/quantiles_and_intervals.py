"""Quantile and predictive interval demonstration."""

from __future__ import annotations

import numpy as np

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


def make_dataset(n_samples: int = 240) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.1 + 0.15 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.4 * np.sin(X[:, 0]) - 0.3 * X[:, 1] + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    split = int(0.75 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train = y[:split]
    grid = make_y_grid(y_train, grid_size=128, mode="quantile")

    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=48,
        epochs=6,
        patience=2,
        sampler="sobol",
        normalization_lambda=0.1,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=12).fit(X_train, y_train)

    q_probs = [0.1, 0.5, 0.9]
    quantiles = estimator.predict_quantile(X_val[:5], q_probs, y_grid=grid)
    interval_low, interval_hi = estimator.predict_interval(X_val[:5], coverage=0.9, y_grid=grid)

    for idx, row in enumerate(quantiles):
        print(f"Row {idx}:")
        for prob, value in zip(q_probs, row, strict=True):
            print(f"  q={prob:.1f}: {value:+.4f}")
        print(f"  90% interval: ({interval_low[idx]:+.4f}, {interval_hi[idx]:+.4f})")


if __name__ == "__main__":
    main()
