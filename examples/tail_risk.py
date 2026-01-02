"""Tail probability and expected shortfall example."""

from __future__ import annotations

import numpy as np

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


def make_dataset(n_samples: int = 220) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.15 + 0.1 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * np.sin(X[:, 0]) - 0.25 * X[:, 1] + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    split = int(0.8 * len(X))
    X_train, X_eval = X[:split], X[split:]
    y_train, y_eval = y[:split], y[split:]
    grid = make_y_grid(y_train, grid_size=96, mode="quantile")

    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=40,
        epochs=6,
        patience=2,
        sampler="sobol",
        normalization_lambda=0.05,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=21).fit(X_train, y_train)

    threshold = np.percentile(y_eval, 95)
    tail_probs = estimator.predict_tail_prob(X_eval[:5], threshold=threshold, y_grid=grid)
    es_values = estimator.expected_shortfall(X_eval[:5], alpha=0.95, y_grid=grid)

    for idx, (prob, es) in enumerate(zip(tail_probs, es_values, strict=True)):
        print(f"Row {idx}: P(Y >= {threshold:+.3f}) ~ {prob:.4f}, ES_0.95 ~ {es:+.4f}")


if __name__ == "__main__":
    main()
