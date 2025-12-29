"""Showcase decision-centric metrics: quantiles, tail probabilities, expected shortfall."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment dependent
    print(f"Torch unavailable: {exc}")
    sys.exit(0)

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


def make_dataset(n_samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(n_samples, 3))
    noise = (0.1 + 0.2 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.6 * np.sin(X[:, 0]) - 0.4 * X[:, 1] + 0.2 * X[:, 2] + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(48, 48),
        m_aux=96,
        epochs=8,
        patience=3,
        sampler="sobol",
        bandwidth=0.1,
        normalization_lambda=0.2,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=21).fit(X, y)
    grid = make_y_grid(y, grid_size=96, mode="quantile")

    samples = X[:3]
    quantiles = estimator.predict_quantile(samples, [0.1, 0.5, 0.9], y_grid=grid)
    right_tail = estimator.predict_tail_prob(samples, threshold=y.mean(), y_grid=grid)
    es = estimator.expected_shortfall(samples, alpha=0.9, y_grid=grid)

    for idx in range(samples.shape[0]):
        q10, q50, q90 = quantiles[idx]
        print(f"Sample {idx}: q10={q10:.3f}, q50={q50:.3f}, q90={q90:.3f}")
        print(f"  Right-tail prob above mean: {right_tail[idx]:.3f}")
        print(f"  Expected shortfall (alpha=0.9): {es[idx]:.3f}")


if __name__ == "__main__":
    main()
