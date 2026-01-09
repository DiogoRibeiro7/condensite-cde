"""Demonstrate probabilistic cross-validation with Condensite."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from condensite_cde import cross_validate
from condensite_torch import CondensiteTorchCDEConfig


def _make_dataset(n_samples: int = 180) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n_samples, 3))
    noise = 0.1 * rng.normal(size=n_samples)
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2] + noise
    return X, y


def main() -> None:
    X, y = _make_dataset()
    config = CondensiteTorchCDEConfig(
        epochs=6,
        patience=3,
        m_aux=24,
        batch_size=64,
        sampler="sobol",
    )
    result = cross_validate(
        config,
        X,
        y,
        cv=3,
        metrics=("nll", "crps", "coverage"),
        seed=21,
        json_path=Path("reports/cross_validation.json"),
    )
    print("Mean metrics:", result.metrics_mean)
    print("Std metrics:", result.metrics_std)


if __name__ == "__main__":
    main()
