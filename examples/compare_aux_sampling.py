"""Compare auxiliary sampling strategies on a toy dataset."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:
    print(f"Torch is unavailable; skipping example. Details: {exc}")
    sys.exit(0)

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, nll_from_pdf


def make_dataset(n_samples: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(21)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.05 + 0.2 * np.cos(X[:, 1])) * rng.normal(size=n_samples)
    y = 0.6 * np.sin(X[:, 0]) + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    split = int(0.75 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    grid = make_y_grid(y_train, grid_size=80, mode="quantile")

    methods = ["iid", "stratified", "lhs", "sobol", "fixed_grid"]
    for method in methods:
        config = CondensiteTorchCDEConfig(
            sampler=method,
            m_aux=64,
            epochs=5,
            patience=2,
            bandwidth=0.1,
            hidden_sizes=(48, 48),
        )
        estimator = CondensiteTorchCDE(config=config, random_seed=5).fit(X_train, y_train)
        pdf = estimator.predict_density(X_val, grid)
        score = nll_from_pdf(y_val, grid, pdf)
        print(f"{method:>10s} sampling -> NLL {score:.4f}")


if __name__ == "__main__":
    main()
