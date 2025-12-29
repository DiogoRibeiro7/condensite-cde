"""Train Condensite Torch on a heteroscedastic synthetic dataset."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:
    print(f"Torch is unavailable; skipping example. Details: {exc}")
    sys.exit(0)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


def make_dataset(n_samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    noise_scale = 0.15 + 0.35 * (np.sin(X[:, 0]) ** 2)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + noise_scale * rng.normal(size=n_samples)
    return X, y


def main() -> None:
    X, y = make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(64, 64),
        m_aux=128,
        bandwidth=0.1,
        epochs=10,
        patience=3,
        sampler="sobol",
        amp=False,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=17).fit(X, y)
    grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 100)
    pdf = estimator.predict_density(X[:2], grid)
    print("PDF summary for two samples:")
    for idx, row in enumerate(pdf):
        mass = np.trapz(row, grid)
        print(f"  Sample {idx}: mass={mass:.3f}, peak={row.max():.3f}")


if __name__ == "__main__":
    main()
