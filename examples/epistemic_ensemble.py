"""Demonstrate ensemble-based epistemic uncertainty."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    print(f"Torch unavailable: {exc}")
    sys.exit(0)

from condensite_torch import CondensiteTorchCDEConfig, EnsembleCondensite


def make_dataset(n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 2))
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * rng.normal(size=n_samples)
    return X, y


def main() -> None:
    X, y = make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=24,
        epochs=4,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
    )
    ensemble = EnsembleCondensite(config, n_models=3, bootstrap=True, random_seed=4).fit(X, y)
    grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 48)
    mean_pdf, var_pdf = ensemble.predict_density(X[:2], grid)
    mean_quant, var_quant = ensemble.predict_quantile(X[:2], [0.5], y_grid=grid)
    for idx in range(2):
        print(f"Sample {idx}: density var mean={var_pdf[idx].mean():.4f}")
        print(f"  Median mean={mean_quant[idx, 0]:.3f}, var={var_quant[idx, 0]:.6f}")


if __name__ == "__main__":
    main()
