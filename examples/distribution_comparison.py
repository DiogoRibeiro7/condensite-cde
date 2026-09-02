"""Compare two models using Wasserstein/Kolmogorov-Smirnov/JS metrics."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    print(f"Torch unavailable: {exc}")
    sys.exit(0)

from condensite_torch import (
    CondensiteTorchCDE,
    CondensiteTorchCDEConfig,
    js_divergence,
    ks_distance,
    wasserstein_1,
)


def make_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(240, 2))
    y = 0.4 * np.sin(X[:, 0]) - 0.3 * X[:, 1] + 0.1 * rng.normal(size=X.shape[0])
    return X, y


def main() -> None:
    X, y = make_dataset()
    split = 180
    config_a = CondensiteTorchCDEConfig(
        hidden_sizes=(24, 24),
        m_aux=24,
        epochs=5,
        patience=2,
        sampler="sobol",
    )
    config_b = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=32,
        epochs=6,
        patience=2,
        sampler="sobol",
    )
    model_a = CondensiteTorchCDE(config=config_a, random_seed=3).fit(X[:split], y[:split])
    model_b = CondensiteTorchCDE(config=config_b, random_seed=4).fit(X[:split], y[:split])
    grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 64)
    pdf_a = model_a.predict_density(X[split:], grid)
    pdf_b = model_b.predict_density(X[split:], grid)
    cdf_a = model_a.predict_cdf(X[split:], grid)
    cdf_b = model_b.predict_cdf(X[split:], grid)
    w1 = wasserstein_1(cdf_a, cdf_b, grid)
    ks = ks_distance(cdf_a, cdf_b)
    js = js_divergence(pdf_a, pdf_b, grid)
    print(f"Wasserstein-1 distance: {w1:.4f}")
    print(f"KS distance: {ks:.4f}")
    print(f"JS divergence: {js:.4f}")


if __name__ == "__main__":
    main()
