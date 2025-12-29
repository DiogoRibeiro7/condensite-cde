from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - depends on runner environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import (
    CondensiteTorchCDE,
    CondensiteTorchCDEConfig,
    crps_from_cdf,
    nll_from_pdf,
    sample_yprime,
)

CDF_MONOTONIC_TOL = 1e-4


def test_sample_yprime_is_deterministic_with_seed() -> None:
    a = sample_yprime("stratified", (2, 3), seed=123)
    b = sample_yprime("stratified", (2, 3), seed=123)
    c = sample_yprime("stratified", (2, 3), seed=321)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_condensite_training_produces_valid_pdf_and_cdf() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 2))
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * rng.normal(size=64)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=8,
        epochs=6,
        patience=3,
        batch_size=16,
        lr=5e-3,
        bandwidth=0.2,
        sampler="sobol",
        amp=False,
        positive_output=True,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=7)
    estimator.fit(X, y)
    grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 64)
    pdf = estimator.predict_density(X[:3], grid)
    areas = np.trapz(pdf, x=grid, axis=1)
    assert np.allclose(areas, 1.0, atol=0.15)
    assert np.all(pdf >= 0)
    cdf = estimator.predict_cdf(X[:3], grid)
    assert np.allclose(cdf[:, 0], 0.0, atol=1e-4)
    assert np.allclose(cdf[:, -1], 1.0, atol=1e-3)
    assert np.all(np.diff(cdf, axis=1) >= -CDF_MONOTONIC_TOL)
    samples = estimator.sample(X[:1], 5, y_grid=grid, seed=5)
    assert samples.shape == (1, 5)
    assert np.all(samples <= grid.max() + 1e-6)


def test_metrics_return_finite_values() -> None:
    y_grid = np.linspace(-1, 1, 32)
    base_pdf = np.exp(-0.5 * y_grid**2)
    base_pdf /= np.trapz(base_pdf, x=y_grid)
    pdf = np.vstack([base_pdf, base_pdf])
    cdf = np.concatenate(
        [np.zeros((2, 1)), np.cumsum(0.5 * (pdf[:, 1:] + pdf[:, :-1]) * np.diff(y_grid), axis=1)],
        axis=1,
    )
    cdf[:, -1] = 1.0
    y_true = np.array([0.1, -0.2])
    assert nll_from_pdf(y_true, y_grid, pdf) > 0
    assert crps_from_cdf(y_true, y_grid, cdf) >= 0
