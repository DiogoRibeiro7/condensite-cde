"""Integration tests for batched inference."""

from __future__ import annotations

import numpy as np
import pytest

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.integration


def _make_dataset(n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, 3))
    y = 0.5 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_chunked_inference_matches_full_result() -> None:
    X, y = _make_dataset()
    grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 64)
    config = CondensiteTorchCDEConfig(
        epochs=3,
        patience=2,
        m_aux=16,
        sampler="sobol",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=9).fit(X, y)
    baseline = estimator.predict_density(X, grid)
    estimator.config.inference_batch_size = 16
    estimator.config.inference_grid_chunk_size = 20
    chunked = estimator.predict_density(X, grid)
    assert np.allclose(baseline, chunked, atol=1e-6)
