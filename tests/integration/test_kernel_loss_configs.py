"""Integration tests for pluggable kernels/losses."""

from __future__ import annotations

import numpy as np
import pytest

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.integration


def _toy_dataset(n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 2))
    y = 0.8 * np.sin(X[:, 0]) - 0.3 * X[:, 1] + 0.2 * rng.normal(size=n)
    return X, y


def test_mae_loss_decreases_training_history() -> None:
    X, y = _toy_dataset()
    config = CondensiteTorchCDEConfig(
        loss="mae",
        epochs=4,
        patience=4,
        m_aux=16,
        batch_size=32,
        sampler="sobol",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=3).fit(X, y)
    history = estimator.training_history
    assert len(history) >= 2
    assert history[0]["train_loss"] > history[-1]["train_loss"]


def test_epanechnikov_kernel_integrates_close_to_one() -> None:
    X, y = _toy_dataset()
    config = CondensiteTorchCDEConfig(
        kernel="epanechnikov",
        epochs=3,
        patience=2,
        m_aux=20,
        batch_size=32,
        sampler="sobol",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=5).fit(X, y)
    grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 72)
    pdf = estimator.predict_density(X[:10], grid)
    masses = np.trapezoid(pdf, x=grid, axis=1)
    assert np.allclose(masses, 1.0, atol=0.05)
