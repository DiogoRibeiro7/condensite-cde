from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDEConfig, EnsembleCondensite

pytestmark = pytest.mark.unit


def _dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 2))
    y = 0.4 * X[:, 0] - 0.1 * X[:, 1] + rng.normal(scale=0.05, size=80)
    return X, y


def test_ensemble_density_mean_and_variance_shapes() -> None:
    X, y = _dataset()
    config = CondensiteTorchCDEConfig(epochs=2, patience=1, m_aux=8, sampler="sobol")
    ensemble = EnsembleCondensite(config, n_models=3, random_seed=5).fit(X, y)
    grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 32)
    mean, var = ensemble.predict_density(X[:4], grid)
    assert mean.shape == (4, grid.size)
    assert var.shape == (4, grid.size)
    assert np.all(var >= 0.0)


def test_ensemble_variance_positive_for_quantiles() -> None:
    X, y = _dataset(seed=1)
    config = CondensiteTorchCDEConfig(epochs=2, patience=1, m_aux=8, sampler="sobol")
    ensemble = EnsembleCondensite(config, n_models=2, random_seed=1, bootstrap=True).fit(X, y)
    grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 32)
    mean, var = ensemble.predict_quantile(X[:5], [0.5], y_grid=grid)
    assert mean.shape == (5, 1)
    assert var.shape == (5, 1)
    assert np.all(var >= 0.0)
