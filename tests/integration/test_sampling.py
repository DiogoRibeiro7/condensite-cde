from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.integration


def _make_dataset(n_samples: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(n_samples, 2))
    y = 0.6 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * rng.normal(size=n_samples)
    split = int(0.7 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def test_sampling_respects_bounds_and_seed() -> None:
    X_train, y_train, X_test, _ = _make_dataset()
    grid = np.linspace(y_train.min() - 0.3, y_train.max() + 0.3, 60)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(24, 24),
        m_aux=16,
        epochs=3,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=6).fit(X_train, y_train)
    samples_a = estimator.sample(X_test, n_samples=50, y_grid=grid, seed=10)
    samples_b = estimator.sample(X_test, n_samples=50, y_grid=grid, seed=10)
    samples_c = estimator.sample(X_test, n_samples=50, y_grid=grid, seed=15)
    assert samples_a.shape == (X_test.shape[0], 50)
    assert np.all(samples_a >= grid.min() - 1e-3)
    assert np.all(samples_a <= grid.max() + 1e-3)
    assert np.allclose(samples_a, samples_b)
    assert not np.allclose(samples_a, samples_c)
