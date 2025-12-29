from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on runner environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.integration


def _make_dataset(n_samples: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_samples, 2))
    noise = 0.2 * rng.normal(size=n_samples)
    y = 0.4 * np.sin(X[:, 0]) - 0.2 * X[:, 1] + noise
    split = int(0.75 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def test_predict_cdf_monotone_and_proper() -> None:
    X_train, y_train, X_test, _ = _make_dataset()
    grid = np.linspace(y_train.min() - 0.4, y_train.max() + 0.4, 72)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=20,
        epochs=4,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=3).fit(X_train, y_train)
    cdf = estimator.predict_cdf(X_test, grid)
    diffs = np.diff(cdf, axis=1)
    assert np.all(diffs >= -1e-6)
    assert np.allclose(cdf[:, 0], 0.0, atol=1e-2)
    assert np.allclose(cdf[:, -1], 1.0, atol=1e-2)
