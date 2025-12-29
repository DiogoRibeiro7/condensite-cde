from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on host environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

pytestmark = pytest.mark.integration


def _make_dataset(n_samples: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n_samples, 3))
    y = 0.3 * X[:, 0] - 0.4 * X[:, 1] + 0.2 * X[:, 2] + 0.15 * rng.normal(size=n_samples)
    split = int(0.75 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def test_device_cpu_training_and_inference() -> None:
    X_train, y_train, X_test, _ = _make_dataset()
    grid = np.linspace(y_train.min() - 0.3, y_train.max() + 0.3, 50)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=16,
        epochs=3,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
        device="cpu",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=9).fit(X_train, y_train)
    pdf = estimator.predict_density(X_test, grid)
    assert pdf.shape == (X_test.shape[0], grid.size)
