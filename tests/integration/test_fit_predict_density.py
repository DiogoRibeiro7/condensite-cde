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
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 2))
    noise_scale = 0.1 + 0.25 * (np.sin(X[:, 0]) ** 2)
    y = np.sin(X[:, 0]) - 0.3 * X[:, 1] + noise_scale * rng.normal(size=n_samples)
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def test_fit_predict_density_produces_valid_pdf() -> None:
    X_train, y_train, X_test, y_test = _make_dataset()
    grid = np.linspace(y_train.min() - 0.5, y_train.max() + 0.5, 64)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=24,
        epochs=4,
        patience=2,
        sampler="sobol",
        bandwidth=0.12,
        positive_output=True,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=4).fit(X_train, y_train)
    pdf = estimator.predict_density(X_test, grid)
    assert pdf.shape == (X_test.shape[0], grid.size)
    assert np.all(pdf >= 0.0)
    mass = np.trapezoid(pdf, x=grid, axis=1)
    assert np.all(np.isfinite(pdf))
    assert np.allclose(mass, 1.0, atol=0.02)
