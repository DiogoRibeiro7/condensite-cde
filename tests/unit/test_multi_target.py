from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDEConfig, MultiTargetCondensite

pytestmark = pytest.mark.unit


def _make_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 2))
    y1 = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * rng.normal(size=X.shape[0])
    y2 = -0.2 * X[:, 0] + 0.4 * X[:, 1] + 0.1 * rng.normal(size=X.shape[0])
    Y = np.stack([y1, y2], axis=1)
    return X, Y


def test_independent_mode_shapes_and_determinism() -> None:
    X, Y = _make_dataset()
    config = CondensiteTorchCDEConfig(epochs=3, patience=2, m_aux=12, sampler="sobol")
    model = MultiTargetCondensite(config, mode="independent", random_seed=5).fit(X, Y)
    grid = np.linspace(Y.min() - 1.0, Y.max() + 1.0, 32)
    pdf = model.predict_density(X[:4], grid)
    assert pdf.shape == (4, 2, grid.size)
    quantiles = model.predict_quantile(X[:4], [0.1, 0.9], y_grid=grid)
    assert quantiles.shape == (4, 2, 2)
    samples_a = model.sample(X[:3], n_samples=6, seed=7)
    samples_b = model.sample(X[:3], n_samples=6, seed=7)
    assert samples_a.shape == (3, 6, 2)
    assert np.allclose(samples_a, samples_b)


def test_autoregressive_mode_requires_context_and_uses_history() -> None:
    X, Y = _make_dataset(seed=2)
    config = CondensiteTorchCDEConfig(epochs=3, patience=2, m_aux=10, sampler="sobol")
    model = MultiTargetCondensite(config, mode="autoregressive", random_seed=1).fit(X, Y)
    grid = np.linspace(Y.min() - 1.0, Y.max() + 1.0, 32)
    context = Y[:5]
    pdf_good = model.predict_density(X[:5], grid, y_context=context)
    assert pdf_good.shape == (5, 2, grid.size)
    swapped = context[:, ::-1]
    pdf_swapped = model.predict_density(X[:5], grid, y_context=swapped)
    assert not np.allclose(pdf_good[:, 1, :], pdf_swapped[:, 1, :])
    with pytest.raises(ValueError):
        model.predict_density(X[:5], grid)


def test_shared_mode_uses_single_estimator() -> None:
    X, Y = _make_dataset(seed=4)
    config = CondensiteTorchCDEConfig(epochs=3, patience=1, m_aux=10, sampler="sobol")
    model = MultiTargetCondensite(config, mode="shared", random_seed=3).fit(X, Y)
    assert model.mode == "shared"
    assert model._shared_estimator is not None
    grid = np.linspace(Y.min() - 0.5, Y.max() + 0.5, 32)
    pdf = model.predict_density(X[:6], grid)
    assert pdf.shape == (6, 2, grid.size)
    quantiles = model.predict_quantile(X[:6], [0.25, 0.75], y_grid=grid)
    assert quantiles.shape == (6, 2, 2)
    samples = model.sample(X[:2], n_samples=4, seed=9)
    assert samples.shape == (2, 4, 2)
