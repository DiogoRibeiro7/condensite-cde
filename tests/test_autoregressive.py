from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on env
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDEConfig
from condensite_torch.autoregressive import AutoregressiveCondensite


def _make_multivariate_dataset(n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(n_samples, 2))
    y1 = 0.5 * np.sin(X[:, 0]) + 0.3 * X[:, 1] + 0.1 * rng.normal(size=n_samples)
    y2 = y1 + 0.2 * X[:, 0] - 0.1 * X[:, 1] + 0.1 * rng.normal(size=n_samples)
    Y = np.stack([y1, y2], axis=1)
    return X, Y


def test_autoregressive_sampling_is_deterministic_and_correlated() -> None:
    X, Y = _make_multivariate_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=24,
        epochs=4,
        patience=2,
        sampler="sobol",
        bandwidth=0.12,
    )
    ar_model = AutoregressiveCondensite(config, random_seed=3).fit(X, Y)
    X_test = X[:12]
    samples_a = ar_model.sample(X_test, n_samples=8, seed=5)
    samples_b = ar_model.sample(X_test, n_samples=8, seed=5)
    samples_c = ar_model.sample(X_test, n_samples=8, seed=15)
    assert samples_a.shape == (X_test.shape[0], 8, 2)
    assert np.allclose(samples_a, samples_b)
    assert not np.allclose(samples_a, samples_c)
    corr = np.corrcoef(samples_a[:, :, 0].ravel(), samples_a[:, :, 1].ravel())[0, 1]
    assert corr > 0.3


def test_predict_marginal_quantile_requires_history_for_later_dims() -> None:
    X, Y = _make_multivariate_dataset()
    config = CondensiteTorchCDEConfig(epochs=3, patience=2, m_aux=16, sampler="sobol")
    ar_model = AutoregressiveCondensite(config, random_seed=2).fit(X, Y)
    with pytest.raises(ValueError):
        ar_model.predict_marginal_quantile(X[:5], dim=1, q=0.5)
    medians_first = ar_model.predict_marginal_quantile(X[:5], dim=0, q=0.5)
    assert medians_first.shape == (5,)
    quantiles_second = ar_model.predict_marginal_quantile(X[:5], dim=1, q=[0.1, 0.9], y_prefix=Y[:5, :1])
    assert quantiles_second.shape == (5, 2)
