from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDEConfig
from condensite_torch.conformal import ConformalCDEWrapper


def _make_dataset(n_samples: int = 320) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 3))
    noise = (0.15 + 0.1 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.4 * np.sin(X[:, 0]) - 0.25 * X[:, 1] + 0.2 * X[:, 2] + noise
    return X, y


@pytest.mark.parametrize("method", ["quantile", "cdf"])
def test_conformal_wrapper_reaches_target_coverage(method: str) -> None:  # noqa: PLR0914
    X, y = _make_dataset()
    train_end = 180
    cal_end = 260
    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X[cal_end:], y[cal_end:]

    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=32,
        epochs=5,
        patience=2,
        batch_size=64,
        bandwidth=0.12,
        sampler="sobol",
    )
    wrapper = ConformalCDEWrapper(config, random_seed=7).fit(
        X_train,
        y_train,
        X_cal,
        y_cal,
        method=method,
    )
    grid = np.linspace(y_train.min() - 0.5, y_train.max() + 0.5, 96)
    target_coverage = 0.9
    lower, upper = wrapper.predict_interval(X_test, coverage=target_coverage, y_grid=grid)
    assert lower.shape == upper.shape == (X_test.shape[0],)
    covered = ((y_test >= lower) & (y_test <= upper)).mean()
    lower_bound = 0.82
    assert lower_bound <= covered <= 1.0 + 1e-6


def test_conformal_wrapper_rejects_unknown_method() -> None:
    X, y = _make_dataset(32)
    config = CondensiteTorchCDEConfig(hidden_sizes=(8,), epochs=1, m_aux=4, batch_size=16)
    wrapper = ConformalCDEWrapper(config, random_seed=0)
    with pytest.raises(ValueError):
        wrapper.fit(X[:16], y[:16], X[16:24], y[16:24], method="bogus")  # type: ignore[arg-type]
