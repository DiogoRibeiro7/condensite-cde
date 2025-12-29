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


def test_conformal_wrapper_reaches_target_coverage() -> None:
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
    wrapper = ConformalCDEWrapper(config, random_seed=7).fit(X_train, y_train, X_cal, y_cal)
    lower, upper = wrapper.predict_interval(X_test, coverage=0.9)
    assert lower.shape == upper.shape == (X_test.shape[0],)
    covered = ((y_test >= lower) & (y_test <= upper)).mean()
    assert 0.82 <= covered <= 0.97
