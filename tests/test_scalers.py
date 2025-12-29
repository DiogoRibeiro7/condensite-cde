"""Unit tests for NumPy-based scalers."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from condensite_torch.scalers import MinMaxScaler1D, StandardScaler


def test_minmax_scaler_round_trip() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(loc=2.0, scale=0.5, size=64)
    scaler = MinMaxScaler1D().fit(y)
    transformed = scaler.transform(y)
    restored = scaler.inverse_transform(transformed)
    assert transformed.min() >= 0.0
    assert transformed.max() <= 1.0
    assert_allclose(restored, y, atol=1e-8)


def test_standard_scaler_outputs_unit_variance() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(32, 4))
    scaler = StandardScaler().fit(X)
    transformed = scaler.transform(X)
    assert transformed.shape == X.shape
    means = transformed.mean(axis=0)
    stds = transformed.std(axis=0)
    assert_allclose(means, np.zeros_like(means), atol=1e-6)
    assert_allclose(stds, np.ones_like(stds), atol=1e-6)
