from __future__ import annotations

import numpy as np
import pytest

from condensite_torch.scalers import MinMaxScaler1D, StandardScaler

pytestmark = pytest.mark.unit


def test_minmax_scaler_maps_bounds_to_unit_interval() -> None:
    data = np.array([-2.0, 0.0, 2.0, 4.0])
    scaler = MinMaxScaler1D().fit(data)
    transformed = scaler.transform(data)
    assert np.isclose(transformed.min(), 0.0)
    assert np.isclose(transformed.max(), 1.0)
    reconstructed = scaler.inverse_transform(transformed)
    assert np.allclose(reconstructed, data, atol=1e-10)


def test_minmax_scaler_handles_constant_input() -> None:
    data = np.ones(5)
    scaler = MinMaxScaler1D().fit(data)
    transformed = scaler.transform(data)
    assert np.all((transformed >= 0.0) & (transformed <= 1.0))


def test_standard_scaler_normalizes_each_feature() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3))
    scaler = StandardScaler().fit(X)
    transformed = scaler.transform(X)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-6)


def test_standard_scaler_handles_near_constant_columns() -> None:
    X = np.column_stack([
        np.linspace(-1.0, 1.0, 10),
        np.full(10, 5.0) + 1e-8,
    ])
    scaler = StandardScaler().fit(X)
    transformed = scaler.transform(X)
    assert np.all(np.isfinite(transformed))
