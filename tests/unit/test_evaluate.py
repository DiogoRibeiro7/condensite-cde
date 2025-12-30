from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_evaluate_returns_metrics(trained_estimator, torch_available) -> None:
    assert torch_available
    estimator, X, y, _ = trained_estimator
    metrics = estimator.evaluate(X[:10], y[:10])
    assert set(metrics) == {"nll", "crps", "integral_error"}
    assert metrics["integral_error"] >= 0.0
    assert np.isfinite(metrics["nll"])
    assert np.isfinite(metrics["crps"])


def test_evaluate_validates_shapes(trained_estimator, torch_available) -> None:
    assert torch_available
    estimator, X, y, _ = trained_estimator
    with pytest.raises(ValueError):
        estimator.evaluate(X[:5], y[:4])
