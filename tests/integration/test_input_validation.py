"""Integration tests for estimator input validation."""

from __future__ import annotations

import numpy as np
import pytest

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig
from condensite_torch.validation import SchemaConstraints, ValidationError

pytestmark = pytest.mark.integration


def _toy_dataset(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 2))
    y = 0.4 * X[:, 0] - 0.1 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_fit_rejects_targets_out_of_bounds() -> None:
    X, y = _toy_dataset()
    schema = SchemaConstraints(y_min=-0.5, y_max=0.5)
    y_bad = y.copy()
    y_bad[0] = 2.0
    config = CondensiteTorchCDEConfig(
        epochs=2,
        patience=1,
        m_aux=16,
        batch_size=32,
        input_schema=schema,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=7)
    with pytest.raises(ValidationError):
        estimator.fit(X, y_bad)


def test_predict_rejects_missing_numeric_when_disallowed() -> None:
    X, y = _toy_dataset()
    schema = SchemaConstraints(numeric_indices=[0, 1], allow_missing_numeric=False)
    config = CondensiteTorchCDEConfig(
        epochs=3,
        patience=1,
        m_aux=16,
        batch_size=32,
        input_schema=schema,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=8).fit(X, y)
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 64)
    with pytest.raises(ValidationError):
        estimator.predict_density(X_bad, grid)
