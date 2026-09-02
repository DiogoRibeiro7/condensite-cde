from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment specific
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, TabularPreprocessorConfig

_CATEGORICAL_THRESHOLD = 0.5
_MISSING_PROBABILITY = 0.1


def test_mixed_tabular_training_with_missing_values() -> None:
    rng = np.random.default_rng(0)
    n = 120
    numeric = rng.normal(size=(n, 2))
    categorical = np.where(
        rng.random(size=(n, 1)) > _CATEGORICAL_THRESHOLD,
        "red",
        "blue",
    ).astype(object)
    categorical[rng.random(size=categorical.shape) < _MISSING_PROBABILITY] = None
    features = np.concatenate([numeric, categorical], axis=1)
    targets = (
        0.4 * numeric[:, 0]
        - 0.2 * numeric[:, 1]
        + (categorical[:, 0] == "red").astype(float)
        + 0.1 * rng.normal(size=n)
    )

    config = CondensiteTorchCDEConfig(
        hidden_sizes=(16, 16),
        m_aux=16,
        epochs=4,
        patience=2,
        batch_size=32,
        sampler="sobol",
        bandwidth=0.1,
        preprocessor=TabularPreprocessorConfig(add_missing_indicator=True),
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=5)
    estimator.fit(features, targets)
    grid = np.linspace(targets.min() - 0.5, targets.max() + 0.5, 48)
    pdf = estimator.predict_density(features[:5], grid)
    assert pdf.shape == (5, grid.size)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0.0)
