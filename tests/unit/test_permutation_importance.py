from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import permutation_importance

pytestmark = pytest.mark.unit


def test_permutation_importance_shapes_and_determinism(trained_estimator) -> None:
    estimator, X, y, grid = trained_estimator
    result = permutation_importance(
        estimator,
        X,
        y,
        metric="crps",
        n_repeats=3,
        random_seed=5,
        y_grid=grid,
    )
    assert result.metric_name == "crps"
    n_features = X.shape[1]
    assert result.importances_mean.shape == (n_features,)
    assert result.importances_std.shape == (n_features,)
    assert result.raw_importances.shape == (n_features, 3)

    repeat = permutation_importance(
        estimator,
        X,
        y,
        metric="crps",
        n_repeats=3,
        random_seed=5,
        y_grid=grid,
    )
    assert np.allclose(result.importances_mean, repeat.importances_mean)
    assert np.allclose(result.importances_std, repeat.importances_std)
    assert np.allclose(result.raw_importances, repeat.raw_importances)


def test_permutation_importance_supports_nll_metric(trained_estimator) -> None:
    estimator, X, y, grid = trained_estimator
    result = permutation_importance(
        estimator,
        X,
        y,
        metric="nll",
        n_repeats=2,
        random_seed=1,
        y_grid=grid,
    )
    assert result.metric_name == "nll"
    assert np.isfinite(result.baseline_score)
    assert result.importances_mean.shape[0] == X.shape[1]
