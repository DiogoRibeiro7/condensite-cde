from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on environment
    pytest.skip(f"Torch unavailable: {exc}", allow_module_level=True)

from condensite_torch import what_if

pytestmark = pytest.mark.unit


def test_what_if_shifts_median_when_feature_changes(trained_estimator) -> None:
    estimator, X, _y, grid = trained_estimator
    row = X[0]
    delta = 1.5
    result = what_if(
        estimator,
        row,
        {0: row[0] + delta},
        outputs=("quantiles",),
        quantile_probs=(0.5,),
        y_grid=grid,
    )
    baseline_median = float(result.baseline["quantiles"]["values"][0])
    modified_median = float(result.modified["quantiles"]["values"][0])
    assert modified_median != pytest.approx(baseline_median)


def test_what_if_returns_pdf_and_tail_probabilities(trained_estimator) -> None:
    estimator, X, _y, grid = trained_estimator
    row = X[1]
    result = what_if(
        estimator,
        row,
        {1: row[1] - 0.7},
        outputs=("pdf", "cdf", "tail_probs"),
        tail_thresholds=(-0.5, 0.0, 0.5),
        y_grid=grid,
    )
    baseline_pdf = result.baseline["pdf"]
    assert np.allclose(baseline_pdf["y_grid"], grid)
    assert baseline_pdf["values"].shape == grid.shape
    baseline_tail = result.baseline["tail_probs"]
    modified_tail = result.modified["tail_probs"]
    assert baseline_tail["thresholds"].shape == (3,)
    assert modified_tail["values"].shape == (3,)
