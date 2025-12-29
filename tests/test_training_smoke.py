"""Smoke test that small training runs complete."""

from __future__ import annotations

import numpy as np


def test_quick_training_predicts_finite_pdf(trained_estimator) -> None:
    estimator, X, _, grid = trained_estimator
    pdf = estimator.predict_density(X[:2], grid)
    assert pdf.shape == (2, grid.shape[0])
    assert np.all(np.isfinite(pdf))
