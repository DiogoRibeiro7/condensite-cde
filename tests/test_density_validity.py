"""Strict density/cdf validity checks."""

from __future__ import annotations

import numpy as np

MAX_AREA_ERROR = 0.25
CDF_END_TOL = 1e-2
CDF_MONO_TOL = -1e-4


def test_pdf_and_cdf_are_well_behaved(trained_estimator) -> None:
    estimator, X, _, grid = trained_estimator
    sample_X = X[:5]
    pdf = estimator.predict_density(sample_X, grid)
    assert np.all(pdf >= 0.0)
    areas = np.trapezoid(pdf, x=grid, axis=1)
    assert np.allclose(areas, np.ones_like(areas), atol=MAX_AREA_ERROR)

    cdf = estimator.predict_cdf(sample_X, grid)
    assert np.all(np.diff(cdf, axis=1) >= CDF_MONO_TOL)
    assert np.allclose(cdf[:, -1], 1.0, atol=CDF_END_TOL)


def test_positive_output_keeps_pdf_non_negative(trained_estimator) -> None:
    estimator, X, _, grid = trained_estimator
    assert estimator.config.positive_output
    pdf = estimator.predict_density(X[:3], grid)
    assert float(pdf.min()) >= 0.0
