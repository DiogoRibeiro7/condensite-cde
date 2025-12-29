"""Verify density and CDF outputs satisfy basic properties."""

from __future__ import annotations

import numpy as np

CDF_NEG_TOL = -1e-3
CDF_MONO_TOL = -1e-4


def test_density_and_cdf_properties(trained_estimator) -> None:
    estimator, X, _, grid = trained_estimator
    sample_X = X[:3]
    pdf = estimator.predict_density(sample_X, grid)
    assert np.all(pdf >= 0)
    areas = np.trapz(pdf, x=grid, axis=1)
    assert np.allclose(areas, np.ones_like(areas), atol=0.25)
    cdf = estimator.predict_cdf(sample_X, grid)
    assert np.all(cdf[:, 0] >= CDF_NEG_TOL)
    assert np.allclose(cdf[:, -1], 1.0, atol=1e-2)
    assert np.all(np.diff(cdf, axis=1) >= CDF_MONO_TOL)
