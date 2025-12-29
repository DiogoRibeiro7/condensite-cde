"""Tests for CRPS and NLL helpers."""

from __future__ import annotations

import numpy as np

from condensite_torch.metrics import crps_from_cdf, nll_from_pdf


def test_metrics_return_finite_values() -> None:
    y_grid = np.linspace(-3, 3, 101)
    pdf = np.exp(-0.5 * y_grid**2)
    pdf /= np.trapezoid(pdf, y_grid)
    pdf_stacked = np.vstack([pdf, pdf])
    cdf_body = 0.5 * (pdf_stacked[:, 1:] + pdf_stacked[:, :-1]) * np.diff(y_grid)
    cdf = np.concatenate([np.zeros((2, 1)), np.cumsum(cdf_body, axis=1)], axis=1)
    cdf[:, -1] = 1.0
    y_true = np.array([0.0, 1.0])
    nll = nll_from_pdf(y_true, y_grid, pdf_stacked)
    crps = crps_from_cdf(y_true, y_grid, cdf)
    assert np.isfinite(nll)
    assert np.isfinite(crps)


def test_better_pdf_has_lower_nll() -> None:
    y_grid = np.linspace(-2, 2, 51)
    pdf_good = np.exp(-0.5 * (y_grid - 0.5) ** 2)
    pdf_good /= np.trapezoid(pdf_good, y_grid)
    pdf_bad = np.ones_like(pdf_good) / (y_grid[-1] - y_grid[0])
    y_true = np.array([0.6])
    nll_good = nll_from_pdf(y_true, y_grid, pdf_good[None, :])
    nll_bad = nll_from_pdf(y_true, y_grid, pdf_bad[None, :])
    assert nll_good < nll_bad
