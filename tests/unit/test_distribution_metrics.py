from __future__ import annotations

import numpy as np
import pytest

from condensite_torch.distribution_metrics import js_divergence, ks_distance, wasserstein_1

pytestmark = pytest.mark.unit


def _cdf_from_pdf(pdf: np.ndarray, grid: np.ndarray) -> np.ndarray:
    widths = np.diff(grid, prepend=grid[0])
    cdf = np.cumsum(pdf * widths, axis=-1)
    cdf /= cdf[:, -1].reshape(-1, 1)
    return cdf


def test_metrics_zero_for_identical_distributions() -> None:
    grid = np.linspace(-1, 1, 32)
    pdf = np.exp(-0.5 * grid**2).reshape(1, -1)
    pdf /= np.trapezoid(pdf, x=grid, axis=-1).reshape(1, 1)
    cdf = _cdf_from_pdf(pdf, grid)
    assert ks_distance(cdf, cdf) == pytest.approx(0.0)
    assert wasserstein_1(cdf, cdf, grid) == pytest.approx(0.0)
    assert js_divergence(pdf, pdf, grid) == pytest.approx(0.0)


def test_metrics_increase_for_shifted_distributions() -> None:
    grid = np.linspace(-3, 3, 64)
    base_pdf = np.exp(-0.5 * ((grid - 0.0) ** 2))
    base_pdf /= np.trapezoid(base_pdf, x=grid)
    shifted_pdf = np.exp(-0.5 * ((grid - 1.0) ** 2))
    shifted_pdf /= np.trapezoid(shifted_pdf, x=grid)
    pdf_a = base_pdf.reshape(1, -1)
    pdf_b = shifted_pdf.reshape(1, -1)
    cdf_a = _cdf_from_pdf(pdf_a, grid)
    cdf_b = _cdf_from_pdf(pdf_b, grid)
    assert ks_distance(cdf_a, cdf_b) > 0.05
    assert wasserstein_1(cdf_a, cdf_b, grid) > 0.1
    assert js_divergence(pdf_a, pdf_b, grid) > 0.01
