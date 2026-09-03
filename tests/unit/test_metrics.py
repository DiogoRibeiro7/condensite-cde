from __future__ import annotations

import numpy as np
import pytest

from condensite_torch.metrics import crps_from_cdf, nll_from_pdf

pytestmark = pytest.mark.unit


def _normalize(pdf: np.ndarray, grid: np.ndarray) -> np.ndarray:
    pdf = pdf.copy()
    area = np.trapezoid(pdf, x=grid)
    return pdf / np.clip(area, 1e-8, None)


def test_nll_penalizes_flatter_pdf() -> None:
    grid = np.linspace(-1.0, 1.0, 200)
    y_true = np.array([0.1])
    sharp = np.exp(-0.5 * ((grid - 0.1) / 0.05) ** 2)
    flat = np.ones_like(grid)
    sharp_pdf = _normalize(sharp, grid)[None, :]
    flat_pdf = _normalize(flat, grid)[None, :]
    assert nll_from_pdf(y_true, grid, sharp_pdf) < nll_from_pdf(y_true, grid, flat_pdf)


def test_nll_penalizes_target_outside_grid_support() -> None:
    grid = np.linspace(-1.0, 1.0, 200)
    y_true = np.array([1.5])
    pdf = _normalize(np.ones_like(grid), grid)[None, :]
    score = nll_from_pdf(y_true, grid, pdf)
    assert score == pytest.approx(-np.log(1e-12))


def test_crps_prefers_aligned_cdf() -> None:
    grid = np.linspace(-1.0, 1.0, 200)
    y_true = np.array([0.0])
    perfect = (grid >= 0.0).astype(float)[None, :]
    shift_point = 0.3
    shifted = (grid >= shift_point).astype(float)[None, :]
    perfect[:, -1] = 1.0
    shifted[:, -1] = 1.0
    assert crps_from_cdf(y_true, grid, perfect) < crps_from_cdf(y_true, grid, shifted)


def test_metrics_support_row_local_grids() -> None:
    grid = np.array([
        [-2.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, 2.0, 3.0],
    ])
    pdf = np.array([
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5, 0.0],
    ])
    pdf /= np.trapz(pdf, x=grid, axis=1)[:, None]
    cdf = np.zeros_like(pdf)
    cdf[:, 1:] = np.cumsum(
        0.5 * (pdf[:, :-1] + pdf[:, 1:]) * np.diff(grid, axis=1),
        axis=1,
    )
    cdf /= cdf[:, -1][:, None]
    y_true = np.array([-0.5, 1.5])

    assert np.isfinite(nll_from_pdf(y_true, grid, pdf))
    assert np.isfinite(crps_from_cdf(y_true, grid, cdf))


def test_metrics_reject_non_increasing_grid() -> None:
    grid = np.array([0.0, 1.0, 0.5])
    values = np.array([[0.2, 0.4, 0.4]])
    with pytest.raises(ValueError, match="strictly increasing"):
        nll_from_pdf(np.array([0.5]), grid, values)
