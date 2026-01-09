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


def test_crps_prefers_aligned_cdf() -> None:
    grid = np.linspace(-1.0, 1.0, 200)
    y_true = np.array([0.0])
    perfect = (grid >= 0.0).astype(float)[None, :]
    shift_point = 0.3
    shifted = (grid >= shift_point).astype(float)[None, :]
    perfect[:, -1] = 1.0
    shifted[:, -1] = 1.0
    assert crps_from_cdf(y_true, grid, perfect) < crps_from_cdf(y_true, grid, shifted)
