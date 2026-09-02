"""Pairwise distribution comparison metrics on a shared grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12
_MIN_GRID_POINTS = 2


def wasserstein_1(
    cdf_a: NDArray[np.floating],
    cdf_b: NDArray[np.floating],
    y_grid: NDArray[np.floating],
) -> float:
    """Return the 1-Wasserstein distance given CDFs on the same grid."""
    cdf_arr_a, cdf_arr_b, grid = _validate_inputs(cdf_a, cdf_b, y_grid, kind="cdf")
    diff = np.abs(cdf_arr_a - cdf_arr_b)
    integral = np.trapz(diff, x=grid, axis=-1)
    return float(np.mean(integral))


def ks_distance(
    cdf_a: NDArray[np.floating],
    cdf_b: NDArray[np.floating],
) -> float:
    """Return the Kolmogorov-Smirnov distance between two CDFs."""
    arr_a = np.asarray(cdf_a, dtype=np.float64)
    arr_b = np.asarray(cdf_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        msg = f"CDF arrays must have identical shape, got {arr_a.shape} vs {arr_b.shape}"
        raise ValueError(msg)
    if arr_a.size == 0:
        msg = "CDF arrays must not be empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(arr_a)) or not np.all(np.isfinite(arr_b)):
        msg = "CDF arrays must contain only finite values."
        raise ValueError(msg)
    return float(np.max(np.abs(arr_a - arr_b)))


def js_divergence(
    pdf_a: NDArray[np.floating],
    pdf_b: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    epsilon: float = EPS,
) -> float:
    """Return Jensen-Shannon divergence between two PDFs on the same grid."""
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        msg = "epsilon must be a positive finite value."
        raise ValueError(msg)
    pdf_arr_a, pdf_arr_b, grid = _validate_inputs(pdf_a, pdf_b, y_grid, kind="pdf")
    pdf_arr_a = _normalize_pdf(pdf_arr_a, grid, epsilon)
    pdf_arr_b = _normalize_pdf(pdf_arr_b, grid, epsilon)
    mixture = 0.5 * (pdf_arr_a + pdf_arr_b)
    kl_a = _kl_divergence(pdf_arr_a, mixture, grid, epsilon)
    kl_b = _kl_divergence(pdf_arr_b, mixture, grid, epsilon)
    return float(0.5 * (kl_a + kl_b))


def _validate_inputs(
    first: NDArray[np.floating],
    second: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    *,
    kind: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    arr_a = np.asarray(first, dtype=np.float64)
    arr_b = np.asarray(second, dtype=np.float64)
    grid = np.asarray(y_grid, dtype=np.float64).reshape(-1)
    if grid.size < _MIN_GRID_POINTS or np.any(np.diff(grid) <= 0.0):
        msg = "y_grid must contain at least two strictly increasing points."
        raise ValueError(msg)
    if not np.all(np.isfinite(grid)):
        msg = "y_grid must contain only finite values."
        raise ValueError(msg)
    if arr_a.shape != arr_b.shape:
        msg = f"{kind} arrays must have the same shape, got {arr_a.shape} vs {arr_b.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr_a)) or not np.all(np.isfinite(arr_b)):
        msg = f"{kind} arrays must contain only finite values."
        raise ValueError(msg)
    if arr_a.ndim == 1:
        arr_a = arr_a.reshape(1, -1)
        arr_b = arr_b.reshape(1, -1)
    if arr_a.ndim != 2:
        msg = f"{kind} arrays must be 1-D or 2-D, got shape {arr_a.shape}."
        raise ValueError(msg)
    if arr_a.shape[-1] != grid.size:
        msg = f"{kind} arrays must align with grid length {grid.size}"
        raise ValueError(msg)
    return arr_a, arr_b, grid


def _normalize_pdf(
    pdf: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    epsilon: float,
) -> NDArray[np.float64]:
    pdf_safe = np.clip(pdf, epsilon, None)
    mass = np.trapz(pdf_safe, x=y_grid, axis=-1)
    mass = mass.reshape((*mass.shape, 1))
    return pdf_safe / np.clip(mass, epsilon, None)


def _kl_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    epsilon: float,
) -> float:
    numerator = np.clip(p, epsilon, None)
    denominator = np.clip(q, epsilon, None)
    integrand = numerator * np.log(numerator / denominator)
    result = np.trapz(integrand, x=y_grid, axis=-1)
    return float(np.mean(result))


__all__ = ("js_divergence", "ks_distance", "wasserstein_1")
