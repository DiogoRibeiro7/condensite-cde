"""Pairwise distribution comparison metrics on a shared grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12


def wasserstein_1(
    cdf_a: NDArray[np.floating],
    cdf_b: NDArray[np.floating],
    y_grid: NDArray[np.floating],
) -> float:
    """Return the 1-Wasserstein distance given CDFs evaluated on the same grid."""
    cdf_arr_a, cdf_arr_b, grid = _validate_inputs(cdf_a, cdf_b, y_grid, kind="cdf")
    diff = np.abs(cdf_arr_a - cdf_arr_b)
    integral = np.trapezoid(diff, x=grid, axis=-1)
    return float(np.mean(integral))


def ks_distance(
    cdf_a: NDArray[np.floating],
    cdf_b: NDArray[np.floating],
) -> float:
    """Return the Kolmogorov–Smirnov distance between two CDFs."""
    if cdf_a.shape != cdf_b.shape:
        msg = f"CDF arrays must have identical shape, got {cdf_a.shape} vs {cdf_b.shape}"
        raise ValueError(msg)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def js_divergence(
    pdf_a: NDArray[np.floating],
    pdf_b: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    epsilon: float = EPS,
) -> float:
    """Return Jensen–Shannon divergence between two PDFs on the same grid."""
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
    if arr_a.shape != arr_b.shape:
        msg = f"{kind} arrays must have the same shape, got {arr_a.shape} vs {arr_b.shape}"
        raise ValueError(msg)
    if arr_a.ndim == 1:
        arr_a = arr_a.reshape(1, -1)
        arr_b = arr_b.reshape(1, -1)
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
    mass = np.trapezoid(pdf_safe, x=y_grid, axis=-1)
    mass = mass.reshape(mass.shape + (1,))
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
    result = np.trapezoid(integrand, x=y_grid, axis=-1)
    return float(np.mean(result))


__all__ = ("wasserstein_1", "ks_distance", "js_divergence")
