"""Evaluation utilities for conditional density models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12


def _validate_shapes(
    y_true: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    values: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_grid_arr = np.asarray(y_grid, dtype=np.float64).reshape(-1)
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.shape != (y_true_arr.shape[0], y_grid_arr.shape[0]):
        msg = (
            "pdf/cdf array must have shape (n_samples, len(y_grid)), "
            f"got {values_arr.shape}"
        )
        raise ValueError(msg)
    return y_true_arr, y_grid_arr, values_arr


def nll_from_pdf(
    y_true: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    pdf: NDArray[np.floating],
    epsilon: float = EPS,
) -> float:
    """Compute mean negative log-likelihood via interpolation on the grid."""
    y_true_arr, y_grid_arr, pdf_arr = _validate_shapes(y_true, y_grid, pdf)
    densities = []
    for row, target in zip(pdf_arr, y_true_arr, strict=True):
        interp = np.interp(target, y_grid_arr, row, left=row[0], right=row[-1])
        densities.append(max(interp, epsilon))
    densities_arr = np.array(densities, dtype=np.float64)
    return float(np.mean(-np.log(densities_arr)))


def crps_from_cdf(
    y_true: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    cdf: NDArray[np.floating],
) -> float:
    """Approximate CRPS by numerically integrating (CDF - indicator)^2."""
    y_true_arr, y_grid_arr, cdf_arr = _validate_shapes(y_true, y_grid, cdf)
    scores: list[float] = []
    for row, target in zip(cdf_arr, y_true_arr, strict=True):
        indicator = (y_grid_arr >= target).astype(np.float64)
        error = (row - indicator) ** 2
        scores.append(float(np.trapezoid(error, y_grid_arr)))
    return float(np.mean(np.asarray(scores, dtype=np.float64)))


__all__: tuple[str, ...] = ("crps_from_cdf", "nll_from_pdf")
