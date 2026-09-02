"""Evaluation utilities for conditional density models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12
_MIN_GRID_POINTS = 2


def _validate_shapes(
    y_true: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    values: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate metric inputs for common or row-local evaluation grids."""
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_grid_arr = np.asarray(y_grid, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)

    if not np.all(np.isfinite(y_true_arr)):
        msg = "y_true must contain only finite values."
        raise ValueError(msg)
    if not np.all(np.isfinite(y_grid_arr)):
        msg = "y_grid must contain only finite values."
        raise ValueError(msg)
    if not np.all(np.isfinite(values_arr)):
        msg = "pdf/cdf values must contain only finite values."
        raise ValueError(msg)

    n_samples = y_true_arr.shape[0]
    if y_grid_arr.ndim == 1:
        if y_grid_arr.size < _MIN_GRID_POINTS:
            msg = "y_grid must contain at least two points."
            raise ValueError(msg)
        expected_shape = (n_samples, y_grid_arr.shape[0])
        if values_arr.shape != expected_shape:
            msg = f"pdf/cdf array must have shape {expected_shape}, got {values_arr.shape}"
            raise ValueError(msg)
        if np.any(np.diff(y_grid_arr) <= 0.0):
            msg = "y_grid must be strictly increasing."
            raise ValueError(msg)
    elif y_grid_arr.ndim == 2:
        if y_grid_arr.shape[0] != n_samples:
            msg = (
                "Row-local y_grid must have one row per target, "
                f"got {y_grid_arr.shape[0]} grid rows for {n_samples} targets."
            )
            raise ValueError(msg)
        if y_grid_arr.shape[1] < _MIN_GRID_POINTS:
            msg = "Each row-local y_grid must contain at least two points."
            raise ValueError(msg)
        if values_arr.shape != y_grid_arr.shape:
            msg = (
                "For row-local grids, pdf/cdf values must match y_grid shape, "
                f"got values={values_arr.shape}, y_grid={y_grid_arr.shape}."
            )
            raise ValueError(msg)
        if np.any(np.diff(y_grid_arr, axis=1) <= 0.0):
            msg = "Each row-local y_grid must be strictly increasing."
            raise ValueError(msg)
    else:
        msg = f"y_grid must be 1-D or 2-D, got shape {y_grid_arr.shape}."
        raise ValueError(msg)

    return y_true_arr, y_grid_arr, values_arr


def _grid_row(y_grid: NDArray[np.float64], index: int) -> NDArray[np.float64]:
    """Return the common grid or the requested row-local grid."""
    if y_grid.ndim == 1:
        return y_grid
    return y_grid[index]


def nll_from_pdf(
    y_true: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    pdf: NDArray[np.floating],
    epsilon: float = EPS,
) -> float:
    """Compute mean negative log-likelihood via interpolation on the grid.

    Targets outside the supplied grid receive the numerical floor ``epsilon`` rather
    than inheriting an endpoint density. This makes truncated support visible in NLL
    instead of silently treating boundary density as extrapolated probability mass.
    """
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        msg = "epsilon must be a positive finite value."
        raise ValueError(msg)

    y_true_arr, y_grid_arr, pdf_arr = _validate_shapes(y_true, y_grid, pdf)
    densities = np.empty(y_true_arr.shape[0], dtype=np.float64)
    for index, (row, target) in enumerate(zip(pdf_arr, y_true_arr, strict=True)):
        grid = _grid_row(y_grid_arr, index)
        interpolated = float(np.interp(target, grid, row, left=0.0, right=0.0))
        densities[index] = max(interpolated, epsilon)
    return float(np.mean(-np.log(densities)))


def crps_from_cdf(
    y_true: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    cdf: NDArray[np.floating],
) -> float:
    """Approximate CRPS by numerically integrating ``(CDF - indicator)^2``."""
    y_true_arr, y_grid_arr, cdf_arr = _validate_shapes(y_true, y_grid, cdf)
    scores = np.empty(y_true_arr.shape[0], dtype=np.float64)
    for index, (row, target) in enumerate(zip(cdf_arr, y_true_arr, strict=True)):
        grid = _grid_row(y_grid_arr, index)
        indicator = (grid >= target).astype(np.float64)
        error = (row - indicator) ** 2
        scores[index] = float(np.trapezoid(error, grid))
    return float(np.mean(scores))


__all__: tuple[str, ...] = ("crps_from_cdf", "nll_from_pdf")
