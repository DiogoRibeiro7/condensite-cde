"""Calibration diagnostics for conditional density estimates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _ensure_shapes(
    y_true: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    values: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if y_true_arr.size == 0:
        msg = "y_true must contain at least one sample."
        raise ValueError(msg)
    y_grid_arr = np.asarray(y_grid, dtype=np.float64).reshape(-1)
    if y_grid_arr.ndim != 1 or y_grid_arr.size < 2:
        msg = "y_grid must be a 1-D array with at least two points."
        raise ValueError(msg)
    if not np.all(np.diff(y_grid_arr) > 0):
        msg = "y_grid must be strictly increasing."
        raise ValueError(msg)
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.shape != (y_true_arr.shape[0], y_grid_arr.shape[0]):
        msg = (
            "values array must have shape (n_samples, len(y_grid)), "
            f"got {values_arr.shape}"
        )
        raise ValueError(msg)
    return y_true_arr, y_grid_arr, values_arr


def pit_values(
    y_true: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    cdf: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return probability integral transform values by interpolating the CDF."""
    y_true_arr, y_grid_arr, cdf_arr = _ensure_shapes(y_true, y_grid, cdf)
    pit = np.empty_like(y_true_arr)
    for idx, (y_val, cdf_row) in enumerate(zip(y_true_arr, cdf_arr, strict=True)):
        pit[idx] = float(np.interp(y_val, y_grid_arr, cdf_row, left=0.0, right=1.0))
    return np.clip(pit, 0.0, 1.0)


def coverage_rate(
    y_true: NDArray[np.floating],
    q_lo: NDArray[np.floating] | float,
    q_hi: NDArray[np.floating] | float,
) -> float:
    """Estimate predictive interval coverage provided lower/upper quantiles."""
    y_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if y_arr.size == 0:
        msg = "y_true must contain at least one sample."
        raise ValueError(msg)
    lo = _broadcast(q_lo, y_arr.size, "q_lo")
    hi = _broadcast(q_hi, y_arr.size, "q_hi")
    if np.any(lo > hi):
        msg = "Lower quantiles must be <= upper quantiles."
        raise ValueError(msg)
    inside = (y_arr >= lo) & (y_arr <= hi)
    return float(np.mean(inside.astype(np.float64)))


def coverage(
    y_true: NDArray[np.floating],
    q_lo: NDArray[np.floating] | float,
    q_hi: NDArray[np.floating] | float,
) -> float:
    """Backward-compatible alias for coverage_rate."""
    return coverage_rate(y_true, q_lo, q_hi)


def _broadcast(
    values: NDArray[np.floating] | float,
    n_samples: int,
    name: str,
) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        msg = f"{name} must contain at least one value."
        raise ValueError(msg)
    if arr.size == 1:
        return np.full(n_samples, float(arr[0]), dtype=np.float64)
    if arr.size != n_samples:
        msg = f"{name} must be scalar or match the number of samples ({n_samples})."
        raise ValueError(msg)
    return arr.astype(np.float64, copy=False)


__all__ = ("coverage", "coverage_rate", "pit_values")
