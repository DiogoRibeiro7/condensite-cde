"""Helpers for building per-row grids tailored to each feature vector."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - avoids circular import at runtime
    from .estimator import CondensiteTorchCDE

_MIN_GRID_SIZE = 2
FeatureArray = NDArray[np.floating] | NDArray[np.object_]


def make_local_grid(  # noqa: PLR0913
    estimator: "CondensiteTorchCDE",
    X: FeatureArray,
    grid_size: int = 64,
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
    padding: float = 0.1,
    y_grid: NDArray[np.floating] | None = None,
    head: int | str | None = None,
) -> NDArray[np.float64]:
    """Construct per-row grids by expanding low/high quantiles with optional padding."""
    if grid_size < _MIN_GRID_SIZE:
        msg = f"grid_size must be >= {_MIN_GRID_SIZE}."
        raise ValueError(msg)
    if not 0.0 <= q_low < q_high <= 1.0:
        msg = "Require 0 <= q_low < q_high <= 1."
        raise ValueError(msg)
    X_arr = np.asarray(X, dtype=object)
    probs = np.array([q_low, q_high], dtype=np.float64)
    quantiles = estimator.predict_quantile(X_arr, probs, y_grid=y_grid, head=head)
    low = quantiles[:, 0]
    high = quantiles[:, 1]
    span = np.maximum(high - low, 1e-6)
    pad = np.maximum(padding, 0.0) * span
    lower = low - pad
    upper = high + pad
    grids = np.stack(
        [
            np.linspace(l_val, u_val, grid_size, dtype=np.float64)
            if u_val > l_val
            else np.linspace(l_val - 1e-3, u_val + 1e-3, grid_size, dtype=np.float64)
            for l_val, u_val in zip(lower, upper, strict=True)
        ],
        axis=0,
    )
    diffs = np.diff(grids, axis=1)
    if np.any(diffs <= 0):
        for row in range(grids.shape[0]):
            if np.any(np.diff(grids[row]) <= 0):
                grids[row] = np.linspace(grids[row, 0], grids[row, -1] + 1e-6, grid_size)
    return grids


__all__ = ("make_local_grid",)
