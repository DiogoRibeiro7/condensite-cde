"""Grid construction utilities for conditional density estimation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

GridMode = Literal["quantile", "linear"]
MIN_GRID_SIZE = 2


def make_y_grid(
    y_train: NDArray[np.floating],
    grid_size: int = 128,
    *,
    mode: GridMode = "quantile",
    clip: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Build a strictly increasing y-grid based on training targets."""
    data = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if data.size == 0:
        msg = "y_train must contain at least one element."
        raise ValueError(msg)
    if grid_size < MIN_GRID_SIZE:
        msg = f"grid_size must be >= {MIN_GRID_SIZE}."
        raise ValueError(msg)

    if clip is not None:
        clip_low, clip_high = clip
        if not np.isfinite(clip_low) or not np.isfinite(clip_high) or clip_high <= clip_low:
            msg = "clip must be (low, high) with finite low < high."
            raise ValueError(msg)
        data = np.clip(data, clip_low, clip_high)

    mode_lower = mode.lower()
    if mode_lower == "quantile":
        quantiles = np.linspace(0.0, 1.0, grid_size)
        grid = np.quantile(data, quantiles, method="linear")  # NumPy>=1.22
    elif mode_lower == "linear":
        grid = np.linspace(float(data.min()), float(data.max()), grid_size)
    else:
        msg = f"Unknown grid mode: {mode!r}."
        raise ValueError(msg)

    if not np.all(np.diff(grid) > 0):
        min_val = float(data.min())
        max_val = float(data.max())
        if min_val == max_val:
            span = 1.0 if min_val == 0.0 else abs(min_val) * 0.1
            span = max(span, 1e-3)
            grid = np.linspace(min_val - span, max_val + span, grid_size)
        else:
            grid = np.linspace(min_val, max_val, grid_size)

    return grid.astype(np.float64, copy=False)
