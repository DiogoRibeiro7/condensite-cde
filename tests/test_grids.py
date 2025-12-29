"""Tests for y-grid helper utilities."""

from __future__ import annotations

import numpy as np

from condensite_cde import make_y_grid

QUANTILE_GRID_SIZE = 32
LINEAR_GRID_SIZE = 8
CLIP_MAX = 2.0


def test_quantile_grid_is_strictly_increasing() -> None:
    y = np.concatenate([np.full(50, -1.0), np.linspace(-1, 2, 150), np.full(50, 2.0)])
    grid = make_y_grid(y, grid_size=QUANTILE_GRID_SIZE, mode="quantile")
    assert grid.shape[0] == QUANTILE_GRID_SIZE
    diffs = np.diff(grid)
    assert np.all(diffs > 0)


def test_linear_grid_handles_constant_targets_and_clip() -> None:
    y = np.ones(20)
    grid = make_y_grid(y, grid_size=LINEAR_GRID_SIZE, mode="linear", clip=(0.0, CLIP_MAX))
    assert grid[0] >= 0.0
    assert grid[-1] <= CLIP_MAX
    assert np.all(np.diff(grid) > 0)
