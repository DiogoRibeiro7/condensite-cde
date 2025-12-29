from __future__ import annotations

import numpy as np
import pytest

from condensite_cde import make_y_grid

pytestmark = pytest.mark.unit


def test_make_y_grid_returns_sorted_unique_values() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=128)
    grid = make_y_grid(y, grid_size=64, mode="quantile")
    assert np.all(np.diff(grid) > 0)
    assert grid[0] <= y.min()
    assert grid[-1] >= y.max()


def test_uniform_grid_covers_bounds() -> None:
    y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    grid = make_y_grid(y, grid_size=32, mode="uniform")
    assert np.isclose(grid[0], -2.0)
    assert np.isclose(grid[-1], 2.0)
    assert np.all(np.diff(grid) > 0)
