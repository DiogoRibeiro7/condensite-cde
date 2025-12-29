"""Sampling behaviour tests."""

from __future__ import annotations

import numpy as np


def test_sampling_shapes_and_ranges(trained_estimator) -> None:
    estimator, X, _, grid = trained_estimator
    n_samples = 5
    draws = estimator.sample(X[:2], n_samples, y_grid=grid, seed=11)
    assert draws.shape == (2, n_samples)
    assert np.all(draws <= grid.max() + 1e-6)
    assert np.all(draws >= grid.min() - 1e-6)


def test_sampling_default_quantile_grid(trained_estimator) -> None:
    estimator, X, y, _ = trained_estimator
    draws = estimator.sample(X[:2], 4, seed=2)
    assert draws.shape == (2, 4)
    assert np.all(draws <= y.max() + 1e-6)
    assert np.all(draws >= y.min() - 1e-6)


def test_sampling_refine_steps(trained_estimator) -> None:
    estimator, X, _, grid = trained_estimator
    draws = estimator.sample(X[:1], 6, y_grid=grid, seed=4, refine_steps=2)
    assert draws.shape == (1, 6)
    assert np.all(draws <= grid.max() + 1e-6)
    assert np.all(draws >= grid.min() - 1e-6)
