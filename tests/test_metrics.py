"""Tests for CRPS and NLL helpers."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf


def test_metrics_return_finite_values() -> None:
    y_grid = np.linspace(-3, 3, 101)
    pdf = np.exp(-0.5 * y_grid**2)
    pdf /= np.trapz(pdf, y_grid)
    pdf_stacked = np.vstack([pdf, pdf])
    cdf_body = 0.5 * (pdf_stacked[:, 1:] + pdf_stacked[:, :-1]) * np.diff(y_grid)
    cdf = np.concatenate([np.zeros((2, 1)), np.cumsum(cdf_body, axis=1)], axis=1)
    cdf[:, -1] = 1.0
    y_true = np.array([0.0, 1.0])
    nll = nll_from_pdf(y_true, y_grid, pdf_stacked)
    crps = crps_from_cdf(y_true, y_grid, cdf)
    assert np.isfinite(nll)
    assert np.isfinite(crps)


def test_better_pdf_has_lower_nll() -> None:
    y_grid = np.linspace(-2, 2, 51)
    pdf_good = np.exp(-0.5 * (y_grid - 0.5) ** 2)
    pdf_good /= np.trapz(pdf_good, y_grid)
    pdf_bad = np.ones_like(pdf_good) / (y_grid[-1] - y_grid[0])
    y_true = np.array([0.6])
    nll_good = nll_from_pdf(y_true, y_grid, pdf_good[None, :])
    nll_bad = nll_from_pdf(y_true, y_grid, pdf_bad[None, :])
    assert nll_good < nll_bad


def test_tail_probability_monotone() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 2))
    y = 0.4 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * rng.normal(size=64)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(16, 16),
        epochs=4,
        patience=2,
        m_aux=16,
        sampler="sobol",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=5).fit(X, y)
    grid = estimator._default_y_grid()
    thresholds = np.linspace(grid.min() - 0.2, grid.max() + 0.2, 5)
    tail_probs = [
        estimator.predict_tail_prob(X[:8], threshold=th, y_grid=grid, side="right")
        for th in thresholds
    ]
    for earlier, later in pairwise(tail_probs):
        assert np.all(earlier >= later)
