from __future__ import annotations

import numpy as np

from condensite_torch.diagnostics import coverage, coverage_rate, pit_values


def test_pit_values_are_bounded() -> None:
    y_true = np.array([0.0, 0.5, 1.0])
    grid = np.linspace(0.0, 1.0, 5)
    cdf = np.vstack([grid, grid**2, np.sqrt(grid + 1e-12)])
    values = pit_values(y_true, grid, cdf)
    assert values.shape == y_true.shape
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


def test_coverage_rate_matches_known_interval() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=1000)
    q_lo = np.full_like(y_true, -1.0)
    q_hi = np.full_like(y_true, 1.0)
    cov = coverage_rate(y_true, q_lo, q_hi)
    assert 0.6 < cov < 0.75


def test_coverage_alias_matches_rate() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=128)
    lo = np.full_like(y_true, -0.5)
    hi = np.full_like(y_true, 0.5)
    assert coverage(y_true, lo, hi) == coverage_rate(y_true, lo, hi)
