from __future__ import annotations

import numpy as np
import pytest

from condensite_torch.monitoring import pit_drift, pit_histogram, population_stability_index, ks_drift

pytestmark = pytest.mark.unit


def test_population_stability_index_bounds() -> None:
    base = np.array([0.0, 0.1, 0.2, 0.3])
    current = base.copy()
    psi = population_stability_index(base, current)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_ks_drift_increases_with_shift() -> None:
    base = np.random.normal(size=200)
    current = base + 0.5
    score = ks_drift(base, base)
    shifted = ks_drift(base, current)
    assert shifted > score


def test_pit_histogram_and_drift() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.random(100)
    current = rng.random(100)
    hist = pit_histogram(baseline, bins=10)
    assert hist["counts"].shape == (10,)
    assert hist["edges"].shape == (11,)
    drift_value = pit_drift(baseline, current, bins=10)
    assert np.isfinite(drift_value)
