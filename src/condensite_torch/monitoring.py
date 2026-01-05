"""Monitoring helpers for feature drift and probabilistic calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def population_stability_index(
    baseline: NDArray[np.floating],
    current: NDArray[np.floating],
    *,
    bins: int = 10,
) -> float:
    """Compute PSI between two numeric samples using equal-width bins."""
    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
    curr = np.asarray(current, dtype=np.float64).reshape(-1)
    if base.size == 0 or curr.size == 0:
        msg = "Both baseline and current samples must contain values."
        raise ValueError(msg)
    edges = np.linspace(min(base.min(), curr.min()), max(base.max(), curr.max()), bins + 1)
    base_hist, _ = np.histogram(base, bins=edges)
    curr_hist, _ = np.histogram(curr, bins=edges)
    base_ratio = np.clip(base_hist / np.clip(base.size, 1, None), 1e-6, None)
    curr_ratio = np.clip(curr_hist / np.clip(curr.size, 1, None), 1e-6, None)
    psi = np.sum((curr_ratio - base_ratio) * np.log(curr_ratio / base_ratio))
    return float(psi)


def ks_drift(
    baseline: NDArray[np.floating],
    current: NDArray[np.floating],
) -> float:
    """Kolmogorov–Smirnov distance between two samples."""
    base = np.sort(np.asarray(baseline, dtype=np.float64).reshape(-1))
    curr = np.sort(np.asarray(current, dtype=np.float64).reshape(-1))
    if base.size == 0 or curr.size == 0:
        msg = "Both baseline and current samples must contain values."
        raise ValueError(msg)
    grid = np.concatenate([base, curr])
    base_cdf = np.searchsorted(base, grid, side="right") / base.size
    curr_cdf = np.searchsorted(curr, grid, side="right") / curr.size
    return float(np.max(np.abs(base_cdf - curr_cdf)))


def pit_histogram(
    pit_values: NDArray[np.floating],
    *,
    bins: int = 20,
) -> dict[str, NDArray[np.float64]]:
    """Return histogram counts and bin edges for PIT values."""
    values = np.asarray(pit_values, dtype=np.float64).reshape(-1)
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return {"counts": counts.astype(np.float64), "edges": edges.astype(np.float64)}


def pit_drift(
    baseline_pit: NDArray[np.floating],
    current_pit: NDArray[np.floating],
    *,
    bins: int = 20,
) -> float:
    """Compute drift between two PIT histograms via PSI."""
    base_hist = pit_histogram(baseline_pit, bins=bins)["counts"]
    curr_hist = pit_histogram(current_pit, bins=bins)["counts"]
    base_ratio = np.clip(base_hist / np.clip(base_hist.sum(), 1, None), 1e-6, None)
    curr_ratio = np.clip(curr_hist / np.clip(curr_hist.sum(), 1, None), 1e-6, None)
    return float(np.sum((curr_ratio - base_ratio) * np.log(curr_ratio / base_ratio)))


__all__ = ("population_stability_index", "ks_drift", "pit_histogram", "pit_drift")
