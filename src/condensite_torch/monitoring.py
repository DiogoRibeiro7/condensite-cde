"""Monitoring helpers for feature drift and probabilistic calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

SchemaDict = dict[str, Any]
_FEATURE_DIM = 2


@dataclass(frozen=True, slots=True)
class DriftThresholds:
    """Threshold configuration for a single drift metric."""

    warn: float
    alert: float

    def __post_init__(self) -> None:
        if self.warn < 0 or self.alert < 0:
            msg = "warn and alert thresholds must be non-negative."
            raise ValueError(msg)
        if self.warn > self.alert:
            msg = "warn threshold must be <= alert threshold."
            raise ValueError(msg)

    def classify(self, value: float) -> str:
        """Return qualitative status for the metric value."""
        if value >= self.alert:
            return "alert"
        if value >= self.warn:
            return "warn"
        return "ok"

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable dictionary."""
        return {"warn": float(self.warn), "alert": float(self.alert)}


@dataclass(frozen=True, slots=True)
class MonitoringThresholds:
    """Threshold bundle for all supported monitoring metrics."""

    psi: DriftThresholds = field(default_factory=lambda: DriftThresholds(0.1, 0.25))
    ks: DriftThresholds = field(default_factory=lambda: DriftThresholds(0.05, 0.1))
    pit: DriftThresholds = field(default_factory=lambda: DriftThresholds(0.05, 0.1))


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
    """Kolmogorov-Smirnov distance between two samples."""
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


def compare_windows(
    baseline_features: NDArray[np.floating],
    current_features: NDArray[np.floating],
    feature_names: Sequence[str],
    *,
    thresholds: MonitoringThresholds | None = None,
) -> list[SchemaDict]:
    """Compare PSI/KS drift for baseline vs current feature windows."""
    thresholds = thresholds or MonitoringThresholds()
    baseline = np.asarray(baseline_features, dtype=np.float64)
    current = np.asarray(current_features, dtype=np.float64)
    if baseline.shape != current.shape:
        msg = "Baseline and current feature matrices must share the same shape."
        raise ValueError(msg)
    if baseline.ndim != _FEATURE_DIM:
        msg = "Feature matrices must be 2D (n_samples, n_features)."
        raise ValueError(msg)
    if len(feature_names) != baseline.shape[1]:
        msg = "Number of feature names must match the number of columns."
        raise ValueError(msg)
    feature_stats: list[SchemaDict] = []
    for idx, name in enumerate(feature_names):
        psi_value = population_stability_index(baseline[:, idx], current[:, idx])
        ks_value = ks_drift(baseline[:, idx], current[:, idx])
        feature_stats.append(
            {
                "feature": name,
                "psi": {
                    "value": psi_value,
                    "status": thresholds.psi.classify(psi_value),
                    "thresholds": thresholds.psi.to_dict(),
                },
                "ks": {
                    "value": ks_value,
                    "status": thresholds.ks.classify(ks_value),
                    "thresholds": thresholds.ks.to_dict(),
                },
            },
        )
    return feature_stats


def compare_pit_windows(
    baseline_pit: NDArray[np.floating],
    current_pit: NDArray[np.floating],
    *,
    thresholds: MonitoringThresholds | None = None,
    bins: int = 20,
) -> SchemaDict:
    """Compare PIT histograms between baseline and current windows."""
    thresholds = thresholds or MonitoringThresholds()
    base_hist = pit_histogram(baseline_pit, bins=bins)
    curr_hist = pit_histogram(current_pit, bins=bins)
    drift_value = pit_drift(baseline_pit, current_pit, bins=bins)
    return {
        "histogram": {
            "baseline": {
                "counts": base_hist["counts"].astype(float).tolist(),
                "bin_edges": base_hist["edges"].astype(float).tolist(),
            },
            "current": {
                "counts": curr_hist["counts"].astype(float).tolist(),
                "bin_edges": curr_hist["edges"].astype(float).tolist(),
            },
        },
        "drift": {
            "value": drift_value,
            "status": thresholds.pit.classify(drift_value),
            "thresholds": thresholds.pit.to_dict(),
        },
    }


def build_monitoring_report(  # noqa: PLR0913
    *,
    baseline_features: NDArray[np.floating],
    current_features: NDArray[np.floating],
    feature_names: Sequence[str],
    baseline_pit: NDArray[np.floating],
    current_pit: NDArray[np.floating],
    thresholds: MonitoringThresholds | None = None,
    bins: int = 20,
    metadata: Mapping[str, Any] | None = None,
) -> SchemaDict:
    """Build a JSON-ready monitoring report payload with drift statuses."""
    thresholds = thresholds or MonitoringThresholds()
    report: SchemaDict = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_drift": compare_windows(
            baseline_features,
            current_features,
            feature_names,
            thresholds=thresholds,
        ),
        "pit": compare_pit_windows(
            baseline_pit,
            current_pit,
            thresholds=thresholds,
            bins=bins,
        ),
    }
    if metadata:
        report["metadata"] = dict(metadata)
    return report


__all__ = (
    "DriftThresholds",
    "MonitoringThresholds",
    "build_monitoring_report",
    "compare_pit_windows",
    "compare_windows",
    "ks_drift",
    "pit_drift",
    "pit_histogram",
    "population_stability_index",
)
