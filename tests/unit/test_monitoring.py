from __future__ import annotations

import numpy as np
import pytest

from condensite_torch.monitoring import (
    DriftThresholds,
    MonitoringThresholds,
    build_monitoring_report,
    compare_pit_windows,
    compare_windows,
    ks_drift,
    pit_drift,
    pit_histogram,
    population_stability_index,
)

pytestmark = pytest.mark.unit


def test_population_stability_index_bounds() -> None:
    base = np.array([0.0, 0.1, 0.2, 0.3])
    current = base.copy()
    psi = population_stability_index(base, current)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_population_stability_index_detects_reference_shift() -> None:
    base = np.linspace(-1.0, 1.0, 200)
    shifted = np.linspace(4.0, 6.0, 80)
    minimum_detectable_psi = 0.25
    assert population_stability_index(base, shifted) > minimum_detectable_psi


def test_ks_drift_increases_with_shift() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=200)
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


def test_pit_histogram_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pit_histogram(np.array([0.1, 1.2]))


def test_drift_thresholds_validate() -> None:
    with pytest.raises(ValueError):
        DriftThresholds(-0.1, 0.2)
    with pytest.raises(ValueError):
        DriftThresholds(0.2, 0.1)
    with pytest.raises(ValueError):
        DriftThresholds(float("nan"), 0.2)


def test_compare_windows_statuses() -> None:
    base = np.zeros((32, 2))
    curr_ok = base.copy()
    thresholds = MonitoringThresholds(
        psi=DriftThresholds(0.05, 0.1),
        ks=DriftThresholds(0.05, 0.1),
        pit=DriftThresholds(0.05, 0.1),
    )
    stats_ok = compare_windows(base, curr_ok, ["f0", "f1"], thresholds=thresholds)
    assert all(item["psi"]["status"] == "ok" for item in stats_ok)
    shifted = base.copy()
    shifted[:, 0] = 5.0
    stats_bad = compare_windows(base, shifted, ["f0", "f1"], thresholds=thresholds)
    assert stats_bad[0]["psi"]["status"] in {"warn", "alert"}


def test_compare_windows_allows_different_window_lengths() -> None:
    rng = np.random.default_rng(11)
    baseline = rng.normal(size=(120, 3))
    current = rng.normal(size=(35, 3))
    feature_names = ["a", "b", "c"]
    stats = compare_windows(baseline, current, feature_names)
    assert len(stats) == len(feature_names)


def test_compare_windows_rejects_feature_dimension_mismatch() -> None:
    baseline = np.zeros((20, 2))
    current = np.zeros((10, 3))
    with pytest.raises(ValueError, match="same number of columns"):
        compare_windows(baseline, current, ["a", "b"])


def test_compare_pit_windows_schema_and_status() -> None:
    base = np.linspace(0, 1, 100)
    current = np.linspace(0.5, 1, 100)
    thresholds = MonitoringThresholds(
        psi=DriftThresholds(1.0, 2.0),
        ks=DriftThresholds(1.0, 2.0),
        pit=DriftThresholds(0.01, 0.02),
    )
    report = compare_pit_windows(base, current, thresholds=thresholds, bins=10)
    assert report["histogram"]["baseline"]["counts"]  # not empty
    assert report["drift"]["status"] in {"warn", "alert"}


def test_build_monitoring_report_contains_metadata() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=(16, 2))
    current = baseline + 0.05
    feature_names = ["a", "b"]
    pit_base = rng.random(16)
    pit_curr = np.clip(pit_base + 0.05, 0, 1)
    meta = {"window": "2025-01"}
    report = build_monitoring_report(
        baseline_features=baseline,
        current_features=current,
        feature_names=feature_names,
        baseline_pit=pit_base,
        current_pit=pit_curr,
        metadata=meta,
    )
    assert report["schema_version"] == "1.0"
    assert report["metadata"]["window"] == "2025-01"
