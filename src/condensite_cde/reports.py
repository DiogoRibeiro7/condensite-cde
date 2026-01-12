"""Utilities for building versioned report payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

CALIBRATION_REPORT_SCHEMA_VERSION = "1.0"
BENCHMARK_REPORT_SCHEMA_VERSION = "1.0"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_calibration_report(
    *,
    pit_histogram: Mapping[str, Sequence[float]],
    coverage: Mapping[str, float],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize PIT histogram/coverage metrics into a schema-stable payload."""
    if "counts" not in pit_histogram or "bin_edges" not in pit_histogram:
        msg = "pit_histogram must provide 'counts' and 'bin_edges'."
        raise ValueError(msg)
    counts = [float(value) for value in pit_histogram["counts"]]
    bin_edges = [float(value) for value in pit_histogram["bin_edges"]]
    if len(bin_edges) != len(counts) + 1:
        msg = "bin_edges must be exactly one element longer than counts."
        raise ValueError(msg)
    coverage_payload = {str(key): float(value) for key, value in coverage.items()}
    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "generated_at": _utc_timestamp(),
        "pit": {
            "counts": counts,
            "bin_edges": bin_edges,
        },
        "coverage": coverage_payload,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def build_benchmark_report(
    *,
    results: Mapping[str, Mapping[str, Mapping[str, float]]],
    quick: bool,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert nested benchmark metrics into a versioned payload."""
    normalized_results: dict[str, dict[str, dict[str, float]]] = {}
    for dataset, baselines in results.items():
        dataset_payload: dict[str, dict[str, float]] = {}
        for model_name, metrics in baselines.items():
            dataset_payload[model_name] = {
                str(metric): float(value) for metric, value in metrics.items()
            }
        normalized_results[dataset] = dataset_payload
    payload: dict[str, Any] = {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "generated_at": _utc_timestamp(),
        "quick": bool(quick),
        "results": normalized_results,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


__all__ = [
    "BENCHMARK_REPORT_SCHEMA_VERSION",
    "CALIBRATION_REPORT_SCHEMA_VERSION",
    "build_benchmark_report",
    "build_calibration_report",
]
