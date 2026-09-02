"""Utilities for building versioned report payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

CALIBRATION_REPORT_SCHEMA_VERSION = "1.0"
BENCHMARK_REPORT_SCHEMA_VERSION = "1.0"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_float(value: float, *, name: str) -> float:
    """Convert a scalar to float and reject non-standard JSON numeric values."""
    converted = float(value)
    if not math.isfinite(converted):
        msg = f"{name} must be finite."
        raise ValueError(msg)
    return converted


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
    if not coverage:
        msg = "coverage must contain at least one metric."
        raise ValueError(msg)

    counts = [_finite_float(value, name="PIT histogram count") for value in pit_histogram["counts"]]
    bin_edges = [
        _finite_float(value, name="PIT histogram bin edge") for value in pit_histogram["bin_edges"]
    ]
    if len(bin_edges) != len(counts) + 1:
        msg = "bin_edges must be exactly one element longer than counts."
        raise ValueError(msg)
    if any(value < 0.0 for value in counts):
        msg = "PIT histogram counts must be non-negative."
        raise ValueError(msg)
    if any(value < 0.0 or value > 1.0 for value in bin_edges):
        msg = "PIT histogram bin edges must lie in [0, 1]."
        raise ValueError(msg)
    if any(right <= left for left, right in zip(bin_edges, bin_edges[1:], strict=True)):
        msg = "PIT histogram bin edges must be strictly increasing."
        raise ValueError(msg)

    coverage_payload: dict[str, float] = {}
    for key, value in coverage.items():
        converted = _finite_float(value, name=f"coverage[{key!r}]")
        if converted < 0.0 or converted > 1.0:
            msg = f"coverage[{key!r}] must lie in [0, 1]."
            raise ValueError(msg)
        coverage_payload[str(key)] = converted

    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "generated_at": _utc_timestamp(),
        "pit": {"counts": counts, "bin_edges": bin_edges},
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
                str(metric): _finite_float(
                    value,
                    name=f"benchmark metric {dataset}/{model_name}/{metric}",
                )
                for metric, value in metrics.items()
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
