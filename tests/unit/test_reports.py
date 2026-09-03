from __future__ import annotations

import json
from pathlib import Path

from condensite_cde.reports import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    CALIBRATION_REPORT_SCHEMA_VERSION,
    build_benchmark_report,
    build_calibration_report,
)


def _load_schema(path: str) -> dict:
    schema_path = Path(path)
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_calibration_report_matches_schema() -> None:
    payload = build_calibration_report(
        pit_histogram={"counts": [5, 3], "bin_edges": [0.0, 0.5, 1.0]},
        coverage={"p50": 0.52, "p90": 0.88},
        metadata={"dataset": "toy"},
    )
    schema = _load_schema("schemas/calibration_report.schema.json")
    assert payload["schema_version"] == CALIBRATION_REPORT_SCHEMA_VERSION
    assert payload["schema_version"] == schema["properties"]["schema_version"]["const"]
    assert "generated_at" in payload
    assert payload["coverage"].keys() == {"p50", "p90"}
    assert len(payload["pit"]["bin_edges"]) == len(payload["pit"]["counts"]) + 1
    assert payload["metadata"]["dataset"] == "toy"


def test_benchmark_report_matches_schema() -> None:
    expected_nll = 1.2
    payload = build_benchmark_report(
        results={"dataset": {"model": {"nll": expected_nll, "crps": 0.5}}},
        quick=True,
        metadata={"datasets": ["dataset"]},
    )
    schema = _load_schema("schemas/benchmark_report.schema.json")
    assert payload["schema_version"] == BENCHMARK_REPORT_SCHEMA_VERSION
    assert payload["schema_version"] == schema["properties"]["schema_version"]["const"]
    assert payload["quick"] is True
    assert payload["results"]["dataset"]["model"]["nll"] == expected_nll
    assert payload["metadata"]["datasets"] == ["dataset"]
