from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    header = [f"f{idx}" for idx in range(X.shape[1])] + ["target"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row, target in zip(X, y, strict=True):
            writer.writerow([float(val) for val in row] + [float(target)])


def _validate_report_schema(payload: dict, schema: dict) -> None:
    assert payload.get("schema_version") == schema["properties"]["schema_version"]["const"]
    assert isinstance(payload.get("feature_drift"), list)
    for entry in payload["feature_drift"]:
        assert {"feature", "psi", "ks"} <= set(entry)
        for metric_name in ("psi", "ks"):
            metric_payload = entry[metric_name]
            assert metric_payload["status"] in {"ok", "warn", "alert"}
            assert {"warn", "alert"} <= set(metric_payload["thresholds"])
    pit = payload["pit"]
    for section in ("baseline", "current"):
        hist = pit["histogram"][section]
        assert len(hist["counts"]) == len(hist["bin_edges"]) - 1
    assert pit["drift"]["status"] in {"ok", "warn", "alert"}


def test_monitor_report_script_generates_valid_payload(tmp_path, trained_estimator, torch_available) -> None:
    estimator, X, y, _grid = trained_estimator
    model_dir = tmp_path / "model"
    estimator.save(model_dir)

    baseline_path = tmp_path / "baseline.csv"
    current_path = tmp_path / "current.csv"
    split = X.shape[0] // 2
    _write_csv(baseline_path, X[:split], y[:split])
    _write_csv(current_path, X[split:], y[split:])

    output_path = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "scripts/monitor_report.py",
        "--model",
        str(model_dir),
        "--baseline",
        str(baseline_path),
        "--current",
        str(current_path),
        "--target",
        "target",
        "--output",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert output_path.exists(), result.stderr

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    schema_path = Path("schemas/monitoring_report.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _validate_report_schema(payload, schema)
