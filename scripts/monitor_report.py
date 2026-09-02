"""Generate a monitoring report with feature drift and PIT drift metrics."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from condensite_torch import CondensiteTorchCDE
from condensite_torch.datasets import load_tabular
from condensite_torch.monitoring import (
    DriftThresholds,
    MonitoringThresholds,
    build_monitoring_report,
)

FloatArray = NDArray[np.float64]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitoring report generator.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--format", default="auto", choices=["auto", "csv", "tsv", "parquet"])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--schema",
        default="schemas/monitoring_report.schema.json",
        help="JSON schema file used to validate the payload.",
    )
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--psi-warn", type=float, default=0.1)
    parser.add_argument("--psi-alert", type=float, default=0.25)
    parser.add_argument("--ks-warn", type=float, default=0.05)
    parser.add_argument("--ks-alert", type=float, default=0.1)
    parser.add_argument("--pit-warn", type=float, default=0.05)
    parser.add_argument("--pit-alert", type=float, default=0.1)
    return parser.parse_args()


def _pit_values_for_dataset(
    targets: FloatArray,
    grid: FloatArray,
    cdf_values: FloatArray,
) -> FloatArray:
    """Interpolate PIT values with explicit distribution-boundary semantics."""
    if not np.all(np.isfinite(targets)):
        msg = "Targets must contain only finite values."
        raise ValueError(msg)
    values = [
        np.interp(y, grid, cdf, left=0.0, right=1.0)
        for y, cdf in zip(targets, cdf_values, strict=True)
    ]
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = _parse_args()
    estimator = CondensiteTorchCDE.load(args.model, map_location="cpu")
    baseline = load_tabular(args.baseline, target_column=args.target, file_format=args.format)
    current = load_tabular(args.current, target_column=args.target, file_format=args.format)
    if baseline[2] != current[2]:
        msg = (
            "Baseline and current datasets must expose identical feature names in the same order. "
            f"Got baseline={baseline[2]!r}, current={current[2]!r}."
        )
        raise ValueError(msg)

    baseline_features = np.asarray(baseline[0], dtype=np.float64)
    current_features = np.asarray(current[0], dtype=np.float64)
    if baseline[1] is None or current[1] is None:
        msg = "Both baseline and current datasets must include targets."
        raise ValueError(msg)
    baseline_targets = np.asarray(baseline[1], dtype=np.float64)
    current_targets = np.asarray(current[1], dtype=np.float64)
    grid = estimator._default_y_grid()  # noqa: SLF001
    cdf_base = estimator.predict_cdf(baseline_features, grid)
    cdf_curr = estimator.predict_cdf(current_features, grid)
    pit_base = _pit_values_for_dataset(baseline_targets, grid, cdf_base)
    pit_curr = _pit_values_for_dataset(current_targets, grid, cdf_curr)

    thresholds = MonitoringThresholds(
        psi=DriftThresholds(args.psi_warn, args.psi_alert),
        ks=DriftThresholds(args.ks_warn, args.ks_alert),
        pit=DriftThresholds(args.pit_warn, args.pit_alert),
    )
    payload = build_monitoring_report(
        baseline_features=baseline_features,
        current_features=current_features,
        feature_names=baseline[2],
        baseline_pit=pit_base,
        current_pit=pit_curr,
        thresholds=thresholds,
        bins=args.bins,
        metadata={
            "model_path": args.model,
            "baseline_path": args.baseline,
            "current_path": args.current,
            "target_column": args.target,
            "baseline_rows": int(baseline[0].shape[0]),
            "current_rows": int(current[0].shape[0]),
        },
    )
    _validate_against_schema(payload, args.schema)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    out_path.write_text(rendered, encoding="utf-8")
    print(rendered)


def _validate_against_schema(payload: dict[str, Any], schema_path: str | None) -> None:
    if schema_path is None:
        return
    schema_file = Path(schema_path)
    if not schema_file.exists():
        msg = f"Schema file {schema_file} not found."
        raise FileNotFoundError(msg)
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    validator = _load_jsonschema()
    if validator is None:
        expected = schema.get("properties", {}).get("schema_version", {}).get("const")
        if expected is not None and payload.get("schema_version") != expected:
            msg = (
                f"schema_version mismatch: payload={payload.get('schema_version')} "
                f"expected={expected}"
            )
            raise ValueError(msg)
        return
    validator.validate(instance=payload, schema=schema)


def _load_jsonschema() -> Any | None:
    try:
        return importlib.import_module("jsonschema")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        return None


if __name__ == "__main__":
    main()
