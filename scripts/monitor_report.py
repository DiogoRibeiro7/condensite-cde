"""Generate a monitoring report with feature drift and PIT drift metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from condensite_torch import CondensiteTorchCDE
from condensite_torch.datasets import load_tabular
from condensite_torch.monitoring import (
    DriftThresholds,
    MonitoringThresholds,
    build_monitoring_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoring report generator.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--format", default="auto", choices=["auto", "csv", "parquet"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--psi-warn", type=float, default=0.1)
    parser.add_argument("--psi-alert", type=float, default=0.25)
    parser.add_argument("--ks-warn", type=float, default=0.05)
    parser.add_argument("--ks-alert", type=float, default=0.1)
    parser.add_argument("--pit-warn", type=float, default=0.05)
    parser.add_argument("--pit-alert", type=float, default=0.1)
    args = parser.parse_args()

    estimator = CondensiteTorchCDE.load(args.model, map_location="cpu")
    X_base, y_base, feature_names = load_tabular(args.baseline, target_column=args.target, file_format=args.format)
    X_curr, y_curr, _ = load_tabular(args.current, target_column=args.target, file_format=args.format)
    grid = estimator._default_y_grid()
    cdf_base = estimator.predict_cdf(X_base, grid)
    cdf_curr = estimator.predict_cdf(X_curr, grid)
    pit_base = np.array([np.interp(y, grid, cdf) for y, cdf in zip(y_base, cdf_base, strict=True)])
    pit_curr = np.array([np.interp(y, grid, cdf) for y, cdf in zip(y_curr, cdf_curr, strict=True)])

    thresholds = MonitoringThresholds(
        psi=DriftThresholds(args.psi_warn, args.psi_alert),
        ks=DriftThresholds(args.ks_warn, args.ks_alert),
        pit=DriftThresholds(args.pit_warn, args.pit_alert),
    )
    payload = build_monitoring_report(
        baseline_features=X_base,
        current_features=X_curr,
        feature_names=feature_names,
        baseline_pit=pit_base,
        current_pit=pit_curr,
        thresholds=thresholds,
        bins=args.bins,
        metadata={
            "model_path": args.model,
            "baseline_path": args.baseline,
            "current_path": args.current,
            "target_column": args.target,
            "baseline_rows": int(X_base.shape[0]),
            "current_rows": int(X_curr.shape[0]),
        },
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
