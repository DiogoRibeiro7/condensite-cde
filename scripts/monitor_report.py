"""Generate a monitoring report with feature drift and PIT drift metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from condensite_torch import (
    CondensiteTorchCDE,
    ks_drift,
    pit_drift,
    pit_histogram,
    population_stability_index,
)
from condensite_torch.datasets import load_tabular


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoring report generator.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--format", default="auto", choices=["auto", "csv", "parquet"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    estimator = CondensiteTorchCDE.load(args.model, map_location="cpu")
    X_base, y_base, feature_names = load_tabular(args.baseline, target_column=args.target, file_format=args.format)
    X_curr, y_curr, _ = load_tabular(args.current, target_column=args.target, file_format=args.format)
    drift_stats = {}
    for idx, name in enumerate(feature_names):
        baseline_col = np.asarray(X_base[:, idx], dtype=np.float64)
        current_col = np.asarray(X_curr[:, idx], dtype=np.float64)
        psi = population_stability_index(baseline_col, current_col)
        ks = ks_drift(baseline_col, current_col)
        drift_stats[name] = {"psi": psi, "ks": ks}

    grid = estimator._default_y_grid()  # noqa: SLF001
    cdf_base = estimator.predict_cdf(X_base, grid)
    cdf_curr = estimator.predict_cdf(X_curr, grid)
    pit_base = np.array([np.interp(y, grid, cdf) for y, cdf in zip(y_base, cdf_base, strict=True)])
    pit_curr = np.array([np.interp(y, grid, cdf) for y, cdf in zip(y_curr, cdf_curr, strict=True)])
    pit_hist_base = pit_histogram(pit_base)
    pit_hist_current = pit_histogram(pit_curr)
    pit_stats = {
        "baseline_counts": pit_hist_base["counts"].tolist(),
        "current_counts": pit_hist_current["counts"].tolist(),
        "bin_edges": pit_hist_base["edges"].tolist(),
        "pit_drift": pit_drift(pit_base, pit_curr),
    }

    payload = {"feature_drift": drift_stats, "pit": pit_stats}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
