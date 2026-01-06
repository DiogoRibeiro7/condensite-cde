"""Benchmark global vs local grid inference across benchmark datasets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    print(json.dumps({"error": f"Torch unavailable: {exc}"}))
    raise SystemExit(0)

from benchmarks.datasets import load_dataset

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, make_local_grid

DEFAULT_DATASETS = ("heteroscedastic", "multimodal", "heavy_tail")


def _fit_estimator(X_train: np.ndarray, y_train: np.ndarray) -> CondensiteTorchCDE:
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(48, 48),
        m_aux=32,
        epochs=6,
        patience=3,
        sampler="sobol",
        bandwidth=0.1,
        inference_batch_size=64,
        inference_grid_chunk_size=256,
    )
    return CondensiteTorchCDE(config=config, random_seed=11).fit(X_train, y_train)


def _time_eval(estimator: CondensiteTorchCDE, X: np.ndarray, y: np.ndarray, **kwargs) -> tuple[dict[str, float], float]:
    start = time.perf_counter()
    metrics = estimator.evaluate(X, y, **kwargs)
    duration = time.perf_counter() - start
    return metrics, duration


def benchmark_dataset(
    name: str,
    *,
    grid_size: int,
    q_low: float,
    q_high: float,
    padding: float,
) -> dict[str, object]:
    bundle = load_dataset(name)
    estimator = _fit_estimator(bundle.X_train, bundle.y_train)
    global_grid = estimator._default_y_grid()
    metrics_global, runtime_global = _time_eval(
        estimator,
        bundle.X_test,
        bundle.y_test,
        y_grid=global_grid,
    )

    local_kwargs = {
        "use_local_grid": True,
        "local_grid_params": {
            "grid_size": grid_size,
            "q_low": q_low,
            "q_high": q_high,
            "padding": padding,
        },
    }
    metrics_local, runtime_local = _time_eval(estimator, bundle.X_test, bundle.y_test, **local_kwargs)

    # Coverage diagnostic to quantify how often y_test sits inside grid bounds.
    local_grids = make_local_grid(
        estimator,
        bundle.X_test,
        grid_size=grid_size,
        q_low=q_low,
        q_high=q_high,
        padding=padding,
    )
    coverage_global = float(
        np.mean((bundle.y_test >= global_grid[0]) & (bundle.y_test <= global_grid[-1])),
    )
    coverage_local = float(
        np.mean(
            (bundle.y_test >= local_grids[:, 0])
            & (bundle.y_test <= local_grids[:, -1]),
        ),
    )

    return {
        "dataset": name,
        "global": {
            "runtime_sec": runtime_global,
            "nll": metrics_global["nll"],
            "crps": metrics_global["crps"],
            "integral_error": metrics_global["integral_error"],
            "grid_points": int(global_grid.size),
            "coverage": coverage_global,
        },
        "local": {
            "runtime_sec": runtime_local,
            "nll": metrics_local["nll"],
            "crps": metrics_local["crps"],
            "integral_error": metrics_local["integral_error"],
            "grid_points": int(grid_size),
            "coverage": coverage_local,
            "q_low": q_low,
            "q_high": q_high,
            "padding": padding,
        },
        "speedup": runtime_global / runtime_local if runtime_local > 0 else None,
        "crps_delta": metrics_global["crps"] - metrics_local["crps"],
        "nll_delta": metrics_global["nll"] - metrics_local["nll"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare global vs local grid inference.")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names from benchmarks.datasets.*",
    )
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    parser.add_argument("--padding", type=float, default=0.1)
    parser.add_argument("--output", default="reports/local_grid_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    results = [
        benchmark_dataset(
            name,
            grid_size=args.grid_size,
            q_low=args.q_low,
            q_high=args.q_high,
            padding=args.padding,
        )
        for name in dataset_names
    ]
    payload = {
        "config": {
            "grid_size": args.grid_size,
            "q_low": args.q_low,
            "q_high": args.q_high,
            "padding": args.padding,
            "datasets": dataset_names,
        },
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
