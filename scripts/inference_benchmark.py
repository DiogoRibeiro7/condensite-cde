"""Benchmark inference throughput with and without chunked evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


def _make_dataset(n_train: int = 1024, n_test: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(21)
    X = rng.normal(size=(n_train + n_test, 4))
    y = 0.6 * np.sin(X[:, 0]) - 0.25 * X[:, 1] + 0.1 * rng.normal(size=X.shape[0])
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _time_predict(estimator: CondensiteTorchCDE, X: np.ndarray, grid: np.ndarray) -> float:
    start = time.perf_counter()
    estimator.predict_density(X, grid)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference batching.")
    parser.add_argument("--output", default="reports/inference_benchmark.json", help="Where to store the JSON results.")
    parser.add_argument("--row-batch", type=int, default=64, help="Row batch size for chunked mode.")
    parser.add_argument("--grid-chunk", type=int, default=64, help="Grid chunk size for chunked mode.")
    args = parser.parse_args()

    X_train, y_train, X_test, _ = _make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(64, 64),
        m_aux=32,
        epochs=6,
        patience=2,
        sampler="sobol",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=13).fit(X_train, y_train)
    grid = make_y_grid(y_train, grid_size=128, mode="quantile")

    # Baseline (no chunking)
    estimator.config.inference_batch_size = None
    estimator.config.inference_grid_chunk_size = None
    baseline = _time_predict(estimator, X_test, grid)

    # Chunked inference
    estimator.config.inference_batch_size = args.row_batch
    estimator.config.inference_grid_chunk_size = args.grid_chunk
    chunked = _time_predict(estimator, X_test, grid)

    payload = {
        "row_batch": args.row_batch,
        "grid_chunk": args.grid_chunk,
        "baseline_seconds": baseline,
        "chunked_seconds": chunked,
        "speedup": baseline / chunked if chunked > 0 else None,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
