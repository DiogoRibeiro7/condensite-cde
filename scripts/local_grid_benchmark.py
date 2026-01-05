"""Benchmark global vs local grid inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    print(json.dumps({"error": f"Torch unavailable: {exc}"}))
    raise SystemExit(0)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, make_local_grid


def make_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(360, 3))
    y = 0.5 * np.sin(X[:, 0]) - 0.3 * X[:, 1] + 0.2 * X[:, 2] + (0.2 + 0.1 * np.abs(X[:, 0])) * rng.normal(
        size=X.shape[0],
    )
    return X, y


def benchmark() -> dict[str, float]:
    X, y = make_dataset()
    split = int(0.75 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=32,
        epochs=5,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=8).fit(X_train, y_train)
    global_grid = estimator._default_y_grid()  # noqa: SLF001

    start = time.perf_counter()
    metrics_global = estimator.evaluate(X_test, y_test, y_grid=global_grid)
    duration_global = time.perf_counter() - start

    local_grids = make_local_grid(estimator, X_test, grid_size=64)
    start = time.perf_counter()
    metrics_local = estimator.evaluate(X_test, y_test, y_grid=local_grids)
    duration_local = time.perf_counter() - start

    return {
        "global_runtime": duration_global,
        "local_runtime": duration_local,
        "global_crps": metrics_global["crps"],
        "local_crps": metrics_local["crps"],
        "global_nll": metrics_global["nll"],
        "local_nll": metrics_local["nll"],
    }


def main() -> None:
    results = benchmark()
    target = Path("reports")
    target.mkdir(parents=True, exist_ok=True)
    out_path = target / "local_grid_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
