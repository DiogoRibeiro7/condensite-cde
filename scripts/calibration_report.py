"""Generate calibration diagnostics (PIT + interval coverage) for a toy dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment dependent
    print(json.dumps({"error": f"Torch unavailable: {exc}"}))
    raise SystemExit(0)

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig
from condensite_torch.diagnostics import coverage_rate, pit_values


def make_dataset(n_samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(n_samples, 3))
    noise = (0.1 + 0.25 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * np.sin(X[:, 0]) - 0.3 * X[:, 1] + 0.2 * X[:, 2] + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    split = int(0.75 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    config = CondensiteTorchCDEConfig(
        hidden_sizes=(48, 48),
        m_aux=64,
        epochs=6,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
        normalization_lambda=0.1,
        val_fraction=0.0,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=13).fit(X_train, y_train)
    grid = make_y_grid(y_train, grid_size=96, mode="quantile")

    cdf = estimator.predict_cdf(X_val, grid)
    pit = pit_values(y_val, grid, cdf)
    hist_counts, hist_edges = np.histogram(pit, bins=20, range=(0.0, 1.0))

    levels = [0.5, 0.8, 0.9, 0.95]
    coverage_results: dict[str, float] = {}
    for level in levels:
        tail = (1.0 - level) / 2.0
        quantiles = estimator.predict_quantile(X_val, [tail, 1.0 - tail], y_grid=grid)
        cov = coverage_rate(y_val, quantiles[:, 0], quantiles[:, 1])
        coverage_results[f"p{int(level * 100):02d}"] = cov

    payload = {
        "pit": {
            "counts": hist_counts.tolist(),
            "bin_edges": hist_edges.tolist(),
        },
        "coverage": coverage_results,
    }

    target = Path("reports") / "calibration.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Calibration report written to {target}")


if __name__ == "__main__":
    main()
