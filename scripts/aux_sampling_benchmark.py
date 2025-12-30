"""Benchmark auxiliary sampling strategies on a synthetic dataset."""

from __future__ import annotations

import json
import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - torch availability is environment-specific
    print(json.dumps({"error": f"Torch unavailable: {exc}"}), file=sys.stderr)
    sys.exit(0)

from condensite_cde import make_y_grid
from condensite_torch import (
    CondensiteTorchCDE,
    CondensiteTorchCDEConfig,
    crps_from_cdf,
    nll_from_pdf,
)


def make_dataset(n_samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 3))
    hetero = (0.2 + 0.1 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.4 * np.sin(X[:, 0]) - 0.3 * X[:, 1] + 0.2 * X[:, 2] + hetero
    return X, y


def evaluate_sampler(
    sampler: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    grid: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, float]:
    nll_scores: list[float] = []
    crps_scores: list[float] = []
    for idx, seed in enumerate(seeds):
        config = CondensiteTorchCDEConfig(
            hidden_sizes=(48, 48),
            m_aux=48,
            epochs=6,
            patience=2,
            batch_size=64,
            lr=3e-3,
            sampler=sampler,
            val_fraction=0.0,
            monitor_metric="val_nll",
            normalization_lambda=0.1,
        )
        estimator = CondensiteTorchCDE(config=config, random_seed=seed + idx).fit(X_train, y_train)
        pdf = estimator.predict_density(X_val, grid)
        cdf = estimator.predict_cdf(X_val, grid)
        nll_scores.append(float(nll_from_pdf(y_val, grid, pdf)))
        crps_scores.append(float(crps_from_cdf(y_val, grid, cdf)))
    nll_arr = np.array(nll_scores, dtype=np.float64)
    crps_arr = np.array(crps_scores, dtype=np.float64)
    return {
        "nll_mean": float(nll_arr.mean()),
        "nll_std": float(nll_arr.std(ddof=0)),
        "crps_mean": float(crps_arr.mean()),
        "crps_std": float(crps_arr.std(ddof=0)),
    }


def main() -> None:
    X, y = make_dataset()
    split = int(0.75 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    grid = make_y_grid(y_train, grid_size=96, mode="quantile")

    samplers = ("iid", "stratified", "lhs", "sobol", "importance")
    seeds = (11, 23, 37)
    results: dict[str, dict[str, float]] = {}
    for sampler in samplers:
        metrics = evaluate_sampler(sampler, X_train, y_train, X_val, y_val, grid, seeds)
        results[sampler] = metrics

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
