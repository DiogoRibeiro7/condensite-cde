from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDEConfig
from condensite_torch.diagnostics import coverage
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf

from benchmarks.datasets import load_dataset
from benchmarks.models import (
    BenchmarkModel,
    ConditionalGaussianBaseline,
    CondensiteBaseline,
    MixtureDensityNetworkBaseline,
    QuantileRegressionBaseline,
)


def _quantile_from_cdf(cdf: np.ndarray, y_grid: np.ndarray, prob: float) -> np.ndarray:
    quantiles = np.empty(cdf.shape[0], dtype=np.float64)
    for idx in range(cdf.shape[0]):
        quantiles[idx] = np.interp(prob, cdf[idx], y_grid, left=y_grid[0], right=y_grid[-1])
    return quantiles


def evaluate_model(model: BenchmarkModel, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    grid = make_y_grid(y_train, grid_size=96, mode="quantile")
    model.fit(X_train, y_train)
    pdf = model.predict_pdf(X_test, grid)
    cdf = model.predict_cdf(X_test, grid)
    nll = float(nll_from_pdf(y_test, grid, pdf))
    crps = float(crps_from_cdf(y_test, grid, cdf))
    mass = np.trapezoid(pdf, x=grid, axis=1)
    integral_error = float(np.mean(np.abs(mass - 1.0)))
    tail = 0.05
    lo = _quantile_from_cdf(cdf, grid, tail)
    hi = _quantile_from_cdf(cdf, grid, 1.0 - tail)
    cov = coverage(y_test, lo, hi)
    return {
        "nll": nll,
        "crps": crps,
        "integral_error": integral_error,
        "coverage90": cov,
    }


def build_baselines(input_dim: int) -> dict[str, BenchmarkModel]:
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(64, 64),
        m_aux=48,
        epochs=6,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
        normalization_lambda=0.1,
    )
    return {
        "condensite": CondensiteBaseline(config=config, random_seed=7),
        "gaussian": ConditionalGaussianBaseline(input_dim=input_dim),
        "quantile": QuantileRegressionBaseline(input_dim=input_dim),
        "mdn": MixtureDensityNetworkBaseline(input_dim=input_dim),
    }


def main() -> None:
    datasets = ["heteroscedastic", "multimodal", "heavy_tail", "skewed", "outliers"]
    results: dict[str, dict[str, dict[str, float]]] = {}
    for name in datasets:
        data = load_dataset(name)
        baselines = build_baselines(data.X_train.shape[1])
        dataset_results: dict[str, dict[str, float]] = {}
        for model_name, model in baselines.items():
            metrics = evaluate_model(model, data.X_train, data.y_train, data.X_test, data.y_test)
            dataset_results[model_name] = metrics
        results[name] = dataset_results

    payload = {"results": results}
    target = Path("benchmarks") / "results.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Benchmark results stored in {target}")


if __name__ == "__main__":
    main()
