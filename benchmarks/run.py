"""Benchmark suite comparing Condensite against probabilistic baselines."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from condensite_cde import make_y_grid
from condensite_cde.reports import build_benchmark_report
from condensite_torch import CondensiteTorchCDEConfig
from condensite_torch.diagnostics import coverage
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf

from .datasets import load_dataset
from .models import (
    BenchmarkModel,
    CondensiteBaseline,
    ConditionalGaussianBaseline,
    MixtureDensityNetworkBaseline,
    QuantileRegressionBaseline,
    TrainingConfig,
)

DATASETS = ("heteroscedastic", "multimodal")


def _evaluate(
    model: BenchmarkModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Fit a model and compute probabilistic evaluation metrics."""
    grid = make_y_grid(y_train, grid_size=96, mode="quantile")
    model.fit(X_train, y_train)
    pdf = model.predict_pdf(X_test, grid)
    cdf = model.predict_cdf(X_test, grid)
    nll = float(nll_from_pdf(y_test, grid, pdf))
    crps = float(crps_from_cdf(y_test, grid, cdf))
    mass = np.trapz(pdf, x=grid, axis=1)
    integral_error = float(np.mean(np.abs(mass - 1.0)))
    tail = 0.05
    lo = np.array([np.interp(tail, cdf_row, grid) for cdf_row in cdf])
    hi = np.array([np.interp(1.0 - tail, cdf_row, grid) for cdf_row in cdf])
    cov = coverage(y_test, lo, hi)
    return {"nll": nll, "crps": crps, "integral_error": integral_error, "coverage90": cov}


def _downsample(
    X: np.ndarray,
    y: np.ndarray,
    target: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample without replacement for the quick benchmark mode."""
    if target < 0:
        msg = "target must be non-negative."
        raise ValueError(msg)
    if target >= X.shape[0]:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=target, replace=False)
    return X[idx], y[idx]


def _build_baselines(input_dim: int, quick: bool) -> dict[str, BenchmarkModel]:
    """Instantiate all baselines with consistent hyper-parameters."""
    condensite_config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32) if quick else (64, 64),
        m_aux=24 if quick else 48,
        epochs=4 if quick else 6,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
        normalization_lambda=0.1,
    )
    baseline_training = TrainingConfig(
        hidden_sizes=(32, 32) if quick else (64, 64),
        epochs=4 if quick else 8,
        batch_size=64 if quick else 128,
    )
    return {
        "condensite": CondensiteBaseline(config=condensite_config, random_seed=7),
        "gaussian": ConditionalGaussianBaseline(input_dim=input_dim, training=baseline_training),
        "quantile": QuantileRegressionBaseline(input_dim=input_dim, training=baseline_training),
        "mdn": MixtureDensityNetworkBaseline(input_dim=input_dim, training=baseline_training),
    }


def run_benchmarks(
    dataset_names: Iterable[str],
    quick: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run each baseline on the requested datasets."""
    results: dict[str, dict[str, dict[str, float]]] = {}
    for name in dataset_names:
        data = load_dataset(name)
        if quick:
            data.X_train, data.y_train = _downsample(data.X_train, data.y_train, 200, seed=0)
            data.X_test, data.y_test = _downsample(data.X_test, data.y_test, 100, seed=1)
        baselines = _build_baselines(data.X_train.shape[1], quick=quick)
        dataset_results: dict[str, dict[str, float]] = {}
        for model_name, model in baselines.items():
            metrics = _evaluate(model, data.X_train, data.y_train, data.X_test, data.y_test)
            dataset_results[model_name] = metrics
        results[name] = dataset_results
    return results


def main() -> None:
    """Parse CLI arguments and run the benchmark suite."""
    parser = argparse.ArgumentParser(description="Benchmark Condensite vs baseline models.")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--quick", action="store_true", help="Run a smaller benchmark for CI.")
    parser.add_argument("--output", default="benchmarks/results.json", help="Path to results JSON.")
    args = parser.parse_args()
    dataset_list = [name.strip() for name in args.datasets.split(",") if name.strip()]
    results = run_benchmarks(dataset_list, quick=args.quick)
    payload = build_benchmark_report(
        results=results,
        quick=args.quick,
        metadata={"datasets": dataset_list},
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Benchmark results saved to {output_path}")


if __name__ == "__main__":
    main()
