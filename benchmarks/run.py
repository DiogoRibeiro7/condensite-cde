"""Benchmark suite comparing Condensite against simple baselines."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDEConfig
from condensite_torch.diagnostics import coverage
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf

from .datasets import load_dataset
from .models import (
    BenchmarkModel,
    CondensiteBaseline,
    ConditionalGaussianBaseline,
    QuantileRegressionBaseline,
)

DATASETS = ("heteroscedastic", "multimodal")


def _evaluate(
    model: BenchmarkModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Fit `model` and compute evaluation metrics on the test set.

    Args:
        model (BenchmarkModel): Baseline to evaluate.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.

    Returns:
        dict[str, float]: Dictionary with `nll`, `crps`, `integral_error`, and `coverage90`.

    Raises:
        ValueError: If grid creation fails.

    Side Effects:
        Trains `model` in-place.

    Complexity:
        O(model.fit + n_test * grid_size).
    """
    grid = make_y_grid(y_train, grid_size=96, mode="quantile")
    model.fit(X_train, y_train)
    pdf = model.predict_pdf(X_test, grid)
    cdf = model.predict_cdf(X_test, grid)
    nll = float(nll_from_pdf(y_test, grid, pdf))
    crps = float(crps_from_cdf(y_test, grid, cdf))
    mass = np.trapezoid(pdf, x=grid, axis=1)
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
    """Subsample without replacement for the quick benchmark mode.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        target (int): Desired number of rows to keep.
        seed (int): RNG seed to ensure determinism.

    Returns:
        tuple[np.ndarray, np.ndarray]: Downsampled `(X, y)` pair.

    Raises:
        ValueError: If `target` is negative.

    Side Effects:
        None.

    Complexity:
        O(n) due to the RNG selection.
    """
    if target >= X.shape[0]:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=target, replace=False)
    return X[idx], y[idx]


def _build_baselines(input_dim: int, quick: bool) -> dict[str, BenchmarkModel]:
    """Instantiate all baselines with consistent hyper-parameters.

    Args:
        input_dim (int): Feature dimension.
        quick (bool): Whether to use faster configs for CI.

    Returns:
        dict[str, BenchmarkModel]: Mapping of baseline name to configured model.

    Raises:
        None.

    Side Effects:
        Seeds are baked into the baseline constructors.

    Complexity:
        O(1) instantiation.
    """
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32) if quick else (64, 64),
        m_aux=24 if quick else 48,
        epochs=4 if quick else 6,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
        normalization_lambda=0.1,
    )
    return {
        "condensite": CondensiteBaseline(config=config, random_seed=7),
        "gaussian": ConditionalGaussianBaseline(
            input_dim=input_dim,
            epochs=4 if quick else 8,
            batch_size=64 if quick else 128,
        ),
        "quantile": QuantileRegressionBaseline(
            input_dim=input_dim,
            epochs=4 if quick else 8,
            batch_size=64 if quick else 128,
        ),
    }


def run_benchmarks(
    dataset_names: Iterable[str],
    quick: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run each baseline on the requested datasets.

    Args:
        dataset_names (Iterable[str]): Names to load via `benchmarks.datasets`.
        quick (bool): Whether to apply downsampling for faster iterations.

    Returns:
        dict[str, dict[str, dict[str, float]]]: Nested mapping dataset -> baseline -> metrics.

    Raises:
        ValueError: If a dataset name is unknown.

    Side Effects:
        Trains models in-place and may allocate GPU/CPU tensors.

    Complexity:
        O(sum_datasets training cost).
    """
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
    """Parse CLI arguments and run the benchmark suite.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: Propagated from argparse on invalid arguments.

    Side Effects:
        Writes JSON results to disk and prints a short summary.

    Complexity:
        O(sum_datasets training cost).
    """
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
    payload = {"quick": args.quick, "results": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Benchmark results saved to {output_path}")


if __name__ == "__main__":
    main()
