"""Demonstrate split-conformal prediction intervals built on Condensite."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment dependent
    print(f"Torch unavailable: {exc}")
    sys.exit(0)

from condensite_torch import CondensiteTorchCDEConfig
from condensite_torch.conformal import ConformalCDEWrapper


@dataclass(slots=True)
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def make_dataset(n_samples: int = 360) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n_samples, 3))
    noise = (0.12 + 0.15 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * np.sin(X[:, 0]) - 0.35 * X[:, 1] + 0.2 * X[:, 2] + noise
    return X, y


def _build_base_config() -> CondensiteTorchCDEConfig:
    return CondensiteTorchCDEConfig(
        hidden_sizes=(48, 48),
        m_aux=64,
        epochs=6,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
    )


def _split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_end: int = 200,
    cal_end: int = 280,
) -> DatasetSplits:
    return DatasetSplits(
        X[:train_end],
        y[:train_end],
        X[train_end:cal_end],
        y[train_end:cal_end],
        X[cal_end:],
        y[cal_end:],
    )


def _fit_wrapper(
    config: CondensiteTorchCDEConfig,
    splits: DatasetSplits,
    *,
    method: str,
) -> ConformalCDEWrapper:
    return ConformalCDEWrapper(config, random_seed=11).fit(
        splits.X_train,
        splits.y_train,
        splits.X_cal,
        splits.y_cal,
        method=method,
    )


def _coverage_rate(
    wrapper: ConformalCDEWrapper,
    splits: DatasetSplits,
    *,
    coverage: float,
    grid: np.ndarray,
) -> float:
    lower, upper = wrapper.predict_interval(
        splits.X_test,
        coverage=coverage,
        y_grid=grid,
    )
    return float(((splits.y_test >= lower) & (splits.y_test <= upper)).mean())


def _print_sample_rows(
    wrappers: dict[str, ConformalCDEWrapper],
    splits: DatasetSplits,
    *,
    coverage: float,
    grid: np.ndarray,
) -> None:
    limit = min(5, splits.X_test.shape[0])
    for idx in range(limit):
        q_lower, q_upper = wrappers["quantile"].predict_interval(
            splits.X_test[idx : idx + 1],
            coverage=coverage,
            y_grid=grid,
        )
        c_lower, c_upper = wrappers["cdf"].predict_interval(
            splits.X_test[idx : idx + 1],
            coverage=coverage,
            y_grid=grid,
        )
        print(
            "Sample "
            f"{idx}: quant=({q_lower[0]:.3f}, {q_upper[0]:.3f}), "
            f"cdf=({c_lower[0]:.3f}, {c_upper[0]:.3f}), obs={splits.y_test[idx]:.3f}",
        )


def main() -> None:
    X, y = make_dataset()
    splits = _split_dataset(X, y)
    coverage_target = 0.9
    config = _build_base_config()
    grid = np.linspace(splits.y_train.min() - 0.5, splits.y_train.max() + 0.5, 96)

    wrappers = {
        "quantile": _fit_wrapper(config, splits, method="quantile"),
        "cdf": _fit_wrapper(config, splits, method="cdf"),
    }
    coverages = {
        name: _coverage_rate(wrapper, splits, coverage=coverage_target, grid=grid)
        for name, wrapper in wrappers.items()
    }

    print(f"Target coverage: {coverage_target:.2f}")
    print(f"Quantile method coverage: {coverages['quantile']:.3f}")
    print(f"CDF method coverage: {coverages['cdf']:.3f}")
    _print_sample_rows(wrappers, splits, coverage=coverage_target, grid=grid)


if __name__ == "__main__":
    main()
