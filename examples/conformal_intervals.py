"""Demonstrate split-conformal prediction intervals built on Condensite."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment dependent
    print(f"Torch unavailable: {exc}")
    sys.exit(0)

from condensite_torch import CondensiteTorchCDEConfig
from condensite_torch.conformal import ConformalCDEWrapper


def make_dataset(n_samples: int = 360) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n_samples, 3))
    noise = (0.12 + 0.15 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * np.sin(X[:, 0]) - 0.35 * X[:, 1] + 0.2 * X[:, 2] + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    train_end = 200
    cal_end = 280
    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X[cal_end:], y[cal_end:]

    config = CondensiteTorchCDEConfig(
        hidden_sizes=(48, 48),
        m_aux=64,
        epochs=6,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
    )
    coverage_target = 0.9
    grid = np.linspace(y_train.min() - 0.5, y_train.max() + 0.5, 96)

    quantile_wrapper = ConformalCDEWrapper(config, random_seed=11).fit(
        X_train,
        y_train,
        X_cal,
        y_cal,
        method="quantile",
    )
    q_lower, q_upper = quantile_wrapper.predict_interval(
        X_test,
        coverage=coverage_target,
        y_grid=grid,
    )
    q_empirical = ((y_test >= q_lower) & (y_test <= q_upper)).mean()

    cdf_wrapper = ConformalCDEWrapper(config, random_seed=11).fit(
        X_train,
        y_train,
        X_cal,
        y_cal,
        method="cdf",
    )
    c_lower, c_upper = cdf_wrapper.predict_interval(
        X_test,
        coverage=coverage_target,
        y_grid=grid,
    )
    c_empirical = ((y_test >= c_lower) & (y_test <= c_upper)).mean()

    print(f"Target coverage: {coverage_target:.2f}")
    print(f"Quantile method coverage: {q_empirical:.3f}")
    print(f"CDF method coverage: {c_empirical:.3f}")
    for idx in range(min(5, X_test.shape[0])):
        print(
            "Sample "
            f"{idx}: quant=({q_lower[idx]:.3f}, {q_upper[idx]:.3f}), "
            f"cdf=({c_lower[idx]:.3f}, {c_upper[idx]:.3f}), obs={y_test[idx]:.3f}",
        )


if __name__ == "__main__":
    main()
