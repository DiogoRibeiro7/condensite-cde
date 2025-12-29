"""Train Condensite Torch on a heteroscedastic synthetic dataset."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:
    print(f"Torch is unavailable; skipping example. Details: {exc}")
    sys.exit(0)

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


def make_dataset(n_samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    noise_scale = 0.15 + 0.35 * (np.sin(X[:, 0]) ** 2)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + noise_scale * rng.normal(size=n_samples)
    return X, y


def main() -> None:
    X, y = make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(64, 64),
        m_aux=128,
        bandwidths=(0.06, 0.12, 0.2),
        bandwidth_strategy="best",
        monitor_metric="val_nll",
        val_fraction=0.2,
        epochs=10,
        patience=3,
        sampler="sobol",
        amp=False,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=17).fit(X, y)
    grid = make_y_grid(y, grid_size=100, mode="quantile")
    pdf_best = estimator.predict_density(X[:2], grid, head="best")
    pdf_mean = estimator.predict_density(X[:2], grid, head="mean")
    print("PDF summary for two samples (head='best' vs. head='mean'):")
    for idx, (row_best, row_mean) in enumerate(zip(pdf_best, pdf_mean, strict=True)):
        mass_best = np.trapezoid(row_best, grid)
        mass_mean = np.trapezoid(row_mean, grid)
        print(
            f"  Sample {idx}: best_mass={mass_best:.3f}, "
            f"mean_mass={mass_mean:.3f}, best_peak={row_best.max():.3f}",
        )


if __name__ == "__main__":
    main()
