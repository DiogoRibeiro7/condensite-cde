"""Compute permutation feature importances for CRPS on a toy dataset."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment dependent
    print(f"Torch unavailable: {exc}")
    sys.exit(0)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, permutation_importance


def make_dataset(n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(n_samples, 3))
    noise = (0.15 + 0.25 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * np.sin(X[:, 0]) - 0.3 * X[:, 1] + 0.2 * X[:, 2] + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=32,
        epochs=5,
        patience=2,
        sampler="sobol",
        val_fraction=0.2,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=7).fit(X, y)
    importance = permutation_importance(
        estimator,
        X,
        y,
        metric="crps",
        n_repeats=5,
        random_seed=42,
    )
    print(f"Baseline {importance.metric_name.upper()}: {importance.baseline_score:.4f}")
    for idx, (mean_imp, std_imp) in enumerate(
        zip(importance.importances_mean, importance.importances_std, strict=True),
    ):
        print(f"Feature {idx}: Δ{importance.metric_name.upper()}={mean_imp:.4f} ± {std_imp:.4f}")


if __name__ == "__main__":
    main()
