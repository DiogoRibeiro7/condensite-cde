"""Train with mixed numeric/categorical data using the built-in preprocessor."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment dependent
    print(f"Torch unavailable: {exc}")
    sys.exit(0)


from condensite_torch import (
    CondensiteTorchCDE,
    CondensiteTorchCDEConfig,
    TabularPreprocessorConfig,
)

CAT_THRESHOLD = 0.5
MISSING_PROB = 0.15


def make_dataset(n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    numeric = rng.normal(size=(n_samples, 2))
    categorical = np.where(
        rng.random(size=(n_samples, 1)) > CAT_THRESHOLD,
        "red",
        "blue",
    ).astype(object)
    categorical[rng.random(size=categorical.shape) < MISSING_PROB] = None
    X = np.concatenate([numeric, categorical], axis=1)
    coef = (categorical[:, 0] == "red").astype(float)
    y = 0.5 * numeric[:, 0] - 0.3 * numeric[:, 1] + coef + 0.1 * rng.normal(size=n_samples)
    return X, y


def main() -> None:
    X, y = make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=32,
        epochs=5,
        patience=2,
        sampler="sobol",
        bandwidth=0.12,
        preprocessor=TabularPreprocessorConfig(add_missing_indicator=True),
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=3).fit(X, y)
    grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 64)
    metrics = estimator.evaluate(X[:50], y[:50], y_grid=grid)
    print("Evaluation metrics on holdout:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
