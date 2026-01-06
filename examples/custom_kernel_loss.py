"""Showcase custom kernels and loss functions."""

from __future__ import annotations

import numpy as np

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig


def main() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(256, 2))
    y = 0.6 * np.sin(X[:, 0]) - 0.2 * X[:, 1] + 0.15 * rng.normal(size=X.shape[0])
    grid = np.linspace(y.min() - 0.5, y.max() + 0.5, 80)

    config = CondensiteTorchCDEConfig(
        kernel="epanechnikov",
        loss="mae",
        hidden_sizes=(64, 64),
        m_aux=32,
        epochs=6,
        patience=3,
        sampler="sobol",
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=5).fit(X, y)
    pdf = estimator.predict_density(X[:3], grid)
    print("PDF shape:", pdf.shape)
    print("Train loss history:", [round(entry["train_loss"], 4) for entry in estimator.training_history])


if __name__ == "__main__":
    main()
