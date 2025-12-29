"""Grid-search bandwidth using CRPS on a validation split."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:
    print(f"Torch is unavailable; skipping example. Details: {exc}")
    sys.exit(0)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, crps_from_cdf


def make_dataset(n_samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.1 + 0.25 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.8 * np.sin(X[:, 0]) + 0.4 * X[:, 1] + noise
    return X, y


def main() -> None:  # noqa: PLR0914
    X, y = make_dataset()
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 80)
    candidates = [0.05, 0.08, 0.12, 0.18]
    results = []
    for bw in candidates:
        config = CondensiteTorchCDEConfig(
            bandwidth=bw,
            m_aux=96,
            epochs=6,
            patience=3,
            sampler="stratified",
            hidden_sizes=(48, 48),
        )
        estimator = CondensiteTorchCDE(config=config, random_seed=3).fit(X_train, y_train)
        cdf = estimator.predict_cdf(X_val, grid)
        score = crps_from_cdf(y_val, grid, cdf)
        results.append((bw, score))
        print(f"Bandwidth {bw:.2f} -> CRPS {score:.4f}")
    best_bw, best_score = min(results, key=lambda item: item[1])
    print(f"Best bandwidth: {best_bw:.2f} with CRPS {best_score:.4f}")


if __name__ == "__main__":
    main()
