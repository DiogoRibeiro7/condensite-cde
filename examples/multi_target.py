"""Train multi-output estimators in independent and autoregressive modes."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    print(f"Torch unavailable: {exc}")
    sys.exit(0)

from condensite_torch import CondensiteTorchCDEConfig, MultiTargetCondensite


def make_dataset(n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 2))
    y1 = 0.4 * np.sin(X[:, 0]) + 0.3 * X[:, 1] + 0.1 * rng.normal(size=n_samples)
    y2 = y1 + 0.2 * X[:, 0] - 0.15 * X[:, 1] + 0.1 * rng.normal(size=n_samples)
    Y = np.stack([y1, y2], axis=1)
    return X, Y


def main() -> None:
    X, Y = make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=32,
        epochs=5,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
    )
    independent = MultiTargetCondensite(config, mode="independent", random_seed=4).fit(X, Y)
    autoreg = MultiTargetCondensite(config, mode="autoregressive", random_seed=4).fit(X, Y)
    shared = MultiTargetCondensite(config, mode="shared", random_seed=4).fit(X, Y)
    grid = np.linspace(Y.min() - 0.5, Y.max() + 0.5, 48)
    indep_metrics = independent.predict_quantile(X[:3], [0.1, 0.5, 0.9], y_grid=grid)
    print("Independent quantiles (first sample):", indep_metrics[0])
    shared_pdf = shared.predict_density(X[:3], grid)
    print("Shared-trunk pdf shape:", shared_pdf.shape)
    context = Y[:3]
    ar_pdf = autoreg.predict_density(X[:3], grid, y_context=context)
    print("Autoregressive pdf shape:", ar_pdf.shape)
    samples = autoreg.sample(X[:2], n_samples=5, seed=12)
    print("Sample[0]:", samples[0])


if __name__ == "__main__":
    main()
