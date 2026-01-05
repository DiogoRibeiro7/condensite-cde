"""Use per-row local grids for faster inference and better tail coverage."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    print(f"Torch unavailable: {exc}")
    sys.exit(0)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, make_local_grid


def make_dataset(n_samples: int = 180) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 2))
    y = (0.2 + 0.3 * np.abs(X[:, 0])) * rng.normal(size=n_samples) + 0.4 * X[:, 0]
    return X, y


def main() -> None:
    X, y = make_dataset()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(24, 24),
        m_aux=24,
        epochs=4,
        patience=2,
        sampler="sobol",
        bandwidth=0.1,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=5).fit(X, y)
    test_X = X[:5]
    global_grid = estimator._default_y_grid()  # noqa: SLF001 demo only
    local_grids = make_local_grid(estimator, test_X, grid_size=64)
    pdf_global = estimator.predict_density(test_X, global_grid)
    pdf_local = estimator.predict_density(test_X, local_grids)
    print("Global grid pdf shape:", pdf_global.shape)
    print("Local grid pdf shape:", pdf_local.shape)
    for idx in range(test_X.shape[0]):
        print(
            f"Row {idx}: local grid min={local_grids[idx, 0]:.3f}, "
            f"max={local_grids[idx, -1]:.3f}",
        )


if __name__ == "__main__":
    main()
