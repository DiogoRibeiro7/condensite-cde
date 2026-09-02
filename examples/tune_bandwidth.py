"""Example showing how to tune bandwidth and m_aux automatically."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:
    print(f"Torch is unavailable; skipping example. Details: {exc}")
    sys.exit(0)

from condensite_cde.tune import tune_bandwidth_m_aux
from condensite_torch import CondensiteTorchCDEConfig


def make_dataset(n_samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(n_samples, 3))
    hetero_noise = (0.1 + 0.25 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.4 * X[:, 0] - 0.2 * X[:, 1] + np.sin(X[:, 2]) + hetero_noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    base_config = CondensiteTorchCDEConfig(
        epochs=6,
        patience=2,
        batch_size=64,
        sampler="stratified",
        val_fraction=0.2,
        amp=False,
    )
    result = tune_bandwidth_m_aux(
        X,
        y,
        bandwidths=[0.08, 0.12, 0.16],
        m_aux_values=[64, 96],
        base_config=base_config,
        metric="val_crps",
        random_seed=5,
    )
    best = result.best_config
    print(
        f"Best configuration: bandwidth={best.bandwidth:.3f}, "
        f"m_aux={best.m_aux}, {result.metric_name}={result.best_metric:.4f}",
    )
    print(f"Run directory: {result.run_dir}")
    print("All trials:")
    for trial in result.history:
        print(
            f"  bw={trial['bandwidth']:.3f} "
            f"m_aux={int(trial['m_aux']):>3} "
            f"{result.metric_name}={trial[result.metric_name]:.4f}",
        )


if __name__ == "__main__":
    main()
