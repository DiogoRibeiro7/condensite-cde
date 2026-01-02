"""Demonstrate a simple what-if analysis by nudging a feature."""

from __future__ import annotations

import sys

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment dependent
    print(f"Torch unavailable: {exc}")
    sys.exit(0)

from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig, what_if


def make_dataset(n_samples: int = 240) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(n_samples, 3))
    noise = (0.2 + 0.15 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.4 * np.sin(X[:, 0]) - 0.35 * X[:, 1] + 0.25 * X[:, 2] + noise
    return X, y


def main() -> None:
    X, y = make_dataset()
    cfg = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=32,
        epochs=5,
        patience=2,
        sampler="sobol",
        val_fraction=0.2,
    )
    estimator = CondensiteTorchCDE(cfg, random_seed=5).fit(X, y)
    idx = 0
    row = X[idx]
    change = {0: row[0] + 1.0}
    result = what_if(
        estimator,
        row,
        change,
        outputs=("quantiles", "tail_probs"),
        quantile_probs=(0.1, 0.5, 0.9),
        tail_thresholds=(-0.5, 0.0, 0.5),
    )
    print(f"Feature change: {change}")
    base_quant = result.baseline["quantiles"]
    mod_quant = result.modified["quantiles"]
    print("Quantiles (baseline -> modified):")
    for prob, base_val, mod_val in zip(
        base_quant["probs"],
        base_quant["values"],
        mod_quant["values"],
        strict=True,
    ):
        print(f"  q={prob:.2f}: {base_val:.3f} -> {mod_val:.3f}")
    base_tail = result.baseline["tail_probs"]
    mod_tail = result.modified["tail_probs"]
    print("Tail probabilities (baseline -> modified):")
    for thr, base_val, mod_val in zip(
        base_tail["thresholds"],
        base_tail["values"],
        mod_tail["values"],
        strict=True,
    ):
        print(f"  P(Y >= {thr:.2f}): {base_val:.3f} -> {mod_val:.3f}")


if __name__ == "__main__":
    main()
