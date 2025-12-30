"""Generate regression baselines for snapshot tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover
    print(f"Torch unavailable: {exc}")
    raise SystemExit(1)

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf


def _make_dataset(n_samples: int = 192) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 2))
    y = 0.4 * np.sin(X[:, 0]) - 0.25 * X[:, 1] + 0.15 * rng.normal(size=n_samples)
    split = int(0.75 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def main() -> None:
    X_train, y_train, X_test, y_test = _make_dataset()
    grid = make_y_grid(y_train, grid_size=64, mode="quantile")
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(32, 32),
        m_aux=24,
        epochs=5,
        patience=2,
        sampler="sobol",
        bandwidth=0.12,
        normalization_lambda=0.1,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=11).fit(X_train, y_train)
    pdf = estimator.predict_density(X_test, grid)
    cdf = estimator.predict_cdf(X_test, grid)
    metrics = {
        "generated": True,
        "nll_expected": float(nll_from_pdf(y_test, grid, pdf)),
        "crps_expected": float(crps_from_cdf(y_test, grid, cdf)),
        "integral_mean_expected": float(np.mean(np.trapezoid(pdf, x=grid, axis=1))),
    }
    base = Path("tests/regression/baselines")
    base.mkdir(parents=True, exist_ok=True)
    np.savez(base / "baseline_pdf.npz", X_test=X_test.astype(np.float32), y_grid=grid.astype(np.float32), pdf_expected=pdf.astype(np.float32))
    (base / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    digest = hashlib.sha256(pdf.tobytes()).hexdigest()
    print(f"Baselines refreshed. PDF hash={digest[:16]}")


if __name__ == "__main__":
    main()
