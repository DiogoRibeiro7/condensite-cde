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
from condensite_torch.diagnostics import coverage_rate, pit_values
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf

QUANTILE_PROBS = np.array([0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975], dtype=np.float64)
QUANTILE_INDEX = {float(prob): idx for idx, prob in enumerate(QUANTILE_PROBS)}
INTERVAL_SPECS = {
    "50": (0.25, 0.75),
    "80": (0.1, 0.9),
    "90": (0.05, 0.95),
    "95": (0.025, 0.975),
}
TAIL_THRESHOLDS = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
PIT_BINS = np.linspace(0.0, 1.0, 11, dtype=np.float64)
DTYPE_SPECS = {"float32": np.float32, "float64": np.float64}


def _make_dataset(n_samples: int = 192) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 2))
    y = 0.4 * np.sin(X[:, 0]) - 0.25 * X[:, 1] + 0.15 * rng.normal(size=n_samples)
    split = int(0.75 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def _tail_matrix(
    estimator: CondensiteTorchCDE,
    X_test: np.ndarray,
    grid: np.ndarray,
    side: str,
) -> np.ndarray:
    values = []
    for threshold in TAIL_THRESHOLDS:
        probs = estimator.predict_tail_prob(X_test, float(threshold), side=side, y_grid=grid)
        values.append(probs.reshape(-1, 1))
    return np.hstack(values).astype(np.float64, copy=False)


def _compute_metrics(
    y_true: np.ndarray,
    grid: np.ndarray,
    pdf: np.ndarray,
    cdf: np.ndarray,
    quantiles: np.ndarray,
) -> dict[str, float | dict[str, float] | list[int]]:
    grid_float = grid.astype(np.float64, copy=False)
    nll = float(nll_from_pdf(y_true, grid_float, pdf))
    crps = float(crps_from_cdf(y_true, grid_float, cdf))
    integral = float(np.mean(np.trapezoid(pdf, x=grid_float, axis=1)))
    coverage_summary: dict[str, float] = {}
    for name, (lo, hi) in INTERVAL_SPECS.items():
        coverage_summary[name] = coverage_rate(
            y_true,
            quantiles[:, QUANTILE_INDEX[lo]],
            quantiles[:, QUANTILE_INDEX[hi]],
        )
    pit = pit_values(y_true, grid_float, cdf)
    counts, _ = np.histogram(pit, bins=PIT_BINS)
    return {
        "nll": nll,
        "crps": crps,
        "integral_mean": integral,
        "coverage": coverage_summary,
        "pit_hist_counts": counts.tolist(),
    }


def _capture_snapshot(
    estimator: CondensiteTorchCDE,
    dtype_name: str,
    dtype: np.dtype,
    X_test: np.ndarray,
    y_test: np.ndarray,
    grid: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float | dict[str, float] | list[int]]]:
    X_cast = X_test.astype(dtype)
    grid_cast = grid.astype(dtype)
    pdf = estimator.predict_density(X_cast, grid_cast).astype(np.float64)
    cdf = estimator.predict_cdf(X_cast, grid_cast).astype(np.float64)
    quantiles = estimator.predict_quantile(X_cast, QUANTILE_PROBS, y_grid=grid_cast).astype(np.float64)
    tail_right = _tail_matrix(estimator, X_cast, grid_cast, side="right")
    tail_left = _tail_matrix(estimator, X_cast, grid_cast, side="left")
    metrics = _compute_metrics(y_test, grid_cast, pdf, cdf, quantiles)
    arrays = {
        "X_test": X_test.astype(np.float32),
        "y_grid": grid_cast.astype(np.float64),
        "pdf": pdf,
        "cdf": cdf,
        "quantiles": quantiles,
        "tail_right": tail_right,
        "tail_left": tail_left,
    }
    return arrays, metrics


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
    base = Path("tests/regression/baselines")
    base.mkdir(parents=True, exist_ok=True)
    metrics_payload: dict[str, object] = {
        "generated": True,
        "quantile_probs": QUANTILE_PROBS.tolist(),
        "tail_prob_thresholds": TAIL_THRESHOLDS.tolist(),
        "pit_bin_edges": PIT_BINS.tolist(),
        "snapshots": {},
    }
    for dtype_name, dtype in DTYPE_SPECS.items():
        arrays, snapshot_metrics = _capture_snapshot(
            estimator,
            dtype_name,
            dtype,
            X_test,
            y_test,
            grid,
        )
        np.savez(
            base / f"baseline_{dtype_name}.npz",
            **arrays,
        )
        metrics_payload["snapshots"][dtype_name] = snapshot_metrics
    digest = hashlib.sha256(np.load(base / "baseline_float32.npz")["pdf"].tobytes()).hexdigest()
    (base / "baseline_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Baselines refreshed. float32 PDF hash={digest[:16]}")


if __name__ == "__main__":
    main()
