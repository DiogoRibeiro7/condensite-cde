"""Generate compliance metrics for article datasets."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:
    raise SystemExit(f"Torch is not available: {exc}") from exc

from condensite_torch import (
    CondensiteTorchCDE,
    CondensiteTorchCDEConfig,
    crps_from_cdf,
    nll_from_pdf,
)

REPORT_PATH = Path("reports/compliance.json")


def heteroscedastic_normal(n_samples: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 2))
    noise_scale = 0.2 + 0.3 * np.abs(X[:, 0])
    y = 0.5 * X[:, 0] - 0.25 * X[:, 1] + noise_scale * rng.normal(size=n_samples)
    return X, y


def multimodal_mixture(n_samples: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.5, 1.5, size=(n_samples, 2))
    mix_prob = (np.sin(X[:, 0]) + 1.0) / 2.0
    component = rng.random(size=n_samples) < mix_prob
    mean_a = -1.0 + 0.4 * X[:, 1]
    mean_b = 1.0 - 0.5 * X[:, 1]
    y = np.where(
        component,
        rng.normal(loc=mean_a, scale=0.25, size=n_samples),
        rng.normal(loc=mean_b, scale=0.35, size=n_samples),
    )
    return X, y


def evaluate_case(
    name: str,
    make_data: Callable[[], tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    X, y = make_data()
    config = CondensiteTorchCDEConfig(
        hidden_sizes=(64, 64),
        m_aux=128,
        bandwidth=0.1,
        epochs=12,
        patience=4,
        sampler="sobol",
        batch_size=64,
        amp=False,
    )
    estimator = CondensiteTorchCDE(config=config, random_seed=42).fit(X, y)
    grid = np.linspace(y.min() - 1.0, y.max() + 1.0, 128)

    eval_X = X[:128]
    eval_y = y[:128]
    pdf = estimator.predict_density(eval_X, grid)
    cdf = estimator.predict_cdf(eval_X, grid)
    masses = np.trapezoid(pdf, x=grid, axis=1)

    results: dict[str, float] = {
        "nll": float(nll_from_pdf(eval_y, grid, pdf)),
        "crps": float(crps_from_cdf(eval_y, grid, cdf)),
        "mean_pdf_mass": float(np.mean(masses)),
        "min_pdf_mass": float(np.min(masses)),
        "mean_cdf_end": float(np.mean(cdf[:, -1])),
        "min_pdf_value": float(np.min(pdf)),
        "max_pdf_value": float(np.max(pdf)),
    }
    return results


def main() -> None:
    Path("reports").mkdir(exist_ok=True)
    results = {
        "heteroscedastic_normal": evaluate_case("heteroscedastic_normal", heteroscedastic_normal),
        "multimodal_mixture": evaluate_case("multimodal_mixture", multimodal_mixture),
    }
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote compliance metrics to {REPORT_PATH}")


if __name__ == "__main__":
    main()
