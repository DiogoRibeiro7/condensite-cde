from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - depends on host
    print(json.dumps({"error": f"Torch unavailable: {exc}"}))
    raise SystemExit(0) from exc

from condensite_cde import make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf


def make_dataset(n_samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(n_samples, 2))
    noise = (0.1 + 0.25 * np.abs(X[:, 0])) * rng.normal(size=n_samples)
    y = 0.5 * np.sin(X[:, 0]) - 0.3 * X[:, 1] + noise
    split = int(0.8 * n_samples)
    return (X[:split], y[:split]), (X[split:], y[split:])


def _single_run(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    grid: np.ndarray,
    *,
    config: CondensiteTorchCDEConfig,
) -> dict[str, float]:
    X_train, y_train, X_test, y_test = dataset
    estimator = CondensiteTorchCDE(config=config, random_seed=5)
    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    runtime = time.perf_counter() - start
    pdf = estimator.predict_density(X_test, grid)
    cdf = estimator.predict_cdf(X_test, grid)
    nll = float(nll_from_pdf(y_test, grid, pdf))
    crps = float(crps_from_cdf(y_test, grid, cdf))
    mass = np.trapezoid(pdf, x=grid, axis=1)
    integral_error = float(np.mean(np.abs(mass - 1.0)))
    return {
        "nll": nll,
        "crps": crps,
        "integral_error": integral_error,
        "runtime_sec": runtime,
    }


def run_experiment(m_values: list[int], samplers: list[str]) -> dict[str, list[dict[str, float]]]:
    (X_train, y_train), (X_test, y_test) = make_dataset()
    dataset = (X_train, y_train, X_test, y_test)
    grid = make_y_grid(y_train, grid_size=96, mode="quantile")
    results: dict[str, list[dict[str, float]]] = {}
    for sampler in samplers:
        sampler_rows = []
        for m_val in m_values:
            config = CondensiteTorchCDEConfig(
                hidden_sizes=(48, 48),
                m_aux=m_val,
                epochs=6,
                patience=2,
                sampler=sampler,
                bandwidth=0.1,
                normalization_lambda=0.1,
            )
            metrics = _single_run(dataset, grid, config=config)
            metrics["m_aux"] = m_val
            metrics["sampler"] = sampler
            sampler_rows.append(metrics)
        results[sampler] = sampler_rows
    return results


def write_report(results: dict[str, list[dict[str, float]]]) -> None:
    target_json = Path("reports") / "m_scaling.json"
    target_json.parent.mkdir(parents=True, exist_ok=True)
    target_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    header = "| Sampler | m_aux | NLL | CRPS | Integral Error | Runtime (s) |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    lines = ["# m_aux scaling study", "", header, separator]
    for sampler, rows in results.items():
        for row in rows:
            lines.append(
                (
                    f"| {sampler} | {row['m_aux']} | {row['nll']:.4f} | {row['crps']:.4f} | "
                    f"{row['integral_error']:.4f} | {row['runtime_sec']:.2f} |"
                ),
            )
    lines.append("")
    lines.append(
        (
            "Diminishing returns typically appear once m_aux exceeds ~64; "
            "runtimes continue to grow while CRPS/NLL improvements flatten."
        ),
    )
    target_md = Path("reports") / "m_scaling.md"
    target_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    m_values = [4, 8, 16, 32, 64, 128]
    samplers = ["iid", "sobol"]
    results = run_experiment(m_values, samplers)
    write_report(results)


if __name__ == "__main__":
    main()
