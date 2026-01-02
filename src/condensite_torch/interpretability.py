"""Interpretability helpers (permutation importance, what-if analysis)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Literal

import numpy as np
from numpy.typing import NDArray

from .estimator import CondensiteTorchCDE

VALID_IMPORTANCE_METRICS: tuple[str, ...] = ("crps", "nll")


@dataclass(slots=True)
class PermutationImportanceResult:
    """Summary of permutation-based feature importance."""

    metric_name: str
    baseline_score: float
    importances_mean: NDArray[np.float64]
    importances_std: NDArray[np.float64]
    raw_importances: NDArray[np.float64]


def permutation_importance(  # noqa: PLR0913
    estimator: CondensiteTorchCDE,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    metric: Literal["crps", "nll"] = "crps",
    n_repeats: int = 5,
    random_seed: int = 0,
    y_grid: NDArray[np.floating] | None = None,
    head: int | str | None = None,
) -> PermutationImportanceResult:
    """Estimate permutation feature importance under the chosen probabilistic metric."""
    estimator._ensure_fitted()  # noqa: SLF001
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if X_arr.ndim != 2:
        msg = f"X must be 2-D, got shape {X_arr.shape}"
        raise ValueError(msg)
    if X_arr.shape[0] != y_arr.shape[0]:
        msg = "X and y must contain the same number of samples."
        raise ValueError(msg)
    if n_repeats <= 0:
        msg = "n_repeats must be positive."
        raise ValueError(msg)
    metric_lower = metric.lower()
    if metric_lower not in VALID_IMPORTANCE_METRICS:
        msg = f"metric must be one of {VALID_IMPORTANCE_METRICS}, got {metric}"
        raise ValueError(msg)

    baseline = _score(estimator, X_arr, y_arr, metric_lower, y_grid, head)
    rng = np.random.default_rng(random_seed)
    n_features = X_arr.shape[1]
    raw = np.empty((n_features, n_repeats), dtype=np.float64)

    for feature_idx in range(n_features):
        for repeat in range(n_repeats):
            shuffled = X_arr.copy()
            permutation = rng.permutation(X_arr.shape[0])
            shuffled[:, feature_idx] = shuffled[permutation, feature_idx]
            perturbed = _score(estimator, shuffled, y_arr, metric_lower, y_grid, head)
            raw[feature_idx, repeat] = perturbed - baseline

    return PermutationImportanceResult(
        metric_name=metric_lower,
        baseline_score=baseline,
        importances_mean=raw.mean(axis=1),
        importances_std=raw.std(axis=1, ddof=0),
        raw_importances=raw,
    )


def _score(
    estimator: CondensiteTorchCDE,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    metric: str,
    y_grid: NDArray[np.floating] | None,
    head: int | str | None,
) -> float:
    metrics = estimator.evaluate(X, y, y_grid=y_grid, head=head)
    return float(metrics[metric])


_ALLOWED_WHAT_IF_OUTPUTS = ("pdf", "cdf", "quantiles", "tail_probs")


@dataclass(slots=True)
class WhatIfResult:
    """Container for baseline vs modified distribution outputs."""

    feature_changes: dict[int, float]
    baseline: dict[str, Any]
    modified: dict[str, Any]


def what_if(  # noqa: PLR0913
    estimator: CondensiteTorchCDE,
    X_row: NDArray[np.floating],
    feature_changes: Mapping[int, float],
    *,
    outputs: Sequence[str] = ("quantiles", "tail_probs"),
    quantile_probs: Sequence[float] = (0.1, 0.5, 0.9),
    tail_thresholds: Sequence[float] | None = None,
    y_grid: NDArray[np.floating] | None = None,
    head: int | str | None = None,
) -> WhatIfResult:
    """Compare baseline vs counterfactual predictions after mutating selected features."""
    estimator._ensure_fitted()  # noqa: SLF001
    base_row = np.asarray(X_row, dtype=np.float64).reshape(1, -1)
    if base_row.ndim != 2 or base_row.shape[0] != 1:
        msg = "X_row must represent a single sample (1-D vector)."
        raise ValueError(msg)
    n_features = base_row.shape[1]
    modified_row = base_row.copy()
    normalized_changes: dict[int, float] = {}
    for key, value in feature_changes.items():
        idx = int(key)
        if idx < 0 or idx >= n_features:
            msg = f"Feature index {idx} is out of bounds for {n_features} features."
            raise ValueError(msg)
        modified_row[0, idx] = float(value)
        normalized_changes[idx] = float(value)
    outputs_set = tuple(outputs)
    for item in outputs_set:
        if item not in _ALLOWED_WHAT_IF_OUTPUTS:
            msg = f"outputs items must be subset of {_ALLOWED_WHAT_IF_OUTPUTS}, got {item}"
            raise ValueError(msg)

    quant_probs_arr = _validate_probs(quantile_probs) if "quantiles" in outputs_set else None
    tail_thresholds_arr: NDArray[np.float64] | None = None
    if "tail_probs" in outputs_set:
        tail_thresholds_arr = _resolve_tail_thresholds(
            tail_thresholds,
            estimator,
            base_row,
            y_grid,
            head,
        )

    contexts = {
        "baseline": base_row,
        "modified": modified_row,
    }
    scenario_outputs: dict[str, dict[str, Any]] = {}
    for label, row in contexts.items():
        scenario_outputs[label] = _compute_what_if_outputs(
            estimator,
            row,
            outputs_set,
            quant_probs_arr,
            tail_thresholds_arr,
            y_grid,
            head,
        )

    return WhatIfResult(
        feature_changes=normalized_changes,
        baseline=scenario_outputs["baseline"],
        modified=scenario_outputs["modified"],
    )


def _compute_what_if_outputs(
    estimator: CondensiteTorchCDE,
    X_row: NDArray[np.float64],
    outputs: Sequence[str],
    quantile_probs: NDArray[np.float64] | None,
    tail_thresholds: NDArray[np.float64] | None,
    y_grid: NDArray[np.floating] | None,
    head: int | str | None,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if "quantiles" in outputs and quantile_probs is not None:
        values = estimator.predict_quantile(X_row, quantile_probs, y_grid=y_grid, head=head)[0]
        data["quantiles"] = {
            "probs": quantile_probs,
            "values": values,
        }
    if "tail_probs" in outputs and tail_thresholds is not None:
        tail_values = np.empty_like(tail_thresholds)
        for idx, threshold in enumerate(tail_thresholds):
            score = estimator.predict_tail_prob(
                X_row,
                float(threshold),
                y_grid=y_grid,
                head=head,
            )[0]
            tail_values[idx] = score
        data["tail_probs"] = {
            "thresholds": tail_thresholds,
            "values": tail_values,
        }
    if "pdf" in outputs:
        grid = _resolve_grid(estimator, y_grid)
        pdf = estimator.predict_density(X_row, grid, head=head)[0]
        data["pdf"] = {
            "y_grid": grid,
            "values": pdf,
        }
    if "cdf" in outputs:
        grid = _resolve_grid(estimator, y_grid)
        cdf = estimator.predict_cdf(X_row, grid, head=head)[0]
        data["cdf"] = {
            "y_grid": grid,
            "values": cdf,
        }
    return data


def _resolve_grid(
    estimator: CondensiteTorchCDE,
    y_grid: NDArray[np.floating] | None,
) -> NDArray[np.float64]:
    if y_grid is not None:
        return estimator._validate_y_grid(y_grid)  # noqa: SLF001
    return estimator._default_y_grid()  # noqa: SLF001


def _validate_probs(probs: Sequence[float]) -> NDArray[np.float64]:
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        msg = "quantile_probs must contain at least one probability."
        raise ValueError(msg)
    if np.any((arr < 0.0) | (arr > 1.0)):
        msg = "quantile_probs must lie within [0, 1]."
        raise ValueError(msg)
    return arr


def _resolve_tail_thresholds(
    thresholds: Sequence[float] | None,
    estimator: CondensiteTorchCDE,
    X_row: NDArray[np.float64],
    y_grid: NDArray[np.floating] | None,
    head: int | str | None,
) -> NDArray[np.float64]:
    if thresholds is not None:
        arr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            msg = "tail_thresholds must contain at least one value."
            raise ValueError(msg)
        return arr
    median = estimator.predict_quantile(X_row, 0.5, y_grid=y_grid, head=head)[0]
    return np.asarray([float(median)], dtype=np.float64)


__all__ = (
    "PermutationImportanceResult",
    "WhatIfResult",
    "permutation_importance",
    "what_if",
)
