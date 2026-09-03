"""Cross-validation utilities for Condensite Torch models."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from condensite_torch.diagnostics import coverage_rate
from condensite_torch.metrics import crps_from_cdf, nll_from_pdf

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from condensite_torch.estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig

FeatureArray = NDArray[np.floating]
TargetArray = NDArray[np.floating]
FoldIndices = list[NDArray[np.int64]]
_DEFAULT_METRICS: tuple[str, ...] = ("nll", "crps", "coverage")
_MIN_CV_FOLDS = 2


@dataclass(slots=True)
class FoldMetrics:
    """Per-fold evaluation summary."""

    fold: int
    metrics: dict[str, float]
    train_size: int
    val_size: int
    seed: int


@dataclass(slots=True)
class CrossValidationResult:
    """Aggregated cross-validation metrics."""

    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    folds: list[FoldMetrics]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "metrics": {"mean": self.metrics_mean, "std": self.metrics_std},
            "folds": [
                {
                    "fold": fold.fold,
                    "train_size": fold.train_size,
                    "val_size": fold.val_size,
                    "seed": fold.seed,
                    "metrics": fold.metrics,
                }
                for fold in self.folds
            ],
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize the CV report to JSON and optionally persist it."""
        payload = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
        return payload


def cross_validate(  # noqa: PLR0913, PLR0914, PLR0915
    model: CondensiteTorchCDE | CondensiteTorchCDEConfig,
    X: FeatureArray,
    y: TargetArray,
    *,
    cv: int = 5,
    metrics: Sequence[str] | None = None,
    seed: int = 0,
    groups: NDArray[np.integer[Any]] | None = None,
    stratify_y: bool = False,
    coverage: float = 0.9,
    json_path: str | Path | None = None,
) -> CrossValidationResult:
    """Run k-fold CV with probabilistic metrics."""
    metric_list = tuple(metrics or _DEFAULT_METRICS)
    if not metric_list:
        msg = "Provide at least one metric."
        raise ValueError(msg)
    metric_set = {name.lower() for name in metric_list}
    invalid = metric_set.difference(_DEFAULT_METRICS)
    if invalid:
        msg = f"Unsupported metrics: {sorted(invalid)}. Valid: {_DEFAULT_METRICS}"
        raise ValueError(msg)
    if "coverage" in metric_set and (not np.isfinite(coverage) or not 0.0 < coverage < 1.0):
        msg = "coverage must be in the open interval (0, 1)."
        raise ValueError(msg)
    if cv < _MIN_CV_FOLDS:
        msg = "cv must be at least 2."
        raise ValueError(msg)

    X_arr = np.asarray(X, dtype=object)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n_samples = X_arr.shape[0]
    if y_arr.shape[0] != n_samples:
        msg = "X and y must contain the same number of rows."
        raise ValueError(msg)
    if not np.all(np.isfinite(y_arr)):
        msg = "y must contain only finite values."
        raise ValueError(msg)
    if cv > n_samples:
        msg = "cv cannot exceed the number of samples."
        raise ValueError(msg)
    if groups is not None and stratify_y:
        msg = "grouped CV and stratify_y cannot be combined."
        raise ValueError(msg)

    group_arr = np.asarray(groups, dtype=object).reshape(-1) if groups is not None else None
    if group_arr is not None and group_arr.shape[0] != n_samples:
        msg = "groups array must match the number of samples."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    stratify_targets = y_arr if stratify_y else None
    fold_indices = _build_folds(n_samples, cv, rng, group_arr, stratify_targets)
    if len(fold_indices) != cv or any(indices.size == 0 for indices in fold_indices):
        msg = "Cross-validation produced an empty validation fold."
        raise RuntimeError(msg)

    base_config = _resolve_config(model)
    fold_summaries: list[FoldMetrics] = []
    for fold_idx, val_idx in enumerate(fold_indices):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.nonzero(train_mask)[0]
        fold_seed = seed + fold_idx
        estimator = _instantiate_estimator(
            config=copy.deepcopy(base_config),
            random_seed=int(fold_seed),
        )
        estimator.fit(X_arr[train_idx], y_arr[train_idx])
        grid = estimator._default_y_grid()
        pdf = estimator.predict_density(X_arr[val_idx], grid)
        cdf = estimator._cdf_from_pdf(pdf, grid)
        fold_metrics = _compute_metrics(
            estimator,
            metric_set,
            y_arr[val_idx],
            grid,
            pdf,
            cdf,
            coverage,
        )
        fold_summaries.append(
            FoldMetrics(
                fold=fold_idx,
                metrics=fold_metrics,
                train_size=int(train_idx.shape[0]),
                val_size=int(val_idx.shape[0]),
                seed=int(fold_seed),
            ),
        )

    means = {
        metric: float(np.mean([fold.metrics[metric] for fold in fold_summaries]))
        for metric in metric_set
    }
    stds = {
        metric: float(np.std([fold.metrics[metric] for fold in fold_summaries], ddof=0))
        for metric in metric_set
    }
    metadata = {
        "cv": cv,
        "seed": int(seed),
        "metrics": sorted(metric_set),
        "coverage": float(coverage),
        "stratified": bool(stratify_y),
        "grouped": group_arr is not None,
        "n_samples": n_samples,
    }
    result = CrossValidationResult(
        metrics_mean=means,
        metrics_std=stds,
        folds=fold_summaries,
        metadata=metadata,
    )
    if json_path is not None:
        result.to_json(json_path)
    return result


def _compute_metrics(  # noqa: PLR0913, PLR0917
    estimator: CondensiteTorchCDE,
    metrics: set[str],
    y_val: NDArray[np.float64],
    grid: NDArray[np.float64],
    pdf: NDArray[np.float64],
    cdf: NDArray[np.float64],
    coverage: float,
) -> dict[str, float]:
    results: dict[str, float] = {}
    if "nll" in metrics:
        results["nll"] = nll_from_pdf(y_val, grid, pdf)
    if "crps" in metrics:
        results["crps"] = crps_from_cdf(y_val, grid, cdf)
    if "coverage" in metrics:
        if not np.isfinite(coverage) or not 0.0 < coverage < 1.0:
            msg = "coverage must be in the open interval (0, 1)."
            raise ValueError(msg)
        tail_mass = (1.0 - float(coverage)) / 2.0
        probs = np.array([tail_mass, 1.0 - tail_mass], dtype=np.float64)
        quantiles = estimator._quantiles_from_cdf(cdf, grid, probs)
        results["coverage"] = coverage_rate(y_val, quantiles[:, 0], quantiles[:, 1])
    return results


def _build_folds(
    n_samples: int,
    cv: int,
    rng: np.random.Generator,
    groups: NDArray[np.object_] | None,
    stratify_targets: NDArray[np.float64] | None,
) -> FoldIndices:
    if groups is not None:
        return _make_group_folds(groups, cv, rng)
    if stratify_targets is not None:
        stratified = _make_stratified_folds(stratify_targets, cv, rng)
        if stratified is not None and all(indices.size > 0 for indices in stratified):
            return stratified
    return _make_random_folds(n_samples, cv, rng)


def _make_random_folds(
    n_samples: int,
    cv: int,
    rng: np.random.Generator,
) -> FoldIndices:
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return [
        indices[start:end].astype(np.int64, copy=False)
        for start, end in _split_indices(indices, cv)
    ]


def _split_indices(indices: NDArray[np.int64], cv: int) -> list[tuple[int, int]]:
    fold_sizes = np.full(cv, indices.size // cv, dtype=int)
    fold_sizes[: indices.size % cv] += 1
    offsets = np.cumsum(fold_sizes)
    starts = np.concatenate(([0], offsets[:-1]))
    return list(zip(starts, offsets, strict=True))


def _make_group_folds(
    groups: NDArray[np.object_],
    cv: int,
    rng: np.random.Generator,
) -> FoldIndices:
    unique_groups = np.unique(groups)
    if unique_groups.size < cv:
        msg = "Number of unique groups must be >= cv."
        raise ValueError(msg)
    rng.shuffle(unique_groups)
    folds: list[list[int]] = [[] for _ in range(cv)]
    fold_sizes = np.zeros(cv, dtype=int)
    for group in unique_groups:
        group_indices = np.nonzero(groups == group)[0].tolist()
        target_fold = int(np.argmin(fold_sizes))
        folds[target_fold].extend(group_indices)
        fold_sizes[target_fold] += len(group_indices)
    return [np.asarray(sorted(indices), dtype=np.int64) for indices in folds]


def _make_stratified_folds(
    targets: NDArray[np.float64],
    cv: int,
    rng: np.random.Generator,
) -> FoldIndices | None:
    unique = np.unique(targets)
    if unique.size <= 1:
        return None
    bin_count = min(max(cv * _MIN_CV_FOLDS, _MIN_CV_FOLDS), unique.size, 20)
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.unique(np.quantile(targets, quantiles))
    if edges.size <= 1:
        return None
    bins = np.digitize(targets, edges[1:-1], right=False)
    folds: list[list[int]] = [[] for _ in range(cv)]
    for bin_id in range(int(np.max(bins)) + 1):
        bin_indices = np.nonzero(bins == bin_id)[0]
        if bin_indices.size == 0:
            continue
        rng.shuffle(bin_indices)
        for offset, idx in enumerate(bin_indices):
            folds[offset % cv].append(int(idx))
    result = [np.asarray(sorted(indices), dtype=np.int64) for indices in folds]
    if any(indices.size == 0 for indices in result):
        return None
    return result


def _resolve_config(
    model: CondensiteTorchCDE | CondensiteTorchCDEConfig,
) -> CondensiteTorchCDEConfig:
    estimator_cls, config_cls = _estimator_classes()
    if isinstance(model, config_cls):
        return copy.deepcopy(model)
    if isinstance(model, estimator_cls):
        return copy.deepcopy(model.config)
    msg = "model must be CondensiteTorchCDE or CondensiteTorchCDEConfig."
    raise TypeError(msg)


def _instantiate_estimator(
    config: CondensiteTorchCDEConfig,
    random_seed: int,
) -> CondensiteTorchCDE:
    estimator_cls, _ = _estimator_classes()
    return estimator_cls(config=config, random_seed=int(random_seed))


@lru_cache(maxsize=1)
def _estimator_classes() -> tuple[type[CondensiteTorchCDE], type[CondensiteTorchCDEConfig]]:
    module = import_module("condensite_torch.estimator")
    return module.CondensiteTorchCDE, module.CondensiteTorchCDEConfig


__all__ = ("CrossValidationResult", "FoldMetrics", "cross_validate")
