"""Simple grid-search tuner for Condensite Torch hyper-parameters."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from condensite_torch import CondensiteTorchCDEConfig

VALID_TUNE_METRICS = {"val_crps", "val_nll"}


@dataclass(slots=True)
class TuneResult:
    """Summary of a bandwidth/m_aux grid search."""

    best_config: CondensiteTorchCDEConfig
    best_metric: float
    metric_name: str
    history: list[dict[str, float]]


def tune_bandwidth_m_aux(  # noqa: PLR0913
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    bandwidths: Sequence[float],
    m_aux_values: Sequence[int],
    *,
    base_config: CondensiteTorchCDEConfig | None = None,
    metric: str = "val_crps",
    random_seed: int = 0,
) -> TuneResult:
    """Grid-search bandwidth/m_aux combinations using validation metrics."""
    if not bandwidths or not m_aux_values:
        msg = "Provide at least one bandwidth and one m_aux value."
        raise ValueError(msg)
    if metric not in VALID_TUNE_METRICS:
        msg = f"metric must be one of {sorted(VALID_TUNE_METRICS)}, got {metric!r}."
        raise ValueError(msg)
    from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig  # noqa: PLC0415

    base = base_config or CondensiteTorchCDEConfig()
    best_config: CondensiteTorchCDEConfig | None = None
    best_score = math.inf
    trials: list[dict[str, float]] = []

    for trial_idx, (bandwidth, m_aux) in enumerate(product(bandwidths, m_aux_values)):
        seed = random_seed + trial_idx
        trial_config = replace(
            base,
            bandwidth=bandwidth,
            m_aux=m_aux,
            monitor_metric=metric,
        )
        estimator = CondensiteTorchCDE(config=trial_config, random_seed=seed)
        estimator.fit(X, y)
        if not estimator.training_history:
            msg = "Training history is empty; ensure the estimator ran for at least one epoch."
            raise RuntimeError(msg)
        metric_value = estimator.training_history[-1].get(metric)
        if metric_value is None:
            msg = (
                f"{metric} not available in training history; provide validation data via "
                "`val_fraction` or explicit X_val/y_val."
            )
            raise RuntimeError(msg)
        trial_record = {"bandwidth": bandwidth, "m_aux": float(m_aux), metric: metric_value}
        trials.append(trial_record)
        if metric_value < best_score:
            best_score = metric_value
            best_config = trial_config

    if best_config is None:
        msg = "Tuner failed to evaluate any configuration."
        raise RuntimeError(msg)

    return TuneResult(
        best_config=best_config,
        best_metric=best_score,
        metric_name=metric,
        history=trials,
    )
