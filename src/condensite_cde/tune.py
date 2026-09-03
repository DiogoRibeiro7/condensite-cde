"""Simple grid-search tuner for Condensite Torch hyper-parameters."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from condensite_torch import CondensiteTorchCDEConfig

VALID_TUNE_METRICS = {"val_crps", "val_nll"}
DEFAULT_RUN_ROOT = Path("runs")


@dataclass(slots=True)
class TuneResult:
    """Summary of a bandwidth/m_aux grid search."""

    best_config: CondensiteTorchCDEConfig
    best_metric: float
    metric_name: str
    history: list[dict[str, Any]]
    run_dir: Path


@dataclass(slots=True)
class _TuneRunContext:
    run_dir: Path
    config_path: Path
    metrics_path: Path
    artifacts_dir: Path
    metadata: dict[str, Any]
    history: list[dict[str, Any]]
    cache: dict[str, dict[str, Any]]

    def record_trial(
        self,
        config_hash: str,
        record: dict[str, Any],
        config_dump: dict[str, Any],
        training_history: list[dict[str, float]],
    ) -> None:
        record_with_hash = record | {"config_hash": config_hash, "status": "completed"}
        self.history.append(record_with_hash)
        self.cache[config_hash] = record_with_hash
        _write_json(self.metrics_path, self.history)
        artifact_dir = self.artifacts_dir / config_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(artifact_dir / "config.json", config_dump)
        _write_json(artifact_dir / "history.json", training_history)


def tune_bandwidth_m_aux(  # noqa: PLR0913, PLR0914
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    bandwidths: Sequence[float],
    m_aux_values: Sequence[int],
    *,
    base_config: CondensiteTorchCDEConfig | None = None,
    metric: str = "val_crps",
    random_seed: int = 0,
    run_root: str | Path | None = DEFAULT_RUN_ROOT,
    run_name: str | None = None,
    resume: bool = False,
) -> TuneResult:
    """Grid-search bandwidth/m_aux combinations using validation metrics."""
    if not bandwidths or not m_aux_values:
        msg = "Provide at least one bandwidth and one m_aux value."
        raise ValueError(msg)
    if metric not in VALID_TUNE_METRICS:
        msg = f"metric must be one of {sorted(VALID_TUNE_METRICS)}, got {metric!r}."
        raise ValueError(msg)
    from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig  # noqa: PLC0415

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    base = base_config or CondensiteTorchCDEConfig()
    metric_literal = cast(Literal["val_crps", "val_nll"], metric)
    metadata = _build_run_metadata(
        base,
        bandwidths,
        m_aux_values,
        metric,
        random_seed,
        X_arr,
        y_arr,
    )
    context = _prepare_run_context(run_root, run_name, resume, metadata)
    best_config: CondensiteTorchCDEConfig | None = None
    best_score = math.inf

    if context.history:
        for record in context.history:
            score = record.get(metric)
            if score is None:
                continue
            score_value = float(score)
            if score_value < best_score:
                best_score = score_value
                best_config = replace(
                    base,
                    bandwidth=float(record["bandwidth"]),
                    m_aux=int(record["m_aux"]),
                    monitor_metric=metric_literal,
                )

    for trial_idx, (bandwidth, m_aux) in enumerate(product(bandwidths, m_aux_values)):
        config_hash = _config_fingerprint(base, bandwidth, m_aux)
        if config_hash in context.cache:
            continue
        seed = random_seed + trial_idx
        trial_config = replace(
            base,
            bandwidth=bandwidth,
            m_aux=m_aux,
            monitor_metric=metric_literal,
        )
        estimator = CondensiteTorchCDE(config=trial_config, random_seed=seed)
        estimator.fit(X_arr, y_arr)
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
        trial_record: dict[str, Any] = {
            "bandwidth": float(bandwidth),
            "m_aux": int(m_aux),
            metric: float(metric_value),
            "seed": seed,
        }
        context.record_trial(
            config_hash,
            trial_record,
            _clean_value(asdict(trial_config)),
            estimator.training_history,
        )
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
        history=context.history,
        run_dir=context.run_dir,
    )


def _prepare_run_context(
    run_root: str | Path | None,
    run_name: str | None,
    resume: bool,
    metadata: dict[str, Any],
) -> _TuneRunContext:
    root = Path(run_root) if run_root is not None else DEFAULT_RUN_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if resume:
        if run_name is None:
            msg = "Provide run_name when resuming a hyper-parameter search."
            raise ValueError(msg)
        run_dir = root / run_name
        if not run_dir.exists():
            msg = f"Run directory {run_dir} not found."
            raise FileNotFoundError(msg)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        folder = run_name or timestamp
        run_dir = root / folder
        if run_dir.exists():
            msg = f"Run directory {run_dir} already exists. Use resume=True to continue."
            raise FileExistsError(msg)
        run_dir.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        if not config_path.exists():
            msg = f"config.json missing from run directory {run_dir}"
            raise FileNotFoundError(msg)
        existing_metadata = json.loads(config_path.read_text(encoding="utf-8"))
        _validate_metadata(existing_metadata, metadata)
        history = _load_json(metrics_path, default=[])
    else:
        _write_json(config_path, metadata)
        history = []

    cache = {entry["config_hash"]: entry for entry in history if "config_hash" in entry}
    return _TuneRunContext(
        run_dir=run_dir,
        config_path=config_path,
        metrics_path=metrics_path,
        artifacts_dir=artifacts_dir,
        metadata=metadata,
        history=history,
        cache=cache,
    )


def _config_fingerprint(
    config: CondensiteTorchCDEConfig,
    bandwidth: float,
    m_aux: int,
) -> str:
    payload = {
        "base_config": _clean_value(asdict(config)),
        "bandwidth": float(bandwidth),
        "m_aux": int(m_aux),
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _array_fingerprint(values: NDArray[Any]) -> str:
    """Return a stable fingerprint including shape, dtype, and array values."""
    arr = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    if arr.dtype == object:
        serialized = json.dumps(_clean_value(arr.tolist()), sort_keys=True, separators=(",", ":"))
        digest.update(serialized.encode("utf-8"))
    else:
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _build_run_metadata(  # noqa: PLR0913, PLR0917
    config: CondensiteTorchCDEConfig,
    bandwidths: Sequence[float],
    m_aux_values: Sequence[int],
    metric: str,
    random_seed: int,
    X: NDArray[Any],
    y: NDArray[Any],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bandwidths": [float(bw) for bw in bandwidths],
        "m_aux_values": [int(val) for val in m_aux_values],
        "metric": metric,
        "random_seed": int(random_seed),
        "base_config": _clean_value(asdict(config)),
        "data_fingerprint": {
            "X": _array_fingerprint(X),
            "y": _array_fingerprint(y),
        },
    }


def _clean_value(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _clean_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_value(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _clean_value(asdict(value))
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def _load_json(path: Path, default: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return default.copy()
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def _validate_metadata(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    fields = (
        "bandwidths",
        "m_aux_values",
        "metric",
        "random_seed",
        "base_config",
        "data_fingerprint",
    )
    for field in fields:
        if existing.get(field) != expected.get(field):
            msg = (
                f"Run metadata mismatch for {field}: existing={existing.get(field)} "
                f"expected={expected.get(field)}"
            )
            raise ValueError(msg)
