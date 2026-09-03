"""Adapters bridging Condensite estimators to common ecosystems."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from condensite_cde.grids import GridMode, make_y_grid
from condensite_torch import CondensiteTorchCDE, CondensiteTorchCDEConfig

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - import optional
    pd = None


class _AdapterBase:
    """Base helper with shared persistence logic."""

    _STATE_FILENAME = "adapter_state.json"
    _ESTIMATOR_DIRNAME = "estimator"

    def __init__(
        self,
        *,
        config: CondensiteTorchCDEConfig | None = None,
        random_seed: int = 0,
        grid_size: int = 128,
        grid_mode: GridMode = "quantile",
        estimator: CondensiteTorchCDE | None = None,
    ) -> None:
        self.grid_size = int(grid_size)
        mode = grid_mode.lower()
        if mode not in {"quantile", "linear"}:
            msg = f"grid_mode must be 'quantile' or 'linear', got {grid_mode!r}"
            raise ValueError(msg)
        self.grid_mode = cast(GridMode, mode)
        self.random_seed = int(random_seed)
        if estimator is None:
            self.config = (
                copy.deepcopy(config) if config is not None else CondensiteTorchCDEConfig()
            )
            self._estimator = CondensiteTorchCDE(
                config=self.config,
                random_seed=self.random_seed,
            )
            self._fitted = False
        else:
            self._estimator = estimator
            self.config = copy.deepcopy(estimator.config)
            self.random_seed = estimator.random_seed
            self._fitted = True
        self._prediction_grid: NDArray[np.float64] | None = None

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            msg = "Call fit() before requesting predictions."
            raise RuntimeError(msg)

    def _fit_numpy(self, X: NDArray[np.object_], y: NDArray[np.float64]) -> None:
        self._estimator.fit(X, y)  # type: ignore[arg-type]
        self._fitted = True
        self._prediction_grid = make_y_grid(y, grid_size=self.grid_size, mode=self.grid_mode)

    def _resolve_prediction_grid(
        self,
        y_grid: NDArray[np.floating] | Sequence[float] | None,
    ) -> NDArray[np.float64] | None:
        if y_grid is None:
            return self._prediction_grid
        return np.asarray(y_grid, dtype=np.float64)

    def _reset_estimator(self) -> None:
        self._estimator = CondensiteTorchCDE(
            config=copy.deepcopy(self.config),
            random_seed=self.random_seed,
        )
        self._fitted = False
        self._prediction_grid = None

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)
        estimator_dir = base / self._ESTIMATOR_DIRNAME
        self._estimator.save(estimator_dir)
        payload: dict[str, Any] = {
            "grid_size": self.grid_size,
            "grid_mode": self.grid_mode,
            "prediction_grid": (
                None if self._prediction_grid is None else self._prediction_grid.tolist()
            ),
        }
        payload.update(self._extra_state())
        (base / self._STATE_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _extra_state(self) -> dict[str, Any]:  # noqa: PLR6301  # pragma: no cover
        return {}

    def _load_extra_state(self, payload: Mapping[str, Any]) -> None:  # noqa: PLR6301  # pragma: no cover
        return None

    @classmethod
    def _load_bundle(
        cls,
        path: str | Path,
        *,
        map_location: str | None = None,
    ) -> tuple[CondensiteTorchCDE, dict[str, Any]]:
        base = Path(path)
        state = json.loads((base / cls._STATE_FILENAME).read_text(encoding="utf-8"))
        estimator = CondensiteTorchCDE.load(
            base / cls._ESTIMATOR_DIRNAME, map_location=map_location
        )
        return estimator, state


class SklearnCondensiteRegressor(_AdapterBase):
    """Sklearn-style wrapper around CondensiteTorchCDE."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        config: CondensiteTorchCDEConfig | None = None,
        random_seed: int = 0,
        prediction_strategy: str = "median",
        grid_size: int = 128,
        grid_mode: GridMode = "quantile",
        estimator: CondensiteTorchCDE | None = None,
    ) -> None:
        super().__init__(
            config=config,
            random_seed=random_seed,
            grid_size=grid_size,
            grid_mode=grid_mode,
            estimator=estimator,
        )
        strategy = prediction_strategy.lower()
        if strategy not in {"median", "mean"}:
            msg = "prediction_strategy must be 'median' or 'mean'."
            raise ValueError(msg)
        self.prediction_strategy = strategy

    def fit(self, X: Any, y: Any) -> SklearnCondensiteRegressor:
        X_arr = np.asarray(X, dtype=object)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_arr.shape[0] != X_arr.shape[0]:
            msg = "X and y must contain the same number of rows."
            raise ValueError(msg)
        self._fit_numpy(X_arr, y_arr)
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=object)
        grid = self._resolve_prediction_grid(None)
        if self.prediction_strategy == "median":
            return self._estimator.predict_quantile(X_arr, 0.5, y_grid=grid)
        grid_arr = grid
        if grid_arr is None:
            target = getattr(self._estimator, "_y_train", None)
            if target is None:
                msg = "Estimator missing training targets for mean prediction."
                raise RuntimeError(msg)
            grid_arr = make_y_grid(target, grid_size=self.grid_size, mode=self.grid_mode)
        pdf = self._estimator.predict_density(X_arr, grid_arr)
        weighted = pdf * grid_arr.reshape(1, -1)
        return np.asarray(np.trapz(weighted, x=grid_arr, axis=1), dtype=np.float64)

    def predict_density(
        self,
        X: Any,
        y_grid: NDArray[np.floating] | Sequence[float] | None = None,
    ) -> NDArray[np.float64]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=object)
        grid = self._resolve_prediction_grid(y_grid)
        return self._estimator.predict_density(X_arr, grid)

    def predict_interval(
        self,
        X: Any,
        coverage: float = 0.9,
        *,
        y_grid: NDArray[np.floating] | Sequence[float] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=object)
        grid = self._resolve_prediction_grid(y_grid)
        return self._estimator.predict_interval(X_arr, coverage, y_grid=grid)

    def score(self, X: Any, y: Any) -> float:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=object)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_arr.shape[0] != X_arr.shape[0]:
            msg = "X and y must contain the same number of rows."
            raise ValueError(msg)
        grid = self._resolve_prediction_grid(None)
        metrics = self._estimator.evaluate(X_arr, y_arr, y_grid=grid)
        return -float(metrics["nll"])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {
            "config": copy.deepcopy(self.config),
            "random_seed": self.random_seed,
            "prediction_strategy": self.prediction_strategy,
            "grid_size": self.grid_size,
            "grid_mode": self.grid_mode,
        }
        if deep:
            for key, value in asdict(self.config).items():
                params[f"config__{key}"] = value
        return params

    def set_params(self, **params: Any) -> SklearnCondensiteRegressor:  # noqa: PLR0912
        config_updates: dict[str, Any] = {}
        for key, value in params.items():
            if key == "config":
                if value is not None and not isinstance(value, CondensiteTorchCDEConfig):
                    msg = "config must be CondensiteTorchCDEConfig or None."
                    raise TypeError(msg)
                self.config = (
                    copy.deepcopy(value) if value is not None else CondensiteTorchCDEConfig()
                )
            elif key == "random_seed":
                self.random_seed = int(value)
            elif key == "prediction_strategy":
                strategy = str(value).lower()
                if strategy not in {"median", "mean"}:
                    msg = "prediction_strategy must be 'median' or 'mean'."
                    raise ValueError(msg)
                self.prediction_strategy = strategy
            elif key == "grid_size":
                self.grid_size = int(value)
            elif key == "grid_mode":
                mode = str(value).lower()
                if mode not in {"quantile", "linear"}:
                    msg = f"Unsupported grid_mode {value!r}"
                    raise ValueError(msg)
                self.grid_mode = cast(GridMode, mode)
            elif key.startswith("config__"):
                config_updates[key.split("__", 1)[1]] = value
            else:
                msg = f"Unknown parameter {key!r}."
                raise ValueError(msg)
        for field, value in config_updates.items():
            if not hasattr(self.config, field):
                msg = f"CondensiteTorchCDEConfig has no field {field!r}."
                raise ValueError(msg)
            setattr(self.config, field, value)
        self._reset_estimator()
        return self

    def _extra_state(self) -> dict[str, Any]:
        return {"prediction_strategy": self.prediction_strategy}

    def _load_extra_state(self, payload: Mapping[str, Any]) -> None:
        strategy = payload.get("prediction_strategy", self.prediction_strategy)
        self.prediction_strategy = str(strategy).lower()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | None = None,
    ) -> SklearnCondensiteRegressor:
        estimator, state = cls._load_bundle(path, map_location=map_location)
        adapter = cls(
            config=copy.deepcopy(estimator.config),
            random_seed=estimator.random_seed,
            prediction_strategy=str(state.get("prediction_strategy", "median")),
            grid_size=int(state.get("grid_size", 128)),
            grid_mode=cast(GridMode, str(state.get("grid_mode", "quantile"))),
            estimator=estimator,
        )
        grid_payload = state.get("prediction_grid")
        if grid_payload is not None:
            adapter._prediction_grid = np.asarray(grid_payload, dtype=np.float64)
        adapter._load_extra_state(state)
        return adapter


class PandasCondensiteAdapter(SklearnCondensiteRegressor):
    """Pandas-first interface returning Series/DataFrame outputs."""

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] | None = None,
        target_column: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._feature_columns = list(feature_columns) if feature_columns is not None else None
        self._target_column = target_column

    @staticmethod
    def _require_pandas() -> None:
        if pd is None:  # pragma: no cover - exercised when pandas missing
            msg = "pandas is required for PandasCondensiteAdapter."
            raise ImportError(msg)

    def _frame_to_numpy(self, frame: pd.DataFrame) -> NDArray[np.object_]:
        if self._feature_columns is None:
            msg = "Fit the adapter before calling predict.*"
            raise RuntimeError(msg)
        return np.asarray(
            frame[self._feature_columns].to_numpy(dtype=object, copy=False), dtype=object
        )

    def fit(
        self,
        data: Any,
        target: str | pd.Series | NDArray[np.floating] | None = None,
        *,
        feature_columns: Sequence[str] | None = None,
    ) -> PandasCondensiteAdapter:
        self._require_pandas()
        if not isinstance(data, pd.DataFrame):
            msg = "data must be a pandas DataFrame."
            raise TypeError(msg)
        frame = data
        y_series = self._resolve_target(frame, target)
        if isinstance(target, str):
            self._target_column = target
        elif y_series.name:
            self._target_column = str(y_series.name)
        features = self._resolve_features(frame, feature_columns)
        self._feature_columns = features
        X_matrix = frame[features].to_numpy(dtype=object, copy=False)
        y_values = y_series.to_numpy(dtype=np.float64, copy=False)
        super().fit(X_matrix, y_values)
        return self

    def _resolve_target(
        self,
        frame: pd.DataFrame,
        target: str | pd.Series | NDArray[np.floating] | None,
    ) -> pd.Series:
        if isinstance(target, str):
            if target not in frame.columns:
                msg = f"Target column {target!r} not in DataFrame."
                raise KeyError(msg)
            return frame[target]
        if target is None:
            if self._target_column is None:
                msg = "Provide a target column or Series when fitting."
                raise ValueError(msg)
            return frame[self._target_column]
        if isinstance(target, np.ndarray):
            return pd.Series(target, index=frame.index, name=self._target_column)
        return target

    def _resolve_features(
        self,
        frame: pd.DataFrame,
        feature_columns: Sequence[str] | None,
    ) -> list[str]:
        if feature_columns is not None:
            missing = [col for col in feature_columns if col not in frame.columns]
            if missing:
                msg = f"Unknown feature columns: {missing}"
                raise KeyError(msg)
            if self._target_column is not None and self._target_column in feature_columns:
                msg = "Target column cannot also be used as a feature."
                raise ValueError(msg)
            return list(feature_columns)
        if self._feature_columns is not None:
            return list(self._feature_columns)
        excluded = {self._target_column} if self._target_column else set()
        candidates = [col for col in frame.columns if col not in excluded]
        if not candidates:
            msg = "No feature columns available after excluding target."
            raise ValueError(msg)
        return candidates

    def predict(self, data: Any) -> Any:
        self._require_pandas()
        if not isinstance(data, pd.DataFrame):
            msg = "data must be a pandas DataFrame."
            raise TypeError(msg)
        values = super().predict(self._frame_to_numpy(data))
        name = self._target_column or "prediction"
        return pd.Series(values, index=data.index, name=name)

    def predict_density(
        self,
        data: Any,
        y_grid: NDArray[np.floating] | Sequence[float] | None = None,
    ) -> Any:
        self._require_pandas()
        if not isinstance(data, pd.DataFrame):
            msg = "data must be a pandas DataFrame."
            raise TypeError(msg)
        grid = self._resolve_prediction_grid(y_grid)
        density = super().predict_density(self._frame_to_numpy(data), grid)
        if grid is None:
            columns = [f"y_{idx}" for idx in range(density.shape[1])]
        else:
            grid_arr = np.asarray(grid)
            columns = (
                grid_arr.tolist()
                if grid_arr.ndim == 1
                else [f"y_{idx}" for idx in range(density.shape[1])]
            )
        return pd.DataFrame(density, index=data.index, columns=columns)

    def predict_interval(
        self,
        data: Any,
        coverage: float = 0.9,
        *,
        y_grid: NDArray[np.floating] | Sequence[float] | None = None,
    ) -> Any:
        self._require_pandas()
        if not isinstance(data, pd.DataFrame):
            msg = "data must be a pandas DataFrame."
            raise TypeError(msg)
        lows, highs = super().predict_interval(
            self._frame_to_numpy(data),
            coverage,
            y_grid=y_grid,
        )
        return pd.DataFrame({"low": lows, "high": highs}, index=data.index)

    def _extra_state(self) -> dict[str, Any]:
        state = super()._extra_state()
        state.update(
            {"feature_columns": self._feature_columns, "target_column": self._target_column},
        )
        return state

    def _load_extra_state(self, payload: Mapping[str, Any]) -> None:
        super()._load_extra_state(payload)
        features = payload.get("feature_columns")
        self._feature_columns = list(features) if features is not None else None
        target = payload.get("target_column")
        self._target_column = None if target is None else str(target)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | None = None,
    ) -> PandasCondensiteAdapter:
        estimator, state = cls._load_bundle(path, map_location=map_location)
        adapter = cls(
            config=copy.deepcopy(estimator.config),
            random_seed=estimator.random_seed,
            prediction_strategy=str(state.get("prediction_strategy", "median")),
            grid_size=int(state.get("grid_size", 128)),
            grid_mode=cast(GridMode, str(state.get("grid_mode", "quantile"))),
            estimator=estimator,
        )
        grid_payload = state.get("prediction_grid")
        if grid_payload is not None:
            adapter._prediction_grid = np.asarray(grid_payload, dtype=np.float64)
        adapter._load_extra_state(state)
        return adapter


__all__ = ("PandasCondensiteAdapter", "SklearnCondensiteRegressor")
