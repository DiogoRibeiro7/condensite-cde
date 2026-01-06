"""Dataset loading utilities for CLI workflows."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


Format = Literal["csv", "parquet"]
ObjectArray = NDArray[np.object_]
FloatArray = NDArray[np.float64]


def load_tabular(
    path: str | Path,
    *,
    target_column: str | None,
    file_format: str = "auto",
) -> tuple[ObjectArray, FloatArray | None, list[str]]:
    """Load tabular data into numpy arrays; returns (X, y, feature_names)."""
    resolved = _resolve_format(path, file_format)
    if resolved == "csv":
        return _load_csv(path, target_column)
    if resolved == "parquet":
        return _load_parquet(path, target_column)
    msg = f"Unsupported format: {resolved}"
    raise ValueError(msg)


def save_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write rows to CSV using pandas if available."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows).to_csv(target, index=False)
        return
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_format(path: str | Path, file_format: str) -> str:
    if file_format != "auto":
        return file_format.lower()
    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    return "csv"


def _load_csv(path: str | Path, target_column: str | None) -> tuple[ObjectArray, FloatArray | None, list[str]]:
    if pd is not None:
        df = pd.read_csv(path)
        y: FloatArray | None = None
        if target_column is not None:
            if target_column not in df.columns:
                msg = f"Target column '{target_column}' not found."
                raise ValueError(msg)
            y_series = df.pop(target_column)
            y = y_series.to_numpy(dtype=np.float64, copy=True)
        feature_names = list(df.columns)
        X = df.astype(object).to_numpy(copy=True, dtype=object)
        return X, y, feature_names
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        msg = "CSV file is empty."
        raise ValueError(msg)
    feature_names = list(rows[0].keys())
    y_values: list[float] = []
    remaining_names = [col for col in feature_names if col != target_column]
    feature_rows: list[list[Any]] = []
    for row in rows:
        row_copy = dict(row)
        if target_column:
            if target_column not in row_copy:
                msg = f"Target column '{target_column}' not found."
                raise ValueError(msg)
            value = row_copy.pop(target_column)
            y_values.append(float(value))
        feature_rows.append([row_copy[col] for col in remaining_names])
    y_arr = np.array(y_values, dtype=np.float64) if target_column else None
    X_arr = np.array(feature_rows, dtype=object)
    return X_arr, y_arr, remaining_names


def _load_parquet(path: str | Path, target_column: str | None) -> tuple[ObjectArray, FloatArray | None, list[str]]:
    if pd is None:
        msg = "Reading Parquet requires pandas; install pandas to enable this format."
        raise RuntimeError(msg)
    df = pd.read_parquet(path)
    y: FloatArray | None = None
    if target_column is not None:
        if target_column not in df.columns:
            msg = f"Target column '{target_column}' not found."
            raise ValueError(msg)
        y_series = df.pop(target_column)
        y = y_series.to_numpy(dtype=np.float64, copy=True)
    feature_names = list(df.columns)
    X = df.astype(object).to_numpy(copy=True, dtype=object)
    return X, y, feature_names


__all__ = ("load_tabular", "save_csv")
