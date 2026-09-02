"""Dataset loading utilities for CLI workflows."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover
    pd = None


Format = Literal["csv", "tsv", "parquet"]
ObjectArray = NDArray[np.object_]
FloatArray = NDArray[np.float64]


def load_tabular(
    path: str | Path,
    *,
    target_column: str | None,
    file_format: str = "auto",
) -> tuple[ObjectArray, FloatArray | None, list[str]]:
    """Load tabular data into NumPy arrays and return ``(X, y, feature_names)``."""
    resolved = _resolve_format(path, file_format)
    if resolved in {"csv", "tsv"}:
        delimiter = "\t" if resolved == "tsv" else ","
        return _load_delimited(path, target_column, delimiter=delimiter)
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
        writer.writerows(rows)


def _resolve_format(path: str | Path, file_format: str) -> str:
    normalized = file_format.lower()
    if normalized != "auto":
        if normalized not in {"csv", "tsv", "parquet"}:
            msg = f"Unsupported format: {file_format}"
            raise ValueError(msg)
        return normalized
    suffix = Path(path).suffix.lower()
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    return "csv"


def _load_delimited(
    path: str | Path,
    target_column: str | None,
    *,
    delimiter: str,
) -> tuple[ObjectArray, FloatArray | None, list[str]]:
    if pd is not None:
        df = pd.read_csv(path, sep=delimiter)
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

    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        msg = "Delimited file is empty."
        raise ValueError(msg)
    if reader.fieldnames is None:
        msg = "Delimited file must contain a header row."
        raise ValueError(msg)

    feature_names = list(reader.fieldnames)
    if target_column is not None and target_column not in feature_names:
        msg = f"Target column '{target_column}' not found."
        raise ValueError(msg)
    remaining_names = [col for col in feature_names if col != target_column]

    y_values: list[float] = []
    raw_columns: dict[str, list[str]] = {name: [] for name in remaining_names}
    for row in rows:
        if target_column is not None:
            value = row.get(target_column, "")
            if value is None or value.strip() == "":
                msg = f"Target column '{target_column}' contains a missing value."
                raise ValueError(msg)
            y_values.append(float(value))
        for name in remaining_names:
            raw_columns[name].append(row.get(name, "") or "")

    converted_columns = [_infer_column(raw_columns[name]) for name in remaining_names]
    feature_rows = (
        list(zip(*converted_columns, strict=True)) if converted_columns else [()] * len(rows)
    )
    X_arr = np.asarray(feature_rows, dtype=object)
    y_arr = np.asarray(y_values, dtype=np.float64) if target_column is not None else None
    return X_arr, y_arr, remaining_names


def _infer_column(values: Sequence[str]) -> list[Any]:
    """Infer a numeric column while preserving empty fields as missing values."""
    non_missing = [value for value in values if value.strip() != ""]
    numeric = bool(non_missing)
    if numeric:
        try:
            for value in non_missing:
                float(value)
        except ValueError:
            numeric = False
    if numeric:
        return [np.nan if value.strip() == "" else float(value) for value in values]
    return [None if value.strip() == "" else value for value in values]


def _load_parquet(
    path: str | Path,
    target_column: str | None,
) -> tuple[ObjectArray, FloatArray | None, list[str]]:
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
