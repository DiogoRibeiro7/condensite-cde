"""Operational failures at tabular file boundaries keep their original cause."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
from dataexcept import DataLoadingError, FileReadError, FileWriteError

from condensite_torch import datasets

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("use_pandas", [False, True])
def test_missing_csv_has_file_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_pandas: bool
) -> None:
    missing = tmp_path / "missing.csv"
    if use_pandas:

        def read_csv(_path: Path, *, sep: str) -> None:
            raise FileNotFoundError(missing)

        monkeypatch.setattr(
            datasets,
            "pd",
            SimpleNamespace(errors=SimpleNamespace(ParserError=ValueError), read_csv=read_csv),
        )
    else:
        monkeypatch.setattr(datasets, "pd", None)

    with pytest.raises(FileReadError) as caught:
        datasets.load_tabular(missing, target_column=None)

    assert caught.value.path == str(missing)
    assert isinstance(caught.value.original, FileNotFoundError)
    assert caught.value.__cause__ is caught.value.original


def test_pandas_csv_parser_error_is_data_loading_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ParserError(Exception):
        pass

    def read_csv(_path: Path, *, sep: str) -> None:
        raise ParserError("bad row")

    monkeypatch.setattr(
        datasets,
        "pd",
        SimpleNamespace(errors=SimpleNamespace(ParserError=ParserError), read_csv=read_csv),
    )
    source = tmp_path / "broken.csv"
    with pytest.raises(DataLoadingError) as caught:
        datasets.load_tabular(source, target_column=None)

    assert caught.value.source == str(source)
    assert isinstance(caught.value.original, ParserError)
    assert caught.value.__cause__ is caught.value.original


def test_parquet_decode_error_is_data_loading_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def read_parquet(_path: Path) -> None:
        raise ValueError("invalid parquet data")

    monkeypatch.setattr(datasets, "pd", SimpleNamespace(read_parquet=read_parquet))
    source = tmp_path / "broken.parquet"
    with pytest.raises(DataLoadingError) as caught:
        datasets.load_tabular(source, target_column=None)

    assert caught.value.source == str(source)
    assert isinstance(caught.value.original, ValueError)
    assert caught.value.__cause__ is caught.value.original


def test_csv_write_error_keeps_path_and_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets, "pd", None)
    destination = tmp_path / "out.csv"

    def fail_writer(*_args: object, **_kwargs: object) -> None:
        raise csv.Error("could not write row")

    monkeypatch.setattr(datasets.csv, "DictWriter", fail_writer)
    with pytest.raises(FileWriteError) as caught:
        datasets.save_csv(destination, [{"feature": 1}])

    assert caught.value.path == str(destination)
    assert isinstance(caught.value.original, csv.Error)
    assert caught.value.__cause__ is caught.value.original


def test_pandas_csv_write_error_keeps_path_and_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "out.csv"

    def to_csv(_path: Path, *, index: bool) -> None:
        raise PermissionError(destination)

    monkeypatch.setattr(
        datasets, "pd", SimpleNamespace(DataFrame=lambda _rows: SimpleNamespace(to_csv=to_csv))
    )
    with pytest.raises(FileWriteError) as caught:
        datasets.save_csv(destination, [{"feature": 1}])

    assert caught.value.path == str(destination)
    assert isinstance(caught.value.original, PermissionError)
    assert caught.value.__cause__ is caught.value.original


def test_missing_target_column_keeps_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets, "pd", None)
    source = tmp_path / "valid.csv"
    source.write_text("feature\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Target column 'target' not found"):
        datasets.load_tabular(source, target_column="target")
