"""Utility helpers shared across Condensite implementations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "PandasCondensiteAdapter": ("condensite_cde.adapters", "PandasCondensiteAdapter"),
    "SklearnCondensiteRegressor": ("condensite_cde.adapters", "SklearnCondensiteRegressor"),
    "CrossValidationResult": ("condensite_cde.cv", "CrossValidationResult"),
    "FoldMetrics": ("condensite_cde.cv", "FoldMetrics"),
    "cross_validate": ("condensite_cde.cv", "cross_validate"),
    "make_y_grid": ("condensite_cde.grids", "make_y_grid"),
    "build_benchmark_report": ("condensite_cde.reports", "build_benchmark_report"),
    "build_calibration_report": ("condensite_cde.reports", "build_calibration_report"),
    "TuneResult": ("condensite_cde.tune", "TuneResult"),
    "tune_bandwidth_m_aux": ("condensite_cde.tune", "tune_bandwidth_m_aux"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_LAZY_IMPORTS.keys()})


if TYPE_CHECKING:
    from .adapters import PandasCondensiteAdapter, SklearnCondensiteRegressor
    from .cv import CrossValidationResult, FoldMetrics, cross_validate
    from .grids import make_y_grid
    from .reports import build_benchmark_report, build_calibration_report
    from .tune import TuneResult, tune_bandwidth_m_aux

__all__ = (
    "CrossValidationResult",
    "FoldMetrics",
    "PandasCondensiteAdapter",
    "SklearnCondensiteRegressor",
    "TuneResult",
    "build_benchmark_report",
    "build_calibration_report",
    "cross_validate",
    "make_y_grid",
    "tune_bandwidth_m_aux",
)
