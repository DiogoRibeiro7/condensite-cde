"""Condensite Torch tabular conditional density estimation utilities."""

from importlib import import_module, metadata
from typing import TYPE_CHECKING, Final

PACKAGE_NAME: Final = "condensite-cde"

try:
    __version__ = metadata.version(PACKAGE_NAME)
except metadata.PackageNotFoundError:  # pragma: no cover - version only set when installed
    __version__ = "0.0.0"

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "sample_yprime": ("condensite_torch.aux_sampling", "sample_yprime"),
    "CondensiteTorchCDE": ("condensite_torch.estimator", "CondensiteTorchCDE"),
    "CondensiteTorchCDEConfig": ("condensite_torch.estimator", "CondensiteTorchCDEConfig"),
    "MLPRegressor": ("condensite_torch.models", "MLPRegressor"),
    "MLPRegressorConfig": ("condensite_torch.models", "MLPRegressorConfig"),
    "StandardScaler": ("condensite_torch.scalers", "StandardScaler"),
    "MinMaxScaler1D": ("condensite_torch.scalers", "MinMaxScaler1D"),
    "gaussian_kernel_np": ("condensite_torch.kernels", "gaussian_kernel_np"),
    "gaussian_kernel_torch": ("condensite_torch.kernels", "gaussian_kernel_torch"),
    "kernel_h_np": ("condensite_torch.kernels", "kernel_h_np"),
    "kernel_h_torch": ("condensite_torch.kernels", "kernel_h_torch"),
    "nll_from_pdf": ("condensite_torch.metrics", "nll_from_pdf"),
    "crps_from_cdf": ("condensite_torch.metrics", "crps_from_cdf"),
    "wasserstein_1": ("condensite_torch.distribution_metrics", "wasserstein_1"),
    "ks_distance": ("condensite_torch.distribution_metrics", "ks_distance"),
    "js_divergence": ("condensite_torch.distribution_metrics", "js_divergence"),
    "pit_values": ("condensite_torch.diagnostics", "pit_values"),
    "coverage": ("condensite_torch.diagnostics", "coverage"),
    "coverage_rate": ("condensite_torch.diagnostics", "coverage_rate"),
    "permutation_importance": ("condensite_torch.interpretability", "permutation_importance"),
    "PermutationImportanceResult": ("condensite_torch.interpretability", "PermutationImportanceResult"),
    "what_if": ("condensite_torch.interpretability", "what_if"),
    "WhatIfResult": ("condensite_torch.interpretability", "WhatIfResult"),
    "ConformalCDEWrapper": ("condensite_torch.conformal", "ConformalCDEWrapper"),
    "AutoregressiveCondensite": ("condensite_torch.autoregressive", "AutoregressiveCondensite"),
    "MultiTargetCondensite": ("condensite_torch.multi_target", "MultiTargetCondensite"),
    "EnsembleCondensite": ("condensite_torch.ensemble", "EnsembleCondensite"),
    "TabularPreprocessor": ("condensite_torch.preprocessing", "TabularPreprocessor"),
    "TabularPreprocessorConfig": ("condensite_torch.preprocessing", "TabularPreprocessorConfig"),
    "make_local_grid": ("condensite_torch.local_grids", "make_local_grid"),
    "export_torchscript": ("condensite_torch.export", "export_torchscript"),
    "export_onnx": ("condensite_torch.export", "export_onnx"),
    "population_stability_index": ("condensite_torch.monitoring", "population_stability_index"),
    "ks_drift": ("condensite_torch.monitoring", "ks_drift"),
    "pit_histogram": ("condensite_torch.monitoring", "pit_histogram"),
    "pit_drift": ("condensite_torch.monitoring", "pit_drift"),
    "SchemaConstraints": ("condensite_torch.validation", "SchemaConstraints"),
    "ValidationError": ("condensite_torch.validation", "ValidationError"),
    "validate_inputs": ("condensite_torch.validation", "validate_inputs"),
    "get_kernel_spec": ("condensite_torch.kernels", "get_kernel_spec"),
    "KernelSpec": ("condensite_torch.kernels", "KernelSpec"),
    "get_loss_spec": ("condensite_torch.losses", "get_loss_spec"),
    "LossSpec": ("condensite_torch.losses", "LossSpec"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_name, attr_name = target
    try:
        module = import_module(module_name)
    except OSError as exc:  # pragma: no cover - environment-specific torch errors
        msg = (
            "PyTorch failed to initialize; install the CPU build or ensure the platform "
            "has the required libraries."
        )
        raise RuntimeError(msg) from exc
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_LAZY_IMPORTS.keys()})


if TYPE_CHECKING:
    from .autoregressive import AutoregressiveCondensite
    from .aux_sampling import sample_yprime
    from .conformal import ConformalCDEWrapper
    from .diagnostics import coverage, coverage_rate, pit_values
    from .distribution_metrics import js_divergence, ks_distance, wasserstein_1
    from .ensemble import EnsembleCondensite
    from .estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig
    from .export import export_onnx, export_torchscript
    from .interpretability import (
        PermutationImportanceResult,
        WhatIfResult,
        permutation_importance,
        what_if,
    )
    from .kernels import (
        KernelSpec,
        gaussian_kernel_np,
        gaussian_kernel_torch,
        get_kernel_spec,
        kernel_h_np,
        kernel_h_torch,
    )
    from .local_grids import make_local_grid
    from .losses import LossSpec, get_loss_spec
    from .metrics import crps_from_cdf, nll_from_pdf
    from .models import MLPRegressor, MLPRegressorConfig
    from .monitoring import ks_drift, pit_drift, pit_histogram, population_stability_index
    from .multi_target import MultiTargetCondensite
    from .preprocessing import TabularPreprocessor, TabularPreprocessorConfig
    from .scalers import MinMaxScaler1D, StandardScaler
    from .validation import SchemaConstraints, ValidationError, validate_inputs


__all__ = (
    "PACKAGE_NAME",
    "AutoregressiveCondensite",
    "CondensiteTorchCDE",
    "CondensiteTorchCDEConfig",
    "ConformalCDEWrapper",
    "EnsembleCondensite",
    "KernelSpec",
    "LossSpec",
    "MLPRegressor",
    "MLPRegressorConfig",
    "MinMaxScaler1D",
    "MultiTargetCondensite",
    "PermutationImportanceResult",
    "SchemaConstraints",
    "StandardScaler",
    "TabularPreprocessor",
    "TabularPreprocessorConfig",
    "ValidationError",
    "WhatIfResult",
    "__version__",
    "coverage",
    "coverage_rate",
    "crps_from_cdf",
    "export_onnx",
    "export_torchscript",
    "gaussian_kernel_np",
    "gaussian_kernel_torch",
    "get_kernel_spec",
    "get_loss_spec",
    "js_divergence",
    "kernel_h_np",
    "kernel_h_torch",
    "ks_distance",
    "ks_drift",
    "make_local_grid",
    "nll_from_pdf",
    "permutation_importance",
    "pit_drift",
    "pit_histogram",
    "pit_values",
    "population_stability_index",
    "sample_yprime",
    "validate_inputs",
    "wasserstein_1",
    "what_if",
)
