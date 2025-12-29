"""Condensite Torch tabular conditional density estimation utilities."""

from importlib import import_module, metadata
from typing import TYPE_CHECKING, Final

PACKAGE_NAME: Final = "condensite-torch"

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
    "pit_values": ("condensite_torch.diagnostics", "pit_values"),
    "coverage": ("condensite_torch.diagnostics", "coverage"),
    "ConformalCDEWrapper": ("condensite_torch.conformal", "ConformalCDEWrapper"),
    "AutoregressiveCondensite": ("condensite_torch.autoregressive", "AutoregressiveCondensite"),
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
    from .aux_sampling import sample_yprime
    from .estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig
    from .kernels import (
        gaussian_kernel_np,
        gaussian_kernel_torch,
        kernel_h_np,
        kernel_h_torch,
    )
    from .metrics import crps_from_cdf, nll_from_pdf
    from .diagnostics import pit_values, coverage
    from .conformal import ConformalCDEWrapper
    from .autoregressive import AutoregressiveCondensite
    from .models import MLPRegressor, MLPRegressorConfig
    from .scalers import MinMaxScaler1D, StandardScaler


__all__ = (
    "PACKAGE_NAME",
    "CondensiteTorchCDE",
    "CondensiteTorchCDEConfig",
    "MLPRegressor",
    "MLPRegressorConfig",
    "MinMaxScaler1D",
    "StandardScaler",
    "__version__",
    "crps_from_cdf",
    "gaussian_kernel_np",
    "gaussian_kernel_torch",
    "kernel_h_np",
    "kernel_h_torch",
    "nll_from_pdf",
    "pit_values",
    "coverage",
    "ConformalCDEWrapper",
    "AutoregressiveCondensite",
    "sample_yprime",
)
