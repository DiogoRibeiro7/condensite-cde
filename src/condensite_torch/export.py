"""Utility functions for exporting PyTorch modules to TorchScript/ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    import onnx
except ModuleNotFoundError:  # pragma: no cover
    onnx = None  # type: ignore[assignment]


def export_torchscript(
    module: nn.Module,
    path: str | Path,
    example_input: Any,
    *,
    strict: bool = True,
) -> Path:
    """Trace `module` with `example_input` and save it as a TorchScript artifact."""
    module.eval()
    example = _to_tensor(example_input)
    traced = torch.jit.trace(module, example, strict=strict)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, target)
    return target


def export_onnx(
    module: nn.Module,
    path: str | Path,
    example_input: Any,
    *,
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    """Export `module` to ONNX if the dependency is installed, otherwise raise."""
    if onnx is None:  # pragma: no cover - optional
        msg = "ONNX is not installed; run `pip install onnx` to enable export."
        raise RuntimeError(msg)
    module.eval()
    example = _to_tensor(example_input)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        example,
        target.as_posix(),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    return target


def _to_tensor(example_input: Any) -> torch.Tensor:
    if isinstance(example_input, torch.Tensor):
        return example_input
    array = np.asarray(example_input, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    return torch.from_numpy(array)


__all__ = ("export_onnx", "export_torchscript")
