"""Utility functions for exporting PyTorch modules to TorchScript/ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    import onnx  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    onnx = None


def export_torchscript(
    module: nn.Module,
    path: str | Path,
    example_input: Any,
    *,
    strict: bool = True,
) -> Path:
    """Trace ``module`` with ``example_input`` and save a TorchScript artifact."""
    original_training = module.training
    example = _to_tensor(example_input)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        module.eval()
        traced = torch.jit.trace(module, example, strict=strict)  # type: ignore[no-untyped-call]
        torch.jit.save(traced, target)  # type: ignore[no-untyped-call]
    finally:
        module.train(original_training)
    return target


def export_onnx(
    module: nn.Module,
    path: str | Path,
    example_input: Any,
    *,
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    """Export ``module`` to ONNX if the optional dependency is installed."""
    if onnx is None:  # pragma: no cover - optional
        msg = "ONNX is not installed; run `pip install onnx` to enable export."
        raise RuntimeError(msg)
    original_training = module.training
    example = _to_tensor(example_input)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        module.eval()
        torch.onnx.export(
            module,
            (example,),
            target.as_posix(),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
    finally:
        module.train(original_training)
    return target


def _to_tensor(example_input: Any) -> torch.Tensor:
    """Convert array-like export input to a float32 tensor."""
    if isinstance(example_input, torch.Tensor):
        return example_input
    array = np.asarray(example_input, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    return torch.from_numpy(array)


__all__ = ("export_onnx", "export_torchscript")
