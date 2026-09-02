"""Kernel utilities shared between NumPy and Torch pipelines."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

SQRT_TWO_PI: float = math.sqrt(2.0 * math.pi)


def _validate_bandwidth(bandwidth: float) -> float:
    if bandwidth <= 0:
        msg = "bandwidth must be positive"
        raise ValueError(msg)
    return float(bandwidth)


def gaussian_kernel_np(delta: NDArray[np.floating], bandwidth: float) -> NDArray[np.floating]:
    """Evaluate a Gaussian kernel on the provided deltas (NumPy)."""
    bw = _validate_bandwidth(bandwidth)
    deltas = np.asarray(delta, dtype=np.float64)
    norm = 1.0 / (bw * SQRT_TWO_PI)
    scaled = (deltas / bw) ** 2 * -0.5
    return (norm * np.exp(scaled)).astype(np.float64)


def gaussian_kernel_torch(delta: Tensor, bandwidth: float | Tensor) -> Tensor:
    """Evaluate a Gaussian kernel on Torch tensors."""
    deltas = delta.float()
    if isinstance(bandwidth, Tensor):
        bw_tensor = torch.clamp(bandwidth.to(deltas.device, deltas.dtype), min=1e-8)
    else:
        bw_scalar = _validate_bandwidth(float(bandwidth))
        bw_tensor = torch.full_like(deltas, bw_scalar)
    deltas_broadcasted, bw_broadcasted = cast(
        tuple[Tensor, Tensor],
        tuple(torch.broadcast_tensors(deltas, bw_tensor)),  # type: ignore[no-untyped-call]
    )
    norm = 1.0 / (bw_broadcasted * SQRT_TWO_PI)
    result = torch.exp(-0.5 * (deltas_broadcasted / bw_broadcasted) ** 2) * norm
    return cast(Tensor, result)


def epanechnikov_kernel_np(delta: NDArray[np.floating], bandwidth: float) -> NDArray[np.floating]:
    """Evaluate the Epanechnikov kernel (parabolic, compact support)."""
    bw = _validate_bandwidth(bandwidth)
    deltas = np.asarray(delta, dtype=np.float64)
    scaled = deltas / bw
    mask = np.abs(scaled) <= 1.0
    values = np.zeros_like(scaled, dtype=np.float64)
    values[mask] = 0.75 * (1.0 - scaled[mask] ** 2) / bw
    return values


def epanechnikov_kernel_torch(delta: Tensor, bandwidth: float | Tensor) -> Tensor:
    """Evaluate the Epanechnikov kernel (Torch)."""
    deltas = delta.float()
    if isinstance(bandwidth, Tensor):
        bw_tensor = torch.clamp(bandwidth.to(deltas.device, deltas.dtype), min=1e-8)
    else:
        bw_scalar = _validate_bandwidth(float(bandwidth))
        bw_tensor = torch.full_like(deltas, bw_scalar)
    deltas_broadcasted, bw_broadcasted = cast(
        tuple[Tensor, Tensor],
        tuple(torch.broadcast_tensors(deltas, bw_tensor)),  # type: ignore[no-untyped-call]
    )
    scaled = deltas_broadcasted / bw_broadcasted
    mask = torch.abs(scaled) <= 1.0
    result = torch.zeros_like(scaled)
    result[mask] = 0.75 * (1.0 - scaled[mask] ** 2) / bw_broadcasted[mask]
    return result


@dataclass(frozen=True)
class KernelSpec:
    """Function bundle describing how to evaluate a kernel."""

    torch_fn: Callable[[Tensor, Tensor, float | Tensor], Tensor]
    numpy_fn: Callable[[NDArray[np.floating], NDArray[np.floating], float], NDArray[np.floating]]


def _kernel_wrapper_torch(
    kernel_fn: Callable[[Tensor, float | Tensor], Tensor],
    y: Tensor,
    y_prime: Tensor,
    bandwidth: float | Tensor,
) -> Tensor:
    y_tensor, y_prime_tensor = cast(
        tuple[Tensor, Tensor],
        tuple(torch.broadcast_tensors(y.float(), y_prime.float())),  # type: ignore[no-untyped-call]
    )
    return kernel_fn(y_tensor - y_prime_tensor, bandwidth)


def _kernel_wrapper_np(
    kernel_fn: Callable[[NDArray[np.floating], float], NDArray[np.floating]],
    y: NDArray[np.floating],
    y_prime: NDArray[np.floating],
    bandwidth: float,
) -> NDArray[np.floating]:
    y_arr, y_prime_arr = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64),
        np.asarray(y_prime, dtype=np.float64),
    )
    return kernel_fn(y_arr - y_prime_arr, bandwidth)


_KERNELS: dict[str, KernelSpec] = {
    "gaussian": KernelSpec(
        torch_fn=lambda y, y_prime, bw: _kernel_wrapper_torch(
            gaussian_kernel_torch,
            y,
            y_prime,
            bw,
        ),
        numpy_fn=lambda y, y_prime, bw: _kernel_wrapper_np(
            gaussian_kernel_np,
            y,
            y_prime,
            bw,
        ),
    ),
    "epanechnikov": KernelSpec(
        torch_fn=lambda y, y_prime, bw: _kernel_wrapper_torch(
            epanechnikov_kernel_torch,
            y,
            y_prime,
            bw,
        ),
        numpy_fn=lambda y, y_prime, bw: _kernel_wrapper_np(
            epanechnikov_kernel_np,
            y,
            y_prime,
            bw,
        ),
    ),
}


def get_kernel_spec(name: str) -> KernelSpec:
    """Return the torch/NumPy evaluation functions for the requested kernel."""
    normalized = name.lower()
    try:
        return _KERNELS[normalized]
    except KeyError as exc:  # pragma: no cover - sanity guard
        msg = f"Unknown kernel '{name}'. Available: {sorted(_KERNELS)}"
        raise ValueError(msg) from exc


def kernel_h_np(
    y: NDArray[np.floating],
    y_prime: NDArray[np.floating],
    bandwidth: float,
) -> NDArray[np.floating]:
    """Compute Gaussian K_h(y, y') for broadcastable NumPy vectors (legacy helper)."""
    return _KERNELS["gaussian"].numpy_fn(y, y_prime, bandwidth)


def kernel_h_torch(y: Tensor, y_prime: Tensor, bandwidth: float | Tensor) -> Tensor:
    """Compute Gaussian K_h(y, y') for Torch tensors (legacy helper)."""
    return _KERNELS["gaussian"].torch_fn(y, y_prime, bandwidth)


__all__: tuple[str, ...] = (
    "KernelSpec",
    "epanechnikov_kernel_np",
    "epanechnikov_kernel_torch",
    "gaussian_kernel_np",
    "gaussian_kernel_torch",
    "get_kernel_spec",
    "kernel_h_np",
    "kernel_h_torch",
)
