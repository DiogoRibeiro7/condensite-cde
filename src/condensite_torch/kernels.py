"""Kernel utilities shared between NumPy and Torch pipelines."""

from __future__ import annotations

import math
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


def kernel_h_np(
    y: NDArray[np.floating],
    y_prime: NDArray[np.floating],
    bandwidth: float,
) -> NDArray[np.floating]:
    """Compute K_h(y, y') for broadcastable NumPy vectors."""
    y_arr, y_prime_arr = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64),
        np.asarray(y_prime, dtype=np.float64),
    )
    return gaussian_kernel_np(y_arr - y_prime_arr, bandwidth)


def gaussian_kernel_torch(delta: Tensor, bandwidth: float | Tensor) -> Tensor:
    """Evaluate a Gaussian kernel on Torch tensors."""
    deltas = delta.float()
    if isinstance(bandwidth, Tensor):
        bw_tensor = bandwidth.to(deltas.device, deltas.dtype)
        bw_tensor = torch.clamp(bw_tensor, min=1e-8)
    else:
        bw_scalar = _validate_bandwidth(float(bandwidth))
        bw_tensor = torch.full_like(deltas, bw_scalar)
    deltas_broadcasted, bw_broadcasted = cast(
        tuple[Tensor, Tensor],
        tuple(torch.broadcast_tensors(deltas, bw_tensor)),  # type: ignore[no-untyped-call]
    )
    norm = 1.0 / (bw_broadcasted * SQRT_TWO_PI)
    return torch.exp(-0.5 * (deltas_broadcasted / bw_broadcasted) ** 2) * norm


def kernel_h_torch(y: Tensor, y_prime: Tensor, bandwidth: float | Tensor) -> Tensor:
    """Compute K_h(y, y') for Torch tensors while supporting broadcasting."""
    y_tensor, y_prime_tensor = cast(
        tuple[Tensor, Tensor],
        tuple(torch.broadcast_tensors(y.float(), y_prime.float())),  # type: ignore[no-untyped-call]
    )
    return gaussian_kernel_torch(y_tensor - y_prime_tensor, bandwidth)


__all__: tuple[str, ...] = (
    "gaussian_kernel_np",
    "gaussian_kernel_torch",
    "kernel_h_np",
    "kernel_h_torch",
)
