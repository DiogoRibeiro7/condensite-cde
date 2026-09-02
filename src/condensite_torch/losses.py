"""Loss registry powering flexible Condensite training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LossSpec:
    """Encapsulate an element-wise loss computation."""

    elementwise_fn: Callable[[Tensor, Tensor], Tensor]


def _mse(predictions: Tensor, targets: Tensor) -> Tensor:
    return torch.square(predictions - targets)


def _mae(predictions: Tensor, targets: Tensor) -> Tensor:
    return torch.abs(predictions - targets)


_LOSSES: dict[str, LossSpec] = {
    "mse": LossSpec(elementwise_fn=_mse),
    "mae": LossSpec(elementwise_fn=_mae),
}


def get_loss_spec(name: str) -> LossSpec:
    """Return the element-wise loss corresponding to ``name``."""
    normalized = name.lower()
    try:
        return _LOSSES[normalized]
    except KeyError as exc:  # pragma: no cover - defensive
        msg = f"Unknown loss '{name}'. Available: {sorted(_LOSSES)}"
        raise ValueError(msg) from exc


__all__ = ("LossSpec", "get_loss_spec")
