"""Model scaffolding for tabular conditional density estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

Tensor = torch.Tensor


@dataclass
class TabularCDEConfig:
    """Minimal configuration for a toy conditional density estimator."""

    input_dim: int
    hidden_dim: int = 32
    activation: torch.nn.Module = torch.nn.ReLU()

    def validate(self) -> None:
        """Validate the configuration eagerly to fail fast."""
        if self.input_dim <= 0:
            msg = "input_dim must be > 0"
            raise ValueError(msg)
        if self.hidden_dim <= 0:
            msg = "hidden_dim must be > 0"
            raise ValueError(msg)


class TabularConditionalDensityEstimator(torch.nn.Module):
    """Placeholder estimator that stores minimal state and basic layers."""

    def __init__(self, config: TabularCDEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(config.input_dim, config.hidden_dim),
            config.activation,
            torch.nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(self, batch: Tensor) -> Tensor:  # pragma: no cover - placeholder
        """Apply the backbone to produce latent features."""
        return self.backbone(batch)

    def parameters_vector(self) -> Tensor:
        """Return all parameters flattened into a single vector for quick checks."""
        tensors: Iterable[Tensor] = (param.view(-1) for param in self.parameters())
        return torch.cat(tuple(tensors))
