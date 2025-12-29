"""Neural network building blocks used by the Condensite estimator."""
from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F

ActivationFactory = Callable[[], nn.Module]


def _make_activation(activation: ActivationFactory | nn.Module | None) -> nn.Module:
    if activation is None:
        return nn.ReLU()
    if isinstance(activation, nn.Module):
        return copy.deepcopy(activation)
    return activation()


@dataclass
class MLPRegressorConfig:
    """Configuration container for the MLPRegressor."""

    input_dim: int
    hidden_sizes: Sequence[int] = (128, 128)
    activation: ActivationFactory | nn.Module | None = None
    dropout: float = 0.0
    positive_output: bool = True


class MLPRegressor(nn.Module):
    """Simple MLP with optional positivity constraint via softplus."""

    def __init__(self, config: MLPRegressorConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = config.input_dim
        for hidden in config.hidden_sizes:
            layers.append(nn.Linear(previous_dim, hidden))
            layers.append(_make_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous_dim = hidden
        layers.append(nn.Linear(previous_dim, 1))
        self.model = nn.Sequential(*layers)
        self.positive_output = config.positive_output

    def forward(self, features: Tensor) -> Tensor:
        out: Tensor = self.model(features)
        if self.positive_output:
            return F.softplus(out)
        return out


__all__: tuple[str, ...] = ("MLPRegressor", "MLPRegressorConfig")
