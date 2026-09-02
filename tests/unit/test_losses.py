"""Unit tests for loss registry."""

from __future__ import annotations

import pytest
import torch

from condensite_torch.losses import get_loss_spec

pytestmark = pytest.mark.unit


def test_mse_loss_elementwise_values() -> None:
    spec = get_loss_spec("mse")
    preds = torch.tensor([[0.0, 0.5], [1.0, -0.5]])
    targets = torch.zeros_like(preds)
    values = spec.elementwise_fn(preds, targets)
    expected = preds ** 2
    assert torch.allclose(values, expected)


def test_mae_loss_nonnegative() -> None:
    spec = get_loss_spec("mae")
    preds = torch.tensor([-1.0, 0.5, 2.0])
    targets = torch.tensor([0.0, 0.0, 1.0])
    values = spec.elementwise_fn(preds, targets)
    assert torch.all(values >= 0)
    assert torch.allclose(values, torch.tensor([1.0, 0.5, 1.0]))


def test_unknown_loss_raises() -> None:
    with pytest.raises(ValueError):
        get_loss_spec("bogus")
