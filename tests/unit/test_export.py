from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from torch import nn

from condensite_torch import export_torchscript

pytestmark = pytest.mark.unit


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)


def test_export_torchscript_creates_file(tmp_path: Path) -> None:
    model = TinyNet()
    example = np.zeros((1, 3), dtype=np.float32)
    target = tmp_path / "tiny.pt"
    export_torchscript(model, target, example)
    assert target.exists()
