from __future__ import annotations

import csv
import sys
from pathlib import Path
from subprocess import run

import numpy as np
import pytest


@pytest.mark.integration
def test_cli_fit_predict_cycle(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = 0.5 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * rng.normal(size=60)
    train_path = tmp_path / "train.csv"
    with train_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f0", "f1", "target"])
        for row, target in zip(X, y, strict=True):
            writer.writerow([row[0], row[1], target])

    model_dir = tmp_path / "model"
    result = run(
        [
            sys.executable,
            "-m",
            "condensite_torch.cli",
            "fit",
            "--train",
            str(train_path),
            "--target",
            "target",
            "--output-model",
            str(model_dir),
            "--epochs",
            "3",
        ],
        check=False,
    )
    assert result.returncode == 0
    assert (model_dir / "model.pt").exists()

    predict_out = tmp_path / "pred.csv"
    result_pred = run(
        [
            sys.executable,
            "-m",
            "condensite_torch.cli",
            "predict",
            "--model",
            str(model_dir),
            "--data",
            str(train_path),
            "--target",
            "target",
            "--output",
            str(predict_out),
            "--probs",
            "0.1,0.5",
        ],
        check=False,
    )
    assert result_pred.returncode == 0
    assert predict_out.exists()
    with predict_out.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert "q_0.100" in header and "q_0.500" in header
