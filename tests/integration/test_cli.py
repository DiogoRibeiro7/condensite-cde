from __future__ import annotations

import csv
import json
import subprocess  # noqa: S404
import sys
from pathlib import Path

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
    subprocess.run(  # noqa: S603
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
        check=True,
    )
    assert (model_dir / "model.pt").exists()

    predict_out = tmp_path / "pred.csv"
    subprocess.run(  # noqa: S603
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
        check=True,
    )
    assert predict_out.exists()
    with predict_out.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert "q_0.100" in header and "q_0.500" in header

    predict_interval_out = tmp_path / "pred_interval.csv"
    subprocess.run(  # noqa: S603
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
            str(predict_interval_out),
            "--probs",
            "0.1",
            "--interval-coverage",
            "0.8",
        ],
        check=True,
    )
    with predict_interval_out.open("r", encoding="utf-8") as handle:
        header_interval = handle.readline().strip().split(",")
    assert "interval_lo_0.80" in header_interval
    assert "interval_hi_0.80" in header_interval


@pytest.mark.integration
def test_cli_tune_supports_resume(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(48, 2))
    y = 0.4 * X[:, 0] - 0.1 * X[:, 1] + 0.05 * rng.normal(size=X.shape[0])
    train_path = tmp_path / "train.csv"
    with train_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f0", "f1", "target"])
        for row, target in zip(X, y, strict=True):
            writer.writerow([row[0], row[1], target])

    run_root = tmp_path / "tune_runs"
    run_name = "cli-run"

    def _run_cli_resume(resume: bool) -> None:
        cmd = [
            sys.executable,
            "-m",
            "condensite_torch.cli",
            "tune",
            "--train",
            str(train_path),
            "--target",
            "target",
            "--format",
            "csv",
            "--bandwidths",
            "0.1",
            "--m-aux-values",
            "12",
            "--epochs",
            "2",
            "--run-root",
            str(run_root),
            "--run-name",
            run_name,
        ]
        if resume:
            cmd.append("--resume")
        subprocess.run(cmd, check=True)  # noqa: S603

    _run_cli_resume(resume=False)
    run_dir = run_root / run_name
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.json").exists()

    _run_cli_resume(resume=True)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics) == 1
