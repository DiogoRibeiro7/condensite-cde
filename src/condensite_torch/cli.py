"""Command-line interface for training, prediction, tuning, and reporting."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from condensite_cde.tune import VALID_TUNE_METRICS, tune_bandwidth_m_aux

from .datasets import load_tabular, save_csv
from .estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig

_FORMAT_CHOICES = ["auto", "csv", "tsv", "parquet"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Condensite command-line interface."""
    parser = argparse.ArgumentParser(prog="condensite-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Train a model on a tabular dataset.")
    fit_parser.add_argument("--train", required=True, help="Training dataset path.")
    fit_parser.add_argument("--target", required=True, help="Target column name.")
    fit_parser.add_argument("--output-model", required=True, help="Directory for the trained model.")
    fit_parser.add_argument("--format", default="auto", choices=_FORMAT_CHOICES)
    fit_parser.add_argument("--epochs", type=int, default=6)
    fit_parser.add_argument("--m-aux", type=int, default=32)
    fit_parser.add_argument("--bandwidth", type=float, default=0.1)
    fit_parser.add_argument("--hidden-sizes", default="32,32", help="Comma-separated layer sizes.")
    fit_parser.add_argument("--seed", type=int, default=7)

    predict_parser = subparsers.add_parser("predict", help="Load a model and produce quantiles.")
    predict_parser.add_argument("--model", required=True, help="Directory containing saved model.")
    predict_parser.add_argument("--data", required=True, help="Dataset for prediction.")
    predict_parser.add_argument("--target", default=None, help="Optional target column to drop.")
    predict_parser.add_argument("--format", default="auto", choices=_FORMAT_CHOICES)
    predict_parser.add_argument("--probs", default="0.1,0.5,0.9", help="Quantile probabilities.")
    predict_parser.add_argument(
        "--interval-coverage",
        type=float,
        default=None,
        help="Optional predictive interval coverage (e.g., 0.9).",
    )
    predict_parser.add_argument("--output", required=True, help="Path to CSV with predictions.")

    report_parser = subparsers.add_parser("report", help="Evaluate a saved model.")
    report_parser.add_argument("--model", required=True)
    report_parser.add_argument("--data", required=True)
    report_parser.add_argument("--target", required=True)
    report_parser.add_argument("--format", default="auto", choices=_FORMAT_CHOICES)
    report_parser.add_argument("--output-json", required=True)
    report_parser.add_argument("--use-local-grid", action="store_true", default=False)

    tune_parser = subparsers.add_parser("tune", help="Grid-search bandwidth/m_aux combinations.")
    tune_parser.add_argument("--train", required=True, help="Training dataset path.")
    tune_parser.add_argument("--target", required=True, help="Target column name.")
    tune_parser.add_argument("--format", default="auto", choices=_FORMAT_CHOICES)
    tune_parser.add_argument("--bandwidths", required=True, help="Comma-separated bandwidth grid.")
    tune_parser.add_argument("--m-aux-values", required=True, help="Comma-separated m_aux grid.")
    tune_parser.add_argument("--metric", choices=sorted(VALID_TUNE_METRICS), default="val_crps")
    tune_parser.add_argument("--epochs", type=int, default=4)
    tune_parser.add_argument("--patience", type=int, default=1)
    tune_parser.add_argument("--batch-size", type=int, default=64)
    tune_parser.add_argument("--val-fraction", type=float, default=0.2)
    tune_parser.add_argument("--sampler", default="sobol")
    tune_parser.add_argument("--seed", type=int, default=0)
    tune_parser.add_argument("--run-root", default="runs")
    tune_parser.add_argument("--run-name", default=None)
    tune_parser.add_argument("--resume", action="store_true", default=False)

    args = parser.parse_args(argv)
    if args.command == "fit":
        return _cmd_fit(args)
    if args.command == "predict":
        return _cmd_predict(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "tune":
        return _cmd_tune(args)
    parser.error("Unknown command")
    return 1


def _cmd_fit(args: argparse.Namespace) -> int:
    X, y, _ = load_tabular(args.train, target_column=args.target, file_format=args.format)
    hidden_sizes = tuple(int(size) for size in args.hidden_sizes.split(",") if size)
    config = CondensiteTorchCDEConfig(
        hidden_sizes=hidden_sizes,
        epochs=args.epochs,
        m_aux=args.m_aux,
        bandwidth=args.bandwidth,
    )
    if y is None:
        msg = "Training data must include the target column."
        raise ValueError(msg)
    estimator = CondensiteTorchCDE(config=config, random_seed=args.seed).fit(
        cast(NDArray[np.floating], X),
        cast(NDArray[np.floating], y),
    )
    estimator.save(args.output_model)
    print(f"Model saved to {args.output_model}")
    return 0


def _quantile_labels(probs: Sequence[float]) -> list[str]:
    """Return stable, collision-free output labels for requested probabilities."""
    base_labels = [f"q_{prob:.3f}" for prob in probs]
    counts: dict[str, int] = {}
    labels: list[str] = []
    for index, (prob, base) in enumerate(zip(probs, base_labels, strict=True)):
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        if base_labels.count(base) == 1:
            labels.append(base)
        else:
            labels.append(f"q_{prob:.17g}_{index}")
    if len(set(labels)) != len(labels):
        labels = [f"{label}_{index}" for index, label in enumerate(labels)]
    return labels


def _cmd_predict(args: argparse.Namespace) -> int:
    estimator = CondensiteTorchCDE.load(args.model, map_location="cpu")
    X, _y, _feature_names = load_tabular(
        args.data,
        target_column=args.target,
        file_format=args.format,
    )
    probs = [float(value) for value in args.probs.split(",") if value]
    labels = _quantile_labels(probs)
    X_cast = cast(NDArray[np.floating], X)
    quantiles = estimator.predict_quantile(X_cast, np.asarray(probs, dtype=np.float64))
    coverage: float | None = args.interval_coverage
    intervals = None if coverage is None else estimator.predict_interval(X_cast, coverage=float(coverage))
    rows = []
    for idx, row in enumerate(quantiles):
        record: dict[str, float] = {"row": float(idx)}
        for label, value in zip(labels, row, strict=True):
            record[label] = float(value)
        if intervals is not None and coverage is not None:
            lo, hi = intervals
            record[f"interval_lo_{coverage:.2f}"] = float(lo[idx])
            record[f"interval_hi_{coverage:.2f}"] = float(hi[idx])
        rows.append(record)
    save_csv(args.output, rows)
    print(f"Predictions written to {args.output}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    estimator = CondensiteTorchCDE.load(args.model, map_location="cpu")
    X, y, _ = load_tabular(args.data, target_column=args.target, file_format=args.format)
    if y is None:
        msg = "Evaluation requires the target column."
        raise ValueError(msg)
    metrics = estimator.evaluate(
        cast(NDArray[np.floating], X),
        cast(NDArray[np.floating], y),
        y_grid=None,
        use_local_grid=args.use_local_grid,
    )
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metrics, indent=2, allow_nan=False)
    path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    X, y, _ = load_tabular(args.train, target_column=args.target, file_format=args.format)
    if y is None:
        msg = "Tuning requires the target column."
        raise ValueError(msg)
    bandwidths = _parse_float_list(args.bandwidths)
    if not bandwidths:
        msg = "Provide at least one bandwidth value."
        raise ValueError(msg)
    m_aux_values = _parse_int_list(args.m_aux_values)
    if not m_aux_values:
        msg = "Provide at least one m_aux value."
        raise ValueError(msg)
    base_config = CondensiteTorchCDEConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        sampler=args.sampler,
    )
    result = tune_bandwidth_m_aux(
        cast(NDArray[np.floating], X),
        cast(NDArray[np.floating], y),
        bandwidths=bandwidths,
        m_aux_values=m_aux_values,
        base_config=base_config,
        metric=args.metric,
        random_seed=args.seed,
        run_root=args.run_root,
        run_name=args.run_name,
        resume=args.resume,
    )
    summary = {
        "best_bandwidth": result.best_config.bandwidth,
        "best_m_aux": result.best_config.m_aux,
        result.metric_name: result.best_metric,
        "run_dir": str(result.run_dir),
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Best configuration saved in {result.run_dir}")
    return 0


def _parse_float_list(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
