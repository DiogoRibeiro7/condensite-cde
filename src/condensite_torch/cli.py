"""Command-line interface for training, prediction, and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .datasets import load_tabular, save_csv
from .estimator import CondensiteTorchCDE, CondensiteTorchCDEConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="condensite-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Train a model on a CSV/Parquet dataset.")
    fit_parser.add_argument("--train", required=True, help="Training dataset path.")
    fit_parser.add_argument("--target", required=True, help="Target column name.")
    fit_parser.add_argument("--output-model", required=True, help="Directory to save the trained model.")
    fit_parser.add_argument("--format", default="auto", choices=["auto", "csv", "parquet"])
    fit_parser.add_argument("--epochs", type=int, default=6)
    fit_parser.add_argument("--m-aux", type=int, default=32)
    fit_parser.add_argument("--bandwidth", type=float, default=0.1)
    fit_parser.add_argument("--hidden-sizes", default="32,32", help="Comma-separated hidden layer sizes.")
    fit_parser.add_argument("--seed", type=int, default=7)

    predict_parser = subparsers.add_parser("predict", help="Load a saved model and produce quantiles.")
    predict_parser.add_argument("--model", required=True, help="Directory containing saved model.")
    predict_parser.add_argument("--data", required=True, help="Dataset for prediction.")
    predict_parser.add_argument("--target", default=None, help="Optional target column to drop.")
    predict_parser.add_argument("--format", default="auto", choices=["auto", "csv", "parquet"])
    predict_parser.add_argument("--probs", default="0.1,0.5,0.9", help="Comma-separated quantile probabilities.")
    predict_parser.add_argument("--output", required=True, help="Path to CSV with predictions.")

    report_parser = subparsers.add_parser("report", help="Evaluate a saved model and emit metrics JSON.")
    report_parser.add_argument("--model", required=True)
    report_parser.add_argument("--data", required=True)
    report_parser.add_argument("--target", required=True)
    report_parser.add_argument("--format", default="auto", choices=["auto", "csv", "parquet"])
    report_parser.add_argument("--output-json", required=True)
    report_parser.add_argument("--use-local-grid", action="store_true", default=False)

    args = parser.parse_args(argv)

    if args.command == "fit":
        return _cmd_fit(args)
    if args.command == "predict":
        return _cmd_predict(args)
    if args.command == "report":
        return _cmd_report(args)
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
    estimator = CondensiteTorchCDE(config=config, random_seed=args.seed).fit(X, y)
    estimator.save(args.output_model)
    print(f"Model saved to {args.output_model}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    estimator = CondensiteTorchCDE.load(args.model, map_location="cpu")
    X, _y, feature_names = load_tabular(args.data, target_column=args.target, file_format=args.format)
    probs = [float(value) for value in args.probs.split(",") if value]
    quantiles = estimator.predict_quantile(X, probs)
    rows = []
    for idx, row in enumerate(quantiles):
        record = {"row": idx}
        for prob, value in zip(probs, row, strict=True):
            record[f"q_{prob:.3f}"] = float(value)
        rows.append(record)
    save_csv(args.output, rows)
    print(f"Predictions written to {args.output}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    estimator = CondensiteTorchCDE.load(args.model, map_location="cpu")
    X, y, _ = load_tabular(args.data, target_column=args.target, file_format=args.format)
    metrics = estimator.evaluate(
        X,
        y,
        y_grid=None,
        use_local_grid=args.use_local_grid,
    )
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
