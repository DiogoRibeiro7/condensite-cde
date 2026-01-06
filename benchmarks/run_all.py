"""Compatibility wrapper for the legacy benchmark entrypoint.

The new runner lives in :mod:`benchmarks.run`. This module just invokes it with
the full dataset list that used to be covered historically so existing docs or
scripts that still rely on ``benchmarks/run_all.py`` keep working.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from .run import run_benchmarks

LEGACY_DATASETS = ("heteroscedastic", "multimodal", "heavy_tail", "skewed", "outliers")


def main() -> None:
    """Retain backward compatibility for legacy scripts.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If a dataset name is unknown.

    Side Effects:
        Runs the full benchmark suite and writes `benchmarks/results.json`.

    Complexity:
        O(sum_datasets training cost).
    """
    warnings.warn(
        "benchmarks/run_all.py is deprecated; run `python -m benchmarks.run` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    results = run_benchmarks(LEGACY_DATASETS, quick=False)
    payload = {"quick": False, "results": results}
    output_path = Path("benchmarks") / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Benchmark results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
