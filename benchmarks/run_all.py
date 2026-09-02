"""Compatibility wrapper for the legacy benchmark entrypoint.

The current runner lives in :mod:`benchmarks.run`. This module invokes it with
the full historical dataset list so existing ``python benchmarks/run_all.py``
usage keeps working, including the legacy MDN result key.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from benchmarks.run import run_benchmarks

LEGACY_DATASETS = ("heteroscedastic", "multimodal", "heavy_tail", "skewed", "outliers")


def main() -> None:
    """Run the historical benchmark set and write ``benchmarks/results.json``."""
    warnings.warn(
        "benchmarks/run_all.py is deprecated; run `python -m benchmarks.run` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    results = run_benchmarks(LEGACY_DATASETS, quick=False)
    payload = {"quick": False, "results": results}
    output_path = Path("benchmarks") / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Benchmark results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
