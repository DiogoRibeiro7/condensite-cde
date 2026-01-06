# Testing Guide

## Fast Suite (default)
- Run all fast layers (unit + integration + regression) with:
  ```bash
  poetry run pytest -q -m "unit or integration or regression"
  ```
  Slow tests are excluded by default.

## Individual Layers
- Unit tests only:
  ```bash
  poetry run pytest -q -m unit
  ```
- Integration tests only:
  ```bash
  poetry run pytest -q -m integration
  ```
- Regression/snapshot tests only:
  ```bash
  poetry run pytest -q -m regression
  ```
- Slow tests (nightly or explicit runs):
  ```bash
  poetry run pytest -q -m slow
  ```

## Snapshots/Baselines
- Refresh regression baselines before committing deterministic output changes:
  ```bash
  poetry run python scripts/generate_regression_baselines.py
  ```

## Baseline Update Protocol
1. Run `poetry run python scripts/generate_regression_baselines.py`. The script trains the canonical estimator, writes `baseline_float32.npz`, `baseline_float64.npz`, and refreshes `baseline_metrics.json` (quantile probs, tail thresholds, PIT histograms, coverage rates).
2. Inspect the script's stdout hash and skim `baseline_metrics.json` to ensure metrics look sensible (no NaNs, coverage within a few percent of nominal).
3. Commit the updated `.npz` + `.json` artifacts alongside the code change, and mention the reported hash in your PR description so reviewers can confirm the refresh.
4. Re-run `poetry run pytest -q -m "regression"` to confirm the snapshots match.

Refer to `tests/CONVENTIONS.md` for additional guidance on seeding, runtime limits, and regeneration policies.
