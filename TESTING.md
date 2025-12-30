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

Refer to `tests/CONVENTIONS.md` for additional guidance on seeding, runtime limits, and regeneration policies.
