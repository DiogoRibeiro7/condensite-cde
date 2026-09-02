# Testing conventions

- **Determinism**: Seed every RNG (`numpy.random.default_rng(seed)`, `torch.manual_seed(seed)`) inside tests and fixtures.
- **Runtime**: Target <5 seconds per test on CPU; mark longer scenarios with `@pytest.mark.slow` and gate them in scheduled jobs.
- **Isolation**: No network calls, file I/O limited to tmp dirs/fixtures, no external datasets.
- **Numerics**: Prefer `np.allclose(..., atol=1e-4)` for float comparisons; justify looser tails (e.g., CRPS/NLL) with comments.
- **Structure**:
  - `tests/unit/`: pure function/class behavior, mocked dependencies.
  - `tests/integration/`: estimator training/prediction loops, CLI scripts.
  - `tests/regression/`: serialized artifacts, invariance checks, deterministic snapshots.
- **Baselines**: Regenerate regression fixtures after intentional behavior changes via `poetry run python scripts/generate_regression_baselines.py` (ensures `tests/regression/baselines/*` stay in sync). The script now writes `baseline_{float32,float64}.npz` plus `baseline_metrics.json` (containing quantile grids, tail thresholds, and calibration histograms). After running it, sanity-check the printed hash, skim the JSON for reasonableness, and mention the refresh in your PR.
- **Fixtures**: Place reusable data/expectations under `tests/fixtures/` and load via helper utilities.
