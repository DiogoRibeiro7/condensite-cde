# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Add `predict_interval` API plus expanded quantile coverage/tests.
- Introduce `examples/quantiles_and_intervals.py` demonstrating quantiles + predictive intervals.
- Tail-risk APIs: document `predict_tail_prob` / `expected_shortfall` and provide `examples/tail_risk.py`.
- Split-conformal wrapper gains `method="quantile"`/`"cdf"` calibration options with updated tests+examples.
- Add permutation importance helper + tests/example for CRPS/NLL interpretability.
- Add `what_if` counterfactual helper plus unit tests and documentation/example.
- Introduce tabular preprocessor (numeric/categorical detection, imputation, encoding) with persistence, tests, integration coverage, and `examples/tabular_preprocessing.py`.
- Add `MultiTargetCondensite` supporting independent/autoregressive multi-output modeling with tests + `examples/multi_target.py`.
- Add `make_local_grid` helper plus per-row grid inference support, tests, and `examples/local_grids.py`.

## [0.1.0] - 2025-12-22
- Initial scaffolding with Poetry, CI, and documentation.
