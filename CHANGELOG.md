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
- Add `EnsembleCondensite` wrapper for epistemic uncertainty with mean/variance predictions and `examples/epistemic_ensemble.py`.
- Introduce reproducible training mode with deterministic seeding, persisted model cards, README/docs updates, and integration tests ensuring identical outputs.
- Document benchmark suite usage, expand module/class/function docstrings (per DOC_COVERAGE.md), and add a CI quick benchmark workflow plus metadata coverage.
- Add pluggable kernel/loss registries (`kernel in {"gaussian","epanechnikov"}`, `loss in {"mse","mae"}`), unit/integration tests, and `examples/custom_kernel_loss.py`.
- Improve inference throughput via `inference_batch_size` / `inference_grid_chunk_size`, chunked prediction loops, regression tests, and `scripts/inference_benchmark.py`.

## [0.1.0] - 2025-12-22
- Initial scaffolding with Poetry, CI, and documentation.
