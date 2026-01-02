# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Add `predict_interval` API plus expanded quantile coverage/tests.
- Introduce `examples/quantiles_and_intervals.py` demonstrating quantiles + predictive intervals.
- Tail-risk APIs: document `predict_tail_prob` / `expected_shortfall` and provide `examples/tail_risk.py`.
- Split-conformal wrapper gains `method="quantile"`/`"cdf"` calibration options with updated tests+examples.

## [0.1.0] - 2025-12-22
- Initial scaffolding with Poetry, CI, and documentation.
