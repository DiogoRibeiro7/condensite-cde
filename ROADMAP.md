# Roadmap

## Next Priorities

To keep the backlog manageable, we are focusing on the following scoped issues before expanding the feature surface:

1. **Decision-quality APIs (M1: Correctness)** – Expose `predict_quantile`, `predict_interval`, tail probabilities, and expected shortfall helpers so downstream apps can request actionable summaries. Tests must enforce monotonicity and tail trends.
2. **Conformal intervals (M1: Correctness)** – Implement a split-conformal wrapper with CLI/example coverage so users can obtain finite-sample guarantees on top of the learned CDF.
3. **Save/load & persistence polish (M3: Reliability & UX)** – Finish the artifact contract (model weights, preprocessors, config metadata) and add regression tests that prove round-trips keep density outputs stable.
4. **Early stopping + validation metrics (M1/M3 overlap)** – Harden the training loop with explicit CRPS/NLL monitoring, patience, and “restore best checkpoint” behaviour. Include an example notebook/script that visualises the recorded metrics.
5. **Sampling defaults + auxiliary improvements (M2: Scalability)** – Promote Sobol/stratified sampling to first-class configuration, add importance sampling, and document how each option affects variance vs. compute on benchmarks.

Each item should have a dedicated GitHub issue linked to the milestone noted above so work stays incremental instead of merging through one mega-change.

## Milestones and Issues

### M1: Correctness

1. **Issue: Calibration diagnostics & PIT coverage**

2. _Scope_: Implement PIT histograms and coverage utilities with CLI hooks.

3. _Acceptance criteria_: `scripts/calibration_report.py` emits PIT histograms + coverage JSON; unit tests cover PIT bounds; docs include usage example.
4. _Labels_: tests, feature, docs.

5. **Issue: Monotone CDF head without numerical fixes**

6. _Scope_: Introduce a strictly monotone CDF output layer with differentiable constraints.

7. _Acceptance criteria_: New head passes integration tests, regression baselines updated, CRPS improves vs current head on synthetic set.
8. _Labels_: feature, tests.

9. **Issue: CI gate for density correctness**

10. _Scope_: Wire regression/constraint suites into CI with failure triage.

11. _Acceptance criteria_: GitHub Actions workflow runs `pytest -m "unit or regression"` plus snapshot hash check; failures block merge; docs list the CI command.
12. _Labels_: tests, ci.

### M2: Scalability

1. **Issue: Dataset utility suite for common benchmarks**

- _Scope_: Add loaders, normalization recipes, and configs for heteroscedastic/multimodal datasets.
- _Acceptance criteria_: `datasets/` module exposes at least three named datasets; example notebook consumes them; docs explain usage.
- _Labels_: feature, docs, performance.

2. **Issue: Lightning & Torch.compile training backend**

- _Scope_: Optional LightningModule wrapper + torch.compile path for faster training.
- _Acceptance criteria_: Flag enables Lightning loop; benchmark script shows ≥15% speedup on GPU; tests ensure determinism.
- _Labels_: feature, performance.

3. **Issue: AutoML-style bandwidth/m_aux search**

- _Scope_: Provide search API that tunes bandwidth/m_aux using validation metrics with parallel execution.
- _Acceptance criteria_: New API runs on CPU in <5 min on toy data; CRPS improves vs default; docs include code sample.
- _Labels_: feature, performance, docs.

### M3: Reliability & UX

1. **Issue: Documentation site with tutorials**

- _Scope_: Publish mkdocs-based site with tutorials, API reference, and how-tos.
- _Acceptance criteria_: `docs/` builds locally and via CI; tutorials cover training, evaluation, save/load; link in README.
- _Labels_: docs, feature, ci.

2. **Issue: Packaging & release automation**

  - _Scope_: Publish pip/conda artifacts and automate release notes.
  - _Acceptance criteria_: `poetry publish` workflow pushes tagged releases; conda recipe available; docs mention installation.
  - _Labels_: docs, ci, feature.

3. **Issue: UX polish for bandwidth predictors**

  - _Scope_: Improve multi-bandwidth/adaptive configuration with presets and warnings.
  - _Acceptance criteria_: Config validation errors surfaced via rich messages; example shows switching between presets; tests cover invalid configs.
  - _Labels_: feature, docs, tests.
