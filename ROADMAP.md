# Roadmap

## Milestones and Issues

### M1: Correctness

1. **Issue: Calibration diagnostics & PIT coverage**

  - _Scope_: Implement PIT histograms and coverage utilities with CLI hooks.
  - _Acceptance criteria_: `scripts/calibration_report.py` emits PIT histograms + coverage JSON; unit tests cover PIT bounds; docs include usage example.
  - _Labels_: tests, feature, docs.

2. **Issue: Monotone CDF head without numerical fixes**

  - _Scope_: Introduce a strictly monotone CDF output layer with differentiable constraints.
  - _Acceptance criteria_: New head passes integration tests, regression baselines updated, CRPS improves vs current head on synthetic set.
  - _Labels_: feature, tests.

3. **Issue: CI gate for density correctness**

  - _Scope_: Wire regression/constraint suites into CI with failure triage.
  - _Acceptance criteria_: GitHub Actions workflow runs `pytest -m "unit or regression"` plus snapshot hash check; failures block merge; docs list the CI command.
  - _Labels_: tests, ci.

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
