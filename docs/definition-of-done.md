# Definition of Done

Every pull request should meet the following criteria before requesting review:

## Tests
- Unit tests added/updated for any new logic or edge cases.
- Integration/regression tests updated when behaviour changes, including snapshot refreshes where applicable.
- `poetry run pytest -q -m "unit or integration or regression"` passes locally.

## Continuous Integration
- CI workflows must pass (linting, type-checking, and test suites).
- New tooling/commands should be wired into CI if they affect quality gates.

## Documentation
- README/examples updated when user-facing features change.
- API docstrings reflect new/modified functions.
- Changelogs or ROADMAP updates provided when appropriate.

## Backward Compatibility
- Public APIs remain backwards compatible unless a breaking change is explicitly communicated.
- Persistence formats (saved checkpoints, configs) retain backwards compatibility or provide migration notes.
