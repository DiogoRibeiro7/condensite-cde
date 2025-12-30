# Contributing

Thanks for taking the time to contribute!

## Workflow
- Fork the repository and create a feature branch.
- Install dependencies with `poetry install`.
- Run the full quality suite: `poetry run ruff check .`, `poetry run mypy src`, and `poetry run pytest`.
- Keep PRs focused and include tests for new functionality.

## Commit Guidelines
- Follow Conventional Commits when possible (`feat:`, `fix:`, etc.).
- Update `CHANGELOG.md` with a short summary under the **Unreleased** or new version heading.

## Code Style & Definition of Done
- The project enforces Ruff and MyPy in CI.
- Use type hints everywhere; prefer explicit return types.
- Document public APIs with docstrings.
- Follow the [Definition of Done](docs/definition-of-done.md): add/update tests, ensure CI passes, update docs, and watch backward compatibility for APIs and saved artifacts.
- See [TESTING.md](TESTING.md) for commands that run unit/integration/regression suites and regenerate baselines before submitting a PR.

## Communication
- For security concerns, follow the instructions in `SECURITY.md`.
- Use GitHub Discussions or Issues for architecture proposals.
