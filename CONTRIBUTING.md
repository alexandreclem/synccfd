# Contributing to SynCCFD

Thanks for your interest in improving SynCCFD. Contributions of all sizes are welcome — bug reports, docs fixes, new fraud scenarios, or performance improvements.

## Setup

```bash
git clone https://github.com/alexandreclem/synccfd
cd synccfd
pip install -e ".[dev]"
```

## Development workflow

Before opening a pull request, make sure the following pass locally (the same checks run in CI):

```bash
ruff check .
ruff format --check .
mypy src
pytest --cov=synccfd
```

## Pull requests

- Keep PRs focused on a single change; avoid bundling unrelated fixes.
- Add or update tests for any behavior change.
- Update the README or docs (`docs/`) if you change public API behavior.
- Describe the motivation for the change in the PR description, not just what changed.

## Reporting bugs

Open an issue at https://github.com/alexandreclem/synccfd/issues with:
- A minimal reproducible example
- Expected vs. actual behavior
- Your Python version and OS

## Releases

Releases are published to PyPI automatically via GitHub Actions when a GitHub Release is published. Maintainers: bump the version in `pyproject.toml`, tag the commit, and publish a release.
