# Contributing to SynCCFD

Contributions are welcome: bug reports, documentation fixes, new fraud scenarios or
performance improvements.

## Setup

```bash
git clone https://github.com/alexandreclem/synccfd
cd synccfd
pip install -e ".[dev,examples]"
```

## Development workflow

Before opening a pull request, run the same checks as CI:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

`pytest` runs the fast tests. `pytest -m slow` runs the long ones, including every
notebook of `examples/` with reduced sizes.

The reference test (`tests/regression/test_reference_output.py`) compares a small output
of `presets.realistic()` with stored files. A change that alters the data on purpose
regenerates them with `python tests/regression/make_regression.py --force` and says so
in the CHANGELOG.

## Pull requests

- One change per pull request.
- Add or update tests for any behavior change.
- Update the README and the docstrings if you change the public API.
- Explain why the change is needed, not only what it does.

## Reporting bugs

Open an issue at https://github.com/alexandreclem/synccfd/issues with:
- A minimal reproducible example
- Expected vs. actual behavior
- Your Python version and OS

## Releases

Publishing a GitHub Release uploads the package to PyPI. Before that, update the
version in `pyproject.toml`, `src/synccfd/__init__.py` and `CITATION.cff` (with
`date-released`), date the `CHANGELOG.md` entry and tag the commit.
