# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] — Unreleased

First public release: the **data simulator** only. The machine-learning pipeline
from the research prototype has been removed from this release and may return in a
future version.

### Added
- `DatasetGenerator` with an explicit, typed constructor and sane defaults.
- Opt-in persistence: `DatasetGenerator.save(...)` and `read_transactions(...)`.
- Top-level exports: `from synccfd import DatasetGenerator, FraudSimulator, LocationFeatures, DataManager`.
- `py.typed` marker (ships type hints), `CITATION.cff`, and full README documentation.

### Changed
- `src/` project layout; packaging moved to `pyproject.toml` (PEP 621, hatchling).
- `generate()` is now pure — it returns the three DataFrames and no longer writes
  to disk as a side effect. Use `save()` to persist.
- Tooling consolidated to Ruff (lint + format) with mypy for types.
- Default `n_terminals` lowered from `1_000_000` to `20_000`.

### Removed
- Machine-learning modules (`synccfd.ml`), ML-only utilities, and their tests.
