# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-11-02

**This release breaks compatibility with 0.1.0.** The v0.1 simulator (the one of the
SBSeg 2024 paper) is still available with `pip install synccfd==0.1.0`.

### Breaking changes
- **One simulation engine** (`synccfd.simulator.engine`), vectorised, whose
  randomness comes only from local random streams derived from `random_state`
  (named `SeedSequence` streams; the global `random`/`numpy.random` states are never
  used). The v0.1 engine is gone, and so are the `engine` parameter and
  `FraudSimulator`. The base model is the v0.1 one, reproduced statistically, not
  draw by draw; times of day outside the day are redrawn rather than discarded, and
  `TRANSACTION_ID` is the chronological position.
- **`DatasetGenerator()` is the complete simulator by default**: merchants, customer
  drift, seasonality and `FraudConfig.realistic()`, for 10 000 customers, 20 000
  terminals and 180 days from 2023-01-01 (the same as `presets.realistic()`).
  `None` switches a component off; `fraud=None` generates no fraud.
- **Frauds are always injected**: fraudsters make new transactions, and legitimate
  transactions are never altered nor labelled. The v0.1 mechanism (altering and
  labelling legitimate transactions) is gone, with its parameters
  `s1_amount_threshold`, `s3_tamper_fraction`, `amount_multiplier` and
  `s4_cnp_prob`. `FraudConfig()` has the five scenarios with rates proportional to
  the population (`rate_mode="proportional"`).
- The transactions always have the fraud ground truth (`GT_FRAUD_SCENARIO`,
  `GT_FRAUDSTER_ID`, `GT_CAMPAIGN_ID`, `GT_COMPROMISE_ID`) and, with each component,
  its columns (see "Added").
- `LocationFeatures` samples with an explicit `numpy.random.Generator`; its
  per-point methods and the `random_state` argument are gone.
- Dependencies: `geopy` and `tqdm` removed; `haversine` moved to the `dev` and
  `examples` extras; `dataframe-image` removed from `examples`.

### Added
- **Fraud scenarios** (`FraudConfig`). Scenario 1 is a purchase far above the
  victim's recent amounts. In scenario 2, a compromised terminal reads cards (the
  purchases stay legitimate); the cards are later cloned or leaked to scenario 3.
  Scenario 3 makes bursts of CNP purchases, and scenario 4 a fast sequence of
  purchases. The new scenario 5 (card testing) can escalate to scenario 3.
  `ground_truth.compromise_table()` lists the compromised cards.
- `IntensitySchedule` and `FraudConfig.intensity_schedule`: fraud rates that change
  over time. `FraudConfig.realistic()`: all five scenarios and fraudster drift.
  `rate_mode="proportional"` draws the daily number of victims from
  `Poisson(fraction × population)`, so the expected fraud rate does not depend on
  the dataset size.
- **Concept drift** (`DriftConfig`). Customers can move from regime A to regime B
  with sudden, gradual, incremental or recurrent drift, and optionally move to
  another city. Population drift (`GlobalDrift`: amount trend and shocks) is also
  available. Ground truth goes in `GT_*` columns of the customer profiles and the
  transactions. Customers without drift keep exactly the same transactions.
- **Seasonality** (`SeasonalityConfig`): weekday, payday and December factors, kept
  separate from drift.
- **Merchants and categories** (`MerchantConfig`). Terminals belong to physical
  merchants; online merchants have virtual terminals. Ten default categories
  (`DEFAULT_CATEGORIES`) set the amount, the hour (CP only in opening hours), the
  weekend effect, the skimming risk and, in mixed categories, the share of online
  purchases (`online_fraction`). Customers get category preferences, a spending
  factor and favourite merchants. New columns: `MERCHANT_ID` and
  `MERCHANT_CATEGORY` in the transactions, `MERCHANT_ID` in the terminal profiles,
  preferences in the customer profiles, and a new `merchant_profiles` table. With
  merchants, regime B also changes the category preferences, and each fraud
  scenario buys in its own categories (scenario 2 compromises whole merchants, by
  the category's skimming risk; scenario 3 can change categories over time).
- **Presets** (`synccfd.simulator.presets`): `realistic()` (the defaults) and
  `ablation(step)`, the components switched on cumulatively (`base`, `drift`,
  `frauds`, `merchants`), with fraud fractions calibrated so that every step has
  about the same fraud rate; `get(name)` and `from_env()`.
- `DatasetGenerator.generate_all()`, returning a `SimulationResult` (customers,
  terminals, merchants, transactions and the configuration), and
  `DatasetGenerator.config`, the effective configuration as JSON types. `save()`
  writes it to `config.json`, with the version of `synccfd`, and saves
  `merchant_profiles`.
- Documentation: presets, output schema, one page per component (drift and
  seasonality, frauds, merchants), architecture (with diagrams) and software
  engineering (tests, CI, release process).
- Example notebooks `drift.ipynb`, `frauds.ipynb` and `merchants.ipynb`; all four
  notebooks generate the same dataset (`presets.from_env()`), and
  `examples/simulator.ipynb` saves the figures of the TCC, in the same order and with
  the same names, to `docs/figures/exploratoria/`.
- `scripts/make_figures.py`: runs the four notebooks, regenerates every figure under
  `docs/figures/`, draws the architecture diagrams (`scripts/diagrams.py`) and writes
  `docs/figures/dataset_stats.json`. `scripts/benchmark.py` times each preset.
- Regression test (`tests/regression/`) freezing a small output of
  `presets.realistic()`, and `scripts/make_regression.py` documenting how it was
  produced.

### Changed
- CI: `mypy` failures now fail the build; slow end-to-end tests (including every
  notebook) are excluded from the default `pytest` run and run in a separate CI job
  (`pytest -m slow`).
- Dependencies: `shapely>=2.0` (vectorised point-in-polygon); `scipy` added to the
  `dev` extra (statistical tests).

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
