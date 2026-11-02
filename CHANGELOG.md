# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Merchants and categories** (`MerchantConfig`, v2 engine). Terminals belong to
  physical merchants; online merchants have virtual terminals. Ten default
  categories (`DEFAULT_CATEGORIES`) set the amount, the hour (CP only in opening
  hours), the weekend effect, the skimming risk and, in mixed categories, the share
  of online purchases (`online_fraction`). Customers get category
  preferences, a spending factor and favourite merchants. New columns:
  `MERCHANT_ID` and `MERCHANT_CATEGORY` in the transactions, `MERCHANT_ID` in the
  terminal profiles, preferences in the customer profiles, and a new
  `merchant_profiles` table (`DatasetGenerator.merchant_profiles`, saved by
  `save()`). Without merchants, the output is unchanged.
- With merchants, regime B of a drifting customer changes the category preferences,
  and the injected frauds buy in the categories of each scenario. Scenario 2
  compromises whole merchants, by the category's skimming risk; scenario 3 can
  change categories over time (`s3_category_schedule`).
- `examples/merchants.ipynb` (figures in `docs/figures/fase4/`), and category views
  in `examples/drift.ipynb` and `examples/frauds.ipynb`.
- **Fraud injection** (`FraudConfig(mechanism="inject")`, v2 engine): fraudsters make
  new transactions instead of altering or labelling legitimate ones. Scenario 1
  becomes a purchase far above the victim's recent amounts. In scenario 2, a
  compromised terminal reads cards (the purchases stay legitimate); the cards are
  later cloned or leaked to scenario 3. Scenario 3 makes bursts of CNP purchases, and
  scenario 4 a fast sequence of purchases. The new scenario 5 (card testing) can
  escalate to scenario 3. Ground truth: `GT_FRAUD_SCENARIO`, `GT_FRAUDSTER_ID`,
  `GT_CAMPAIGN_ID`, `GT_COMPROMISE_ID`; `ground_truth.compromise_table()` lists the
  compromised cards.
- `IntensitySchedule` and `FraudConfig.intensity_schedule`: fraud rates that change
  over time. `FraudConfig.realistic()`: all five scenarios, proportional rates and
  fraudster drift.
- `examples/frauds.ipynb`, with the figures in `docs/figures/fase3/`.
- **Concept drift** (`DriftConfig`, v2 engine). Customers can move from regime A to
  regime B with sudden, gradual, incremental or recurrent drift, and optionally move
  to another city. Population drift (`GlobalDrift`: amount trend and shocks) is also
  available. Ground truth goes in `GT_*` columns of the customer profiles and the
  transactions. Customers without drift keep exactly the same transactions.
- **Seasonality** (`SeasonalityConfig`, v2 engine): weekday, payday and December
  factors, kept separate from drift.
- `DatasetGenerator(drift=..., seasonality=...)`: `engine="auto"` switches to v2 when
  either is set.
- `examples/drift.ipynb`, with the figures in `docs/figures/fase2/`. The slow test now
  runs both example notebooks.
- **v2 engine** (`DatasetGenerator(engine="v2")`, package `synccfd.simulator.engine`):
  a vectorised engine with the v0.1 model, using only local random streams
  (`numpy.random.Generator`, derived from `random_state` through named
  `SeedSequence` streams). It generates 10 000 customers × 180 days in about 20 s.
  Times of day outside the day are redrawn rather than discarded, and transactions are
  re-sorted chronologically after the fraud scenarios. `engine="auto"` (default)
  still uses the legacy engine.
- `LocationFeatures` methods accept an optional `rng` (vectorised sampling, used by
  v2). Without it, the behaviour is unchanged.
- `scripts/benchmark.py`; a slow test that runs `examples/simulator.ipynb` end to end
  with reduced sizes; a statistical comparison of v2 against legacy.
- `FraudConfig` (`synccfd.simulator.config`): every fraud parameter that was hard-coded
  (scenario selection, amount threshold, compromises per day, durations, tamper
  fraction, amount multiplier, CNP probability) is now configurable through
  `DatasetGenerator(fraud=...)` or `FraudSimulator(config=...)`. Defaults reproduce
  v0.1.0 exactly.
- `rate_mode="proportional"`: the number of compromised terminals/customers per day
  is drawn from `Poisson(fraction × population)`, so the expected fraud rate no longer
  depends on the dataset size (in `"absolute"` mode, the default, small datasets get
  a very high fraud rate).
- Golden regression test (`tests/golden/`) freezing the v0.1.0 output, and
  `scripts/make_golden.py` documenting how it was produced.

### Fixed
- Fraud scenarios 3 and 4 no longer crash with `IndexError` when the compromised
  customers queue is empty. This happened when, on some day, fewer customers than
  needed were available, so nothing was added to the queue but an entry was still
  removed later. Results are unchanged whenever v0.1 did not crash.
- `mypy src` now passes (annotation-only changes, no behaviour change).

### Changed
- CI: `mypy` failures now fail the build; slow end-to-end tests are excluded from the
  default `pytest` run and run in a separate CI job (`pytest -m slow`).
- The v0.1 code moved to `synccfd.simulator.legacy` (logic unchanged).
  `synccfd.simulator.generator.DatasetGenerator`, `synccfd.simulator.fraud.FraudSimulator`
  and the top-level imports keep working.
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
