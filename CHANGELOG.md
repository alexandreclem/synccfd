# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-13

First public release. SynCCFD generates synthetic credit card transactions with
labelled frauds, for research on fraud detection models.

### Added
- `DatasetGenerator`: generates customers, terminals, merchants and labelled
  transactions (`generate()`, `generate_all()`), saves them with their
  `config.json` (`save()`) and reads them back (`read_transactions()`).
- Reproducible output: every draw comes from local random streams derived from
  `random_state`, and a component that is off changes nothing.
- CP and CNP transactions with billing, terminal and shipping locations around
  Brazilian cities, drawn by population.
- Fraud injection (`FraudConfig`): fraudsters add new transactions in five
  scenarios (anomalous amount, compromised terminal, stolen card number, stolen
  physical card and card testing), with rates proportional to the population and
  fraudster drift (`IntensitySchedule`, `FraudConfig.realistic()`).
- Concept drift (`DriftConfig`): customers move from regime A to regime B with
  sudden, gradual, incremental or recurrent drift, plus population drift
  (`GlobalDrift`).
- Seasonality (`SeasonalityConfig`): weekday, payday and December factors.
- Merchants and categories (`MerchantConfig`): physical and online merchants, ten
  default categories, customer preferences, spending factors and favourite
  merchants.
- Ground truth in the `GT_*` columns and `TX_FRAUD_SCENARIO`, for evaluating
  models; `ground_truth.compromise_table()` lists the compromised cards.
- Presets (`synccfd.simulator.presets`): `realistic()` and `ablation(step)`.
- Example notebooks: `simulator`, `drift`, `frauds` and `merchants`.
