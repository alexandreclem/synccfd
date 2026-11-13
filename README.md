# SynCCFD

**Synthetic Credit Card Fraud Detection: a transaction and fraud data simulator.**

[![CI](https://github.com/alexandreclem/synccfd/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandreclem/synccfd/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/synccfd.svg)](https://pypi.org/project/synccfd/)
[![Downloads](https://img.shields.io/pypi/dm/synccfd.svg)](https://pypi.org/project/synccfd/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

SynCCFD generates synthetic credit card transactions with labelled frauds, for
research on fraud detection models: customers, terminals, merchants and a labelled
transaction stream, with no real cardholder data.

The simulator extends the one in the
[Reproducible Machine Learning for Credit Card Fraud Detection handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
(Le Borgne et al., 2022), extended in our
[SBSeg 2024 paper](https://sol.sbc.org.br/index.php/sbseg/article/view/30067/29874).
It models:

- **Card-Present (CP) and Card-Not-Present (CNP) transactions**, with billing,
  terminal and shipping locations.
- **A realistic geography**: locations are drawn from Brazilian municipalities,
  weighted by population.
- **Merchants and categories**, which set the amount and the time of a purchase.
- **Concept drift and seasonality**: customers change behaviour over time, and the
  calendar adds patterns.
- **Five fraud scenarios**, injected by fraudsters who also change over time.

The public classes have short docstrings: `help(synccfd.DatasetGenerator)`,
`help(synccfd.simulator.config.FraudConfig)`, and so on. The model itself is described
below.

## Installation

```bash
pip install synccfd
```

SynCCFD needs Python 3.10 or later. To run the example notebooks, install the
`examples` extra (`pip install "synccfd[examples]"`).

From source, for development:

```bash
git clone https://github.com/alexandreclem/synccfd
cd synccfd
pip install -e ".[dev,examples]"
```

## Minimal example

```python
from synccfd import DatasetGenerator

generator = DatasetGenerator(
    n_customers=1000,
    n_terminals=2000,
    nb_days=30,
    start_date="2025-01-01",
    random_state=42,
)

customer_profiles, terminal_profiles, transactions = generator.generate()

print(transactions.shape)
print(f"Fraud rate: {100 * transactions.TX_FRAUD.mean():.2f}%")
```

`generate()` returns three DataFrames and does **not** touch the disk. Save a dataset
explicitly when you want it, and load a date range back later:

```python
generator.save(output_dir="data", save_format="parquet")  # also: "pkl", "csv"

txs = generator.read_transactions(
    begin_date="2025-01-01",
    end_date="2025-01-30",
    save_format="parquet",
    data_dir="data",
)
```

`save()` writes `profiles/`, `transactions/raw/` (one file per day; with
`save_by_day=False`, a single file) and `config.json`.

## Presets and components

`presets.realistic()` is the complete simulator, and the same as `DatasetGenerator()`:
every component on, for 10 000 customers, 20 000 terminals and 180 days from
2023-01-01, with `random_state=42`. Any argument can be overridden:

```python
from synccfd import DatasetGenerator
from synccfd.simulator import presets

generator = DatasetGenerator(**presets.realistic(n_customers=1_000))

# Everything in one object: the tables and the effective configuration
result = generator.generate_all()
result.customers, result.terminals, result.merchants, result.transactions
result.config
```

Each component is one argument of `DatasetGenerator`, with a configuration object of
`synccfd.simulator.config`. **`None` switches a component off**, and a component that
is off draws nothing and changes nothing:

| Argument | Configuration | Default | With `None` |
| --- | --- | --- | --- |
| `fraud` | `FraudConfig` | `FraudConfig.realistic()` | no fraud at all |
| `drift` | `DriftConfig` | `DriftConfig()` | customers with a stationary behaviour |
| `seasonality` | `SeasonalityConfig` | `SeasonalityConfig()` | no calendar patterns |
| `merchants` | `MerchantConfig` | `MerchantConfig()` | no merchants, no categories |

```python
# Stationary customers, no merchants, the five fraud scenarios
generator = DatasetGenerator(
    n_customers=1_000, drift=None, seasonality=None, merchants=None
)
```

`presets.ablation(step)` switches the components on cumulatively, to measure the
effect of each one on a model. Every step has the same size and seed, and the fraud
fractions are scaled so that the fraud rate is about the same (0.34–0.35%):

| `step` | Components |
| --- | --- |
| `"base"` | stationary customers, fraud scenarios 1–4 at constant rates |
| `"drift"` | plus customer drift and seasonality |
| `"frauds"` | with the five scenarios and fraudster drift instead (`FraudConfig.realistic()`) |
| `"merchants"` | plus merchants; the same as `realistic()` |

## Output tables and columns

**`transactions`**

| Column | Description |
| --- | --- |
| `TRANSACTION_ID` | Unique transaction identifier (chronological position) |
| `TX_DATETIME` | Transaction timestamp |
| `TX_TIME_SECONDS`, `TX_TIME_DAYS` | Time since `start_date`, in seconds / days |
| `CUSTOMER_ID`, `TERMINAL_ID` | Customer and terminal involved |
| `MERCHANT_ID`, `MERCHANT_CATEGORY` | Merchant and its category (with merchants; observable, like an MCC) |
| `TX_AMOUNT` | Transaction amount (R$) |
| `TX_TYPE` | `CP` (Card Present) or `CNP` (Card Not Present) |
| `TX_BILL_LAT`, `TX_BILL_LONG` | Billing location |
| `TX_TERM_LAT`, `TX_TERM_LONG` | Terminal location |
| `TX_SHIPP_LAT`, `TX_SHIPP_LONG` | Shipping / delivery location |
| `TX_FRAUD` | **The label**: 0 = legitimate, 1 = fraud |
| `TX_FRAUD_SCENARIO` | Ground truth: 0 = legitimate, 1–5 = fraud scenario |
| `GT_FRAUD_SCENARIO`, `GT_FRAUDSTER_ID`, `GT_CAMPAIGN_ID`, `GT_COMPROMISE_ID` | Ground truth of the frauds: scenario, fraudster, attack, compromised card |
| `GT_REGIME`, `GT_REGIME_WEIGHT` | Ground truth of drift: regime (`A`/`B`) of a legitimate transaction, weight of regime B on its day |
| `GT_SHOCK` | Ground truth of global drift: the day is inside a shock |

**`customer_profiles`**

| Column | Description |
| --- | --- |
| `CUSTOMER_ID` | Unique customer identifier |
| `billing_lat`, `billing_long` | Billing location |
| `mean_amount`, `std_amount` | Mean and standard deviation of the amounts |
| `mean_nb_tx_per_day` | Mean number of transactions per day |
| `cnp_prob` | Probability of a CNP purchase |
| `available_terminals` | Ids of the terminals in reach |
| `spending_factor`, `pref_<category>` | With merchants: spending factor and category preferences |
| `GT_DRIFT_*`, `GT_B_*`, `GT_RELOCATED` | With drift: ground truth of the drift (see below) |

**`terminal_profiles`**: `TERMINAL_ID`, `terminal_lat`, `terminal_long`; with
merchants, `MERCHANT_ID`.

**`merchant_profiles`** (with merchants; `generator.merchant_profiles` or
`result.merchants`): `MERCHANT_ID`, `MERCHANT_CATEGORY`, `channel` (`CP`/`CNP`),
`n_terminals`, `lat`, `long`.

The columns of each component are described in its section below.

> **Ground truth is not a feature.** Columns starting with `GT_` and
> `TX_FRAUD_SCENARIO` describe how the data was generated: fraud scenarios, fraudsters,
> campaigns, drift regimes. A model could not observe them in practice, so **never use
> them as model inputs**. They are there to evaluate models, for example per scenario
> or per regime. `TX_FRAUD` is the label.

## Concept drift and seasonality

Customers can change behaviour over time (`drift`, on by default) and the calendar
adds patterns (`seasonality`, on by default):

```python
from synccfd import DatasetGenerator
from synccfd.simulator.config import DriftConfig, SeasonalityConfig

generator = DatasetGenerator(
    n_customers=10_000,
    nb_days=365,
    drift=DriftConfig(customer_fraction=0.3),
    seasonality=SeasonalityConfig(),
)
customer_profiles, terminal_profiles, transactions = generator.generate()
```

Each customer has a **regime A** (its profile). A drifting customer also gets a
**regime B**, drawn from A: amount and frequency multiplied by log-normal factors,
CNP probability shifted on the logit scale and, with a small probability, a new
city. The weight of regime B over time, `w(t)`, depends on the drift type:

| Type | `w(t)` | Effect |
| --- | --- | --- |
| `sudden` | 0 before `t0`, 1 after | regime B from `t0` on |
| `gradual` | rises from 0 to 1 over `[t0, t0 + L]` | each transaction comes from B with probability `w(t)` |
| `incremental` | same | parameters interpolated between A and B |
| `recurrent` | 1 for `D` days every `P` days, from `t0` | e.g. holidays, travel |

`DriftConfig(global_drift=GlobalDrift(...))` adds population drift (an amount trend,
shocks such as Black Friday). `SeasonalityConfig` adds calendar patterns (weekday,
paydays, December). These patterns repeat, so they are labelled separately from
drift. Customers without drift keep exactly the same transactions when drift is
enabled.

Ground-truth columns:

- **`customer_profiles`**:
  - `GT_DRIFT_TYPE` (`none`, `sudden`, `gradual`, `incremental`, `recurrent`);
  - `GT_DRIFT_T0`, `GT_DRIFT_L`, `GT_DRIFT_PERIOD`, `GT_DRIFT_ACTIVE_DAYS`;
  - the regime B parameters `GT_B_*`, and `GT_RELOCATED`.
- **`transactions`**:
  - `GT_REGIME_WEIGHT` (`w` on the transaction's day);
  - `GT_REGIME` (`A`/`B`; for incremental drift, `B` once `w >= 0.5`);
  - with global drift, `GT_SHOCK`.

See [`examples/drift.ipynb`](examples/drift.ipynb) for one customer of each drift
type.

## Fraud scenarios

Fraudsters make **new** transactions, added to the customers' own ones. Legitimate
transactions are never altered nor labelled as fraud, so a victim's history stays
intact. `FraudConfig()` has the five scenarios at constant rates;
`FraudConfig.realistic()` (the default) adds fraudster drift:

```python
from synccfd import DatasetGenerator
from synccfd.simulator.config import FraudConfig

generator = DatasetGenerator(
    n_customers=10_000, nb_days=180, fraud=FraudConfig.realistic()
)
customer_profiles, terminal_profiles, transactions = generator.generate()
```

| # | Scenario | What the fraudster does |
| --- | --- | --- |
| 1 | Anomalous amount | One purchase far above the victim's recent amounts (2 × their 95th percentile over the last 30 days, or more). Unlike a labelling rule with a fixed threshold, it does not label customers who legitimately spend more. |
| 2 | Compromised terminal | A compromised terminal reads the cards of the customers who buy there (those purchases stay legitimate). Days later, some read cards are cloned (a fast sequence of CP purchases near the fraudster) and others leak to scenario 3 |
| 3 | Stolen card number | Bursts of 2–5 CNP purchases within hours, shipped to the fraudster, often at night, above the victim's usual amounts |
| 4 | Stolen physical card | A fast sequence of purchases near the victim's city, until the card is blocked, a spending limit is reached or 24 h have passed |
| 5 | Card testing | 3–10 CNP purchases of R$ 1–5 within minutes; half of the time the fraudster goes on to scenario 3 with the same card |

The number of new victims per day is drawn from `Poisson(fraction × population)`
(`rate_mode="proportional"`, the default), so the expected fraud rate does not depend
on the dataset size; `rate_mode="absolute"` uses fixed counts per day.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `scenarios` | `(1, 2, 3, 4, 5)` | Scenarios to apply |
| `s1_customer_fraction_per_day` | `3e-4` | Scenario 1 victims per day, as a fraction of the customers |
| `s2_terminal_fraction_per_day`, `s2_duration_days` | `1e-4`, `28` | Terminals compromised per day, and for how long |
| `s3_customer_fraction_per_day`, `s3_duration_days` | `3e-4`, `14` | Card numbers stolen per day, and the campaign length |
| `s4_customer_fraction_per_day` | `3e-4` | Physical cards stolen per day |
| `s5_customer_fraction_per_day` | `2e-4` | Card-testing victims per day |
| `intensity_schedule` | none (`realistic()`: scenarios 2 and 5) | Multiplier of each scenario's daily rate over time |

The other parameters of each scenario (amount factors, burst sizes, blocking
probability, ...) are fields of `FraudConfig`, with a comment on each default in
`simulator/config.py`.

`intensity_schedule` changes a scenario's daily rate over time (fraudster drift):
`IntensitySchedule("step" | "ramp" | "pulse" | "constant", start, end, base, level)`,
with times as fractions of the period. In `FraudConfig.realistic()`, scenario 2
halves from 60% of the period on and scenario 5 starts at 30%. The fraudulent
transactions get `GT_FRAUD_SCENARIO`, `GT_FRAUDSTER_ID`, `GT_CAMPAIGN_ID` (the
transactions of one attack, e.g. card testing and the purchases that follow) and
`GT_COMPROMISE_ID`. The last one marks the legitimate purchase where a card was read
at a compromised terminal and the frauds later made with that card;
`synccfd.simulator.ground_truth.compromise_table(transactions)` lists the
compromises. Injected transactions have no `GT_REGIME`. See
[`examples/frauds.ipynb`](examples/frauds.ipynb).

## Merchants and categories

With `merchants=MerchantConfig()` (on by default), the amount and the time of a purchase
depend on the merchant's **category**, not only on the customer:

```python
from synccfd import DatasetGenerator
from synccfd.simulator.config import MerchantConfig

generator = DatasetGenerator(
    n_customers=10_000, nb_days=180, merchants=MerchantConfig()
)
customer_profiles, terminal_profiles, transactions = generator.generate()
merchant_profiles = generator.merchant_profiles
```

- The terminals belong to physical merchants (1 + Poisson(0.5) terminals in one city).
  Online merchants (20 per category with CNP) have one virtual terminal at their head
  office; their popularity follows a Zipf law.
- Ten default categories (`DEFAULT_CATEGORIES`): supermarket, restaurant, fuel
  station, pharmacy, clothing, electronics (CP and/or CNP), travel, marketplace,
  digital services and ride-hailing (CNP only). Each has a median amount and a
  spread, a weight, a 24-hour profile, opening hours (CP purchases only happen
  then), a weekend factor and a skimming risk. All are configurable.
- The category names are **values of the data, in Portuguese**, as in the Brazilian
  market the model describes; they appear in `MERCHANT_CATEGORY` and in the
  `pref_<category>` columns: `supermercado`, `restaurante_bar`, `posto_combustivel`,
  `farmacia`, `vestuario`, `eletronicos`, `viagem_turismo`,
  `ecommerce_marketplace`, `servicos_digitais` and `transporte_app`.
- Mixed categories (CP and CNP) have an online fraction (supermarket 0.10, pharmacy
  0.15, clothing 0.30, electronics 0.50), combined with the customer's `cnp_prob` on
  the logit scale, so that the mean stays at the category's fraction. Customers with
  no merchant of the category in reach always buy it online.
- Each customer has a preference over the categories, `Dirichlet(10 × weights)`, a
  spending factor, `LogNormal(0, 0.5)`, and favourite merchants per category (60% of
  the purchases in a category go to 2 favourites). A purchase costs
  `median × spending factor × LogNormal(0, σ)` of its category.
- With drift, regime B also changes the category preferences. With fraud injection,
  each scenario buys in its own categories, and scenario 2 compromises whole merchants,
  chosen by the category's skimming risk.

New columns:
- `transactions` get `MERCHANT_ID` and `MERCHANT_CATEGORY`. They are observable,
  like an MCC, so they are not `GT_*`.
- `terminal_profiles` get `MERCHANT_ID`.
- `customer_profiles` get `spending_factor` and `pref_<category>`; `mean_amount` and
  `std_amount` are recomputed from the generated transactions.
- `merchant_profiles` (saved by `save()`) lists `MERCHANT_ID`, `MERCHANT_CATEGORY`,
  `channel`, `n_terminals`, `lat` and `long`.

See [`examples/merchants.ipynb`](examples/merchants.ipynb).

## Reproducibility

- Runs are deterministic: the same parameters and the same version of `synccfd` always
  produce the same dataset. `random_state` controls every random draw, and the global
  `random`/`numpy.random` states are never used. A regression test freezes a small
  output. Between platforms, floats may differ in the last bit.
- `save()` writes `config.json` next to the data, and `DatasetGenerator.config` and
  `SimulationResult.config` give the same content: the version of `synccfd`, the size,
  the seed and every configuration object, with all its fields. It records everything
  needed to regenerate the dataset.
- Locations are drawn from **Brazilian** municipality data bundled with the package
  (population-weighted). SynCCFD models a Brazil-based network, and amounts are in R$.

## Example notebooks

The notebooks of [`examples/`](examples) use the `examples` extra. All of them read the
same configuration (`presets.realistic()` and its default size, through
`presets.from_env()`).

| Notebook | Content |
| --- | --- |
| [`simulator.ipynb`](examples/simulator.ipynb) | Walkthrough: tables, locations, amounts, times, frauds; the generation time; the dataset statistics |
| [`drift.ipynb`](examples/drift.ipynb) | Customer concept drift: one customer of each type, regime B |
| [`frauds.ipynb`](examples/frauds.ipynb) | The five fraud scenarios, campaigns, compromised terminals |
| [`merchants.ipynb`](examples/merchants.ipynb) | Merchants and categories: amounts, hours, shares, loyalty |

Each notebook writes its figures to `examples/figures/` (not versioned), one subfolder
per notebook: `overview`, `drift`, `frauds` and `merchants`. The file
names are the same in every language; only the labels change. In Portuguese the numbers
use the decimal comma and a space between thousands, except in `transactions_table`,
which shows the data as the library outputs them.
`simulator.ipynb` also writes `dataset_stats.json` there. These environment variables
configure a run:

| Variable | Meaning | Default |
| --- | --- | --- |
| `SYNCCFD_LANG` | Language of the figure labels (`LANG` in the notebooks): `en` or `pt` | `en` |
| `SYNCCFD_FIG_DIR` | Output folder | `figures` |
| `SYNCCFD_PRESET` | `realistic` or `ablation:<step>` | `realistic` |
| `SYNCCFD_N_CUSTOMERS`, `SYNCCFD_N_TERMINALS`, `SYNCCFD_NB_DAYS`, `SYNCCFD_SEED` | Size and seed | `presets.DEFAULT_SIZE` |

To generate every figure and the statistics with the labels in Portuguese, from a
terminal and without changing the notebooks (`--output-dir` keeps the executed copies
elsewhere):

```bash
cd examples
SYNCCFD_LANG=pt jupyter nbconvert --to notebook --execute --output-dir executed \
    simulator.ipynb drift.ipynb frauds.ipynb merchants.ipynb
```

In PowerShell, set the variable first: `$env:SYNCCFD_LANG = "pt"`. The figures are
deterministic: an unchanged figure is written byte for byte the same.

## Extending

Every default value and configuration object lives in `simulator/config.py`, and every
random draw must come from the stream passed in, so that the other components keep
their draws. Without a new component, the regression test must keep passing.

**A merchant category** needs no code change: pass the categories to `MerchantConfig`.
The weights are normalised.

```python
from synccfd import DatasetGenerator
from synccfd.simulator.config import (
    DEFAULT_CATEGORIES,
    MerchantCategory,
    MerchantConfig,
    hourly_profile,
)

pet_shop = MerchantCategory(
    name="pet_shop",
    channel="CP",  # "CP", "CNP" or "CP+CNP"
    median_amount=90,
    log_sigma=0.6,
    weight=0.03,
    hourly_profile=hourly_profile(((11, 3, 1.0), (17, 3, 1.0))),
    opening_hours=(9, 20),
)
generator = DatasetGenerator(
    n_customers=1_000,
    n_terminals=2_000,
    nb_days=30,
    merchants=MerchantConfig(categories=(*DEFAULT_CATEGORIES, pet_shop)),
)
```

**A fraud scenario:**

1. In `config.py`, add its number to `KNOWN_FRAUD_SCENARIOS`, its rate parameters
   (`sN_customers_per_day`, `sN_customer_fraction_per_day`) and its own parameters to
   `FraudConfig`, with validation in `__post_init__`. Add a branch for it in
   `FraudConfig.daily_counts`.
2. In `engine/injection.py`, write a `_scenario_N` function that gets the victims of
   each day from `config.daily_counts(N, ...)` (which follows `rate_mode` and
   `intensity_schedule`) and adds the new transactions with `out.add`, with
   `GT_FRAUD_SCENARIO`, `GT_FRAUDSTER_ID` and `GT_CAMPAIGN_ID`. Call it from
   `inject_frauds`, with the `fraud` stream only.
3. With merchants, add its categories (`sN_categories`, also in
   `FraudConfig.fraud_categories`) and use `_Shop` to pick merchants and amounts.
4. Test that it only adds transactions (the legitimate ones stay unchanged), that its
   rate follows the configured fraction, and its characteristic behaviour.

**A drift type:**

1. In `config.py`, add it to `DRIFT_TYPES` and give it a weight in the default
   `type_weights`.
2. In `engine/drift.py`, give its weight `w(t)` in `regime_weight`, and draw its own
   parameters in `assign_drift`, if any.
3. Say how a transaction uses it (each one from a regime, or interpolated parameters)
   in **both** paths of the generation: `engine/transactions.py` and, with merchants,
   `engine/merchants.py`.

## Citation

If you use SynCCFD in academic work, please cite the paper it is based on
(a machine-readable [`CITATION.cff`](CITATION.cff) is included):

> A. C. B. dos Santos, R. de S. Passos, L. D. T. J. Tarrataca, D. de O. Cardoso,
> D. B. Haddad, F. da R. Henriques. *Construção de um Modelo Orientado a Dados
> para Detecção de Fraudes em Cartões de Crédito utilizando Dados Sintéticos.*
> Anais do SBSeg 2024 (Artigos Curtos).
> https://sol.sbc.org.br/index.php/sbseg/article/view/30067/29874

## Contributing

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
