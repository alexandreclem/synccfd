# SynCCFD

**Synthetic Credit Card Fraud Detection — a transaction & fraud data simulator.**

[![CI](https://github.com/alexandreclem/synccfd/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandreclem/synccfd/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/synccfd.svg)](https://pypi.org/project/synccfd/)
[![Downloads](https://img.shields.io/pypi/dm/synccfd.svg)](https://pypi.org/project/synccfd/)
[![Docs](https://readthedocs.org/projects/synccfd/badge/?version=latest)](https://synccfd.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

SynCCFD generates realistic synthetic credit card transaction data with labelled
fraud, for research and benchmarking of fraud-detection models. It produces
customer profiles, terminal profiles, and a fully labelled transaction stream —
no real cardholder data required.

📖 **Documentation:** https://synccfd.readthedocs.io/en/latest/

The simulator is a direct evolution of the one in the
[Reproducible Machine Learning for Credit Card Fraud Detection handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
(Le Borgne et al., 2022), extended for the work described in our
[SBSeg 2024 paper](https://sol.sbc.org.br/index.php/sbseg/article/view/30067/29874) with three contributions:

- **Transaction type** — Card-Present (CP) vs Card-Not-Present (CNP), affecting
  timing, locations, and fraud behaviour.
- **A realistic geography** — locations are sampled from Brazilian municipalities,
  weighted by population, instead of a uniform grid.
- **A new fraud scenario** — lost/stolen physical card, plus billing / terminal /
  shipping locations per transaction.

## Installation

```bash
pip install synccfd
```

From source, for development:

```bash
git clone https://github.com/alexandreclem/synccfd
cd synccfd
pip install -e ".[dev]"
```

Optional extras: `.[examples]` (plotting + notebooks), `.[docs]` (Sphinx).

## Quickstart

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

`generate()` returns three DataFrames and does **not** touch the disk. Persist a
dataset explicitly when you want it (day-partitioned files, following the handbook
convention):

```python
generator.save(output_dir="data", save_format="parquet")

# ...later, load a date range back:
txs = generator.read_transactions(
    begin_date="2025-01-01",
    end_date="2025-01-30",
    save_format="parquet",
    data_dir="data",
)
```

See [`examples/simulator.ipynb`](examples/simulator.ipynb) for a full walkthrough
with plots.

## Output schema

**`transactions`**

| Column | Description |
| --- | --- |
| `TRANSACTION_ID` | Unique transaction identifier (chronological) |
| `TX_DATETIME` | Transaction timestamp |
| `TX_TIME_SECONDS`, `TX_TIME_DAYS` | Time since `start_date`, in seconds / days |
| `CUSTOMER_ID`, `TERMINAL_ID` | Customer and terminal involved |
| `TX_AMOUNT` | Transaction amount |
| `TX_TYPE` | `CP` (Card Present) or `CNP` (Card Not Present) |
| `TX_BILL_LAT`, `TX_BILL_LONG` | Billing location |
| `TX_TERM_LAT`, `TX_TERM_LONG` | Terminal location |
| `TX_SHIPP_LAT`, `TX_SHIPP_LONG` | Shipping / delivery location |
| `TX_FRAUD` | Binary fraud label (0 = genuine, 1 = fraud) |
| `TX_FRAUD_SCENARIO` | 0 = genuine, 1–4 = fraud scenario (see below; 5 with fraud injection) |

**`customer_profiles`**: `CUSTOMER_ID`, `billing_lat`, `billing_long`,
`mean_amount`, `std_amount`, `mean_nb_tx_per_day`, `cnp_prob`.

**`terminal_profiles`**: `TERMINAL_ID`, `terminal_lat`, `terminal_long`.

> **Ground truth is not a feature.** Columns starting with `GT_` (and
> `TX_FRAUD_SCENARIO`) describe how the data was generated: drift regimes, fraud
> scenarios. A model could not observe them in practice, so never use them as model
> inputs. They are there to evaluate models (for example, per scenario or per
> regime).

## Fraud scenarios

| # | Scenario | Behaviour |
| --- | --- | --- |
| 1 | High amount | Any transaction above a fixed amount threshold is fraudulent |
| 2 | Compromised terminal | A few terminals are compromised each day for a period; transactions there are fraudulent |
| 3 | Stolen card number (CNP) | A customer is compromised for ~2 weeks; a fraction of their transactions are tampered — forced to CNP, amount inflated, shipped to the fraudster's location |
| 4 | Stolen physical card (CP) | A customer's card is stolen for a few days; the fraudster spends as much as possible, as fast as possible, from their own location |

## Fraud configuration

All fraud parameters are exposed through `FraudConfig`. The defaults reproduce
v0.1.0 exactly.

```python
from synccfd import DatasetGenerator
from synccfd.simulator.config import FraudConfig

generator = DatasetGenerator(
    n_customers=1000,
    n_terminals=2000,
    nb_days=30,
    fraud=FraudConfig(
        scenarios=(1, 3, 4),  # disable scenario 2
        rate_mode="proportional",  # scale compromises with the population
        s3_customer_fraction_per_day=3e-3,
        s4_customer_fraction_per_day=3e-3,
    ),
)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `scenarios` | `(1, 2, 3, 4)` | Scenarios to apply |
| `rate_mode` | `"absolute"` | `"absolute"`: fixed counts per day; `"proportional"`: daily count drawn from `Poisson(fraction × population)` |
| `s1_amount_threshold` | `220.0` | Scenario 1 amount threshold |
| `s2_terminals_per_day` / `s2_terminal_fraction_per_day` | `2` / `1e-4` | Terminals compromised per day |
| `s2_duration_days` | `28` | Days a terminal stays compromised |
| `s3_customers_per_day` / `s3_customer_fraction_per_day` | `3` / `3e-4` | Card numbers stolen per day |
| `s4_customers_per_day` / `s4_customer_fraction_per_day` | `3` / `3e-4` | Physical cards stolen per day |
| `s3_duration_days`, `s4_duration_days` | `14`, `5` | Days a customer stays compromised |
| `s3_tamper_fraction` | `1/3` | Fraction of a scenario-3 victim's transactions that become fraud |
| `amount_multiplier` | `5.0` | Amount factor of tampered transactions (scenarios 3 and 4) |
| `s4_cnp_prob` | `0.5` | Probability that a scenario-4 transaction is CNP |

With `rate_mode="absolute"` the number of compromised terminals and customers per
day does not depend on the dataset size, so small datasets get a very high fraud
rate (about 26% with 100 customers and 15 days, against under 1% with 5 000
customers). With `"proportional"`, each day's count is drawn from a Poisson
distribution with mean `fraction × population`, so the *expected* fraud rate does not
depend on the dataset size. On small datasets most days have no new compromise and
the realised rate varies more from seed to seed. The default fractions give, on
average, the absolute counts at the default size (10 000 customers, 20 000
terminals). In this mode `random_state` controls the daily counts.

## Engines

`DatasetGenerator(..., engine=...)` selects the simulation engine:

| `engine` | What it is |
| --- | --- |
| `"legacy"` | The v0.1.0 code, unchanged. With the default configuration it reproduces v0.1.0 exactly. |
| `"v2"` | A vectorised engine with the same model, much faster. It uses only local random streams derived from `random_state`, never the global `random`/`numpy.random` state. It reproduces the v0.1 model statistically, not draw by draw. |
| `"auto"` (default) | `"legacy"`, unless a component that only v2 supports is enabled: drift, seasonality or fraud injection. |

```python
generator = DatasetGenerator(n_customers=10_000, nb_days=180, engine="v2")
customer_profiles, terminal_profiles, transactions = generator.generate()
```

Both engines produce the same columns and dtypes. Differences in v2:

- A transaction whose time of day falls outside the day is redrawn, not discarded,
  so the number of transactions per day follows the Poisson parameter exactly (v0.1
  loses about 1.5% of the transactions, mostly CP ones).
- Transactions are sorted chronologically again after the fraud scenarios (scenario
  4 moves transactions in time), and `TRANSACTION_ID` is the chronological position.
- Random points around a location use a spherical model instead of geopy's
  ellipsoid; the difference is negligible at the radii used (≤ 12 km).

## Concept drift and seasonality

Customers can change behaviour over time (v2 engine; `engine="auto"` switches to it
when `drift` or `seasonality` is set):

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

## Fraud injection

With `FraudConfig(mechanism="inject")` (v2 engine), fraudsters make **new**
transactions, added to the customers' own ones. Legitimate transactions are never
altered nor labelled as fraud, so a victim's history stays intact. `FraudConfig.realistic()` enables all
five scenarios with rates proportional to the population:

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
| 1 | Anomalous amount | One purchase far above the victim's recent amounts (2 × their 95th percentile over the last 30 days, or more). Unlike the fixed threshold, it does not label customers who legitimately spend more. |
| 2 | Compromised terminal | A compromised terminal reads the cards of the customers who buy there (those purchases stay legitimate). Days later, some read cards are cloned (a fast sequence of CP purchases near the fraudster) and others leak to scenario 3 |
| 3 | Stolen card number | Bursts of 2–5 CNP purchases within hours, shipped to the fraudster, often at night, above the victim's usual amounts |
| 4 | Stolen physical card | A fast sequence of purchases near the victim's city, until the card is blocked, a spending limit is reached or 24 h have passed |
| 5 | Card testing | 3–10 CNP purchases of R$ 1–5 within minutes; half of the time the fraudster goes on to scenario 3 with the same card |

`intensity_schedule` changes a scenario's daily rate over time (fraudster drift):
`IntensitySchedule("step" | "ramp" | "pulse" | "constant", start, end, base, level)`,
with times as fractions of the period. In `FraudConfig.realistic()`, scenario 2
halves from 60% of the period on and scenario 5 starts at 30%. With injection, the
transactions get `GT_FRAUD_SCENARIO`, `GT_FRAUDSTER_ID`, `GT_CAMPAIGN_ID` (the
transactions of one attack, e.g. card testing and the purchases that follow) and
`GT_COMPROMISE_ID`. The last one marks the legitimate purchase where a card was read
at a compromised terminal and the frauds later made with that card;
`synccfd.simulator.ground_truth.compromise_table(transactions)` lists the
compromises. Injected transactions have no `GT_REGIME`. See
[`examples/frauds.ipynb`](examples/frauds.ipynb).

## Merchants and categories

With `merchants=MerchantConfig()` (v2 engine), the amount and the time of a purchase
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

## Reproducibility & scope

- Runs are deterministic: the same parameters always produce the same dataset.
- With `engine="v2"`, `random_state` controls every random draw: different seeds give
  different datasets. In the legacy engine the random seeds are fixed internally,
  so `random_state` has little effect on the generated data.
- Locations are drawn from **Brazilian** municipality data bundled with the package
  (population-weighted). SynCCFD currently models a Brazil-based network.

## Citation

If you use SynCCFD in academic work, please cite the paper it is based on
(a machine-readable [`CITATION.cff`](CITATION.cff) is included):

> A. C. B. dos Santos, R. de S. Passos, L. D. T. J. Tarrataca, D. de O. Cardoso,
> D. B. Haddad, F. da R. Henriques. *Construção de um Modelo Orientado a Dados
> para Detecção de Fraudes em Cartões de Crédito utilizando Dados Sintéticos.*
> Anais do SBSeg 2024 (Artigos Curtos).
> https://sol.sbc.org.br/index.php/sbseg/article/view/30067/29874

## Contributing

Contributions are welcome. Please open an issue to discuss substantial changes
first, run `ruff check .` and `pytest` before submitting, and update tests as
appropriate.

## License

[MIT](LICENSE)
