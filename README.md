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

The defaults are the complete simulator: merchants and categories, customer concept
drift, seasonality and five fraud scenarios injected by fraudsters (see below).
`None` switches a component off, e.g. `DatasetGenerator(drift=None, merchants=None)`;
`fraud=None` generates no fraud at all.

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

## Version 0.2.0 and v0.1

**0.2.0 breaks compatibility with 0.1.0.** The v0.1 simulator (the one of the SBSeg
2024 paper) is still available with `pip install synccfd==0.1.0`. In 0.2.0:

- one simulation engine, vectorised, whose randomness comes only from local streams
  derived from `random_state`; the `engine` parameter and the v0.1 engine are gone;
- the defaults of `DatasetGenerator` switch every component on (merchants, customer
  drift, seasonality, fraud injection), for 10 000 customers, 20 000 terminals and
  180 days from 2023-01-01;
- frauds are always injected as new transactions: the v0.1 mechanism that altered
  and labelled legitimate transactions (`mechanism="tamper"`, with
  `s1_amount_threshold`, `s3_tamper_fraction`, `amount_multiplier`, `s4_cnp_prob`)
  is gone, as are `FraudSimulator` and `presets.legacy()`;
- `FraudConfig()` has the five scenarios with rates proportional to the population.

## Presets

`synccfd.simulator.presets` returns ready arguments for `DatasetGenerator`; any of
them can be overridden:

```python
from synccfd import DatasetGenerator
from synccfd.simulator import presets

# Every component (the defaults of DatasetGenerator), for 1 000 customers
generator = DatasetGenerator(**presets.realistic(n_customers=1_000))
# customers, terminals, merchants, transactions and the configuration
result = generator.generate_all()
generator.save(output_dir="data", save_format="parquet")  # also writes config.json
```

| Preset | What it sets |
| --- | --- |
| `realistic()` | `MerchantConfig()`, `DriftConfig()`, `SeasonalityConfig()` and `FraudConfig.realistic()`, 10 000 customers, 20 000 terminals, 180 days from 2023-01-01, `random_state=42` (the defaults of `DatasetGenerator`) |
| `ablation(step)` | The components switched on cumulatively: `"base"` (stationary customers, fraud scenarios 1–4 at constant rates), `"drift"` (+ drift and seasonality), `"frauds"` (the five scenarios with fraudster drift), `"merchants"` (+ merchants, same as `realistic()`). The fraud fractions are scaled per step so that the fraud rate is about the same (0.34–0.35%) |

`save()` writes `config.json` next to the data: the version of `synccfd`, the
size, the seed and every configuration object.

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
| `TX_FRAUD_SCENARIO` | Ground truth: 0 = genuine, 1–5 = fraud scenario (see below) |
| `MERCHANT_ID`, `MERCHANT_CATEGORY` | Merchant and its category (with merchants; observable, like an MCC) |
| `GT_REGIME`, `GT_REGIME_WEIGHT`, `GT_SHOCK` | Ground truth of drift: regime (`A`/`B`) of a legitimate transaction, weight of regime B on its day, global shock |
| `GT_FRAUD_SCENARIO`, `GT_FRAUDSTER_ID`, `GT_CAMPAIGN_ID`, `GT_COMPROMISE_ID` | Ground truth of the frauds: scenario, fraudster, attack, compromised card |

**`customer_profiles`**: `CUSTOMER_ID`, `billing_lat`, `billing_long`,
`mean_amount`, `std_amount`, `mean_nb_tx_per_day`, `cnp_prob`,
`available_terminals`. With merchants, `spending_factor` and `pref_<category>`.
With drift, `GT_DRIFT_*`, `GT_B_*` and `GT_RELOCATED`.

**`terminal_profiles`**: `TERMINAL_ID`, `terminal_lat`, `terminal_long`; with
merchants, `MERCHANT_ID`.

**`merchant_profiles`** (with merchants): `MERCHANT_ID`, `MERCHANT_CATEGORY`,
`channel`, `n_terminals`, `lat`, `long`.

The columns of each component are described in its section below.

> **Ground truth is not a feature.** Columns starting with `GT_` (and
> `TX_FRAUD_SCENARIO`) describe how the data was generated: drift regimes, fraud
> scenarios. A model could not observe them in practice, so never use them as model
> inputs. They are there to evaluate models (for example, per scenario or per
> regime).

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

Every other parameter of each scenario (amount factors, burst sizes, blocking
probability, ...) is documented in `FraudConfig`.

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

- Runs are deterministic: the same parameters always produce the same dataset
  (a regression test freezes a small output). `random_state` controls every random
  draw, and the global `random`/`numpy.random` states are never used.
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
