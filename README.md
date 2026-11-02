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
| `TX_FRAUD_SCENARIO` | 0 = genuine, 1–4 = fraud scenario (see below) |

**`customer_profiles`**: `CUSTOMER_ID`, `billing_lat`, `billing_long`,
`mean_amount`, `std_amount`, `mean_nb_tx_per_day`, `cnp_prob`.

**`terminal_profiles`**: `TERMINAL_ID`, `terminal_lat`, `terminal_long`.

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
        scenarios=(1, 3, 4),          # disable scenario 2
        rate_mode="proportional",     # scale compromises with the population
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

## Reproducibility & scope

- Runs are deterministic: the same parameters always produce the same dataset. In
  the current engine the random seeds are fixed internally, so `random_state` has
  little effect on the generated data.
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
