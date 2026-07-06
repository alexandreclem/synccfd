# SynCCFD

**Synthetic Credit Card Fraud Detection — a transaction & fraud data simulator.**

[![CI](https://github.com/alexandreclem/synccfd/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandreclem/synccfd/actions/workflows/ci.yml)
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

## Reproducibility & scope

- Passing `random_state` makes a run fully reproducible.
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
