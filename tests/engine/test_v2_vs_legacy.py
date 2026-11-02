"""Statistical comparison of the v2 engine with the legacy (v0.1) engine.

Three layers, to separate the sources of variation:

- A: transaction model — v2 transactions generated from the *legacy* profiles;
- B: fraud — v2 and legacy fraud applied to the *same* legitimate transactions
  (v2 averaged over seeds; the legacy draws do not depend on the seed);
- C: end to end — full v2 dataset against the full legacy dataset.

The legacy engine discards transactions whose time of day falls outside the day,
while v2 redraws the time (see ``engine/transactions.py``). Legacy therefore has
about 1.5% fewer transactions, mostly CP ones, which also raises its CNP share
slightly; the tolerances below allow for this.
"""

import contextlib
import datetime
import io

import numpy as np
import pytest
from scipy import stats

from synccfd import DatasetGenerator, FraudSimulator
from synccfd.simulator.config import FraudConfig
from synccfd.simulator.engine.fraud import apply_frauds
from synccfd.simulator.engine.rng import RandomStreams
from synccfd.simulator.engine.transactions import generate_transactions

pytestmark = pytest.mark.slow

PARAMS = {
    "n_customers": 1000,
    "n_terminals": 2000,
    "nb_days": 30,
    "start_date": "2023-01-01",
    "random_state": 42,
}
START = datetime.datetime(2023, 1, 1)
CUSTOMER_DAYS = PARAMS["n_customers"] * PARAMS["nb_days"]


@pytest.fixture(scope="module")
def legacy():
    with contextlib.redirect_stdout(io.StringIO()):
        generator = DatasetGenerator(**PARAMS)
        customers = generator.generator.generate_customer_profiles()
        terminals = generator.generator.generate_terminal_profiles()
        legit = generator.generator.generate_transactions()
        labelled = FraudSimulator(config=FraudConfig()).add_frauds(
            customers, terminals, legit.copy(), generator.generator.loc_features
        )
    return {
        "customers": customers,
        "terminals": terminals,
        "legit": legit,
        "labelled": labelled,
        "loc_features": generator.generator.loc_features,
    }


def _time_of_day(transactions):
    return transactions.TX_TIME_SECONDS - 86400 * transactions.TX_TIME_DAYS


# --- A: transaction model, same profiles ------------------------------------------


@pytest.fixture(scope="module")
def v2_legit(legacy):
    rngs = RandomStreams(1).child_generators("transactions", PARAMS["n_customers"])
    return generate_transactions(
        legacy["customers"],
        legacy["terminals"],
        legacy["loc_features"],
        PARAMS["nb_days"],
        START,
        rngs,
    )


def test_transactions_per_customer_day(legacy, v2_legit):
    ratio = len(v2_legit) / len(legacy["legit"])
    # v2 keeps the transactions that legacy discards (~1.5%)
    assert 1.0 <= ratio <= 1.04


def test_mean_amount(legacy, v2_legit):
    assert v2_legit.TX_AMOUNT.mean() == pytest.approx(
        legacy["legit"].TX_AMOUNT.mean(), rel=0.02
    )
    assert stats.ks_2samp(v2_legit.TX_AMOUNT, legacy["legit"].TX_AMOUNT).pvalue > 0.01


def test_cnp_share(legacy, v2_legit):
    v2_share = (v2_legit.TX_TYPE == "CNP").mean()
    legacy_share = (legacy["legit"].TX_TYPE == "CNP").mean()
    assert v2_share == pytest.approx(legacy_share, abs=0.02)


@pytest.mark.parametrize("tx_type", ["CP", "CNP"])
def test_time_of_day_distribution(legacy, v2_legit, tx_type):
    legacy_times = _time_of_day(legacy["legit"][legacy["legit"].TX_TYPE == tx_type])
    v2_times = _time_of_day(v2_legit[v2_legit.TX_TYPE == tx_type])
    # Size of the difference, not a p-value: with ~20 000 transactions per type a
    # p-value flags differences too small to matter. D is the largest gap between
    # the two empirical CDFs.
    assert stats.ks_2samp(legacy_times, v2_times).statistic < 0.02


# --- B: fraud, same legitimate transactions ---------------------------------------


def _scenario_rates(transactions):
    counts = transactions.TX_FRAUD_SCENARIO.value_counts(normalize=True)
    return counts.reindex(range(5), fill_value=0.0).to_numpy()


def test_fraud_rate_per_scenario(legacy):
    v2_rates = np.mean(
        [
            _scenario_rates(
                apply_frauds(
                    legacy["legit"],
                    legacy["customers"],
                    legacy["terminals"],
                    legacy["loc_features"],
                    FraudConfig(),
                    RandomStreams(seed),
                    START,
                )
            )
            for seed in range(10)
        ],
        axis=0,
    )
    legacy_rates = _scenario_rates(legacy["labelled"])
    for scenario in range(1, 5):
        assert v2_rates[scenario] == pytest.approx(legacy_rates[scenario], rel=0.20), (
            f"scenario {scenario}"
        )


# --- C: end to end ------------------------------------------------------------------


@pytest.fixture(scope="module")
def v2_full():
    return DatasetGenerator(**PARAMS, engine="v2").generate()


@pytest.mark.parametrize(
    "column", ["mean_amount", "mean_nb_tx_per_day", "cnp_prob", "billing_lat"]
)
def test_customer_profiles_distribution(legacy, v2_full, column):
    v2_customers = v2_full[0]
    assert (
        stats.ks_2samp(legacy["customers"][column], v2_customers[column]).pvalue > 0.01
    )


def test_end_to_end_totals(legacy, v2_full):
    v2_transactions = v2_full[2]
    assert len(v2_transactions) / CUSTOMER_DAYS == pytest.approx(
        len(legacy["labelled"]) / CUSTOMER_DAYS, rel=0.05
    )
    assert v2_transactions.TX_FRAUD.mean() == pytest.approx(
        legacy["labelled"].TX_FRAUD.mean(), rel=0.20
    )
