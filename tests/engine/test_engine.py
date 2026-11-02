"""Behaviour of the simulation engine (fast tests, small datasets)."""

import datetime
import random

import numpy as np
import pandas as pd
import pytest

from synccfd import DatasetGenerator
from synccfd.simulator import presets
from synccfd.simulator.config import FraudConfig
from synccfd.simulator.engine.rng import RandomStreams
from synccfd.simulator.engine.transactions import generate_transactions

SMALL = {
    "n_customers": 100,
    "n_terminals": 200,
    "nb_days": 15,
    "start_date": "2023-01-01",
}
# The base model: no drift, no seasonality, no merchants
BASE = {"drift": None, "seasonality": None, "merchants": None}
# Columns of the base model with injected frauds
BASE_TRANSACTION_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TYPE",
    "TX_SHIPP_LAT",
    "TX_SHIPP_LONG",
    "TX_BILL_LAT",
    "TX_BILL_LONG",
    "TX_TERM_LAT",
    "TX_TERM_LONG",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
    "GT_FRAUD_SCENARIO",
    "GT_FRAUDSTER_ID",
    "GT_CAMPAIGN_ID",
    "GT_COMPROMISE_ID",
]
# Many victims, so each scenario has frauds in a small dataset
DENSE_FRAUD = {
    "s1_customer_fraction_per_day": 0.01,
    "s2_terminal_fraction_per_day": 0.01,
    "s3_customer_fraction_per_day": 0.01,
    "s4_customer_fraction_per_day": 0.01,
    "s5_customer_fraction_per_day": 0.01,
}


def _generator(**kwargs):
    params = {**SMALL, **BASE, "random_state": 42, **kwargs}
    return DatasetGenerator(**params)


@pytest.fixture(scope="module")
def base_data():
    generator = _generator(fraud=FraudConfig(**DENSE_FRAUD))
    customers, terminals, transactions = generator.generate()
    return generator, customers, terminals, transactions


# --- defaults -------------------------------------------------------------------


def test_defaults_are_the_realistic_preset():
    generator = DatasetGenerator()
    assert generator.config == DatasetGenerator(**presets.realistic()).config
    assert generator.fraud_config == FraudConfig.realistic()
    assert generator.n_customers == 10_000 and generator.nb_days == 180


def test_none_switches_a_component_off():
    transactions = _generator(fraud=None).generate()[2]
    assert transactions.TX_FRAUD.sum() == 0
    assert "GT_REGIME" not in transactions and "MERCHANT_ID" not in transactions


def test_location_features_require_generate():
    with pytest.raises(RuntimeError):
        _generator().get_location_features()


# --- schema and invariants ------------------------------------------------------


def test_base_schema(base_data):
    _, customers, terminals, transactions = base_data
    assert list(transactions.columns) == BASE_TRANSACTION_COLUMNS
    assert list(terminals.columns) == ["TERMINAL_ID", "terminal_lat", "terminal_long"]
    assert list(customers.columns[:8]) == [
        "CUSTOMER_ID",
        "billing_lat",
        "billing_long",
        "mean_amount",
        "std_amount",
        "mean_nb_tx_per_day",
        "cnp_prob",
        "available_terminals",
    ]


def test_transactions_are_chronological(base_data):
    *_, transactions = base_data
    assert transactions.TX_TIME_SECONDS.is_monotonic_increasing
    assert (transactions.TRANSACTION_ID == np.arange(len(transactions))).all()
    expected = pd.to_datetime(
        transactions.TX_TIME_SECONDS, unit="s", origin=datetime.datetime(2023, 1, 1)
    )
    assert (transactions.TX_DATETIME == expected).all()


def test_times_of_day_are_valid(base_data):
    *_, transactions = base_data
    seconds_in_day = transactions.TX_TIME_SECONDS - 86400 * transactions.TX_TIME_DAYS
    assert seconds_in_day.between(0, 86400).all()
    assert transactions.TX_TIME_DAYS.between(0, SMALL["nb_days"] - 1).all()


def test_cp_transactions_use_reachable_terminals(base_data):
    _, customers, _, transactions = base_data
    cp = transactions[(transactions.TX_TYPE == "CP") & (transactions.TX_FRAUD == 0)]
    reachable = customers.set_index("CUSTOMER_ID").available_terminals
    assert all(t in reachable[c] for c, t in zip(cp.CUSTOMER_ID, cp.TERMINAL_ID))


def test_location_features_are_filled(base_data):
    generator, customers, terminals, _ = base_data
    loc = generator.get_location_features()
    assert loc.n_customers == len(customers)
    assert loc.n_terminals == len(terminals)
    assert len(loc.billing_cities_locations) == len(customers)
    assert len(loc.customers_available_terminals) == len(customers)


# --- randomness -----------------------------------------------------------------


def test_reproducible_with_same_seed(base_data):
    _, customers, terminals, transactions = base_data
    again = _generator(fraud=FraudConfig(**DENSE_FRAUD)).generate()
    pd.testing.assert_frame_equal(customers, again[0])
    pd.testing.assert_frame_equal(terminals, again[1])
    pd.testing.assert_frame_equal(transactions, again[2])


def test_different_seeds_give_different_data(base_data):
    *_, transactions = base_data
    other = _generator(fraud=FraudConfig(**DENSE_FRAUD), random_state=7).generate()[2]
    assert len(other) != len(transactions) or not other.equals(transactions)


def test_global_random_state_is_untouched():
    random.seed(123)
    np.random.seed(123)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    DatasetGenerator(**SMALL).generate()
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


def test_customer_transactions_do_not_depend_on_other_customers(base_data):
    generator, customers, terminals, _ = base_data
    loc = generator.get_location_features()
    start = datetime.datetime(2023, 1, 1)
    streams = RandomStreams(5)
    rngs = streams.child_generators("transactions", 10)
    few = generate_transactions(customers.iloc[:3], terminals, loc, 10, start, rngs[:3])
    rngs = streams.child_generators("transactions", 10)
    many = generate_transactions(customers.iloc[:10], terminals, loc, 10, start, rngs)
    subset = many[many.CUSTOMER_ID < 3].drop(columns="TRANSACTION_ID")
    pd.testing.assert_frame_equal(
        few.drop(columns="TRANSACTION_ID").reset_index(drop=True),
        subset.reset_index(drop=True),
    )


# --- fraud ----------------------------------------------------------------------


@pytest.mark.parametrize("scenarios", [(), (1,), (2,), (3,), (4,), (5,), (2, 4)])
def test_only_selected_scenarios_are_labelled(scenarios):
    config = FraudConfig(scenarios=scenarios, **DENSE_FRAUD)
    transactions = _generator(fraud=config).generate()[2]
    assert set(transactions.TX_FRAUD_SCENARIO.unique()) <= {0, *scenarios}
    assert (transactions.TX_FRAUD == (transactions.TX_FRAUD_SCENARIO > 0)).all()


def test_frauds_are_added_and_legitimate_transactions_unchanged(base_data):
    *_, transactions = base_data
    no_fraud = _generator(fraud=None).generate()[2]
    legit = transactions[transactions.TX_FRAUD == 0]
    assert len(transactions) > len(no_fraud) == len(legit)
    columns = ["CUSTOMER_ID", "TERMINAL_ID", "TX_TIME_SECONDS", "TX_AMOUNT", "TX_TYPE"]
    pd.testing.assert_frame_equal(
        legit[columns].reset_index(drop=True), no_fraud[columns].reset_index(drop=True)
    )


def test_scenario_3_transactions_are_cnp(base_data):
    *_, transactions = base_data
    assert (transactions[transactions.TX_FRAUD_SCENARIO == 3].TX_TYPE == "CNP").all()
