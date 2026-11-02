"""Behaviour of the v2 engine (fast tests, small datasets)."""

import contextlib
import datetime
import io
import random

import numpy as np
import pandas as pd
import pytest

from synccfd import DatasetGenerator
from synccfd.simulator.config import FraudConfig
from synccfd.simulator.engine.rng import RandomStreams
from synccfd.simulator.engine.transactions import generate_transactions

SMALL = {
    "n_customers": 100,
    "n_terminals": 200,
    "nb_days": 15,
    "start_date": "2023-01-01",
}


def _v2(**kwargs):
    params = {**SMALL, "random_state": 42, **kwargs}
    return DatasetGenerator(**params, engine="v2")


@pytest.fixture(scope="module")
def v2_data():
    generator = _v2()
    customers, terminals, transactions = generator.generate()
    return generator, customers, terminals, transactions


@pytest.fixture(scope="module")
def legacy_data():
    with contextlib.redirect_stdout(io.StringIO()):
        return DatasetGenerator(**SMALL, random_state=42).generate()


# --- engine selection -----------------------------------------------------------


def test_auto_uses_legacy():
    assert DatasetGenerator(**SMALL).engine == "legacy"
    assert DatasetGenerator(**SMALL, engine="auto").engine == "legacy"


def test_invalid_engine_raises():
    with pytest.raises(ValueError):
        DatasetGenerator(**SMALL, engine="v3")


def test_location_features_require_generate_in_v2():
    generator = _v2()
    with pytest.raises(RuntimeError):
        generator.get_location_features()


# --- schema and invariants ------------------------------------------------------


def test_same_schema_as_legacy(v2_data, legacy_data):
    _, *v2_frames = v2_data
    for v2_frame, legacy_frame in zip(v2_frames, legacy_data):
        assert list(v2_frame.columns) == list(legacy_frame.columns)
        assert dict(v2_frame.dtypes) == dict(legacy_frame.dtypes)


def test_transactions_are_chronological(v2_data):
    *_, transactions = v2_data
    assert transactions.TX_TIME_SECONDS.is_monotonic_increasing
    assert (transactions.TRANSACTION_ID == np.arange(len(transactions))).all()
    expected = pd.to_datetime(
        transactions.TX_TIME_SECONDS, unit="s", origin=datetime.datetime(2023, 1, 1)
    )
    assert (transactions.TX_DATETIME == expected).all()


def test_times_of_day_are_valid(v2_data):
    *_, transactions = v2_data
    seconds_in_day = transactions.TX_TIME_SECONDS - 86400 * transactions.TX_TIME_DAYS
    assert seconds_in_day.between(0, 86400).all()
    assert transactions.TX_TIME_DAYS.between(0, SMALL["nb_days"] - 1).all()


def test_cp_transactions_use_reachable_terminals(v2_data):
    _, customers, _, transactions = v2_data
    cp = transactions[(transactions.TX_TYPE == "CP") & (transactions.TX_FRAUD == 0)]
    reachable = customers.set_index("CUSTOMER_ID").available_terminals
    assert all(t in reachable[c] for c, t in zip(cp.CUSTOMER_ID, cp.TERMINAL_ID))


def test_location_features_are_filled(v2_data):
    generator, customers, terminals, _ = v2_data
    loc = generator.get_location_features()
    assert loc.n_customers == len(customers)
    assert loc.n_terminals == len(terminals)
    assert len(loc.billing_cities_locations) == len(customers)
    assert len(loc.customers_available_terminals) == len(customers)


# --- randomness -----------------------------------------------------------------


def test_reproducible_with_same_seed(v2_data):
    _, customers, terminals, transactions = v2_data
    again = _v2().generate()
    pd.testing.assert_frame_equal(customers, again[0])
    pd.testing.assert_frame_equal(terminals, again[1])
    pd.testing.assert_frame_equal(transactions, again[2])


def test_different_seeds_give_different_data(v2_data):
    *_, transactions = v2_data
    other = _v2(random_state=7).generate()[2]
    assert len(other) != len(transactions) or not other.equals(transactions)


def test_global_random_state_is_untouched():
    random.seed(123)
    np.random.seed(123)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    _v2().generate()
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


def test_customer_transactions_do_not_depend_on_other_customers(v2_data):
    generator, customers, terminals, _ = v2_data
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


@pytest.mark.parametrize("scenarios", [(), (1,), (2,), (3,), (4,), (2, 4)])
def test_only_selected_scenarios_are_labelled(scenarios):
    transactions = _v2(fraud=FraudConfig(scenarios=scenarios)).generate()[2]
    assert set(transactions.TX_FRAUD_SCENARIO.unique()) <= {0, *scenarios}
    assert (transactions.TX_FRAUD == (transactions.TX_FRAUD_SCENARIO > 0)).all()


def test_scenario_1_threshold():
    config = FraudConfig(scenarios=(1,), s1_amount_threshold=80.0)
    transactions = _v2(fraud=config).generate()[2]
    assert (transactions.TX_FRAUD == (transactions.TX_AMOUNT > 80.0)).all()


def test_scenario_3_transactions_are_cnp(v2_data):
    *_, transactions = v2_data
    assert (transactions[transactions.TX_FRAUD_SCENARIO == 3].TX_TYPE == "CNP").all()


def test_fraud_does_not_change_the_number_of_transactions(v2_data):
    *_, transactions = v2_data
    no_fraud = _v2(fraud=FraudConfig(scenarios=())).generate()[2]
    assert len(no_fraud) == len(transactions)
    assert no_fraud.TX_FRAUD.sum() == 0


def test_scenario_2_does_not_depend_on_scenarios_3_and_4():
    with_all = _v2().generate()[2]
    only_2 = _v2(fraud=FraudConfig(scenarios=(1, 2))).generate()[2]
    # Scenario 2 runs after 1 and before 3 and 4, from its own stream: same
    # compromised terminals, hence the same scenario-2 transactions (identified by
    # customer, terminal and original time; scenario 2 does not tamper).
    key = ["CUSTOMER_ID", "TERMINAL_ID", "TX_TIME_SECONDS"]
    a = with_all[with_all.TX_FRAUD_SCENARIO == 2][key].sort_values(key)
    b = only_2[only_2.TX_FRAUD_SCENARIO == 2][key].sort_values(key)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
