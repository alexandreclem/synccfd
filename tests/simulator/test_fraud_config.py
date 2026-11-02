"""Tests for FraudConfig and its use by FraudSimulator."""

import contextlib
import io

import numpy as np
import pytest

from synccfd import DatasetGenerator, FraudSimulator
from synccfd.simulator.config import FraudConfig


def _base_data(n_customers, n_terminals, nb_days):
    """Legitimate transactions (no fraud yet) plus what FraudSimulator needs."""
    gen = DatasetGenerator(
        n_customers=n_customers,
        n_terminals=n_terminals,
        nb_days=nb_days,
        start_date="2023-01-01",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        customers = gen.generator.generate_customer_profiles()
        terminals = gen.generator.generate_terminal_profiles()
        transactions = gen.generator.generate_transactions()
    return customers, terminals, transactions, gen.generator.loc_features


def _add_frauds(base, config):
    customers, terminals, transactions, loc_features = base
    with contextlib.redirect_stdout(io.StringIO()):
        return FraudSimulator(config=config).add_frauds(
            customers, terminals, transactions.copy(), loc_features
        )


# --- validation -----------------------------------------------------------------


def test_defaults_match_v0_1():
    config = FraudConfig()
    assert config.scenarios == (1, 2, 3, 4)
    assert config.rate_mode == "absolute"
    assert (config.daily_counts(2, 20_000, n_days=5) == 2).all()
    assert (config.daily_counts(3, 10_000, n_days=5) == 3).all()
    assert (config.daily_counts(4, 10_000, n_days=5) == 3).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenarios": (1, 5)},
        {"scenarios": (1, 1)},
        {"rate_mode": "relative"},
        {"s2_terminal_fraction_per_day": 1.5},
        {"s3_customer_fraction_per_day": -0.1},
        {"s3_tamper_fraction": 2.0},
        {"s4_cnp_prob": -1.0},
        {"s2_duration_days": 0},
        {"s3_duration_days": 1.5},
        {"s4_customers_per_day": -1},
        {"s1_amount_threshold": -1.0},
        {"amount_multiplier": 0.0},
    ],
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        FraudConfig(**kwargs)


def test_scenarios_are_stored_as_tuple():
    assert FraudConfig(scenarios=[3, 1]).scenarios == (3, 1)


def test_config_is_frozen():
    with pytest.raises(AttributeError):
        FraudConfig().rate_mode = "proportional"  # type: ignore[misc]


# --- per-day counts --------------------------------------------------------------


def test_absolute_counts_ignore_population():
    config = FraudConfig()
    assert (config.daily_counts(3, 100, n_days=10) == 3).all()
    assert (config.daily_counts(3, 1_000_000, n_days=10) == 3).all()


def test_disabled_scenarios_have_zero_count():
    rng = np.random.default_rng(0)
    for rate_mode in ("absolute", "proportional"):
        config = FraudConfig(scenarios=(1, 3), rate_mode=rate_mode)
        assert (config.daily_counts(2, 20_000, n_days=10, rng=rng) == 0).all()
        assert (config.daily_counts(4, 10_000, n_days=10, rng=rng) == 0).all()
        assert config.daily_counts(3, 10_000, n_days=10, rng=rng).sum() > 0


@pytest.mark.parametrize("population", [500, 5_000, 50_000])
def test_proportional_counts_have_poisson_mean(population):
    config = FraudConfig(rate_mode="proportional")
    n_days = 200_000
    counts = config.daily_counts(3, population, n_days, np.random.default_rng(1))
    expected = config.s3_customer_fraction_per_day * population
    # mean of n_days Poisson draws: standard error sqrt(expected / n_days)
    assert counts.mean() == pytest.approx(expected, abs=5 * np.sqrt(expected / n_days))
    assert counts.var() == pytest.approx(expected, rel=0.05)


def test_proportional_counts_can_be_zero():
    config = FraudConfig(rate_mode="proportional")
    counts = config.daily_counts(3, 500, 1_000, np.random.default_rng(2))
    # expected 0.15 per day: most days have no new victim
    assert (counts == 0).mean() == pytest.approx(np.exp(-0.15), abs=0.05)


def test_proportional_counts_require_rng():
    with pytest.raises(ValueError):
        FraudConfig(rate_mode="proportional").daily_counts(3, 500, 10)


def test_unknown_count_scenario_raises():
    with pytest.raises(ValueError):
        FraudConfig().daily_counts(1, 500, 10)


def test_simulator_counts_are_reproducible_and_independent():
    config = FraudConfig(rate_mode="proportional")
    kwargs = {"n_days": 365, "n_terminals": 20_000, "n_customers": 10_000}
    a = FraudSimulator(random_state=7, config=config).daily_counts(**kwargs)
    b = FraudSimulator(random_state=7, config=config).daily_counts(**kwargs)
    c = FraudSimulator(random_state=8, config=config).daily_counts(**kwargs)
    for scenario in (2, 3, 4):
        np.testing.assert_array_equal(a[scenario], b[scenario])
    assert any((a[k] != c[k]).any() for k in (2, 3, 4))

    # Each scenario has its own random stream: disabling one leaves the others as-is.
    only_4 = FraudConfig(rate_mode="proportional", scenarios=(1, 4))
    d = FraudSimulator(random_state=7, config=only_4).daily_counts(**kwargs)
    np.testing.assert_array_equal(d[4], a[4])
    assert (d[3] == 0).all()


# --- behaviour --------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_base():
    return _base_data(n_customers=100, n_terminals=200, nb_days=15)


@pytest.mark.parametrize("scenarios", [(), (1,), (2,), (3,), (4,), (2, 4)])
def test_only_selected_scenarios_are_labelled(small_base, scenarios):
    transactions = _add_frauds(small_base, FraudConfig(scenarios=scenarios))
    assert set(transactions.TX_FRAUD_SCENARIO.unique()) <= {0, *scenarios}
    assert (transactions.TX_FRAUD == (transactions.TX_FRAUD_SCENARIO > 0)).all()


def test_scenario_1_only_through_dataset_generator():
    gen = DatasetGenerator(
        n_customers=100,
        n_terminals=200,
        nb_days=15,
        start_date="2023-01-01",
        fraud=FraudConfig(scenarios=(1,)),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, transactions = gen.generate()
    assert set(transactions.TX_FRAUD_SCENARIO.unique()) <= {0, 1}
    fraud = transactions[transactions.TX_FRAUD == 1]
    assert (fraud.TX_AMOUNT > 220).all()


def test_amount_threshold_is_configurable(small_base):
    transactions = _add_frauds(
        small_base, FraudConfig(scenarios=(1,), s1_amount_threshold=100.0)
    )
    assert (transactions.TX_FRAUD == (transactions.TX_AMOUNT > 100.0)).all()


@pytest.mark.slow
def test_proportional_rate_is_stable_across_sizes():
    # With the default fractions, 500 customers get on average 0.15 new victims per
    # scenario per day, so a single run is noisy; average over seeds (which, in the
    # legacy engine, only change the Poisson daily counts) and over 60 days.
    config = FraudConfig(rate_mode="proportional")
    seeds = range(20)
    mean_rates = {}
    for n_customers in (500, 5000):
        customers, terminals, transactions, loc_features = _base_data(
            n_customers, n_terminals=2 * n_customers, nb_days=60
        )
        rates = []
        for seed in seeds:
            with contextlib.redirect_stdout(io.StringIO()):
                frauds = FraudSimulator(random_state=seed, config=config).add_frauds(
                    customers, terminals, transactions.copy(), loc_features
                )
            rates.append(frauds.TX_FRAUD.mean())
        mean_rates[n_customers] = float(np.mean(rates))

    assert mean_rates[500] == pytest.approx(mean_rates[5000], rel=0.30)
