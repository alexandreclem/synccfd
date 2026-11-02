"""Tests for FraudConfig: validation, daily counts and rates."""

import numpy as np
import pytest

from synccfd import DatasetGenerator
from synccfd.simulator.config import KNOWN_FRAUD_SCENARIOS, FraudConfig

# The base model: no drift, no seasonality, no merchants
BASE = {"drift": None, "seasonality": None, "merchants": None}

# --- validation -----------------------------------------------------------------


def test_defaults():
    config = FraudConfig()
    assert config.scenarios == KNOWN_FRAUD_SCENARIOS
    assert config.rate_mode == "proportional"
    assert not config.intensity_schedule
    absolute = FraudConfig(rate_mode="absolute")
    assert (absolute.daily_counts(2, 20_000, n_days=5) == 2).all()
    assert (absolute.daily_counts(3, 10_000, n_days=5) == 3).all()
    assert (absolute.daily_counts(4, 10_000, n_days=5) == 3).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenarios": (1, 6)},
        {"scenarios": (1, 1)},
        {"rate_mode": "relative"},
        {"s2_terminal_fraction_per_day": 1.5},
        {"s3_customer_fraction_per_day": -0.1},
        {"s2_duration_days": 0},
        {"s3_duration_days": 1.5},
        {"s4_customers_per_day": -1},
    ],
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        FraudConfig(**kwargs)


def test_removed_tamper_parameters_are_rejected():
    for name in ("mechanism", "s1_amount_threshold", "amount_multiplier"):
        with pytest.raises(TypeError):
            FraudConfig(**{name: 1})


def test_scenarios_are_stored_as_tuple():
    assert FraudConfig(scenarios=[3, 1]).scenarios == (3, 1)


def test_config_is_frozen():
    with pytest.raises(AttributeError):
        FraudConfig().rate_mode = "absolute"  # type: ignore[misc]


# --- per-day counts --------------------------------------------------------------


def test_absolute_counts_ignore_population():
    config = FraudConfig(rate_mode="absolute")
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
    config = FraudConfig()
    n_days = 200_000
    counts = config.daily_counts(3, population, n_days, np.random.default_rng(1))
    expected = config.s3_customer_fraction_per_day * population
    # mean of n_days Poisson draws: standard error sqrt(expected / n_days)
    assert counts.mean() == pytest.approx(expected, abs=5 * np.sqrt(expected / n_days))
    assert counts.var() == pytest.approx(expected, rel=0.05)


def test_proportional_counts_can_be_zero():
    counts = FraudConfig().daily_counts(3, 500, 1_000, np.random.default_rng(2))
    # expected 0.15 per day: most days have no new victim
    assert (counts == 0).mean() == pytest.approx(np.exp(-0.15), abs=0.05)


def test_proportional_counts_require_rng():
    with pytest.raises(ValueError):
        FraudConfig().daily_counts(3, 500, 10)


def test_unknown_count_scenario_raises():
    with pytest.raises(ValueError):
        FraudConfig().daily_counts(6, 500, 10, np.random.default_rng(0))


# --- rates ------------------------------------------------------------------------


def test_scenario_1_through_dataset_generator():
    fraud = FraudConfig(scenarios=(1,), s1_customer_fraction_per_day=0.01)
    _, _, transactions = DatasetGenerator(
        n_customers=100, n_terminals=200, nb_days=15, fraud=fraud, **BASE
    ).generate()
    assert set(transactions.TX_FRAUD_SCENARIO.unique()) == {0, 1}
    # One anomalous purchase per victim
    frauds = transactions[transactions.TX_FRAUD == 1]
    assert frauds.GT_CAMPAIGN_ID.is_unique


@pytest.mark.slow
def test_proportional_rate_is_stable_across_sizes():
    # With the default fractions, 500 customers get on average 0.15 new victims per
    # scenario per day, so a single run is noisy; average over seeds and 60 days.
    mean_rates = {}
    for n_customers in (500, 5000):
        rates = [
            DatasetGenerator(
                n_customers=n_customers,
                n_terminals=2 * n_customers,
                nb_days=60,
                random_state=seed,
                fraud=FraudConfig(scenarios=(1, 2, 3, 4)),
                **BASE,
            )
            .generate()[2]
            .TX_FRAUD.mean()
            for seed in range(10)
        ]
        mean_rates[n_customers] = float(np.mean(rates))

    assert mean_rates[500] == pytest.approx(mean_rates[5000], rel=0.30)
