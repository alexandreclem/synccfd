"""Fraud injection (v2 engine, ``FraudConfig(mechanism="inject")``)."""

import numpy as np
import pandas as pd
import pytest

from synccfd import DatasetGenerator, FraudSimulator
from synccfd.simulator.config import DriftConfig, FraudConfig, IntensitySchedule
from synccfd.simulator.engine.drift import CustomerDrift, regime_weight
from synccfd.simulator.location.sampling import haversine_km

PARAMS = {
    "n_customers": 1000,
    "n_terminals": 2000,
    "nb_days": 90,
    "start_date": "2023-01-01",
    "random_state": 42,
}
TX_COLUMNS = [
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
]
# Many victims per day, so each scenario has enough campaigns in a small dataset
DENSE = {
    "s1_customer_fraction_per_day": 0.01,
    "s2_terminal_fraction_per_day": 1e-3,
    "s3_customer_fraction_per_day": 2e-3,
    "s4_customer_fraction_per_day": 2e-3,
    "s5_customer_fraction_per_day": 2e-3,
    "intensity_schedule": {},
}


def _generate(**kwargs):
    return DatasetGenerator(**{**PARAMS, **kwargs}).generate()


@pytest.fixture(scope="module")
def clean():
    return _generate(engine="v2", fraud=FraudConfig(scenarios=()))


@pytest.fixture(scope="module")
def injected():
    return _generate(fraud=FraudConfig.realistic(**DENSE))


def _frauds(transactions, scenario):
    return transactions[transactions.TX_FRAUD_SCENARIO == scenario]


# --- configuration --------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mechanism": "replace"},
        {"scenarios": (1, 5)},  # scenario 5 needs injection
        {"intensity_schedule": {2: IntensitySchedule()}},  # needs injection
        {"mechanism": "inject", "intensity_schedule": {6: IntensitySchedule()}},
        {"mechanism": "inject", "intensity_schedule": {2: 0.5}},
        {"mechanism": "inject", "s1_quantile": 1.5},
        {"mechanism": "inject", "s4_max_txs": 0},
        {"mechanism": "inject", "s3_burst_size_range": (5, 2)},
        {"mechanism": "inject", "s3_mean_bursts": 0.5},
        {"mechanism": "inject", "s4_spending_limit": 0.0},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        FraudConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "linear"},
        {"start": 0.8, "end": 0.2},
        {"level": -1.0},
    ],
)
def test_invalid_schedule_raises(kwargs):
    with pytest.raises(ValueError):
        IntensitySchedule(**kwargs)


def test_defaults_stay_tamper():
    config = FraudConfig()
    assert config.mechanism == "tamper"
    assert config.scenarios == (1, 2, 3, 4)
    assert dict(config.intensity_schedule) == {}


def test_realistic_config():
    config = FraudConfig.realistic(s2_leak_prob=0.1)
    assert config.mechanism == "inject"
    assert config.rate_mode == "proportional"
    assert config.scenarios == (1, 2, 3, 4, 5)
    assert config.s2_leak_prob == 0.1
    assert set(config.intensity_schedule) == {2, 5}
    with pytest.raises(TypeError):
        config.intensity_schedule[3] = IntensitySchedule()  # type: ignore[index]


def test_injection_requires_v2():
    config = FraudConfig(mechanism="inject")
    assert DatasetGenerator(n_customers=5, n_terminals=5, fraud=config).engine == "v2"
    with pytest.raises(ValueError):
        DatasetGenerator(n_customers=5, n_terminals=5, fraud=config, engine="legacy")
    with pytest.raises(ValueError):
        FraudSimulator(config=config)


# --- intensity schedule ------------------------------------------------------------


def test_schedule_shapes():
    n = 10  # day midpoints 0.05, 0.15, ..., 0.95
    np.testing.assert_allclose(IntensitySchedule(level=2.0).multipliers(n), 2.0)
    step = IntensitySchedule("step", start=0.5, base=1.0, level=0.5).multipliers(n)
    np.testing.assert_allclose(step, [1.0] * 5 + [0.5] * 5)
    pulse = IntensitySchedule("pulse", start=0.2, end=0.4, level=3.0).multipliers(n)
    np.testing.assert_allclose(pulse, [1, 1, 3, 3, 1, 1, 1, 1, 1, 1])
    ramp = IntensitySchedule("ramp", start=0.0, end=1.0, base=0.0, level=1.0)
    np.testing.assert_allclose(ramp.multipliers(n), np.arange(n) / n + 0.05)


def test_step_schedule_changes_daily_rate_on_the_planned_day():
    # Step at 40% of 100 days: days 0-39 before, days 40-99 after
    step = IntensitySchedule("step", start=0.4, base=0.0, level=1.0)
    absolute = FraudConfig(
        mechanism="inject", s3_customers_per_day=4, intensity_schedule={3: step}
    )
    counts = absolute.daily_counts(3, 10_000, 100, np.random.default_rng(0))
    assert (counts[:40] == 0).all() and (counts[40:] == 4).all()

    halving = IntensitySchedule("step", start=0.4, base=1.0, level=0.5)
    proportional = FraudConfig(
        mechanism="inject",
        rate_mode="proportional",
        s2_terminal_fraction_per_day=0.01,
        intensity_schedule={2: halving},
    )
    counts = proportional.daily_counts(2, 10_000, 1000, np.random.default_rng(0))
    ratio = counts[400:].mean() / counts[:400].mean()
    assert ratio == pytest.approx(0.5, abs=0.02)


def test_schedule_in_generated_data():
    # The realistic config starts card testing at 30% of the period
    _, _, transactions = _generate(
        fraud=FraudConfig.realistic(s5_customer_fraction_per_day=0.01)
    )
    first_day = _frauds(transactions, 5).TX_TIME_DAYS.min()
    assert first_day == 27  # first day with (d + 0.5) / 90 >= 0.3


# --- injection keeps the legitimate transactions ------------------------------------


def test_legitimate_transactions_are_unchanged(clean, injected):
    _, _, before = clean
    _, _, after = injected
    legit = after[after.TX_FRAUD_SCENARIO.isin([0, 2])]
    pd.testing.assert_frame_equal(
        legit[TX_COLUMNS].reset_index(drop=True), before[TX_COLUMNS]
    )
    # Every other fraud is a new transaction
    assert len(after) == len(before) + (~after.TX_FRAUD_SCENARIO.isin([0, 2])).sum()
    assert (
        after.TX_FRAUD_SCENARIO.value_counts().index.sort_values() == range(6)
    ).all()


def test_ground_truth_columns(injected):
    _, _, transactions = injected
    assert (transactions.GT_FRAUD_SCENARIO == transactions.TX_FRAUD_SCENARIO).all()
    fraud = transactions.TX_FRAUD == 1
    for column in ("GT_FRAUDSTER_ID", "GT_CAMPAIGN_ID"):
        assert (transactions[column].notna() == fraud).all()
    assert (transactions.TX_FRAUD == (transactions.TX_FRAUD_SCENARIO > 0)).all()
    assert transactions.TRANSACTION_ID.is_monotonic_increasing
    assert transactions.TX_TIME_SECONDS.is_monotonic_increasing
    assert transactions.TX_TIME_DAYS.max() < PARAMS["nb_days"]


def test_reproducible(injected):
    _, _, again = _generate(fraud=FraudConfig.realistic(**DENSE))
    pd.testing.assert_frame_equal(injected[2], again)


# --- scenario 1 ----------------------------------------------------------------------


def _scenario_1_thresholds(transactions, config):
    """``factor x quantile`` of each scenario-1 victim's purchases in the
    ``history_days`` before the day of the fraud."""
    legit = transactions[transactions.TX_FRAUD_SCENARIO.isin([0, 2])]
    by_customer = dict(tuple(legit.groupby("CUSTOMER_ID")))
    thresholds = []
    for _, fraud in _frauds(transactions, 1).iterrows():
        own = by_customer[fraud.CUSTOMER_ID]
        day = fraud.TX_TIME_DAYS
        window = own[
            (own.TX_TIME_DAYS >= day - config.history_days) & (own.TX_TIME_DAYS < day)
        ]
        assert len(window) >= config.min_history_tx
        thresholds.append(
            config.s1_amount_factor * window.TX_AMOUNT.quantile(config.s1_quantile)
        )
    return np.array(thresholds)


def test_scenario_1_is_far_above_the_customer_history(injected):
    _, _, transactions = injected
    config = FraudConfig.realistic(**DENSE)
    frauds = _frauds(transactions, 1)
    assert len(frauds) > 300
    thresholds = _scenario_1_thresholds(transactions, config)
    assert (frauds.TX_AMOUNT.to_numpy() > thresholds).all()
    upper = thresholds * (1 + config.s1_amount_spread)
    assert (frauds.TX_AMOUNT.to_numpy() <= upper + 0.01).all()
    # One purchase per campaign
    assert frauds.GT_CAMPAIGN_ID.is_unique


def _victim_weights(customers, frauds):
    """Regime-B weight of each victim on the day of its fraud."""
    weights = []
    for _, fraud in frauds.iterrows():
        row = customers.loc[fraud.CUSTOMER_ID]

        def value(name, row=row):
            return 0 if pd.isna(row[name]) else int(row[name])

        drift = CustomerDrift(
            kind=row.GT_DRIFT_TYPE,
            t0=value("GT_DRIFT_T0"),
            transition_days=value("GT_DRIFT_L"),
            period_days=value("GT_DRIFT_PERIOD"),
            active_days=value("GT_DRIFT_ACTIVE_DAYS"),
        )
        weights.append(regime_weight(drift, np.array([fraud.TX_TIME_DAYS + 0.5]))[0])
    return np.array(weights)


def test_scenario_1_with_drift_does_not_label_regime_b():
    customers, _, transactions = _generate(
        drift=DriftConfig(customer_fraction=0.5),
        fraud=FraudConfig.realistic(scenarios=(1,), s1_customer_fraction_per_day=0.02),
    )
    frauds = _frauds(transactions, 1)
    legit = transactions[transactions.TX_FRAUD == 0]
    # Legitimate transactions are never labelled: every fraud is injected
    assert frauds.GT_REGIME.isna().all() and legit.GT_REGIME.notna().all()
    assert len(transactions) == len(legit) + len(frauds)
    # Scenario 1 hits regime B about as often as regime B occurs
    b_share_of_legit = legit.GT_REGIME_WEIGHT.mean()
    b_share_of_s1 = _victim_weights(customers, frauds).mean()
    assert b_share_of_s1 == pytest.approx(b_share_of_legit, abs=0.05)


# --- scenarios 2-5 ------------------------------------------------------------------


def test_scenario_2_labels_cp_purchases_and_leaks_cards(injected):
    _, _, transactions = injected
    s2 = _frauds(transactions, 2)
    assert len(s2) > 0 and (s2.TX_TYPE == "CP").all()
    # Each compromised terminal is one campaign
    assert (s2.groupby("GT_CAMPAIGN_ID").TERMINAL_ID.nunique() == 1).all()
    # Some leaked cards are used in scenario 3 by the same fraudster, later
    s3 = _frauds(transactions, 3)
    leaked = s3[s3.GT_FRAUDSTER_ID.isin(s2.GT_FRAUDSTER_ID)]
    assert len(leaked) > 0
    first_skim = s2.groupby(["GT_FRAUDSTER_ID", "CUSTOMER_ID"]).TX_TIME_DAYS.min()
    for (fraudster, customer), uses in leaked.groupby(
        ["GT_FRAUDSTER_ID", "CUSTOMER_ID"]
    ):
        assert uses.TX_TIME_DAYS.min() > first_skim.loc[(fraudster, customer)]


def test_scenario_3_bursts(injected):
    _, _, transactions = injected
    s3 = _frauds(transactions, 3)
    assert (s3.TX_TYPE == "CNP").all()
    # Shipped to one fraudster address per campaign, far from the billing address
    per_campaign = s3.groupby("GT_CAMPAIGN_ID")
    assert (per_campaign.TX_SHIPP_LAT.nunique() == 1).all()
    assert (s3.TX_SHIPP_LAT != s3.TX_BILL_LAT).all()
    assert per_campaign.size().min() >= 2


def test_scenario_4_within_24h_and_spending_limit(injected):
    _, _, transactions = injected
    config = FraudConfig.realistic(**DENSE)
    s4 = _frauds(transactions, 4)
    assert len(s4) > 0
    per_campaign = s4.groupby("GT_CAMPAIGN_ID")
    span = per_campaign.TX_TIME_SECONDS.max() - per_campaign.TX_TIME_SECONDS.min()
    assert (span < 24 * 3600).all()
    assert (per_campaign.TX_AMOUNT.sum() <= config.s4_spending_limit).all()
    assert (per_campaign.size() <= config.s4_max_txs).all()
    assert per_campaign.CUSTOMER_ID.nunique().max() == 1
    # CP purchases happen near the victim: the fraudster is within the city radius
    # (12 km) of the billing address and buys within 10 km of where it is
    cp = s4[s4.TX_TYPE == "CP"]
    assert len(cp) > 0
    distance = haversine_km(
        cp.TX_BILL_LAT, cp.TX_BILL_LONG, cp.TX_TERM_LAT, cp.TX_TERM_LONG
    )
    assert (distance <= 22 + 1e-6).all()


def test_scenario_5_card_testing_and_escalation(injected):
    _, _, transactions = injected
    config = FraudConfig.realistic(**DENSE)
    s5 = _frauds(transactions, 5)
    assert len(s5) > 0
    assert (s5.TX_AMOUNT <= config.s5_amount_range[1]).all()
    assert (s5.TX_AMOUNT >= config.s5_amount_range[0]).all()
    assert (s5.TX_TYPE == "CNP").all()
    sizes = s5.groupby("GT_CAMPAIGN_ID").size()
    assert sizes.between(*config.s5_n_tests_range).all()

    # Escalated campaigns: scenario 3 follows, same campaign, fraudster and card
    s3 = _frauds(transactions, 3)
    escalated = s3[s3.GT_CAMPAIGN_ID.isin(s5.GT_CAMPAIGN_ID)]
    share = escalated.GT_CAMPAIGN_ID.nunique() / s5.GT_CAMPAIGN_ID.nunique()
    assert share == pytest.approx(config.s5_escalate_prob, abs=0.15)
    tests_by_campaign = s5.groupby("GT_CAMPAIGN_ID")
    for campaign, follow_up in escalated.groupby("GT_CAMPAIGN_ID"):
        tests = tests_by_campaign.get_group(campaign)
        assert follow_up.TX_TIME_SECONDS.min() > tests.TX_TIME_SECONDS.max()
        assert set(follow_up.CUSTOMER_ID) == set(tests.CUSTOMER_ID)
        assert set(follow_up.GT_FRAUDSTER_ID) == set(tests.GT_FRAUDSTER_ID)


def test_all_scenarios_work_without_merchants(injected):
    # There are no merchants yet (Phase 4): every scenario still produces frauds
    _, _, transactions = injected
    counts = transactions.TX_FRAUD_SCENARIO.value_counts()
    assert all(counts.get(s, 0) > 0 for s in (1, 2, 3, 4, 5))


def test_realistic_fraud_rate():
    _, _, transactions = _generate(fraud=FraudConfig.realistic())
    assert 0.002 <= transactions.TX_FRAUD.mean() <= 0.01
