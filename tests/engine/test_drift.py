"""Concept drift, seasonality and global drift (v2 engine)."""

import datetime

import numpy as np
import pandas as pd
import pytest

from synccfd import DatasetGenerator
from synccfd.simulator.config import (
    DriftConfig,
    FraudConfig,
    GlobalDrift,
    SeasonalityConfig,
    Shock,
)
from synccfd.simulator.engine.daily_factors import daily_factors
from synccfd.simulator.engine.drift import CustomerDrift, regime_weight
from synccfd.simulator.location.sampling import haversine_km

PARAMS = {
    "n_customers": 1000,
    "n_terminals": 2000,
    "nb_days": 120,
    "start_date": "2023-01-01",
    "random_state": 42,
    # Components off unless a test switches them on
    "fraud": None,
    "drift": None,
    "seasonality": None,
    "merchants": None,
}
NO_FRAUD = FraudConfig(scenarios=())


def _generate(**kwargs):
    return DatasetGenerator(**{**PARAMS, **kwargs}).generate()


def _gt_columns(frame):
    return [c for c in frame.columns if c.startswith("GT_")]


@pytest.fixture(scope="module")
def drifted():
    # Half of the customers drift, so every type has enough customers; no fraud,
    # so amounts and labels come only from the drift model.
    return _generate(drift=DriftConfig(customer_fraction=0.5), fraud=NO_FRAUD)


# --- configuration --------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"customer_fraction": 1.5},
        {"type_weights": {"sudden": 0.5, "gradual": 0.6}},
        {"type_weights": {"sudden": 1.0, "seasonal": 0.0}},
        {"start_day_range": (0.8, 0.2)},
        {"transition_days_range": (0, 10)},
        {"recurrent_active_days_range": (5, 40)},
        {"amount_log_sigma": -1.0},
        {"relocation_prob": 2.0},
    ],
)
def test_invalid_drift_config_raises(kwargs):
    with pytest.raises(ValueError):
        DriftConfig(**kwargs)


def test_type_weights_are_read_only():
    config = DriftConfig()
    with pytest.raises(TypeError):
        config.type_weights["sudden"] = 1.0  # type: ignore[index]


def test_invalid_seasonality_and_shock_raise():
    with pytest.raises(ValueError):
        SeasonalityConfig(weekday_frequency=(1.0,) * 6)
    with pytest.raises(ValueError):
        Shock("2023-02-30")
    with pytest.raises(ValueError):
        Shock("2023-11-24", duration_days=0)


# --- w(t) -------------------------------------------------------------------------


def test_regime_weight_shapes():
    t = np.arange(100) + 0.5
    sudden = regime_weight(CustomerDrift(kind="sudden", t0=40), t)
    assert (sudden[:40] == 0).all() and (sudden[40:] == 1).all()

    gradual = regime_weight(CustomerDrift(kind="gradual", t0=40, transition_days=10), t)
    assert (gradual[:40] == 0).all() and (gradual[50:] == 1).all()
    assert np.all(np.diff(gradual[40:50]) > 0)
    np.testing.assert_allclose(gradual[44], 0.45)

    recurrent = regime_weight(
        CustomerDrift(kind="recurrent", t0=10, period_days=30, active_days=5), t
    )
    assert np.flatnonzero(recurrent).tolist() == [
        d for d in range(100) if d >= 10 and (d - 10) % 30 < 5
    ]
    assert (regime_weight(CustomerDrift(), t) == 0).all()


# --- acceptance criteria ----------------------------------------------------------


def test_no_drift_customers_are_unchanged(drifted):
    """customer_fraction=0 + no seasonality gives exactly the data without drift; with
    drift on, customers without drift keep exactly their transactions."""
    base = _generate(fraud=NO_FRAUD)
    zero = _generate(drift=DriftConfig(customer_fraction=0.0), fraud=NO_FRAUD)
    for base_frame, zero_frame in zip(base, zero):
        # Same data; the drift adds only ground-truth columns
        pd.testing.assert_frame_equal(
            base_frame.drop(columns=_gt_columns(base_frame)),
            zero_frame.drop(columns=_gt_columns(zero_frame)),
        )

    customers, _, transactions = drifted
    unchanged = customers.CUSTOMER_ID[customers.GT_DRIFT_TYPE == "none"]
    key = ["CUSTOMER_ID", "TX_TIME_SECONDS", "TERMINAL_ID", "TX_AMOUNT", "TX_TYPE"]
    a = base[2][base[2].CUSTOMER_ID.isin(unchanged)][key]
    b = transactions[transactions.CUSTOMER_ID.isin(unchanged)][key]
    pd.testing.assert_frame_equal(
        a.sort_values(key).reset_index(drop=True),
        b.sort_values(key).reset_index(drop=True),
    )


def test_no_drift_customers_have_no_trend(drifted):
    customers, _, transactions = drifted
    stable = transactions.merge(
        customers[customers.GT_DRIFT_TYPE == "none"][["CUSTOMER_ID", "mean_amount"]]
    )
    ratio = stable.TX_AMOUNT / stable.mean_amount
    slope = np.polyfit(stable.TX_TIME_DAYS, ratio, 1)[0]
    # change of the normalised mean amount over the whole period
    assert abs(slope * PARAMS["nb_days"]) < 0.02
    assert (stable.GT_REGIME == "A").all() and (stable.GT_REGIME_WEIGHT == 0).all()


def test_sudden_drift_changes_level_at_t0(drifted):
    customers, _, transactions = drifted
    observed, expected = [], []
    for row in customers[customers.GT_DRIFT_TYPE == "sudden"].itertuples():
        own = transactions[transactions.CUSTOMER_ID == row.CUSTOMER_ID]
        before = own[own.TX_TIME_DAYS < row.GT_DRIFT_T0].TX_AMOUNT
        after = own[own.TX_TIME_DAYS >= row.GT_DRIFT_T0].TX_AMOUNT
        if len(before) >= 10 and len(after) >= 10:
            observed.append(np.log(after.mean() / before.mean()))
            expected.append(np.log(row.GT_B_mean_amount / row.mean_amount))
        assert (own[own.TX_TIME_DAYS < row.GT_DRIFT_T0].GT_REGIME == "A").all()
        assert (own[own.TX_TIME_DAYS >= row.GT_DRIFT_T0].GT_REGIME == "B").all()
    assert len(observed) > 50
    assert np.corrcoef(observed, expected)[0, 1] > 0.85


def test_gradual_regime_share_follows_weight(drifted):
    customers, _, transactions = drifted
    gradual = transactions[
        transactions.CUSTOMER_ID.isin(
            customers.CUSTOMER_ID[customers.GT_DRIFT_TYPE == "gradual"]
        )
    ]
    assert (gradual[gradual.GT_REGIME_WEIGHT == 0].GT_REGIME == "A").all()
    assert (gradual[gradual.GT_REGIME_WEIGHT == 1].GT_REGIME == "B").all()
    transition = gradual[
        (gradual.GT_REGIME_WEIGHT > 0) & (gradual.GT_REGIME_WEIGHT < 1)
    ]
    bins = pd.cut(transition.GT_REGIME_WEIGHT, np.linspace(0, 1, 6))
    share = transition.groupby(bins, observed=True).agg(
        w=("GT_REGIME_WEIGHT", "mean"), b=("GT_REGIME", lambda r: (r == "B").mean())
    )
    np.testing.assert_allclose(share.b, share.w, atol=0.05)


# --- other drift behaviour ---------------------------------------------------------


def test_incremental_parameters_are_interpolated(drifted):
    customers, _, transactions = drifted
    rows = customers[customers.GT_DRIFT_TYPE == "incremental"]
    merged = transactions.merge(
        rows[["CUSTOMER_ID", "mean_amount", "GT_B_mean_amount"]]
    )
    expected = (
        1 - merged.GT_REGIME_WEIGHT
    ) * merged.mean_amount + merged.GT_REGIME_WEIGHT * merged.GT_B_mean_amount
    # amounts are drawn around the interpolated mean (std = mean / 2)
    assert (merged.TX_AMOUNT / expected).mean() == pytest.approx(1.0, abs=0.05)
    assert (
        merged.GT_REGIME == np.where(merged.GT_REGIME_WEIGHT >= 0.5, "B", "A")
    ).all()


def test_recurrent_episodes(drifted):
    customers, _, transactions = drifted
    for row in customers[customers.GT_DRIFT_TYPE == "recurrent"].head(30).itertuples():
        own = transactions[transactions.CUSTOMER_ID == row.CUSTOMER_ID]
        days = own.TX_TIME_DAYS
        active = (days >= row.GT_DRIFT_T0) & (
            (days - row.GT_DRIFT_T0) % row.GT_DRIFT_PERIOD < row.GT_DRIFT_ACTIVE_DAYS
        )
        assert ((own.GT_REGIME == "B") == active).all()


def test_relocated_customers_buy_near_their_new_address(drifted):
    customers, terminals, transactions = drifted
    moved = customers[customers.GT_RELOCATED]
    assert len(moved) > 0
    merged = transactions.merge(
        moved[["CUSTOMER_ID", "GT_B_billing_lat", "GT_B_billing_long"]]
    )
    in_b = merged[merged.GT_REGIME == "B"]
    np.testing.assert_allclose(in_b.TX_BILL_LAT, in_b.GT_B_billing_lat)
    cp = in_b[in_b.TX_TYPE == "CP"]
    distance = haversine_km(
        cp.GT_B_billing_lat, cp.GT_B_billing_long, cp.TX_TERM_LAT, cp.TX_TERM_LONG
    )
    assert (distance <= 10.0 + 1e-9).all()  # available_terminals_radius


def test_ground_truth_columns(drifted):
    customers, _, transactions = drifted
    assert set(customers.GT_DRIFT_TYPE) == {
        "none",
        "sudden",
        "gradual",
        "incremental",
        "recurrent",
    }
    assert (customers.GT_DRIFT_TYPE != "none").sum() == PARAMS["n_customers"] // 2
    assert customers[customers.GT_DRIFT_TYPE == "none"].GT_DRIFT_T0.isna().all()
    assert transactions.GT_REGIME_WEIGHT.between(0, 1).all()
    assert set(transactions.GT_REGIME) <= {"A", "B"}
    # transactions columns: the attributes first, then the ground truth
    columns = transactions.columns.tolist()
    first_gt = min(columns.index(c) for c in _gt_columns(transactions))
    assert all(c.startswith("GT_") for c in columns[first_gt:])
    assert columns.index("GT_REGIME") == columns.index("GT_REGIME_WEIGHT") + 1


def test_drift_does_not_create_fraud_labels():
    transactions = _generate(drift=DriftConfig(customer_fraction=0.5), fraud=NO_FRAUD)[
        2
    ]
    assert transactions.TX_FRAUD.sum() == 0


def test_drift_with_fraud_keeps_ground_truth():
    customers, _, transactions = _generate(drift=DriftConfig(), fraud=FraudConfig())
    assert transactions.TX_FRAUD.sum() > 0
    assert {"GT_REGIME", "GT_REGIME_WEIGHT"} <= set(transactions.columns)


def test_reproducible_with_same_seed(drifted):
    again = _generate(drift=DriftConfig(customer_fraction=0.5), fraud=NO_FRAUD)
    for a, b in zip(drifted, again):
        pd.testing.assert_frame_equal(a, b)


# --- seasonality and global drift ---------------------------------------------------


def test_daily_factors():
    start = datetime.datetime(2023, 11, 27)  # a Monday
    factors = daily_factors(
        40,
        start,
        SeasonalityConfig(),
        GlobalDrift(
            amount_trend_per_year=0.05,
            shocks=(Shock("2023-12-01", duration_days=2, frequency_multiplier=2.0),),
        ),
    )
    config = SeasonalityConfig()
    # Monday 27 Nov 2023: weekday factor only
    assert factors.frequency[0] == pytest.approx(config.weekday_frequency[0])
    # Thursday 30 Nov 2023: last business day of the month
    assert factors.frequency[3] == pytest.approx(
        config.weekday_frequency[3] * config.payday_frequency_multiplier
    )
    # Friday 1 Dec 2023: December and the shock
    assert factors.frequency[4] == pytest.approx(
        config.weekday_frequency[4] * config.december_frequency_multiplier * 2.0
    )
    assert factors.shock.tolist()[:7] == [False] * 4 + [True, True, False]
    # Tuesday 5 Dec 2023: payday in December
    assert factors.frequency[8] == pytest.approx(
        config.weekday_frequency[1]
        * config.payday_frequency_multiplier
        * config.december_frequency_multiplier
    )
    assert factors.amount[4] == pytest.approx(
        config.december_amount_multiplier * 1.05 ** (4 / 365.25)
    )


def test_seasonality_changes_daily_volume():
    transactions = _generate(seasonality=SeasonalityConfig(), fraud=NO_FRAUD)[2]
    per_day = transactions.groupby("TX_TIME_DAYS").size()
    weekday = (
        pd.Timestamp(PARAMS["start_date"]) + pd.to_timedelta(per_day.index, "D")
    ).weekday
    by_weekday = per_day.groupby(weekday).mean()
    assert by_weekday[5] > by_weekday[6] * 1.2  # Saturday vs Sunday (1.15 / 0.85)
    assert "GT_REGIME" not in transactions.columns  # seasonality is not drift


def test_global_shock_is_labelled():
    shock = Shock("2023-02-01", duration_days=3, frequency_multiplier=2.0)
    drift = DriftConfig(
        customer_fraction=0.0, global_drift=GlobalDrift(shocks=(shock,))
    )
    transactions = _generate(drift=drift, fraud=NO_FRAUD)[2]
    on_shock = transactions.TX_TIME_DAYS.between(31, 33)
    assert (transactions.GT_SHOCK == on_shock).all()
    per_day = transactions.groupby("TX_TIME_DAYS").size()
    assert per_day.loc[31:33].mean() > 1.6 * per_day.loc[20:30].mean()
