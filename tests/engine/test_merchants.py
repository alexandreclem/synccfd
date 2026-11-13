"""Merchants and categories."""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from synccfd import DatasetGenerator
from synccfd.simulator.config import (
    DEFAULT_CATEGORIES,
    DriftConfig,
    FraudConfig,
    MerchantCategory,
    MerchantConfig,
    hourly_profile,
)
from synccfd.simulator.engine.merchants import cnp_logit_offset
from synccfd.simulator.ground_truth import compromise_table

PARAMS = {
    "n_customers": 1000,
    "n_terminals": 2000,
    "nb_days": 90,
    "start_date": "2023-01-01",
    "random_state": 42,
    # Components off unless a test switches them on
    "fraud": None,
    "drift": None,
    "seasonality": None,
    "merchants": None,
}
CATEGORIES = {c.name: c for c in DEFAULT_CATEGORIES}
# Many victims, so each scenario has enough frauds in a small dataset
DENSE_FRAUD = {
    "s1_customer_fraction_per_day": 0.01,
    "s2_terminal_fraction_per_day": 2e-3,
    "s3_customer_fraction_per_day": 2e-3,
    "s4_customer_fraction_per_day": 2e-3,
    "s5_customer_fraction_per_day": 2e-3,
    "intensity_schedule": {},
}


def _generate(**kwargs):
    generator = DatasetGenerator(**{**PARAMS, **kwargs})
    customers, terminals, transactions = generator.generate()
    return customers, terminals, transactions, generator.merchant_profiles


@pytest.fixture(scope="module")
def shop():
    return _generate(merchants=MerchantConfig(), fraud=FraudConfig(scenarios=()))


@pytest.fixture(scope="module")
def shop_fraud():
    return _generate(
        merchants=MerchantConfig(), fraud=FraudConfig.realistic(**DENSE_FRAUD)
    )


def _legit(transactions):
    return transactions[transactions.TX_FRAUD == 0]


# --- configuration --------------------------------------------------------------


def test_default_categories():
    weights = {c.name: c.weight for c in DEFAULT_CATEGORIES}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert CATEGORIES["eletronicos"].median_amount == 900
    assert CATEGORIES["viagem_turismo"].channel == "CNP"
    assert CATEGORIES["restaurante_bar"].channel == "CP"
    assert all(len(c.hourly_profile) == 24 for c in DEFAULT_CATEGORIES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel": "ATM"},
        {"median_amount": 0},
        {"hourly_profile": (1.0,) * 23},
        {"opening_hours": (22, 8)},
        {"hourly_profile": (1.0,) * 6 + (0.0,) * 18, "opening_hours": (8, 20)},
        {"online_fraction": 0.0},
        {"online_fraction": 1.0},
    ],
)
def test_invalid_category_raises(kwargs):
    values = {
        "name": "x",
        "channel": "CP",
        "median_amount": 10,
        "log_sigma": 0.5,
        "weight": 1.0,
        "hourly_profile": hourly_profile(((12, 2, 1),)),
    }
    with pytest.raises(ValueError):
        MerchantCategory(**{**values, **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"categories": ()},
        {"categories": DEFAULT_CATEGORIES + DEFAULT_CATEGORIES[:1]},
        {"categories": tuple(c for c in DEFAULT_CATEGORIES if not c.has_cp)},
        {"loyalty": 1.5},
        {"n_favorites": 0},
        {"preference_concentration": 0},
    ],
)
def test_invalid_merchant_config_raises(kwargs):
    with pytest.raises(ValueError):
        MerchantConfig(**kwargs)


def test_fraud_categories_must_be_known():
    with pytest.raises(ValueError):
        _generate(
            n_customers=20,
            n_terminals=40,
            nb_days=5,
            merchants=MerchantConfig(),
            fraud=FraudConfig.realistic(s1_categories=("cassino",)),
        )


def test_without_merchants_nothing_changes():
    customers, terminals, transactions, merchants = _generate(
        n_customers=100, n_terminals=200, nb_days=10
    )
    assert merchants is None
    assert "MERCHANT_ID" not in transactions and "MERCHANT_ID" not in terminals
    assert not any(c.startswith("pref_") for c in customers.columns)


# --- merchants and terminals -------------------------------------------------------


def test_merchant_profiles_and_terminals(shop):
    _, terminals, transactions, merchants = shop
    assert list(merchants.columns) == [
        "MERCHANT_ID",
        "MERCHANT_CATEGORY",
        "channel",
        "n_terminals",
        "lat",
        "long",
    ]
    physical = merchants[merchants.channel == "CP"]
    online = merchants[merchants.channel == "CNP"]
    # Every physical terminal belongs to a physical merchant of a CP category
    assert physical.n_terminals.sum() == PARAMS["n_terminals"]
    assert physical.MERCHANT_CATEGORY.map(lambda c: CATEGORIES[c].has_cp).all()
    # One virtual terminal per online merchant, 20 per category with CNP
    assert (online.n_terminals == 1).all()
    n_cnp_categories = sum(c.has_cnp for c in DEFAULT_CATEGORIES)
    assert len(online) == 20 * n_cnp_categories
    assert len(terminals) == PARAMS["n_terminals"] + len(online)
    assert terminals.MERCHANT_ID.isin(merchants.MERCHANT_ID).all()
    # Transactions: merchant and category follow the terminal
    by_terminal = terminals.set_index("TERMINAL_ID").MERCHANT_ID
    assert (
        by_terminal[transactions.TERMINAL_ID].to_numpy() == transactions.MERCHANT_ID
    ).all()
    category = merchants.set_index("MERCHANT_ID").MERCHANT_CATEGORY
    assert (
        category[transactions.MERCHANT_ID].to_numpy() == transactions.MERCHANT_CATEGORY
    ).all()


def test_customer_preferences(shop):
    customers, _, _, _ = shop
    prefs = customers[[f"pref_{name}" for name in CATEGORIES]]
    np.testing.assert_allclose(prefs.sum(axis=1), 1.0)
    weights = np.array([c.weight for c in DEFAULT_CATEGORIES])
    np.testing.assert_allclose(prefs.mean(), weights, atol=0.02)
    assert (customers.spending_factor > 0).all()
    # mean_amount is recomputed from the generated transactions
    assert customers.mean_amount.notna().mean() > 0.9


def test_cp_purchases_use_reachable_terminals_of_their_category(shop):
    customers, _, transactions, _ = shop
    cp = transactions[transactions.TX_TYPE == "CP"]
    reachable = customers.set_index("CUSTOMER_ID").available_terminals
    for customer, rows in cp.groupby("CUSTOMER_ID"):
        assert set(rows.TERMINAL_ID) <= set(reachable[customer])


# --- acceptance criteria ---------------------------------------------------------


def test_cp_hours_are_within_opening_hours(shop_fraud):
    _, _, transactions, _ = shop_fraud
    cp = transactions[transactions.TX_TYPE == "CP"]
    hours = (cp.TX_TIME_SECONDS % 86400) // 3600
    opening = cp.MERCHANT_CATEGORY.map(lambda c: CATEGORIES[c].opening_hours)
    inside = [o[0] <= h < o[1] for h, o in zip(hours, opening)]
    assert all(inside)  # legitimate and injected CP purchases


def test_cnp_only_categories_are_never_cp(shop_fraud):
    _, _, transactions, _ = shop_fraud
    cnp_only = [c.name for c in DEFAULT_CATEGORIES if c.channel == "CNP"]
    assert (
        transactions[transactions.MERCHANT_CATEGORY.isin(cnp_only)].TX_TYPE == "CNP"
    ).all()
    cp_only = [c.name for c in DEFAULT_CATEGORIES if c.channel == "CP"]
    assert (
        transactions[transactions.MERCHANT_CATEGORY.isin(cp_only)].TX_TYPE == "CP"
    ).all()


@pytest.mark.parametrize("fraction", [0.1, 0.15, 0.3, 0.5, 0.8])
def test_cnp_logit_offset_keeps_the_mean(fraction):
    # Mean of sigmoid(c + logit(u)) over u ~ U(0, 1), by the midpoint rule
    u = (np.arange(200_000) + 0.5) / 200_000
    c = cnp_logit_offset(fraction)
    assert np.mean(1.0 / (1.0 + np.exp(-c) * (1.0 - u) / u)) == pytest.approx(
        fraction, abs=1e-4
    )
    assert (c == pytest.approx(0.0, abs=1e-9)) == (fraction == 0.5)


def test_online_fraction_of_mixed_categories(shop):
    # Customers who reach a merchant of the category: share of online purchases
    # near the category's online_fraction (the others always buy online)
    customers, terminals, transactions, merchants = shop
    terminal_category = terminals.MERCHANT_ID.map(merchants.MERCHANT_CATEGORY)
    shares = {}
    for name, category in CATEGORIES.items():
        if category.channel != "CP+CNP":
            continue
        reaches = customers.available_terminals.map(
            lambda t, name=name: bool((terminal_category.iloc[t] == name).any())
        )
        rows = transactions[
            (transactions.MERCHANT_CATEGORY == name)
            & transactions.CUSTOMER_ID.isin(customers.CUSTOMER_ID[reaches])
        ]
        shares[name] = (rows.TX_TYPE == "CNP").mean()
        assert shares[name] == pytest.approx(category.online_fraction, abs=0.05), name
    assert list(shares.values()) == sorted(shares.values())


def test_median_amount_per_category():
    # Large sample: medians within 15% of the configured ones
    _, _, transactions, _ = _generate(
        n_customers=3000,
        n_terminals=6000,
        nb_days=60,
        merchants=MerchantConfig(),
        fraud=FraudConfig(scenarios=()),
    )
    medians = transactions.groupby("MERCHANT_CATEGORY").TX_AMOUNT.median()
    for name, category in CATEGORIES.items():
        assert medians[name] == pytest.approx(category.median_amount, rel=0.15), name


def test_hourly_profiles_and_weekends(shop):
    _, _, transactions, _ = shop
    hours = (transactions.TX_TIME_SECONDS % 86400) // 3600
    restaurants = transactions.MERCHANT_CATEGORY == "restaurante_bar"
    # Restaurants peak at lunch and dinner
    peak = hours[restaurants].value_counts()
    assert peak.idxmax() in (12, 20)
    # More restaurants and fewer ride-hailing purchases on weekends (2023-01-01 was a
    # Sunday)
    weekday = (6 + transactions.TX_TIME_DAYS) % 7
    weekend = weekday >= 5
    share = transactions.groupby(weekend).MERCHANT_CATEGORY.value_counts(normalize=True)
    # (other categories also rise on weekends, so the relative effect is smaller than
    # the factor 1.4)
    assert share[True]["restaurante_bar"] > 1.1 * share[False]["restaurante_bar"]
    assert share[True]["transporte_app"] < 0.85 * share[False]["transporte_app"]


def test_loyalty_concentrates_purchases():
    def top2_share(loyalty):
        _, _, tx, _ = _generate(
            nb_days=60,
            merchants=MerchantConfig(loyalty=loyalty),
            fraud=FraudConfig(scenarios=()),
        )
        counts = tx.groupby(["CUSTOMER_ID", "MERCHANT_CATEGORY", "MERCHANT_ID"]).size()
        groups = counts.groupby(level=[0, 1])
        top2 = groups.apply(lambda s: s.nlargest(2).sum()) / groups.sum()
        return top2.mean()

    assert top2_share(0.6) > top2_share(0.0) + 0.1


def test_drift_changes_category_preferences():
    customers, _, transactions, _ = _generate(
        merchants=MerchantConfig(),
        drift=DriftConfig(customer_fraction=0.5, type_weights={"sudden": 1.0}),
        fraud=FraudConfig(scenarios=()),
    )
    names = list(CATEGORIES)
    drifting = customers[customers.GT_DRIFT_TYPE == "sudden"].set_index("CUSTOMER_ID")
    pref_a = drifting[[f"pref_{n}" for n in names]].to_numpy()
    pref_b = drifting[[f"GT_B_pref_{n}" for n in names]].to_numpy()
    # Regime B prefers other categories: total variation distance > 0.1 on average
    assert (0.5 * np.abs(pref_b - pref_a).sum(axis=1)).mean() > 0.1

    # The observed category shares follow the regime: after t0 they are closer to B
    tx = transactions[transactions.CUSTOMER_ID.isin(drifting.index)]
    shares = (
        tx.groupby(["CUSTOMER_ID", "GT_REGIME"]).MERCHANT_CATEGORY.value_counts(
            normalize=True
        )
    ).unstack(fill_value=0.0)
    shares = shares.reindex(columns=names, fill_value=0.0)
    closer_to_b = []
    for customer, row in drifting.iterrows():
        if (customer, "B") not in shares.index or (customer, "A") not in shares.index:
            continue
        b_observed = shares.loc[(customer, "B")].to_numpy()
        a = row[[f"pref_{n}" for n in names]].to_numpy(dtype=float)
        b = row[[f"GT_B_pref_{n}" for n in names]].to_numpy(dtype=float)
        closer_to_b.append(np.abs(b_observed - b).sum() < np.abs(b_observed - a).sum())
    assert np.mean(closer_to_b) > 0.75


# --- fraud with merchants ---------------------------------------------------------


def test_scenario_1_in_high_value_categories(shop_fraud):
    _, _, transactions, _ = shop_fraud
    s1 = transactions[transactions.TX_FRAUD_SCENARIO == 1]
    assert len(s1) > 100
    assert s1.MERCHANT_CATEGORY.isin(["eletronicos", "viagem_turismo"]).all()


def test_scenario_3_follows_its_category_schedule(shop_fraud):
    _, _, transactions, _ = shop_fraud
    s3 = transactions[transactions.TX_FRAUD_SCENARIO == 3]
    assert (s3.TX_TYPE == "CNP").all()
    # realistic(): electronics and marketplaces, then digital services and
    # marketplaces from 50% of the period (day 45 of 90); bursts may run over
    early = s3[s3.TX_TIME_DAYS < 44]
    late = s3[s3.TX_TIME_DAYS >= 46]
    assert early.MERCHANT_CATEGORY.isin(["eletronicos", "ecommerce_marketplace"]).all()
    assert late.MERCHANT_CATEGORY.isin(
        ["servicos_digitais", "ecommerce_marketplace"]
    ).all()
    assert (early.MERCHANT_CATEGORY == "eletronicos").any()
    assert (late.MERCHANT_CATEGORY == "servicos_digitais").any()
    # Amounts above the category median, on average
    median = s3.MERCHANT_CATEGORY.map(lambda c: CATEGORIES[c].median_amount)
    assert (s3.TX_AMOUNT / median).median() > 1.5


def test_scenario_2_compromises_whole_merchants_by_risk(shop_fraud):
    _, terminals, transactions, merchants = shop_fraud
    reads = transactions[
        transactions.GT_COMPROMISE_ID.notna() & (transactions.TX_FRAUD == 0)
    ]
    assert len(reads) > 0 and (reads.TX_TYPE == "CP").all()
    # The read happens at a terminal of the compromised merchant; one fraudster per
    # merchant, whatever terminal read the card
    table = compromise_table(transactions)
    table["MERCHANT_ID"] = (
        terminals.set_index("TERMINAL_ID").MERCHANT_ID[table.TERMINAL_ID].to_numpy()
    )
    used = table.dropna(subset=["GT_FRAUDSTER_ID"])
    assert (used.groupby("GT_FRAUDSTER_ID").MERCHANT_ID.nunique() == 1).all()
    # Riskier categories are compromised more often than their share of merchants
    physical = merchants[merchants.channel == "CP"].set_index("MERCHANT_ID")
    compromised = physical.loc[table.MERCHANT_ID.unique()]
    share = compromised.MERCHANT_CATEGORY.value_counts(normalize=True)
    base = physical.MERCHANT_CATEGORY.value_counts(normalize=True)
    # (only merchants that read some card appear here, so the effect is diluted)
    assert share["posto_combustivel"] > 1.2 * base["posto_combustivel"]
    low_risk = [c.name for c in DEFAULT_CATEGORIES if c.has_cp and c.skimming_risk == 1]
    assert share.reindex(low_risk, fill_value=0).sum() < 0.9 * base[low_risk].sum()
    # Cloned cards are used in resellable categories
    s2 = transactions[transactions.TX_FRAUD_SCENARIO == 2]
    assert s2.MERCHANT_CATEGORY.isin(FraudConfig().s4_categories).all()


def test_scenarios_4_and_5_categories(shop_fraud):
    _, _, transactions, _ = shop_fraud
    s4 = transactions[transactions.TX_FRAUD_SCENARIO == 4]
    assert len(s4) > 0 and s4.MERCHANT_CATEGORY.isin(FraudConfig().s4_categories).all()
    s5 = transactions[transactions.TX_FRAUD_SCENARIO == 5]
    assert len(s5) > 0 and (s5.MERCHANT_CATEGORY == "servicos_digitais").all()
    # Card testing at one merchant per campaign
    assert (s5.groupby("GT_CAMPAIGN_ID").MERCHANT_ID.nunique() == 1).all()


def test_reproducible_and_saved(shop, tmp_path):
    again = _generate(merchants=MerchantConfig(), fraud=FraudConfig(scenarios=()))
    pd.testing.assert_frame_equal(shop[2], again[2])
    pd.testing.assert_frame_equal(shop[3], again[3])

    generator = DatasetGenerator(
        n_customers=50, n_terminals=100, nb_days=5, merchants=MerchantConfig()
    )
    generator.generate()
    with contextlib.redirect_stdout(io.StringIO()):
        generator.save(output_dir=str(tmp_path), save_by_day=False)
    assert (tmp_path / "profiles" / "merchant_profiles.pkl").exists()
