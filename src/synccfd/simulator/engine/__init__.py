"""v2 simulation engine: vectorised, with local random streams only."""

import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DriftConfig, FraudConfig, MerchantConfig, SeasonalityConfig
from ..location.features import LocationFeatures
from . import merchants as merchant_model
from .customers import generate_customer_profiles
from .daily_factors import daily_factors
from .drift import assign_drift, drift_columns
from .fraud import apply_frauds
from .rng import RandomStreams
from .terminals import generate_terminal_profiles
from .transactions import generate_transactions

__all__ = ["RandomStreams", "V2Result", "simulate"]


@dataclass
class V2Result:
    customer_profiles: pd.DataFrame
    terminal_profiles: pd.DataFrame
    transactions: pd.DataFrame
    loc_features: LocationFeatures
    merchant_profiles: Optional[pd.DataFrame] = None


def simulate(
    n_customers: int,
    n_terminals: int,
    nb_days: int,
    city_radius: float,
    available_terminals_radius: float,
    start_date: datetime.datetime,
    random_state: Optional[int],
    fraud: FraudConfig,
    drift: Optional[DriftConfig] = None,
    seasonality: Optional[SeasonalityConfig] = None,
    merchants: Optional[MerchantConfig] = None,
) -> V2Result:
    """Generate profiles and labelled transactions with the v2 engine."""
    if merchants is not None:
        unknown = fraud.fraud_categories() - set(merchants.names)
        if fraud.mechanism == "inject" and unknown:
            raise ValueError(
                f"FraudConfig names categories that the merchants do not have: "
                f"{sorted(unknown)}."
            )
    streams = RandomStreams(random_state)
    # random_state=None: LocationFeatures must not seed the global ``random`` module
    loc_features = LocationFeatures(
        city_radius=city_radius,
        available_terminals_radius=available_terminals_radius,
        random_state=None,
    )

    customers = generate_customer_profiles(
        n_customers, loc_features, streams.generator("customers")
    )
    terminals = generate_terminal_profiles(
        n_terminals, loc_features, streams.generator("terminals")
    )
    loc_features.gen_customer_available_terminals(vectorized=True)
    customers["available_terminals"] = loc_features.customers_available_terminals

    drifts = None
    if drift is not None:
        drifts = assign_drift(
            customers, drift, nb_days, loc_features, streams.generator("drift")
        )
    global_drift = drift.global_drift if drift is not None else None
    factors = None
    if seasonality is not None or global_drift is not None:
        factors = daily_factors(nb_days, start_date, seasonality, global_drift)

    world = None
    if merchants is None:
        transactions = generate_transactions(
            customers,
            terminals,
            loc_features,
            nb_days,
            start_date,
            streams.child_generators("transactions", n_customers),
            drifts=drifts,
            factors=factors,
            shock_column=global_drift is not None,
        )
        if drifts is not None:
            customers = pd.concat([customers, drift_columns(drifts)], axis=1)
    else:
        world, terminals = merchant_model.build_merchants(
            merchants, terminals, loc_features, streams.generator("merchants", 0)
        )
        preferences, spending = merchant_model.customer_preferences(
            n_customers, merchants, streams.generator("merchants", 1)
        )
        preferences_b = None
        if drifts is not None and drift is not None:
            preferences_b = merchant_model.regime_b_preferences(
                preferences, drifts, merchants, drift, streams.generator("drift", 1)
            )
        transactions = merchant_model.generate_transactions(
            customers,
            preferences,
            spending,
            world,
            terminals,
            loc_features,
            nb_days,
            start_date,
            streams.child_generators("transactions", n_customers),
            drifts=drifts,
            preferences_b=preferences_b,
            factors=factors,
            shock_column=global_drift is not None,
        )
        customers = _merchant_customer_columns(
            customers, transactions, world, preferences, spending
        )
        if drifts is not None and preferences_b is not None:
            gt = drift_columns(drifts).drop(
                columns=["GT_B_mean_amount", "GT_B_std_amount"]
            )
            # Regime B spending factor: the drift's amount factor applied to it
            amount_b = np.array([d.mean_amount for d in drifts])
            gt.insert(
                5,
                "GT_B_spending_factor",
                spending * amount_b / customers["_original_mean_amount"].to_numpy(),
            )
            customers = pd.concat(
                [
                    customers,
                    gt,
                    merchant_model.preference_columns(
                        world, preferences_b, "GT_B_pref_"
                    ),
                ],
                axis=1,
            )
        customers = customers.drop(columns="_original_mean_amount")
    transactions = apply_frauds(
        transactions,
        customers,
        terminals,
        loc_features,
        fraud,
        streams,
        start_date,
        nb_days=nb_days,
        shock_by_day=factors.shock if factors is not None else None,
        merchants=world,
    )
    merchant_profiles = None
    if world is not None:
        columns = merchant_model.merchant_columns(
            world, transactions["TERMINAL_ID"].to_numpy()
        )
        position = transactions.columns.get_loc("TERMINAL_ID") + 1
        transactions.insert(position, "MERCHANT_ID", columns["MERCHANT_ID"])
        transactions.insert(
            position + 1, "MERCHANT_CATEGORY", columns["MERCHANT_CATEGORY"]
        )
        merchant_profiles = world.merchant_profiles
    return V2Result(customers, terminals, transactions, loc_features, merchant_profiles)


def _merchant_customer_columns(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    world: "merchant_model.MerchantWorld",
    preferences: np.ndarray,
    spending: np.ndarray,
) -> pd.DataFrame:
    """With merchants, the amounts come from the categories: ``mean_amount`` and
    ``std_amount`` are recomputed from the generated transactions (to keep the
    columns), and the spending factor and the preferences are added."""
    customers = customers.copy()
    customers["_original_mean_amount"] = customers["mean_amount"]
    amounts = transactions.groupby("CUSTOMER_ID")["TX_AMOUNT"]
    index = customers["CUSTOMER_ID"]
    customers["mean_amount"] = amounts.mean().reindex(index).to_numpy()
    customers["std_amount"] = amounts.std().reindex(index).to_numpy()
    customers["spending_factor"] = spending
    return pd.concat(
        [customers, merchant_model.preference_columns(world, preferences)], axis=1
    )
