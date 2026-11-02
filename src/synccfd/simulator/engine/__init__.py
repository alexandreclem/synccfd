"""v2 simulation engine: vectorised, with local random streams only."""

import datetime
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..config import DriftConfig, FraudConfig, SeasonalityConfig
from ..location.features import LocationFeatures
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
) -> V2Result:
    """Generate profiles and labelled transactions with the v2 engine."""
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
    )
    return V2Result(customers, terminals, transactions, loc_features)
