"""v2 simulation engine: vectorised, with local random streams only."""

import datetime
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..config import FraudConfig
from ..location.features import LocationFeatures
from .customers import generate_customer_profiles
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

    transactions = generate_transactions(
        customers,
        terminals,
        loc_features,
        nb_days,
        start_date,
        streams.child_generators("transactions", n_customers),
    )
    transactions = apply_frauds(
        transactions, customers, terminals, loc_features, fraud, streams, start_date
    )
    return V2Result(customers, terminals, transactions, loc_features)
