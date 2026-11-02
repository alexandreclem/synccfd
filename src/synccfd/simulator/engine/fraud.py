"""Fraud scenarios 1-4 (v2 engine), tamper mechanism.

With ``FraudConfig(mechanism="inject")``, ``apply_frauds`` delegates to
``injection.py`` (scenarios 1-5, new transactions). Otherwise: same scenarios and
parameters as v0.1 (``legacy/fraud.py``, configured by ``FraudConfig``), with local
random streams: scenario ``k`` draws everything (daily counts, compromised
terminals/customers, tampering) from sub-stream ``k`` of the ``fraud`` stream, so the
scenarios are independent of each other.

- Scenario 1: every transaction above ``s1_amount_threshold`` is fraud.
- Scenario 2: each day, ``n2`` terminals are compromised for ``s2_duration_days``;
  their not-yet-fraudulent transactions in that window become fraud.
- Scenarios 3 and 4: each day, ``n3`` and ``n4`` customers not currently
  compromised are chosen.
  - Scenario 3 (card number stolen): over ``s3_duration_days``, a fraction
    ``s3_tamper_fraction`` of the chosen customers' transactions become CNP, at a
    random terminal, shipped to the fraudster's location (random point around a
    population-weighted city), with the amount multiplied by ``amount_multiplier``.
  - Scenario 4 (physical card stolen): over ``s4_duration_days``, all the chosen
    customers' transactions are moved to the first of those days (same time of
    day), become CNP with probability ``s4_cnp_prob``, and are made near the
    fraudster (random point around the customer's city): CP at a terminal near the
    fraudster, CNP at a random terminal shipped to the fraudster; amounts are
    multiplied by ``amount_multiplier``.

Unlike v0.1, the transactions are sorted chronologically again at the end (scenario
4 moves transactions in time) and ``TRANSACTION_ID`` is reassigned.
"""

import datetime
from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ..config import FraudConfig
from ..location import sampling
from ..location.features import LocationFeatures
from .injection import FRAUD_GT_COLUMNS, inject_frauds
from .rng import RandomStreams

if TYPE_CHECKING:
    from .merchants import MerchantWorld
from .transactions import SECONDS_PER_DAY, assemble_transactions


class _Grouped:
    """Row indices grouped by an integer key (e.g. customer or terminal id)."""

    def __init__(self, keys: np.ndarray, n_keys: int):
        self.order = np.argsort(keys, kind="stable")
        sorted_keys = keys[self.order]
        self.starts = np.searchsorted(sorted_keys, np.arange(n_keys), side="left")
        self.stops = np.searchsorted(sorted_keys, np.arange(n_keys), side="right")

    def rows(self, key: int) -> np.ndarray:
        return self.order[self.starts[key] : self.stops[key]]


def apply_frauds(
    transactions: pd.DataFrame,
    customer_profiles: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    loc_features: LocationFeatures,
    config: FraudConfig,
    streams: RandomStreams,
    start_date: datetime.datetime,
    nb_days: Optional[int] = None,
    shock_by_day: Optional[np.ndarray] = None,
    merchants: Optional["MerchantWorld"] = None,
) -> pd.DataFrame:
    """Label (and tamper) transactions according to ``config``; returns a new frame.

    With ``config.mechanism == "inject"``, fraud transactions are added instead (see
    ``injection.py``); ``nb_days`` bounds them, ``shock_by_day`` fills
    ``GT_SHOCK`` for them, and ``merchants`` puts them in merchant categories.
    """
    tx = {c: transactions[c].to_numpy().copy() for c in transactions.columns}
    tx["IS_CNP"] = tx.pop("TX_TYPE") == "CNP"
    n_tx = len(transactions)
    tx["TX_FRAUD"] = np.zeros(n_tx, dtype=np.int64)
    tx["TX_FRAUD_SCENARIO"] = np.zeros(n_tx, dtype=np.int64)
    n_days = int(tx["TX_TIME_DAYS"].max()) + 1 if n_tx else 0
    if config.mechanism == "inject":
        return _inject(
            tx,
            transactions,
            customer_profiles,
            terminal_profiles,
            loc_features,
            config,
            streams,
            start_date,
            nb_days if nb_days is not None else n_days,
            shock_by_day,
            merchants,
        )
    scenarios = config.scenarios

    if 1 in scenarios:
        _scenario_1(tx, config)
    if 2 in scenarios:
        _scenario_2(
            tx, config, len(terminal_profiles), n_days, streams.generator("fraud", 2)
        )
    if 3 in scenarios or 4 in scenarios:
        _scenarios_3_and_4(
            tx,
            config,
            len(customer_profiles),
            terminal_profiles,
            loc_features,
            n_days,
            streams.generator("fraud", 3),
            streams.generator("fraud", 4),
        )

    # Ground-truth columns of the legitimate transactions (drift) are kept as they are
    ground_truth = tuple(c for c in transactions.columns if c.startswith("GT_"))
    return assemble_transactions(
        tx, start_date, extra_columns=("TX_FRAUD", "TX_FRAUD_SCENARIO", *ground_truth)
    )


def _inject(
    tx: dict,
    transactions: pd.DataFrame,
    customer_profiles: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    loc_features: LocationFeatures,
    config: FraudConfig,
    streams: RandomStreams,
    start_date: datetime.datetime,
    n_days: int,
    shock_by_day: Optional[np.ndarray],
    merchants: Optional["MerchantWorld"],
) -> pd.DataFrame:
    n_tx = len(transactions)
    tx["GT_FRAUDSTER_ID"] = np.full(n_tx, -1, dtype=np.int64)
    tx["GT_CAMPAIGN_ID"] = np.full(n_tx, -1, dtype=np.int64)
    tx["GT_COMPROMISE_ID"] = np.full(n_tx, -1, dtype=np.int64)
    rngs = {k: streams.generator("fraud", k) for k in config.scenarios}
    injected = inject_frauds(
        tx,
        customer_profiles,
        terminal_profiles,
        loc_features,
        config,
        rngs,
        n_days,
        merchants=merchants,
    )
    drift_gt = tuple(c for c in transactions.columns if c.startswith("GT_"))
    if injected:
        n_new = len(injected["TX_TIME_SECONDS"])
        injected["TX_TIME_DAYS"] = injected["TX_TIME_SECONDS"] // SECONDS_PER_DAY
        injected["TX_FRAUD"] = np.ones(n_new, dtype=np.int64)
        # Injected transactions do not come from the customer's regimes
        if "GT_REGIME_WEIGHT" in tx:
            injected["GT_REGIME_WEIGHT"] = np.full(n_new, np.nan)
        if "GT_REGIME" in tx:
            tx["GT_REGIME"] = tx["GT_REGIME"].astype(object)
            injected["GT_REGIME"] = np.full(n_new, None, dtype=object)
        if "GT_SHOCK" in tx:
            injected["GT_SHOCK"] = (
                shock_by_day[injected["TX_TIME_DAYS"]]
                if shock_by_day is not None
                else np.zeros(n_new, dtype=bool)
            )
        tx = {
            key: np.concatenate([value, injected[key].astype(value.dtype, copy=False)])
            if key in injected
            else value
            for key, value in tx.items()
            if key not in ("TX_DATETIME", "TRANSACTION_ID")
        }
    tx["GT_FRAUD_SCENARIO"] = tx["TX_FRAUD_SCENARIO"].copy()
    frame = assemble_transactions(
        tx,
        start_date,
        extra_columns=("TX_FRAUD", "TX_FRAUD_SCENARIO", *drift_gt, *FRAUD_GT_COLUMNS),
    )
    for column in ("GT_FRAUDSTER_ID", "GT_CAMPAIGN_ID", "GT_COMPROMISE_ID"):
        values = frame[column].astype("Int64")
        frame[column] = values.mask(values < 0)
    return frame


def _mark(tx: dict, rows: np.ndarray, scenario: int) -> None:
    tx["TX_FRAUD"][rows] = 1
    tx["TX_FRAUD_SCENARIO"][rows] = scenario


def _scenario_1(tx: dict, config: FraudConfig) -> None:
    _mark(tx, np.flatnonzero(tx["TX_AMOUNT"] > config.s1_amount_threshold), 1)


def _scenario_2(
    tx: dict,
    config: FraudConfig,
    n_terminals: int,
    n_days: int,
    rng: np.random.Generator,
) -> None:
    counts = config.daily_counts(2, n_terminals, n_days, rng)
    by_terminal = _Grouped(tx["TERMINAL_ID"], n_terminals)
    days = tx["TX_TIME_DAYS"]
    for day in range(n_days):
        n = min(int(counts[day]), n_terminals)
        for terminal in rng.choice(n_terminals, size=n, replace=False):
            rows = by_terminal.rows(terminal)
            rows = rows[
                (days[rows] >= day)
                & (days[rows] < day + config.s2_duration_days)
                & (tx["TX_FRAUD"][rows] == 0)
            ]
            _mark(tx, rows, 2)


def _scenarios_3_and_4(
    tx: dict,
    config: FraudConfig,
    n_customers: int,
    terminal_profiles: pd.DataFrame,
    loc_features: LocationFeatures,
    n_days: int,
    rng3: np.random.Generator,
    rng4: np.random.Generator,
) -> None:
    counts3 = config.daily_counts(3, n_customers, n_days, rng3)
    counts4 = config.daily_counts(4, n_customers, n_days, rng4)
    by_customer = _Grouped(tx["CUSTOMER_ID"], n_customers)
    terminals = (
        terminal_profiles["terminal_lat"].to_numpy(),
        terminal_profiles["terminal_long"].to_numpy(),
    )
    all_customers = np.arange(n_customers)
    queue3: deque[np.ndarray] = deque(maxlen=config.s3_duration_days)
    queue4: deque[np.ndarray] = deque(maxlen=config.s4_duration_days)

    for day in range(n_days):
        # Customers compromised in the last ``duration`` days cannot be chosen again
        if day >= config.s4_duration_days and queue4:
            queue4.popleft()
        if day >= config.s3_duration_days and queue3:
            queue3.popleft()
        excluded = np.concatenate([*queue3, *queue4, np.empty(0, dtype=np.int64)])
        available = np.setdiff1d(all_customers, excluded)
        n3, n4 = int(counts3[day]), int(counts4[day])
        if len(available) < n3 + n4:
            continue

        chosen3 = rng3.choice(available, size=n3, replace=False)
        chosen4 = rng4.choice(np.setdiff1d(available, chosen3), size=n4, replace=False)
        queue3.append(chosen3)
        queue4.append(chosen4)

        rows3 = _window_rows(tx, by_customer, chosen3, day, config.s3_duration_days)
        if len(rows3):
            k = int(len(rows3) * config.s3_tamper_fraction)
            picked = rng3.choice(rows3, size=k, replace=False)
            _tamper_scenario_3(tx, picked, terminals, loc_features, config, rng3)

        for customer in chosen4:
            rows4 = _window_rows(
                tx, by_customer, [customer], day, config.s4_duration_days
            )
            if len(rows4):
                _tamper_scenario_4(
                    tx, rows4, int(customer), terminals, loc_features, config, rng4
                )


def _window_rows(
    tx: dict, by_customer: _Grouped, customers, day: int, duration: int
) -> np.ndarray:
    """Not-yet-fraudulent transactions of ``customers`` in ``[day, day + duration)``."""
    if len(customers) == 0:
        return np.empty(0, dtype=np.int64)
    rows = np.sort(np.concatenate([by_customer.rows(int(c)) for c in customers]))
    days = tx["TX_TIME_DAYS"][rows]
    return rows[(days >= day) & (days < day + duration) & (tx["TX_FRAUD"][rows] == 0)]


def _set_terminals(
    tx: dict, rows: np.ndarray, terminal_ids: np.ndarray, terminals
) -> None:
    tx["TERMINAL_ID"][rows] = terminal_ids
    tx["TX_TERM_LAT"][rows] = terminals[0][terminal_ids]
    tx["TX_TERM_LONG"][rows] = terminals[1][terminal_ids]


def _tamper_scenario_3(
    tx: dict,
    rows: np.ndarray,
    terminals,
    loc_features: LocationFeatures,
    config: FraudConfig,
    rng: np.random.Generator,
) -> None:
    tx["IS_CNP"][rows] = True
    _set_terminals(tx, rows, rng.integers(0, len(terminals[0]), len(rows)), terminals)
    # One fraudster location per victim: a random point around a
    # population-weighted city anywhere in Brazil.
    victims = np.unique(tx["CUSTOMER_ID"][rows])
    fraudsters = sampling.random_points_around_cities(
        len(victims),
        loc_features.city_radius,
        loc_features.data["brazil_polygon"],
        loc_features.data["brazil_cities"],
        rng,
    )["random_points"]
    position = np.searchsorted(victims, tx["CUSTOMER_ID"][rows])
    tx["TX_SHIPP_LAT"][rows] = fraudsters["lat"].to_numpy()[position]
    tx["TX_SHIPP_LONG"][rows] = fraudsters["long"].to_numpy()[position]
    tx["TX_AMOUNT"][rows] = tx["TX_AMOUNT"][rows] * config.amount_multiplier
    _mark(tx, rows, 3)


def _tamper_scenario_4(
    tx: dict,
    rows: np.ndarray,
    customer: int,
    terminals,
    loc_features: LocationFeatures,
    config: FraudConfig,
    rng: np.random.Generator,
) -> None:
    # Move every transaction to the first day, keeping the time of day
    first_day = tx["TX_TIME_DAYS"][rows].min()
    tx["TX_TIME_SECONDS"][rows] = (
        first_day * SECONDS_PER_DAY + tx["TX_TIME_SECONDS"][rows] % SECONDS_PER_DAY
    )
    tx["TX_TIME_DAYS"][rows] = first_day

    is_cnp = rng.random(len(rows)) < config.s4_cnp_prob

    # The fraudster is somewhere in the victim's city
    city = loc_features.billing_cities_locations.iloc[customer]
    lat, long = sampling.random_points_around(
        np.array([city["lat"]]),
        np.array([city["long"]]),
        loc_features.city_radius,
        loc_features.data["brazil_polygon"],
        rng,
    )
    nearby = np.asarray(
        sampling.points_within_radius(
            lat,
            long,
            terminals[0],
            terminals[1],
            loc_features.available_terminals_radius,
        )[0],
        dtype=np.int64,
    )
    if len(nearby) == 0:
        is_cnp[:] = True

    terminal_ids = rng.integers(0, len(terminals[0]), len(rows))
    if len(nearby):
        cp = ~is_cnp
        terminal_ids[cp] = nearby[rng.integers(0, len(nearby), int(cp.sum()))]
    _set_terminals(tx, rows, terminal_ids, terminals)
    tx["IS_CNP"][rows] = is_cnp
    tx["TX_SHIPP_LAT"][rows] = np.where(is_cnp, lat[0], tx["TX_TERM_LAT"][rows])
    tx["TX_SHIPP_LONG"][rows] = np.where(is_cnp, long[0], tx["TX_TERM_LONG"][rows])
    tx["TX_AMOUNT"][rows] = tx["TX_AMOUNT"][rows] * config.amount_multiplier
    _mark(tx, rows, 4)
