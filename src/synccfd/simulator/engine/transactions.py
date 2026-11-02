"""Legitimate transactions (v2 engine).

Same model as v0.1 (``legacy/generator.py``), vectorised per customer: all days of a
customer are drawn at once, from that customer's own random stream, so a customer's
transactions do not depend on the other customers.

Model, per customer and day:

- number of transactions ~ Poisson(``mean_nb_tx_per_day``);
- amount ~ Normal(``mean_amount``, ``std_amount``); a negative draw is replaced by
  Uniform(0, 2 * ``mean_amount``); rounded to cents;
- type: CNP with probability ``cnp_prob`` (always CNP if no terminal is reachable);
- terminal: CP, uniform among the reachable terminals; CNP, uniform among all;
- shipping address: CP, terminal (0.8), billing (0.15) or a random point around the
  billing address (0.05); CNP, billing (0.9) or a random point (0.1);
- time of day (seconds): CP ~ Normal(12h, 20 000 s); CNP, with equal probability,
  Normal(10h, 12 500 s) or Normal(20h, 12 500 s) modulo one day; truncated to an
  integer.

Deliberate difference from v0.1: a time outside ``[0, 86400]`` is *redrawn*
instead of the transaction being discarded, so the number of transactions per day
follows the Poisson parameter exactly.

With concept drift (``drift.py``), the parameters of each day and transaction follow
the customer's regimes; with seasonality or global drift (``daily_factors.py``), the
daily rate and the amounts are multiplied by per-day factors. A customer without
drift and without daily factors draws exactly the same numbers as without these
components.
"""

import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..location import sampling
from ..location.features import LocationFeatures
from .daily_factors import DailyFactors
from .drift import NO_DRIFT, CustomerDrift, regime_weight

TRANSACTION_COLUMNS = [
    "TRANSACTION_ID",
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

SECONDS_PER_DAY = 86400
# Time-of-day model (seconds), as in v0.1
CP_TIME_MEAN, CP_TIME_STD = SECONDS_PER_DAY / 2, 20000
CNP_TIME_MEANS, CNP_TIME_STD = (36000, 72000), 12500
# Shipping address probabilities, as in v0.1
CP_SHIPPING_PROBS = (0.8, 0.15, 0.05)  # terminal, billing, random point
CNP_SHIPPING_PROBS = (0.9, 0.1)  # billing, random point


def draw_times_of_day(is_cnp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Integer seconds in ``[0, 86400]``; invalid draws are redrawn."""
    times = np.empty(len(is_cnp), dtype=np.int64)
    pending: np.ndarray = np.arange(len(is_cnp))
    while len(pending):
        cnp = is_cnp[pending]
        cp_times = np.trunc(rng.normal(CP_TIME_MEAN, CP_TIME_STD, len(pending)))
        evening = rng.random(len(pending)) < 0.5
        cnp_means = np.where(evening, CNP_TIME_MEANS[1], CNP_TIME_MEANS[0])
        cnp_times = np.trunc(rng.normal(cnp_means, CNP_TIME_STD))
        # v0.1 wraps only the evening mode around midnight
        cnp_times = np.where(evening, np.mod(cnp_times, SECONDS_PER_DAY), cnp_times)
        draw = np.where(cnp, cnp_times, cp_times)
        valid = (draw >= 0) & (draw <= SECONDS_PER_DAY)
        times[pending[valid]] = draw[valid].astype(np.int64)
        pending = pending[~valid]
    return times


def draw_shipping(
    is_cnp: np.ndarray,
    term_lat: np.ndarray,
    term_long: np.ndarray,
    bill_lat: np.ndarray,
    bill_long: np.ndarray,
    loc_features: LocationFeatures,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Shipping address: CP, terminal (0.8), billing (0.15) or a random point around
    the billing address (0.05); CNP, billing (0.9) or a random point (0.1)."""
    u = rng.random(len(is_cnp))
    at_terminal = ~is_cnp & (u < CP_SHIPPING_PROBS[0])
    at_billing = np.where(
        is_cnp,
        u < CNP_SHIPPING_PROBS[0],
        (u >= CP_SHIPPING_PROBS[0]) & (u < CP_SHIPPING_PROBS[0] + CP_SHIPPING_PROBS[1]),
    )
    random_point = ~at_terminal & ~at_billing
    shipp_lat = np.where(at_terminal, term_lat, bill_lat)
    shipp_long = np.where(at_terminal, term_long, bill_long)
    if random_point.any():
        lat, long = sampling.random_points_around(
            bill_lat[random_point],
            bill_long[random_point],
            loc_features.available_terminals_radius,
            loc_features.data["brazil_polygon"],
            rng,
        )
        shipp_lat[random_point] = lat
        shipp_long[random_point] = long
    return shipp_lat, shipp_long


def _regime_parameter(
    a: float, b: float, kind: str, is_b: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Per-transaction parameter: interpolated (incremental) or A/B by regime."""
    if kind == "incremental":
        return (1.0 - w) * a + w * b
    return np.where(is_b, b, a)


def _customer_transactions(
    customer: dict,
    available: np.ndarray,
    nb_days: int,
    terminal_lat: np.ndarray,
    terminal_long: np.ndarray,
    loc_features: LocationFeatures,
    rng: np.random.Generator,
    drift: Optional[CustomerDrift] = None,
    factors: Optional[DailyFactors] = None,
) -> dict[str, np.ndarray]:
    # A customer without drift and without daily factors follows exactly the draws
    # of the Phase 1 engine (scalar parameters, same calls in the same order).
    drifting = drift is not None and drift.kind != NO_DRIFT

    lam: Union[float, np.ndarray] = customer["mean_nb_tx_per_day"]
    if drifting:
        assert drift is not None
        w_day = regime_weight(drift, np.arange(nb_days) + 0.5)
        lam = (1.0 - w_day) * customer[
            "mean_nb_tx_per_day"
        ] + w_day * drift.mean_nb_tx_per_day
    if factors is not None:
        lam = lam * factors.frequency
    counts = rng.poisson(lam, nb_days)
    n = int(counts.sum())
    days = np.repeat(np.arange(nb_days, dtype=np.int64), counts)

    mean: Union[float, np.ndarray] = customer["mean_amount"]
    std: Union[float, np.ndarray] = customer["std_amount"]
    cnp_prob: Union[float, np.ndarray] = customer["cnp_prob"]
    w: np.ndarray = np.zeros(n)
    is_b: np.ndarray = np.zeros(n, dtype=bool)
    if drifting:
        assert drift is not None
        w = w_day[days]
        if drift.kind == "gradual":
            # Mixture of concepts: each transaction comes from B with probability w
            is_b = rng.random(n) < w
        elif drift.kind == "incremental":
            is_b = w >= 0.5  # label only; the parameters are interpolated
        else:  # sudden, recurrent: w is 0 or 1
            is_b = w >= 1.0
        mean = _regime_parameter(
            customer["mean_amount"], drift.mean_amount, drift.kind, is_b, w
        )
        std = _regime_parameter(
            customer["std_amount"], drift.std_amount, drift.kind, is_b, w
        )
        cnp_prob = _regime_parameter(
            customer["cnp_prob"], drift.cnp_prob, drift.kind, is_b, w
        )

    amounts = rng.normal(mean, std, n)
    negative = amounts < 0
    upper = 2 * (mean[negative] if isinstance(mean, np.ndarray) else mean)
    amounts[negative] = rng.uniform(0, upper, int(negative.sum()))
    if factors is not None:
        amounts = amounts * factors.amount[days]
    amounts = np.round(amounts, 2)

    relocated = drifting and drift is not None and drift.relocated
    bill_lat = np.full(n, customer["billing_lat"])
    bill_long = np.full(n, customer["billing_long"])
    if not relocated:
        if len(available):
            is_cnp = rng.random(n) < cnp_prob
        else:
            is_cnp = np.ones(n, dtype=bool)
        cnp_terminals = rng.integers(0, len(terminal_lat), n)
        if len(available):
            cp_terminals = available[rng.integers(0, len(available), n)]
        else:
            cp_terminals = cnp_terminals  # unused: every transaction is CNP
    else:
        # Regime B lives elsewhere: billing address and reachable terminals follow
        # the regime of each transaction.
        assert drift is not None and drift.available_terminals is not None
        available_b = drift.available_terminals
        bill_lat[is_b] = drift.billing_lat
        bill_long[is_b] = drift.billing_long
        has_terminal = np.where(is_b, len(available_b) > 0, len(available) > 0)
        is_cnp = (rng.random(n) < cnp_prob) | ~has_terminal
        cnp_terminals = rng.integers(0, len(terminal_lat), n)
        cp_a = (
            available[rng.integers(0, len(available), n)]
            if len(available)
            else cnp_terminals
        )
        cp_b = (
            available_b[rng.integers(0, len(available_b), n)]
            if len(available_b)
            else cnp_terminals
        )
        cp_terminals = np.where(is_b, cp_b, cp_a)
    terminals = np.where(is_cnp, cnp_terminals, cp_terminals)
    term_lat, term_long = terminal_lat[terminals], terminal_long[terminals]
    shipp_lat, shipp_long = draw_shipping(
        is_cnp, term_lat, term_long, bill_lat, bill_long, loc_features, rng
    )

    seconds = draw_times_of_day(is_cnp, rng) + days * SECONDS_PER_DAY

    columns = {
        "TX_TIME_SECONDS": seconds,
        "TX_TIME_DAYS": days,
        "CUSTOMER_ID": np.full(n, customer["CUSTOMER_ID"], dtype=np.int64),
        "TERMINAL_ID": terminals.astype(np.int64),
        "TX_AMOUNT": amounts,
        "IS_CNP": is_cnp,
        "TX_SHIPP_LAT": shipp_lat,
        "TX_SHIPP_LONG": shipp_long,
        "TX_BILL_LAT": bill_lat,
        "TX_BILL_LONG": bill_long,
        "TX_TERM_LAT": term_lat,
        "TX_TERM_LONG": term_long,
    }
    if drift is not None:
        columns["GT_REGIME_WEIGHT"] = w
        columns["GT_IS_B"] = is_b
    if factors is not None:
        columns["GT_SHOCK"] = factors.shock[days]
    return columns


def generate_transactions(
    customer_profiles: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    loc_features: LocationFeatures,
    nb_days: int,
    start_date: datetime.datetime,
    customer_rngs: list[np.random.Generator],
    drifts: Optional[list[CustomerDrift]] = None,
    factors: Optional[DailyFactors] = None,
    shock_column: bool = False,
) -> pd.DataFrame:
    """Legitimate transactions of all customers, in chronological order.

    ``customer_rngs[i]`` is the random stream of the i-th customer. With ``drifts``
    (one per customer), the transactions follow each customer's regimes and get the
    ground-truth columns ``GT_REGIME_WEIGHT`` (``w`` on the transaction's day) and
    ``GT_REGIME`` (``"A"``/``"B"``; for incremental drift, ``"B"`` once ``w >= 0.5``).
    ``factors`` applies seasonality and global drift; with ``shock_column``, the
    column ``GT_SHOCK`` flags transactions on global-shock days.
    """
    terminal_lat = terminal_profiles["terminal_lat"].to_numpy()
    terminal_long = terminal_profiles["terminal_long"].to_numpy()
    per_customer_drift: list[Optional[CustomerDrift]] = (
        list(drifts) if drifts is not None else [None] * len(customer_profiles)
    )
    parts = [
        _customer_transactions(
            customer,
            np.asarray(available, dtype=np.int64),
            nb_days,
            terminal_lat,
            terminal_long,
            loc_features,
            rng,
            drift=drift if drifts is not None else None,
            factors=factors,
        )
        for customer, available, rng, drift in zip(
            customer_profiles.to_dict("records"),
            customer_profiles["available_terminals"],
            customer_rngs,
            per_customer_drift,
        )
    ]
    return finish_transactions(
        parts, start_date, drift_columns=drifts is not None, shock_column=shock_column
    )


def finish_transactions(
    parts: list[dict[str, np.ndarray]],
    start_date: datetime.datetime,
    drift_columns: bool,
    shock_column: bool,
) -> pd.DataFrame:
    """Join the per-customer columns and build the chronological frame."""
    columns = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    extra: list[str] = []
    if drift_columns:
        columns["GT_REGIME"] = np.where(columns.pop("GT_IS_B"), "B", "A")
        extra += ["GT_REGIME_WEIGHT", "GT_REGIME"]
    if shock_column:
        extra.append("GT_SHOCK")
    else:
        columns.pop("GT_SHOCK", None)
    return assemble_transactions(columns, start_date, extra_columns=tuple(extra))


def assemble_transactions(
    columns: dict[str, np.ndarray],
    start_date: datetime.datetime,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build the transactions frame (v0.1 schema) sorted chronologically.

    ``columns`` holds the per-transaction arrays, with the type as a boolean
    ``IS_CNP``; ``TX_DATETIME`` and ``TRANSACTION_ID`` are (re)computed. Ties in time
    are broken by customer and then by original order, so the result is
    deterministic. ``TRANSACTION_ID`` is the chronological position.
    """
    columns = {
        k: v for k, v in columns.items() if k not in ("TX_DATETIME", "TRANSACTION_ID")
    }
    order = np.lexsort(
        (
            np.arange(len(columns["TX_TIME_SECONDS"])),
            columns["CUSTOMER_ID"],
            columns["TX_TIME_SECONDS"],
        )
    )
    frame = pd.DataFrame({key: value[order] for key, value in columns.items()})
    frame["TX_TYPE"] = np.where(frame.pop("IS_CNP").to_numpy(), "CNP", "CP")
    frame["TX_DATETIME"] = pd.to_datetime(
        frame["TX_TIME_SECONDS"], unit="s", origin=start_date
    )
    frame["TRANSACTION_ID"] = np.arange(len(frame), dtype=np.int64)
    return frame[[*TRANSACTION_COLUMNS, *extra_columns]]
