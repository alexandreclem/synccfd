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
"""

import datetime

import numpy as np
import pandas as pd

from ..location import sampling
from ..location.features import LocationFeatures

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
    pending = np.arange(len(is_cnp))
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


def _customer_transactions(
    customer: dict,
    available: np.ndarray,
    nb_days: int,
    terminal_lat: np.ndarray,
    terminal_long: np.ndarray,
    loc_features: LocationFeatures,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    counts = rng.poisson(customer["mean_nb_tx_per_day"], nb_days)
    n = int(counts.sum())
    days = np.repeat(np.arange(nb_days, dtype=np.int64), counts)

    mean, std = customer["mean_amount"], customer["std_amount"]
    amounts = rng.normal(mean, std, n)
    negative = amounts < 0
    amounts[negative] = rng.uniform(0, 2 * mean, int(negative.sum()))
    amounts = np.round(amounts, 2)

    if len(available):
        is_cnp = rng.random(n) < customer["cnp_prob"]
    else:
        is_cnp = np.ones(n, dtype=bool)

    cnp_terminals = rng.integers(0, len(terminal_lat), n)
    if len(available):
        cp_terminals = available[rng.integers(0, len(available), n)]
    else:
        cp_terminals = cnp_terminals  # unused: every transaction is CNP
    terminals = np.where(is_cnp, cnp_terminals, cp_terminals)
    term_lat, term_long = terminal_lat[terminals], terminal_long[terminals]

    bill_lat = np.full(n, customer["billing_lat"])
    bill_long = np.full(n, customer["billing_long"])
    u = rng.random(n)
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

    seconds = draw_times_of_day(is_cnp, rng) + days * SECONDS_PER_DAY

    return {
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


def generate_transactions(
    customer_profiles: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    loc_features: LocationFeatures,
    nb_days: int,
    start_date: datetime.datetime,
    customer_rngs: list[np.random.Generator],
) -> pd.DataFrame:
    """Legitimate transactions of all customers, in chronological order.

    ``customer_rngs[i]`` is the random stream of the i-th customer.
    """
    terminal_lat = terminal_profiles["terminal_lat"].to_numpy()
    terminal_long = terminal_profiles["terminal_long"].to_numpy()
    parts = [
        _customer_transactions(
            customer,
            np.asarray(available, dtype=np.int64),
            nb_days,
            terminal_lat,
            terminal_long,
            loc_features,
            rng,
        )
        for customer, available, rng in zip(
            customer_profiles.to_dict("records"),
            customer_profiles["available_terminals"],
            customer_rngs,
        )
    ]
    columns = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return assemble_transactions(columns, start_date)


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
