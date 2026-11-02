"""Fraud injection (v2 engine, ``FraudConfig(mechanism="inject")``).

Fraudsters make *new* transactions, added to the customers' legitimate ones; a
legitimate transaction is never altered, and only scenario 2 labels legitimate
transactions (the CP purchases at a compromised terminal). Each scenario draws from
its own sub-stream of the ``fraud`` stream.

- Scenario 1 (anomalous amount): each victim gets one purchase far above its recent
  amounts: ``s1_amount_factor`` times the ``s1_quantile`` quantile of the victim's
  purchases in the ``history_days`` before the day, times ``1 + U * s1_amount_spread``.
  Everything else (type, terminal for CP, addresses, time-of-day model) copies a
  random recent purchase of the victim, so only the amount is anomalous. A victim
  with fewer than ``min_history_tx`` recent purchases is skipped.
- Scenario 2 (skimming): a compromised terminal for ``s2_duration_days``; its CP
  purchases are labelled, and each victim's card leaks with probability
  ``s2_leak_prob`` into a scenario-3 campaign some days later (same fraudster).
- Scenario 3 (card number stolen): ``1 + Poisson(s3_mean_bursts - 1)`` bursts on
  random days of ``s3_duration_days``; each burst has 2-5 CNP purchases within
  ``s3_burst_hours``, shipped to the fraudster, at night with probability
  ``night_prob``, with amounts around ``s3_amount_factor`` times the victim's recent
  median.
- Scenario 4 (physical card stolen): from an instant ``tau``, purchases every
  ``Exp(s4_mean_gap_minutes)`` near the victim's city (CP, or a CNP attempt with
  probability ``s4_cnp_attempt_prob``), until ``s4_max_txs``, a block
  (``s4_block_prob_per_tx`` after each purchase), the spending limit or 24 h.
- Scenario 5 (card testing): 3-10 CNP purchases of R$ 1-5 within minutes; with
  probability ``s5_escalate_prob`` the fraudster goes on to scenario 3 with the same
  card some hours later, in the same campaign.

Each fraud has a fraudster (``GT_FRAUDSTER_ID``) and belongs to a campaign
(``GT_CAMPAIGN_ID``): the transactions of one attack.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..config import FraudConfig
from ..location import sampling
from ..location.features import LocationFeatures
from .transactions import SECONDS_PER_DAY, draw_times_of_day

FRAUD_GT_COLUMNS = ("GT_FRAUD_SCENARIO", "GT_FRAUDSTER_ID", "GT_CAMPAIGN_ID")
NIGHT_END = 6 * 3600


@dataclass
class _S3Campaign:
    """A planned scenario-3 campaign (direct, leaked by scenario 2, or escalated
    from scenario 5)."""

    customer: int
    start: int  # seconds
    fraudster: int
    campaign: int
    # Escalation from card testing: the first burst starts at ``start``, and the
    # purchases are shipped to the tester's location
    first_burst_at_start: bool = False
    location: Optional[tuple[float, float]] = None


@dataclass
class _Injected:
    """Columns of the injected transactions."""

    columns: dict[str, list] = field(default_factory=dict)

    def add(self, **values) -> None:
        n = len(values["TX_TIME_SECONDS"])
        for key, value in values.items():
            self.columns.setdefault(key, []).append(
                np.broadcast_to(np.asarray(value), (n,))
            )

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: np.concatenate(parts) for key, parts in self.columns.items()}


class _Ids:
    def __init__(self) -> None:
        self.campaigns = 0
        self.fraudsters = 0

    def campaign(self) -> int:
        self.campaigns += 1
        return self.campaigns - 1

    def fraudster(self) -> int:
        self.fraudsters += 1
        return self.fraudsters - 1


class _History:
    """The legitimate purchases of each customer, in chronological order."""

    def __init__(self, tx: dict, customers: pd.DataFrame):
        self.tx = tx
        keys = tx["CUSTOMER_ID"]
        self.order = np.argsort(keys, kind="stable")  # chronological per customer
        sorted_keys = keys[self.order]
        ids = np.arange(len(customers))
        self.starts = np.searchsorted(sorted_keys, ids, side="left")
        self.stops = np.searchsorted(sorted_keys, ids, side="right")
        self.seconds = tx["TX_TIME_SECONDS"][self.order]
        self.mean_amount = customers["mean_amount"].to_numpy()
        self.billing = (
            customers["billing_lat"].to_numpy(),
            customers["billing_long"].to_numpy(),
        )

    def _span(self, customer: int, start: float, stop: float) -> tuple[int, int]:
        lo, hi = self.starts[customer], self.stops[customer]
        times = self.seconds[lo:hi]
        return (
            lo + int(np.searchsorted(times, start, side="left")),
            lo + int(np.searchsorted(times, stop, side="left")),
        )

    def recent(self, customer: int, t: float, days: int) -> np.ndarray:
        """Rows of the customer's purchases in ``[t - days, t)``."""
        lo, hi = self._span(customer, t - days * SECONDS_PER_DAY, t)
        return self.order[lo:hi]

    def reference_amount(self, customer: int, t: float, config: FraudConfig) -> float:
        """Median of the recent amounts; the profile mean with too little history."""
        rows = self.recent(customer, t, config.history_days)
        if len(rows) >= max(config.min_history_tx, 1):
            return float(np.median(self.tx["TX_AMOUNT"][rows]))
        return float(self.mean_amount[customer])

    def billing_at(self, customer: int, t: float) -> tuple[float, float]:
        """Billing address of the last purchase before ``t`` (it changes when a
        drifting customer moves), or the profile's."""
        lo, hi = self._span(customer, -np.inf, t)
        if hi > self.starts[customer]:
            row = self.order[hi - 1]
            return float(self.tx["TX_BILL_LAT"][row]), float(
                self.tx["TX_BILL_LONG"][row]
            )
        return float(self.billing[0][customer]), float(self.billing[1][customer])


def inject_frauds(
    tx: dict,
    customers: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    loc_features: LocationFeatures,
    config: FraudConfig,
    rngs: dict[int, np.random.Generator],
    n_days: int,
) -> dict[str, np.ndarray]:
    """Label scenario 2 in ``tx`` (in place) and return the injected transactions.

    ``tx`` holds the legitimate transactions in chronological order, with
    ``TX_FRAUD``, ``TX_FRAUD_SCENARIO``, ``GT_FRAUDSTER_ID`` and ``GT_CAMPAIGN_ID``.
    """
    terminals = (
        terminal_profiles["terminal_lat"].to_numpy(),
        terminal_profiles["terminal_long"].to_numpy(),
    )
    history = _History(tx, customers)
    ids = _Ids()
    out = _Injected()
    s3: list[_S3Campaign] = []
    scenarios = config.scenarios
    end = n_days * SECONDS_PER_DAY

    if 2 in scenarios:
        s3 += _scenario_2(tx, config, len(terminals[0]), n_days, rngs[2], ids)
    if 5 in scenarios:
        s3 += _scenario_5(
            out, history, config, terminals, loc_features, n_days, rngs[5], ids
        )
    if 1 in scenarios:
        _scenario_1(out, history, config, terminals, n_days, rngs[1], ids)
    if 3 in scenarios:
        counts = config.daily_counts(3, len(customers), n_days, rngs[3])
        for day in range(n_days):
            for customer in _victims(len(customers), counts[day], rngs[3]):
                s3.append(
                    _S3Campaign(
                        int(customer),
                        day * SECONDS_PER_DAY,
                        ids.fraudster(),
                        ids.campaign(),
                    )
                )
    if s3:
        _scenario_3(out, history, config, terminals, loc_features, s3, rngs[3])
    if 4 in scenarios:
        _scenario_4(out, history, config, terminals, loc_features, n_days, rngs[4], ids)

    if not out.columns:
        return {}
    injected = out.arrays()
    inside = injected["TX_TIME_SECONDS"] < end
    return {key: value[inside] for key, value in injected.items()}


def _victims(n_customers: int, count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_customers, size=min(int(count), n_customers), replace=False)


def _night_biased_times(n: int, config: FraudConfig, rng: np.random.Generator):
    """Seconds into the day: ``[0, 6 h)`` with probability ``night_prob``, else
    ``[6 h, 24 h)``."""
    night = rng.random(n) < config.night_prob
    u = rng.random(n)
    return np.where(
        night, u * NIGHT_END, NIGHT_END + u * (SECONDS_PER_DAY - NIGHT_END)
    ).astype(np.int64)


def _cents_up(amount: np.ndarray) -> np.ndarray:
    # Rounded up, so an amount drawn above a threshold stays above it
    return np.ceil(np.round(amount * 100, 6)) / 100


def _fraudster_locations(
    n: int, loc_features: LocationFeatures, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random points around population-weighted cities anywhere in Brazil."""
    points = sampling.random_points_around_cities(
        n,
        loc_features.city_radius,
        loc_features.data["brazil_polygon"],
        loc_features.data["brazil_cities"],
        rng,
    )["random_points"]
    return points["lat"].to_numpy(), points["long"].to_numpy()


def _scenario_1(
    out: _Injected,
    history: _History,
    config: FraudConfig,
    terminals,
    n_days: int,
    rng: np.random.Generator,
    ids: _Ids,
) -> None:
    tx = history.tx
    n_customers = len(history.mean_amount)
    counts = config.daily_counts(1, n_customers, n_days, rng)
    for day in range(n_days):
        for customer in _victims(n_customers, counts[day], rng):
            day_start = day * SECONDS_PER_DAY
            rows = history.recent(int(customer), day_start, config.history_days)
            if len(rows) < config.min_history_tx or len(rows) == 0:
                continue
            threshold = config.s1_amount_factor * float(
                np.quantile(tx["TX_AMOUNT"][rows], config.s1_quantile)
            )
            u = 1.0 - rng.random()  # (0, 1]
            amount = _cents_up(
                np.array([threshold * (1 + u * config.s1_amount_spread)])
            )
            model = int(
                rows[rng.integers(len(rows))]
            )  # purchase whose context is copied
            is_cnp = np.array([tx["IS_CNP"][model]])
            seconds = day_start + draw_times_of_day(is_cnp, rng)
            terminal = (
                np.array([rng.integers(len(terminals[0]))])
                if is_cnp[0]
                else np.array([tx["TERMINAL_ID"][model]])
            )
            out.add(
                TX_TIME_SECONDS=seconds,
                CUSTOMER_ID=int(customer),
                TERMINAL_ID=terminal,
                TX_AMOUNT=amount,
                IS_CNP=is_cnp,
                TX_SHIPP_LAT=tx["TX_SHIPP_LAT"][model],
                TX_SHIPP_LONG=tx["TX_SHIPP_LONG"][model],
                TX_BILL_LAT=tx["TX_BILL_LAT"][model],
                TX_BILL_LONG=tx["TX_BILL_LONG"][model],
                TX_TERM_LAT=terminals[0][terminal],
                TX_TERM_LONG=terminals[1][terminal],
                TX_FRAUD_SCENARIO=1,
                GT_FRAUDSTER_ID=ids.fraudster(),
                GT_CAMPAIGN_ID=ids.campaign(),
            )


def _scenario_2(
    tx: dict,
    config: FraudConfig,
    n_terminals: int,
    n_days: int,
    rng: np.random.Generator,
    ids: _Ids,
) -> list[_S3Campaign]:
    """Label the CP purchases at compromised terminals; return the leaked cards."""
    counts = config.daily_counts(2, n_terminals, n_days, rng)
    keys = tx["TERMINAL_ID"]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.searchsorted(sorted_keys, np.arange(n_terminals), side="left")
    stops = np.searchsorted(sorted_keys, np.arange(n_terminals), side="right")
    days = tx["TX_TIME_DAYS"]
    low, high = config.s2_leak_delay_days_range
    leaked: list[_S3Campaign] = []
    for day in range(n_days):
        n = min(int(counts[day]), n_terminals)
        for terminal in rng.choice(n_terminals, size=n, replace=False):
            rows = order[starts[terminal] : stops[terminal]]
            rows = rows[
                (days[rows] >= day)
                & (days[rows] < day + config.s2_duration_days)
                & ~tx["IS_CNP"][rows]
                & (tx["TX_FRAUD"][rows] == 0)
            ]
            fraudster, campaign = ids.fraudster(), ids.campaign()
            tx["TX_FRAUD"][rows] = 1
            tx["TX_FRAUD_SCENARIO"][rows] = 2
            tx["GT_FRAUDSTER_ID"][rows] = fraudster
            tx["GT_CAMPAIGN_ID"][rows] = campaign
            # Each victim's card is read once (its first purchase there)
            victims, first = np.unique(tx["CUSTOMER_ID"][rows], return_index=True)
            leak = rng.random(len(victims)) < config.s2_leak_prob
            delays = rng.integers(low, high + 1, len(victims))
            for customer, row, delay in zip(
                victims[leak], rows[first][leak], delays[leak]
            ):
                start_day = int(days[row]) + int(delay)
                leaked.append(
                    _S3Campaign(
                        int(customer),
                        start_day * SECONDS_PER_DAY,
                        fraudster,
                        ids.campaign(),
                    )
                )
    return leaked


def _scenario_3(
    out: _Injected,
    history: _History,
    config: FraudConfig,
    terminals,
    loc_features: LocationFeatures,
    campaigns: list[_S3Campaign],
    rng: np.random.Generator,
) -> None:
    missing = [c for c in campaigns if c.location is None]
    lat, long = _fraudster_locations(len(missing), loc_features, rng)
    for campaign, la, lo in zip(missing, lat, long):
        campaign.location = (float(la), float(lo))

    low, high = config.s3_burst_size_range
    for c in campaigns:
        assert c.location is not None
        n_bursts = 1 + int(rng.poisson(config.s3_mean_bursts - 1))
        first_day = c.start // SECONDS_PER_DAY
        burst_days = first_day + rng.integers(0, config.s3_duration_days, n_bursts)
        starts = burst_days * SECONDS_PER_DAY + _night_biased_times(
            n_bursts, config, rng
        )
        starts = np.maximum(starts, c.start)
        if c.first_burst_at_start:
            starts[0] = c.start
        sizes = rng.integers(low, high + 1, n_bursts)
        offsets = [
            np.concatenate(
                [[0.0], np.sort(rng.uniform(0, config.s3_burst_hours * 3600, s - 1))]
            )
            for s in sizes
        ]
        seconds = np.concatenate(
            [start + offset for start, offset in zip(starts, offsets)]
        ).astype(np.int64)
        n = len(seconds)
        reference = history.reference_amount(c.customer, c.start, config)
        amounts = _cents_up(
            reference
            * config.s3_amount_factor
            * rng.lognormal(0.0, config.amount_log_sigma, n)
        )
        terminal = rng.integers(0, len(terminals[0]), n)
        bill_lat, bill_long = history.billing_at(c.customer, c.start)
        out.add(
            TX_TIME_SECONDS=seconds,
            CUSTOMER_ID=c.customer,
            TERMINAL_ID=terminal,
            TX_AMOUNT=amounts,
            IS_CNP=True,
            TX_SHIPP_LAT=c.location[0],
            TX_SHIPP_LONG=c.location[1],
            TX_BILL_LAT=bill_lat,
            TX_BILL_LONG=bill_long,
            TX_TERM_LAT=terminals[0][terminal],
            TX_TERM_LONG=terminals[1][terminal],
            TX_FRAUD_SCENARIO=3,
            GT_FRAUDSTER_ID=c.fraudster,
            GT_CAMPAIGN_ID=c.campaign,
        )


def _scenario_4(
    out: _Injected,
    history: _History,
    config: FraudConfig,
    terminals,
    loc_features: LocationFeatures,
    n_days: int,
    rng: np.random.Generator,
    ids: _Ids,
) -> None:
    n_customers = len(history.mean_amount)
    counts = config.daily_counts(4, n_customers, n_days, rng)
    mean_gap = config.s4_mean_gap_minutes * 60
    for day in range(n_days):
        for customer in _victims(n_customers, counts[day], rng):
            customer = int(customer)
            tau = day * SECONDS_PER_DAY + int(rng.integers(0, SECONDS_PER_DAY))
            bill_lat, bill_long = history.billing_at(customer, tau)
            # The fraudster is in the victim's city
            lat, long = sampling.random_points_around(
                np.array([bill_lat]),
                np.array([bill_long]),
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
            reference = history.reference_amount(customer, tau, config)

            seconds, amounts, cnp = [], [], []
            t, total = float(tau), 0.0
            for _ in range(config.s4_max_txs):
                amount = float(
                    _cents_up(
                        np.array(
                            [
                                reference
                                * config.s4_amount_factor
                                * rng.lognormal(0.0, config.amount_log_sigma)
                            ]
                        )
                    )[0]
                )
                if total + amount > config.s4_spending_limit:
                    break
                seconds.append(int(t))
                amounts.append(amount)
                cnp.append(
                    len(nearby) == 0 or rng.random() < config.s4_cnp_attempt_prob
                )
                total += amount
                if rng.random() < config.s4_block_prob_per_tx:
                    break
                t += rng.exponential(mean_gap)
                if t >= tau + SECONDS_PER_DAY:
                    break
            if not seconds:
                continue
            is_cnp = np.array(cnp)
            n = len(seconds)
            terminal = rng.integers(0, len(terminals[0]), n)
            if len(nearby):
                terminal[~is_cnp] = nearby[
                    rng.integers(0, len(nearby), int((~is_cnp).sum()))
                ]
            term_lat, term_long = terminals[0][terminal], terminals[1][terminal]
            out.add(
                TX_TIME_SECONDS=np.array(seconds, dtype=np.int64),
                CUSTOMER_ID=customer,
                TERMINAL_ID=terminal,
                TX_AMOUNT=np.array(amounts),
                IS_CNP=is_cnp,
                TX_SHIPP_LAT=np.where(is_cnp, lat[0], term_lat),
                TX_SHIPP_LONG=np.where(is_cnp, long[0], term_long),
                TX_BILL_LAT=bill_lat,
                TX_BILL_LONG=bill_long,
                TX_TERM_LAT=term_lat,
                TX_TERM_LONG=term_long,
                TX_FRAUD_SCENARIO=4,
                GT_FRAUDSTER_ID=ids.fraudster(),
                GT_CAMPAIGN_ID=ids.campaign(),
            )


def _scenario_5(
    out: _Injected,
    history: _History,
    config: FraudConfig,
    terminals,
    loc_features: LocationFeatures,
    n_days: int,
    rng: np.random.Generator,
    ids: _Ids,
) -> list[_S3Campaign]:
    """Card testing; returns the scenario-3 campaigns that follow it."""
    n_customers = len(history.mean_amount)
    counts = config.daily_counts(5, n_customers, n_days, rng)
    low, high = config.s5_n_tests_range
    min_amount, max_amount = config.s5_amount_range
    min_delay, max_delay = config.s5_escalation_delay_hours_range
    plans = [
        (day, int(customer))
        for day in range(n_days)
        for customer in _victims(n_customers, counts[day], rng)
    ]
    lat, long = _fraudster_locations(len(plans), loc_features, rng)
    escalated: list[_S3Campaign] = []
    for (day, customer), la, lo in zip(plans, lat, long):
        n = int(rng.integers(low, high + 1))
        start = day * SECONDS_PER_DAY + int(_night_biased_times(1, config, rng)[0])
        gaps = rng.exponential(config.s5_mean_gap_seconds, n - 1)
        seconds = (start + np.concatenate([[0.0], np.cumsum(gaps)])).astype(np.int64)
        amounts = np.round(rng.uniform(min_amount, max_amount, n), 2)
        terminal = rng.integers(0, len(terminals[0]), n)
        bill_lat, bill_long = history.billing_at(customer, start)
        fraudster, campaign = ids.fraudster(), ids.campaign()
        out.add(
            TX_TIME_SECONDS=seconds,
            CUSTOMER_ID=customer,
            TERMINAL_ID=terminal,
            TX_AMOUNT=amounts,
            IS_CNP=True,
            TX_SHIPP_LAT=la,
            TX_SHIPP_LONG=lo,
            TX_BILL_LAT=bill_lat,
            TX_BILL_LONG=bill_long,
            TX_TERM_LAT=terminals[0][terminal],
            TX_TERM_LONG=terminals[1][terminal],
            TX_FRAUD_SCENARIO=5,
            GT_FRAUDSTER_ID=fraudster,
            GT_CAMPAIGN_ID=campaign,
        )
        if rng.random() < config.s5_escalate_prob:
            delay = rng.uniform(min_delay, max_delay) * 3600
            escalated.append(
                _S3Campaign(
                    customer,
                    int(seconds[-1] + delay),
                    fraudster,
                    campaign,
                    first_burst_at_start=True,
                    location=(float(la), float(lo)),
                )
            )
    return escalated
