"""Fraud injection (v2 engine, ``FraudConfig(mechanism="inject")``).

Fraudsters make *new* transactions, added to the customers' legitimate ones; a
legitimate transaction is never altered nor labelled as fraud. Each scenario draws
from its own sub-stream of the ``fraud`` stream.

- Scenario 1 (anomalous amount): each victim gets one purchase far above its recent
  amounts: ``s1_amount_factor`` times the ``s1_quantile`` quantile of the victim's
  purchases in the ``history_days`` before the day, times ``1 + U * s1_amount_spread``.
  With fewer than ``min_history_tx`` recent purchases, the quantile of all legitimate
  purchases of the period is used. Everything else (type, terminal for CP,
  addresses, time-of-day model) copies a purchase of the victim, so only the amount
  is anomalous.
- Scenario 2 (skimming): a terminal is compromised for ``s2_duration_days``; it reads
  the card of each customer making a CP purchase there (the purchase stays
  legitimate; ``GT_COMPROMISE_ID`` marks it). Some ``s2_use_delay_days_range`` days
  later, a read card is cloned with probability ``s2_clone_prob`` (a fast sequence of
  CP purchases near the fraudster, like scenario 4) or, with probability
  ``s2_leak_prob``, used in a scenario-3 campaign. Both frauds carry the
  ``GT_COMPROMISE_ID`` of the read.
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

FRAUD_GT_COLUMNS = (
    "GT_FRAUD_SCENARIO",
    "GT_FRAUDSTER_ID",
    "GT_CAMPAIGN_ID",
    "GT_COMPROMISE_ID",
)
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
    compromise: int = -1  # scenario-2 read that leaked the card


@dataclass
class _Injected:
    """Columns of the injected transactions."""

    columns: dict[str, list] = field(default_factory=dict)

    def add(self, **values) -> None:
        n = len(values["TX_TIME_SECONDS"])
        values.setdefault("GT_COMPROMISE_ID", -1)
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
        self.compromises = 0

    def campaign(self) -> int:
        self.campaigns += 1
        return self.campaigns - 1

    def fraudster(self) -> int:
        self.fraudsters += 1
        return self.fraudsters - 1

    def compromise(self) -> int:
        self.compromises += 1
        return self.compromises - 1


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

    def nearest(self, customer: int, t: float) -> Optional[int]:
        """The customer's last purchase before ``t``, else its first one after;
        ``None`` if the customer has no purchase."""
        lo, hi = self._span(customer, -np.inf, t)
        if hi > self.starts[customer]:
            return int(self.order[hi - 1])
        if self.stops[customer] > self.starts[customer]:
            return int(self.order[self.starts[customer]])
        return None

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
    """Return the injected transactions.

    ``tx`` holds the legitimate transactions in chronological order, with
    ``GT_COMPROMISE_ID`` (``-1``), which scenario 2 fills in place for the purchases
    whose card was read at a compromised terminal.
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
        s3 += _scenario_2(
            out, history, config, terminals, loc_features, n_days, rngs[2], ids
        )
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


def _nearby_terminals(
    lat: float, long: float, terminals, radius_km: float, nearest_fallback: bool
) -> np.ndarray:
    """Terminals within ``radius_km`` of a point; with ``nearest_fallback``, the
    nearest terminal when none is that close."""
    nearby = np.asarray(
        sampling.points_within_radius(
            np.array([lat]), np.array([long]), terminals[0], terminals[1], radius_km
        )[0],
        dtype=np.int64,
    )
    if len(nearby) == 0 and nearest_fallback:
        distances = sampling.haversine_km(
            np.array(lat), np.array(long), terminals[0], terminals[1]
        )
        nearby = np.array([int(np.argmin(distances))])
    return nearby


def _card_sequence(
    out: _Injected,
    customer: int,
    tau: int,
    location: tuple[float, float],
    nearby: np.ndarray,
    reference: float,
    cnp_attempt_prob: float,
    bill: tuple[float, float],
    terminals,
    config: FraudConfig,
    rng: np.random.Generator,
    **ground_truth,
) -> None:
    """A fraudster holding a card (stolen or cloned) buys fast, from ``tau`` on,
    until ``s4_max_txs``, a block, the spending limit or 24 h. CP purchases at the
    ``nearby`` terminals; CNP attempts (probability ``cnp_attempt_prob``, or always
    if there is no nearby terminal) at random terminals, shipped to ``location``."""
    mean_gap = config.s4_mean_gap_minutes * 60
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
        cnp.append(len(nearby) == 0 or rng.random() < cnp_attempt_prob)
        total += amount
        if rng.random() < config.s4_block_prob_per_tx:
            break
        t += rng.exponential(mean_gap)
        if t >= tau + SECONDS_PER_DAY:
            break
    if not seconds:
        return
    is_cnp = np.array(cnp)
    n = len(seconds)
    terminal = rng.integers(0, len(terminals[0]), n)
    if len(nearby):
        terminal[~is_cnp] = nearby[rng.integers(0, len(nearby), int((~is_cnp).sum()))]
    term_lat, term_long = terminals[0][terminal], terminals[1][terminal]
    out.add(
        TX_TIME_SECONDS=np.array(seconds, dtype=np.int64),
        CUSTOMER_ID=customer,
        TERMINAL_ID=terminal,
        TX_AMOUNT=np.array(amounts),
        IS_CNP=is_cnp,
        TX_SHIPP_LAT=np.where(is_cnp, location[0], term_lat),
        TX_SHIPP_LONG=np.where(is_cnp, location[1], term_long),
        TX_BILL_LAT=bill[0],
        TX_BILL_LONG=bill[1],
        TX_TERM_LAT=term_lat,
        TX_TERM_LONG=term_long,
        **ground_truth,
    )


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
    # Reference for victims with too little recent history: all legitimate purchases
    # of the period (``tx`` holds only legitimate transactions here)
    population_quantile = (
        float(np.quantile(tx["TX_AMOUNT"], config.s1_quantile))
        if len(tx["TX_AMOUNT"])
        else 0.0
    )
    for day in range(n_days):
        for customer in _victims(n_customers, counts[day], rng):
            customer = int(customer)
            day_start = day * SECONDS_PER_DAY
            rows = history.recent(customer, day_start, config.history_days)
            if len(rows) >= max(config.min_history_tx, 1):
                quantile = float(np.quantile(tx["TX_AMOUNT"][rows], config.s1_quantile))
            else:
                quantile = population_quantile
            threshold = config.s1_amount_factor * quantile
            u = 1.0 - rng.random()  # (0, 1]
            amount = _cents_up(
                np.array([threshold * (1 + u * config.s1_amount_spread)])
            )
            # Purchase whose context is copied: a recent one, else the nearest in
            # time; a customer without any purchase gets a CNP purchase from its
            # profile
            model: Optional[int] = (
                int(rows[rng.integers(len(rows))])
                if len(rows)
                else history.nearest(customer, day_start)
            )
            if model is not None:
                is_cnp = np.array([tx["IS_CNP"][model]])
                shipping = (tx["TX_SHIPP_LAT"][model], tx["TX_SHIPP_LONG"][model])
                billing = (tx["TX_BILL_LAT"][model], tx["TX_BILL_LONG"][model])
            else:
                is_cnp = np.array([True])
                billing = history.billing_at(customer, day_start)
                shipping = billing
            seconds = day_start + draw_times_of_day(is_cnp, rng)
            terminal = (
                np.array([rng.integers(len(terminals[0]))])
                if is_cnp[0] or model is None
                else np.array([tx["TERMINAL_ID"][model]])
            )
            out.add(
                TX_TIME_SECONDS=seconds,
                CUSTOMER_ID=customer,
                TERMINAL_ID=terminal,
                TX_AMOUNT=amount,
                IS_CNP=is_cnp,
                TX_SHIPP_LAT=shipping[0],
                TX_SHIPP_LONG=shipping[1],
                TX_BILL_LAT=billing[0],
                TX_BILL_LONG=billing[1],
                TX_TERM_LAT=terminals[0][terminal],
                TX_TERM_LONG=terminals[1][terminal],
                TX_FRAUD_SCENARIO=1,
                GT_FRAUDSTER_ID=ids.fraudster(),
                GT_CAMPAIGN_ID=ids.campaign(),
            )


def _scenario_2(
    out: _Injected,
    history: _History,
    config: FraudConfig,
    terminals,
    loc_features: LocationFeatures,
    n_days: int,
    rng: np.random.Generator,
    ids: _Ids,
) -> list[_S3Campaign]:
    """Compromised terminals read cards; cloned cards are used near the fraudster.
    Returns the scenario-3 campaigns of the leaked cards."""
    tx = history.tx
    n_terminals = len(terminals[0])
    counts = config.daily_counts(2, n_terminals, n_days, rng)
    keys = tx["TERMINAL_ID"]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.searchsorted(sorted_keys, np.arange(n_terminals), side="left")
    stops = np.searchsorted(sorted_keys, np.arange(n_terminals), side="right")
    days = tx["TX_TIME_DAYS"]
    low, high = config.s2_use_delay_days_range
    compromised_until = np.zeros(n_terminals, dtype=np.int64)
    leaked: list[_S3Campaign] = []
    for day in range(n_days):
        n = min(int(counts[day]), n_terminals)
        for terminal in rng.choice(n_terminals, size=n, replace=False):
            if compromised_until[terminal] > day:
                continue  # already compromised
            compromised_until[terminal] = day + config.s2_duration_days
            rows = order[starts[terminal] : stops[terminal]]
            rows = rows[
                (days[rows] >= day)
                & (days[rows] < day + config.s2_duration_days)
                & ~tx["IS_CNP"][rows]
            ]
            # Each customer's card is read once: at its first purchase there
            customers, first = np.unique(tx["CUSTOMER_ID"][rows], return_index=True)
            reads = rows[first]
            fraudster = ids.fraudster()
            location: Optional[tuple[float, float]] = None
            u = rng.random(len(reads))
            delays = rng.integers(low, high + 1, len(reads))
            for customer, row, ui, delay in zip(customers, reads, u, delays):
                compromise = ids.compromise()
                tx["GT_COMPROMISE_ID"][row] = compromise
                use_day = int(days[row]) + int(delay)
                if use_day >= n_days:
                    continue
                customer = int(customer)
                if ui < config.s2_clone_prob:
                    if location is None:
                        lat, long = _fraudster_locations(1, loc_features, rng)
                        location = (float(lat[0]), float(long[0]))
                    tau = use_day * SECONDS_PER_DAY + int(
                        rng.integers(0, SECONDS_PER_DAY)
                    )
                    _card_sequence(
                        out,
                        customer,
                        tau,
                        location,
                        _nearby_terminals(
                            location[0],
                            location[1],
                            terminals,
                            loc_features.available_terminals_radius,
                            nearest_fallback=True,
                        ),
                        history.reference_amount(customer, tau, config),
                        0.0,  # a cloned card is a physical card: CP only
                        history.billing_at(customer, tau),
                        terminals,
                        config,
                        rng,
                        TX_FRAUD_SCENARIO=2,
                        GT_FRAUDSTER_ID=fraudster,
                        GT_CAMPAIGN_ID=ids.campaign(),
                        GT_COMPROMISE_ID=compromise,
                    )
                elif ui < config.s2_clone_prob + config.s2_leak_prob:
                    leaked.append(
                        _S3Campaign(
                            customer,
                            use_day * SECONDS_PER_DAY,
                            fraudster,
                            ids.campaign(),
                            compromise=compromise,
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
            GT_COMPROMISE_ID=c.compromise,
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
    for day in range(n_days):
        for customer in _victims(n_customers, counts[day], rng):
            customer = int(customer)
            tau = day * SECONDS_PER_DAY + int(rng.integers(0, SECONDS_PER_DAY))
            bill = history.billing_at(customer, tau)
            # The fraudster is in the victim's city
            lat, long = sampling.random_points_around(
                np.array([bill[0]]),
                np.array([bill[1]]),
                loc_features.city_radius,
                loc_features.data["brazil_polygon"],
                rng,
            )
            location = (float(lat[0]), float(long[0]))
            _card_sequence(
                out,
                customer,
                tau,
                location,
                _nearby_terminals(
                    location[0],
                    location[1],
                    terminals,
                    loc_features.available_terminals_radius,
                    nearest_fallback=False,
                ),
                history.reference_amount(customer, tau, config),
                config.s4_cnp_attempt_prob,
                bill,
                terminals,
                config,
                rng,
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
