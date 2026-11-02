"""Configuration objects for the simulator.

Every default value lives here, next to a short note on why it was chosen. As in the
original paper, defaults are reasonable choices, not calibrated on real data.
"""

import datetime
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Optional

import numpy as np

KNOWN_FRAUD_SCENARIOS = (1, 2, 3, 4, 5)
# Scenarios of v0.1; scenario 5 (card testing) only exists with injection
TAMPER_FRAUD_SCENARIOS = (1, 2, 3, 4)
RATE_MODES = ("absolute", "proportional")
FRAUD_MECHANISMS = ("tamper", "inject")
SCHEDULE_KINDS = ("constant", "step", "ramp", "pulse")


@dataclass(frozen=True)
class IntensitySchedule:
    """Multiplier of a fraud scenario's daily rate over time (fraudster drift).

    Times are fractions of the simulated period, evaluated at the middle of each
    day, ``x = (d + 0.5) / nb_days``:

    - ``constant``: ``level`` every day;
    - ``step``: ``base`` before ``start``, ``level`` from ``start`` on;
    - ``ramp``: ``base`` before ``start``, linear from ``base`` to ``level`` over
      ``[start, end]``, ``level`` after;
    - ``pulse``: ``level`` in ``[start, end)``, ``base`` outside.

    Example: ``IntensitySchedule("step", start=0.6, level=0.5)`` halves the rate from
    60% of the period on.
    """

    kind: str = "constant"
    start: float = 0.0
    end: float = 1.0
    base: float = 1.0
    level: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_KINDS:
            raise ValueError(
                f"kind must be one of {SCHEDULE_KINDS}, got {self.kind!r}."
            )
        if not 0.0 <= self.start <= self.end <= 1.0:
            raise ValueError(
                f"Need 0 <= start <= end <= 1, got start={self.start}, end={self.end}."
            )
        if self.base < 0 or self.level < 0:
            raise ValueError("base and level must be >= 0.")

    def multipliers(self, nb_days: int) -> np.ndarray:
        """Rate multiplier of each of ``nb_days`` days."""
        x = (np.arange(nb_days) + 0.5) / nb_days
        if self.kind == "constant":
            return np.full(nb_days, float(self.level))
        if self.kind == "step":
            return np.where(x < self.start, self.base, self.level).astype(float)
        if self.kind == "pulse":
            inside = (x >= self.start) & (x < self.end)
            return np.where(inside, self.level, self.base).astype(float)
        # ramp
        width = self.end - self.start
        if width == 0:
            share = (x >= self.start).astype(float)
        else:
            share = np.clip((x - self.start) / width, 0.0, 1.0)
        return self.base + share * (self.level - self.base)


def _empty_schedule() -> Mapping[int, IntensitySchedule]:
    return {}


@dataclass(frozen=True)
class FraudConfig:
    """Parameters of the fraud scenarios.

    The defaults reproduce v0.1.0 exactly.

    Two mechanisms:

    - ``"tamper"`` (default; v0.1): scenario 1 labels the legitimate transactions
      above a fixed amount; scenarios 3 and 4 alter legitimate transactions.
    - ``"inject"`` (v2 engine only): fraudsters make *new* transactions, added to
      the customers' legitimate ones, which are never altered. Scenario 1 is a
      purchase far above the customer's recent amounts; scenario 2 labels the CP
      purchases at a compromised terminal and leaks some cards to scenario 3;
      scenario 3 makes bursts of CNP purchases; scenario 4 a fast sequence of
      purchases near the victim; scenario 5 (card testing) a few tiny CNP
      purchases, sometimes followed by scenario 3. Each scenario's daily rate can
      change over time (``intensity_schedule``).

    Args:
        scenarios: Fraud scenarios to apply (subset of 1-4; 1-5 with injection).
        mechanism: ``"tamper"`` or ``"inject"`` (see above).
        rate_mode: How many terminals/customers are compromised per day.
            ``"absolute"`` uses the fixed ``*_per_day`` counts (v0.1 behaviour);
            ``"proportional"`` draws each day's count from
            ``Poisson(fraction * population)`` with the ``*_fraction_per_day``
            values, so the expected fraud rate does not depend on the dataset size.
        s1_amount_threshold: Scenario 1 — transactions above this amount are fraud.
        s2_terminals_per_day: Scenario 2 — terminals compromised per day (absolute).
        s2_terminal_fraction_per_day: Scenario 2 — fraction of terminals compromised
            per day (proportional).
        s2_duration_days: Scenario 2 — days a terminal stays compromised.
        s3_customers_per_day: Scenario 3 — customers whose card number is stolen per
            day (absolute).
        s4_customers_per_day: Scenario 4 — customers whose physical card is stolen
            per day (absolute).
        s3_customer_fraction_per_day: Scenario 3 — fraction of customers per day
            (proportional).
        s4_customer_fraction_per_day: Scenario 4 — fraction of customers per day
            (proportional).
        s3_duration_days: Scenario 3 — days a customer stays compromised.
        s4_duration_days: Scenario 4 — days a customer stays compromised.
        s3_tamper_fraction: Scenario 3 — fraction of the compromised customer's
            transactions that become fraudulent.
        amount_multiplier: Scenarios 3 and 4 — factor applied to the amount of a
            tampered transaction.
        s4_cnp_prob: Scenario 4 — probability that a tampered transaction is CNP.

    Injection only (``mechanism="inject"``):
        s1_customers_per_day, s1_customer_fraction_per_day: Scenario 1 — victims per
            day (absolute / proportional).
        s5_customers_per_day, s5_customer_fraction_per_day: Scenario 5 — victims per
            day (absolute / proportional).
        history_days: Window of the victim's recent purchases used as reference
            (scenarios 1, 3 and 4).
        min_history_tx: Scenario 1 — a victim with fewer recent purchases is skipped
            (no reference amount).
        s1_quantile, s1_amount_factor: Scenario 1 — the fraud amount is above
            ``s1_amount_factor`` times the ``s1_quantile`` quantile of the victim's
            recent amounts...
        s1_amount_spread: ...namely ``factor * quantile * (1 + U * spread)``,
            ``U ~ Uniform(0, 1]``.
        s2_leak_prob: Scenario 2 — probability that a victim's card, read at a
            compromised terminal, is later used in scenario 3.
        s2_leak_delay_days_range: Scenario 2 — days until a leaked card is used.
        s3_mean_bursts: Scenario 3 — mean number of bursts per campaign
            (``1 + Poisson(mean - 1)``), on random days of ``s3_duration_days``.
        s3_burst_size_range: Scenario 3 — purchases per burst.
        s3_burst_hours: Scenario 3 — length of a burst.
        s3_amount_factor: Scenario 3 — amounts around this factor times the
            median of the victim's recent amounts.
        s4_max_txs, s4_block_prob_per_tx, s4_spending_limit: Scenario 4 — the
            sequence stops after ``s4_max_txs`` purchases, when the card is blocked
            (probability per purchase), when the next purchase would exceed the
            spending limit, or 24 h after it started.
        s4_mean_gap_minutes: Scenario 4 — mean time between purchases.
        s4_amount_factor: Scenario 4 — amounts around this factor times the median
            of the victim's recent amounts.
        s4_cnp_attempt_prob: Scenario 4 — probability that a purchase is a CNP
            attempt instead of CP.
        s5_n_tests_range: Scenario 5 — number of test purchases.
        s5_amount_range: Scenario 5 — amount of each test purchase.
        s5_mean_gap_seconds: Scenario 5 — mean time between test purchases.
        s5_escalate_prob: Scenario 5 — probability that the fraudster goes on to
            scenario 3 with the same card (same campaign).
        s5_escalation_delay_hours_range: Scenario 5 — delay until scenario 3.
        night_prob: Scenarios 3 and 5 — probability that a burst starts at night
            (0-6 h); otherwise uniform over 6-24 h.
        amount_log_sigma: Scenarios 3 and 4 — spread of the amounts (log scale).
        intensity_schedule: Per scenario, a multiplier of its daily rate over time.
    """

    scenarios: tuple[int, ...] = TAMPER_FRAUD_SCENARIOS
    rate_mode: str = "absolute"
    mechanism: str = "tamper"
    # Scenario 1 — high amount. v0.1 value; well above most customers' spending
    # (mean amounts are drawn from U(5, 100)).
    s1_amount_threshold: float = 220.0
    # Scenario 2 — compromised terminals. The proportional fraction equals the
    # absolute count at the default size (2 / 20 000 terminals).
    s2_terminals_per_day: int = 2
    s2_terminal_fraction_per_day: float = 1e-4
    s2_duration_days: int = 28
    # Scenarios 3 and 4 — compromised customers. The proportional fractions equal the
    # absolute counts at the default size (3 / 10 000 customers).
    s3_customers_per_day: int = 3
    s4_customers_per_day: int = 3
    s3_customer_fraction_per_day: float = 3e-4
    s4_customer_fraction_per_day: float = 3e-4
    s3_duration_days: int = 14
    s4_duration_days: int = 5
    # v0.1 tampers 1/3 of the stolen-card-number customer's transactions...
    s3_tamper_fraction: float = 1 / 3
    # ...and multiplies the tampered amounts by 5.
    amount_multiplier: float = 5.0
    # v0.1 draws CP/CNP uniformly for stolen physical cards.
    s4_cnp_prob: float = 0.5

    # --- Injection (v2 engine) ---
    # Same order of magnitude as scenarios 3 and 4 (3 / 10 000 customers).
    s1_customers_per_day: int = 3
    s1_customer_fraction_per_day: float = 3e-4
    s5_customers_per_day: int = 2
    s5_customer_fraction_per_day: float = 2e-4
    # About a month of purchases: long enough for a stable quantile, short enough
    # to follow the customer's drift.
    history_days: int = 30
    min_history_tx: int = 5
    # Twice the customer's 95th percentile: rare for the customer, whatever regime.
    s1_quantile: float = 0.95
    s1_amount_factor: float = 2.0
    s1_amount_spread: float = 0.5
    # Skimmed cards are sold and used days later; only some are used.
    s2_leak_prob: float = 0.05
    s2_leak_delay_days_range: tuple[int, int] = (1, 14)
    # A few bursts of 2-5 purchases within hours, well above the usual amount.
    s3_mean_bursts: float = 3.0
    s3_burst_size_range: tuple[int, int] = (2, 5)
    s3_burst_hours: float = 3.0
    s3_amount_factor: float = 2.0
    # A stolen card is used quickly, until it is blocked or the limit is reached.
    s4_max_txs: int = 10
    s4_block_prob_per_tx: float = 0.15
    s4_spending_limit: float = 2000.0
    s4_mean_gap_minutes: float = 20.0
    s4_amount_factor: float = 3.0
    s4_cnp_attempt_prob: float = 0.2
    # Card testing: a handful of tiny CNP purchases within minutes.
    s5_n_tests_range: tuple[int, int] = (3, 10)
    s5_amount_range: tuple[float, float] = (1.0, 5.0)
    s5_mean_gap_seconds: float = 60.0
    s5_escalate_prob: float = 0.5
    s5_escalation_delay_hours_range: tuple[float, float] = (1.0, 72.0)
    # Online fraud is more frequent at night.
    night_prob: float = 0.5
    amount_log_sigma: float = 0.3
    intensity_schedule: Mapping[int, IntensitySchedule] = field(
        default_factory=_empty_schedule, hash=False
    )

    @classmethod
    def realistic(cls, **overrides) -> "FraudConfig":
        """Injection with all five scenarios, rates proportional to the population
        and fraudster drift: scenario 2 halves from 60% of the period on (chip
        cards) and scenario 5 appears at 30% of the period. Any field can be
        overridden.
        """
        values: dict = {
            "scenarios": KNOWN_FRAUD_SCENARIOS,
            "rate_mode": "proportional",
            "mechanism": "inject",
            "intensity_schedule": {
                2: IntensitySchedule("step", start=0.6, base=1.0, level=0.5),
                5: IntensitySchedule("step", start=0.3, base=0.0, level=1.0),
            },
        }
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        unknown = set(scenarios) - set(KNOWN_FRAUD_SCENARIOS)
        if unknown:
            raise ValueError(
                f"Unknown fraud scenarios {sorted(unknown)}; "
                f"expected a subset of {KNOWN_FRAUD_SCENARIOS}."
            )
        if self.mechanism not in FRAUD_MECHANISMS:
            raise ValueError(
                f"mechanism must be one of {FRAUD_MECHANISMS}, got {self.mechanism!r}."
            )
        if self.mechanism == "tamper" and 5 in scenarios:
            raise ValueError('Scenario 5 requires mechanism="inject".')
        if len(set(scenarios)) != len(scenarios):
            raise ValueError(f"Duplicate fraud scenarios in {scenarios}.")
        # Accept any iterable (e.g. a list) but store a tuple, so the config stays
        # hashable and serialisable.
        object.__setattr__(self, "scenarios", scenarios)

        if self.rate_mode not in RATE_MODES:
            raise ValueError(
                f"rate_mode must be one of {RATE_MODES}, got {self.rate_mode!r}."
            )

        for name in (
            "s2_terminal_fraction_per_day",
            "s3_customer_fraction_per_day",
            "s4_customer_fraction_per_day",
            "s3_tamper_fraction",
            "s4_cnp_prob",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

        for name in ("s2_duration_days", "s3_duration_days", "s4_duration_days"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}.")

        for name in (
            "s2_terminals_per_day",
            "s3_customers_per_day",
            "s4_customers_per_day",
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value}.")

        if self.s1_amount_threshold < 0:
            raise ValueError(
                f"s1_amount_threshold must be >= 0, got {self.s1_amount_threshold}."
            )
        if self.amount_multiplier <= 0:
            raise ValueError(
                f"amount_multiplier must be > 0, got {self.amount_multiplier}."
            )
        self._validate_injection()

    def _validate_injection(self) -> None:
        for name in (
            "s1_customer_fraction_per_day",
            "s5_customer_fraction_per_day",
            "s1_quantile",
            "s2_leak_prob",
            "s4_block_prob_per_tx",
            "s4_cnp_attempt_prob",
            "s5_escalate_prob",
            "night_prob",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        for name in (
            "s1_customers_per_day",
            "s5_customers_per_day",
            "min_history_tx",
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value}.")
        for name in ("history_days", "s4_max_txs"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}.")
        for name in (
            "s1_amount_factor",
            "s3_burst_hours",
            "s3_amount_factor",
            "s4_spending_limit",
            "s4_mean_gap_minutes",
            "s4_amount_factor",
            "s5_mean_gap_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}.")
        if self.s1_amount_spread < 0 or self.amount_log_sigma < 0:
            raise ValueError("s1_amount_spread and amount_log_sigma must be >= 0.")
        if self.s3_mean_bursts < 1:
            raise ValueError(f"s3_mean_bursts must be >= 1, got {self.s3_mean_bursts}.")
        _check_range(
            "s2_leak_delay_days_range", self.s2_leak_delay_days_range, 0, np.inf, True
        )
        _check_range("s3_burst_size_range", self.s3_burst_size_range, 1, np.inf, True)
        _check_range("s5_n_tests_range", self.s5_n_tests_range, 1, np.inf, True)
        _check_range("s5_amount_range", self.s5_amount_range, 0.01, np.inf, False)
        _check_range(
            "s5_escalation_delay_hours_range",
            self.s5_escalation_delay_hours_range,
            0.0,
            np.inf,
            False,
        )

        schedule = dict(self.intensity_schedule)
        unknown = set(schedule) - set(KNOWN_FRAUD_SCENARIOS)
        if unknown:
            raise ValueError(f"intensity_schedule has unknown scenarios {unknown}.")
        if any(not isinstance(s, IntensitySchedule) for s in schedule.values()):
            raise ValueError("intensity_schedule values must be IntensitySchedule.")
        if schedule and self.mechanism != "inject":
            raise ValueError('intensity_schedule requires mechanism="inject".')
        # Read-only copy, so the frozen config cannot be changed through it
        object.__setattr__(self, "intensity_schedule", MappingProxyType(schedule))

    def daily_counts(
        self,
        scenario: int,
        population: int,
        n_days: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Number of terminals (scenario 2) or customers (other scenarios)
        compromised on each of ``n_days`` days.

        ``"absolute"`` returns the fixed ``*_per_day`` count every day.
        ``"proportional"`` draws each day's count from
        ``Poisson(fraction * population)``, so the expected number per day is
        proportional to the population and may be zero; ``rng`` is required.
        Disabled scenarios return zeros.

        With an ``intensity_schedule`` for the scenario, the expected count of day
        ``d`` is multiplied by ``m(d)``: the Poisson mean in ``"proportional"``
        mode; in ``"absolute"`` mode, ``count * m(d)`` is rounded up or down at
        random, keeping its mean (``rng`` required).
        Scenarios 1 and 5 have daily counts only with injection.
        """
        if scenario in (1, 5) and self.mechanism != "inject":
            raise ValueError(
                f'Scenario {scenario} has daily counts only with mechanism="inject".'
            )
        if scenario == 1:
            count, fraction = (
                self.s1_customers_per_day,
                self.s1_customer_fraction_per_day,
            )
        elif scenario == 5:
            count, fraction = (
                self.s5_customers_per_day,
                self.s5_customer_fraction_per_day,
            )
        elif scenario == 2:
            count, fraction = (
                self.s2_terminals_per_day,
                self.s2_terminal_fraction_per_day,
            )
        elif scenario == 3:
            count, fraction = (
                self.s3_customers_per_day,
                self.s3_customer_fraction_per_day,
            )
        elif scenario == 4:
            count, fraction = (
                self.s4_customers_per_day,
                self.s4_customer_fraction_per_day,
            )
        else:
            raise ValueError(f"scenario must be one of 1-5, got {scenario}.")

        if scenario not in self.scenarios:
            return np.zeros(n_days, dtype=int)
        schedule = self.intensity_schedule.get(scenario)
        if self.rate_mode == "absolute" and schedule is None:
            return np.full(n_days, count, dtype=int)
        if rng is None:
            raise ValueError(
                'rate_mode="proportional" and intensity_schedule require an rng.'
            )
        if schedule is None:
            return rng.poisson(fraction * population, size=n_days)
        multipliers = schedule.multipliers(n_days)
        if self.rate_mode == "absolute":
            expected = count * multipliers
            return np.floor(expected + rng.random(n_days)).astype(int)
        return rng.poisson(fraction * population * multipliers, size=n_days)


# --- Concept drift (Phase 2) ------------------------------------------------------

DRIFT_TYPES = ("sudden", "gradual", "incremental", "recurrent")


def _check_range(name: str, value, low: float, high: float, integer: bool) -> None:
    if len(value) != 2 or not low <= value[0] <= value[1] <= high:
        raise ValueError(
            f"{name} must be (a, b) with {low} <= a <= b <= {high}, got {value}."
        )
    if integer and not all(isinstance(v, numbers.Integral) for v in value):
        raise ValueError(f"{name} must hold integers, got {value}.")


@dataclass(frozen=True)
class Shock:
    """A temporary, population-wide change (e.g. Black Friday).

    Args:
        start_date: First day of the shock (``YYYY-MM-DD``).
        duration_days: Number of days the shock lasts.
        frequency_multiplier: Factor on every customer's transactions per day.
        amount_multiplier: Factor on every transaction amount.
    """

    start_date: str
    duration_days: int = 1
    frequency_multiplier: float = 1.0
    amount_multiplier: float = 1.0

    def __post_init__(self) -> None:
        datetime.datetime.strptime(self.start_date, "%Y-%m-%d")  # validates the date
        if (
            not isinstance(self.duration_days, numbers.Integral)
            or self.duration_days <= 0
        ):
            raise ValueError(
                f"duration_days must be a positive integer, got {self.duration_days}."
            )
        if self.frequency_multiplier < 0 or self.amount_multiplier <= 0:
            raise ValueError("Shock multipliers must be positive.")


@dataclass(frozen=True)
class GlobalDrift:
    """Population drift: changes that affect every customer.

    Args:
        amount_trend_per_year: Yearly relative growth of every amount (e.g. 0.05 for
            5% inflation), applied continuously: factor ``(1 + r) ** (day / 365.25)``.
        shocks: Temporary changes at given dates.
    """

    amount_trend_per_year: float = 0.0
    shocks: tuple[Shock, ...] = ()

    def __post_init__(self) -> None:
        if self.amount_trend_per_year <= -1:
            raise ValueError("amount_trend_per_year must be > -1.")
        object.__setattr__(self, "shocks", tuple(self.shocks))


def _default_type_weights() -> Mapping[str, float]:
    return {"sudden": 0.3, "gradual": 0.3, "incremental": 0.2, "recurrent": 0.2}


@dataclass(frozen=True)
class DriftConfig:
    """Customer concept drift: some customers move from regime A to regime B.

    Regime A is the customer's original profile; regime B is drawn from A:
    ``mean_amount`` and ``mean_nb_tx_per_day`` are multiplied by log-normal factors,
    ``cnp_prob`` is shifted on the logit scale and, with probability
    ``relocation_prob``, the customer moves to another city. The weight of regime B,
    ``w(t)`` in ``[0, 1]``, depends on the drift type:

    - ``sudden``: 0 before ``t0``, 1 from ``t0`` on;
    - ``gradual``: rises linearly from 0 to 1 over ``[t0, t0 + L]``; each transaction
      comes from regime B with probability ``w(t)`` (mixture of two concepts);
    - ``incremental``: same ``w(t)``, but the parameters are interpolated,
      ``theta(t) = (1 - w) theta_A + w theta_B`` (one concept moving);
    - ``recurrent``: from ``t0`` on, regime B is active for ``D`` days in every
      period of ``P`` days (e.g. holidays, travel).

    Args:
        customer_fraction: Fraction of customers with drift (exact count, rounded).
        type_weights: Probability of each drift type among drifting customers.
        start_day_range: Range of ``t0``, as a fraction of ``nb_days``.
        transition_days_range: Range of ``L`` in days (gradual and incremental).
        recurrent_period_days_range: Range of ``P`` in days (recurrent).
        recurrent_active_days_range: Range of ``D`` in days (recurrent); must be
            shorter than the shortest period.
        amount_log_mu, amount_log_sigma: Regime B amount factor ~ LogNormal(mu, sigma).
        frequency_log_sigma: Regime B frequency factor ~ LogNormal(0, sigma).
        cnp_logit_sigma: Regime B ``cnp_prob = sigmoid(logit(p_A) + N(0, sigma))``.
        category_shift_range: Shift of category preferences (used with merchants,
            Phase 4; ignored for now).
        relocation_prob: Probability that regime B lives in another city.
        global_drift: Optional population drift, applied to every customer.
    """

    customer_fraction: float = 0.3
    type_weights: Mapping[str, float] = field(
        default_factory=_default_type_weights, hash=False
    )
    # t0 between 20% and 80% of the period, so both regimes are observed.
    start_day_range: tuple[float, float] = (0.2, 0.8)
    # One week to one month of transition.
    transition_days_range: tuple[int, int] = (7, 30)
    # Episodes every one to three months, lasting one to two weeks (holidays, travel).
    recurrent_period_days_range: tuple[int, int] = (30, 90)
    recurrent_active_days_range: tuple[int, int] = (5, 15)
    # sigma = 0.5: regime B spends between ~0.4x and ~2.7x as much (95% range).
    amount_log_mu: float = 0.0
    amount_log_sigma: float = 0.5
    # sigma = 0.4: between ~0.5x and ~2.2x as many transactions (95% range).
    frequency_log_sigma: float = 0.4
    cnp_logit_sigma: float = 1.0
    category_shift_range: tuple[float, float] = (0.3, 0.8)
    relocation_prob: float = 0.05
    global_drift: Optional[GlobalDrift] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.customer_fraction <= 1.0:
            raise ValueError(
                f"customer_fraction must be in [0, 1], got {self.customer_fraction}."
            )
        weights = dict(self.type_weights)
        unknown = set(weights) - set(DRIFT_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown drift types {sorted(unknown)}; expected {DRIFT_TYPES}."
            )
        if any(w < 0 for w in weights.values()):
            raise ValueError("type_weights must be non-negative.")
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError(
                f"type_weights must sum to 1, got {sum(weights.values())}."
            )
        # Read-only copy, so the frozen config cannot be changed through it
        object.__setattr__(self, "type_weights", MappingProxyType(weights))

        _check_range("start_day_range", self.start_day_range, 0.0, 1.0, False)
        _check_range(
            "transition_days_range", self.transition_days_range, 1, np.inf, True
        )
        _check_range(
            "recurrent_period_days_range",
            self.recurrent_period_days_range,
            1,
            np.inf,
            True,
        )
        _check_range(
            "recurrent_active_days_range",
            self.recurrent_active_days_range,
            1,
            np.inf,
            True,
        )
        if self.recurrent_active_days_range[1] >= self.recurrent_period_days_range[0]:
            raise ValueError(
                "recurrent_active_days_range must be shorter than the shortest period."
            )
        _check_range("category_shift_range", self.category_shift_range, 0.0, 1.0, False)
        for name in ("amount_log_sigma", "frequency_log_sigma", "cnp_logit_sigma"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")
        if not 0.0 <= self.relocation_prob <= 1.0:
            raise ValueError(
                f"relocation_prob must be in [0, 1], got {self.relocation_prob}."
            )


# --- Seasonality (Phase 2) ----------------------------------------------------------


@dataclass(frozen=True)
class SeasonalityConfig:
    """Calendar patterns. They repeat every week/month/year, so they are stationary
    and are *not* concept drift; they apply to every customer.

    Args:
        weekday_frequency: Transactions-per-day factor for Monday..Sunday.
        payday_days: Days of the month with a payday peak.
        payday_last_business_day: Also peak on the last weekday of each month.
        payday_frequency_multiplier: Transactions-per-day factor on paydays.
        december_frequency_multiplier: Transactions-per-day factor in December.
        december_amount_multiplier: Amount factor in December.
    """

    # More purchases at the end of the week, fewer on Sundays; mean = 1.
    weekday_frequency: tuple[float, ...] = (0.95, 0.95, 1.0, 1.0, 1.1, 1.15, 0.85)
    # Salaries are often paid on the 5th and 20th (advance) and at month end.
    payday_days: tuple[int, ...] = (5, 20)
    payday_last_business_day: bool = True
    payday_frequency_multiplier: float = 1.3
    # Christmas shopping: more and larger purchases.
    december_frequency_multiplier: float = 1.2
    december_amount_multiplier: float = 1.15

    def __post_init__(self) -> None:
        object.__setattr__(self, "weekday_frequency", tuple(self.weekday_frequency))
        object.__setattr__(self, "payday_days", tuple(self.payday_days))
        if len(self.weekday_frequency) != 7 or min(self.weekday_frequency) < 0:
            raise ValueError("weekday_frequency must hold 7 non-negative factors.")
        if any(not 1 <= d <= 31 for d in self.payday_days):
            raise ValueError("payday_days must be days of the month (1-31).")
        for name in (
            "payday_frequency_multiplier",
            "december_frequency_multiplier",
            "december_amount_multiplier",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0.")
