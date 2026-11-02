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

KNOWN_FRAUD_SCENARIOS = (1, 2, 3, 4)
RATE_MODES = ("absolute", "proportional")


@dataclass(frozen=True)
class FraudConfig:
    """Parameters of the fraud scenarios.

    The defaults reproduce v0.1.0 exactly.

    Args:
        scenarios: Fraud scenarios to apply (subset of 1-4).
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
    """

    scenarios: tuple[int, ...] = KNOWN_FRAUD_SCENARIOS
    rate_mode: str = "absolute"
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

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        unknown = set(scenarios) - set(KNOWN_FRAUD_SCENARIOS)
        if unknown:
            raise ValueError(
                f"Unknown fraud scenarios {sorted(unknown)}; "
                f"expected a subset of {KNOWN_FRAUD_SCENARIOS}."
            )
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

    def daily_counts(
        self,
        scenario: int,
        population: int,
        n_days: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Number of terminals (scenario 2) or customers (scenarios 3 and 4)
        compromised on each of ``n_days`` days.

        ``"absolute"`` returns the fixed ``*_per_day`` count every day.
        ``"proportional"`` draws each day's count from
        ``Poisson(fraction * population)``, so the expected number per day is
        proportional to the population and may be zero; ``rng`` is required.
        Disabled scenarios return zeros.
        """
        if scenario == 2:
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
            raise ValueError(f"scenario must be 2, 3 or 4, got {scenario}.")

        if scenario not in self.scenarios:
            return np.zeros(n_days, dtype=int)
        if self.rate_mode == "absolute":
            return np.full(n_days, count, dtype=int)
        if rng is None:
            raise ValueError('rate_mode="proportional" requires an rng.')
        return rng.poisson(fraction * population, size=n_days)


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
