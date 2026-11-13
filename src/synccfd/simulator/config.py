"""Configuration of the simulator.

Defaults are reasonable choices, not calibrated on real data.
"""

import dataclasses
import datetime
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Optional

import numpy as np

KNOWN_FRAUD_SCENARIOS = (1, 2, 3, 4, 5)
RATE_MODES = ("absolute", "proportional")
SCHEDULE_KINDS = ("constant", "step", "ramp", "pulse")


@dataclass(frozen=True)
class IntensitySchedule:
    """Multiplier of a fraud scenario's daily rate over time (fraudster drift).

    Args:
        kind: "constant" (level every day), "step" (base, then level from start),
            "ramp" (base, then linear up to level between start and end) or
            "pulse" (level between start and end, base outside).
        start, end: Fractions of the simulated period.

    Example: IntensitySchedule("step", start=0.6, level=0.5) halves the rate from
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
        """Rate multiplier of each day."""
        x = (np.arange(nb_days) + 0.5) / nb_days  # Middle of each day
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

    Fraudsters add new transactions; legitimate ones are never altered. Scenario 1
    is an anomalous amount, 2 skimming, 3 a stolen card number, 4 a stolen physical
    card and 5 card testing. FraudConfig.realistic() adds fraudster drift. The
    category parameters (s1_categories, s3_category_schedule, s4_categories,
    s5_categories) are used only with merchants.

    Args:
        scenarios: Scenarios to apply (subset of 1 to 5). Leaked cards (scenario 2)
            and escalations (scenario 5) need scenario 3 enabled.
        rate_mode: "proportional" draws each day's count from a Poisson with mean
            fraction * population (the *_fraction_per_day fields); "absolute" uses
            the fixed *_per_day counts.
        history_days: Window of recent purchases used as the victim's reference.
        min_history_tx: Scenario 1: with fewer recent purchases, the quantile of
            all legitimate purchases is used instead.
        s1_amount_spread: Scenario 1: the amount is factor * quantile * (1 + u *
            spread), with u uniform in (0, 1].
        s2_clone_prob, s2_leak_prob: Scenario 2: probability that a read card is
            cloned (CP purchases near the fraudster) or leaked to scenario 3.
        s3_category_schedule: Scenario 3: ((start, categories), ...), with start
            as a fraction of the period.
        s4_block_prob_per_tx: Scenario 4: probability that the card is blocked
            after each purchase.
        s5_escalate_prob: Scenario 5: probability that the fraudster goes on to
            scenario 3 with the same card.
        night_prob: Scenarios 3 and 5: probability that a burst starts between 0 h
            and 6 h.
        amount_log_sigma: Scenarios 3 and 4: spread of the amounts (log scale).
        intensity_schedule: Per scenario, an IntensitySchedule of its daily rate.
    """

    scenarios: tuple[int, ...] = KNOWN_FRAUD_SCENARIOS
    # Proportional: the expected fraud rate does not depend on the dataset size
    rate_mode: str = "proportional"
    # Scenario 2: compromised terminals (2 out of 20 000 at the default size)
    s2_terminals_per_day: int = 2
    s2_terminal_fraction_per_day: float = 1e-4
    s2_duration_days: int = 28
    # Scenarios 3 and 4: compromised customers (3 out of 10 000 at the default size)
    s3_customers_per_day: int = 3
    s4_customers_per_day: int = 3
    s3_customer_fraction_per_day: float = 3e-4
    s4_customer_fraction_per_day: float = 3e-4
    s3_duration_days: int = 14
    s4_duration_days: int = 5
    # Scenarios 1 and 5: same order of magnitude as scenarios 3 and 4
    s1_customers_per_day: int = 3
    s1_customer_fraction_per_day: float = 3e-4
    s5_customers_per_day: int = 2
    s5_customer_fraction_per_day: float = 2e-4
    # About a month: enough for a stable quantile, short enough to follow drift
    history_days: int = 30
    min_history_tx: int = 5
    # Twice the customer's 95th percentile: rare for the customer, whatever regime
    s1_quantile: float = 0.95
    s1_amount_factor: float = 2.0
    s1_amount_spread: float = 0.5
    # Only some skimmed cards are used, days later: mostly cloned, sometimes leaked
    s2_clone_prob: float = 0.2
    s2_leak_prob: float = 0.05
    s2_use_delay_days_range: tuple[int, int] = (1, 14)
    # A few bursts of 2 to 5 purchases within hours, well above the usual amount
    s3_mean_bursts: float = 3.0
    s3_burst_size_range: tuple[int, int] = (2, 5)
    s3_burst_hours: float = 3.0
    s3_amount_factor: float = 2.0
    # A stolen card is used quickly, until it is blocked or the limit is reached
    s4_max_txs: int = 10
    s4_block_prob_per_tx: float = 0.15
    s4_spending_limit: float = 2000.0
    s4_mean_gap_minutes: float = 20.0
    s4_amount_factor: float = 3.0
    s4_cnp_attempt_prob: float = 0.2
    # Card testing: a handful of tiny CNP purchases within minutes
    s5_n_tests_range: tuple[int, int] = (3, 10)
    s5_amount_range: tuple[float, float] = (1.0, 5.0)
    s5_mean_gap_seconds: float = 60.0
    s5_escalate_prob: float = 0.5
    s5_escalation_delay_hours_range: tuple[float, float] = (1.0, 72.0)
    # Online fraud is more frequent at night
    night_prob: float = 0.5
    amount_log_sigma: float = 0.3
    intensity_schedule: Mapping[int, IntensitySchedule] = field(
        default_factory=_empty_schedule, hash=False
    )
    # Used only with merchants
    # Scenario 1: a category where large amounts exist
    s1_categories: tuple[str, ...] = ("eletronicos", "viagem_turismo")
    # Scenario 3: resellable or digital goods; the targets can change over time
    s3_category_schedule: tuple[tuple[float, tuple[str, ...]], ...] = (
        (
            0.0,
            (
                "eletronicos",
                "ecommerce_marketplace",
                "servicos_digitais",
                "viagem_turismo",
            ),
        ),
    )
    # Scenario 4 and cloned cards (scenario 2): goods that are easy to resell
    s4_categories: tuple[str, ...] = (
        "eletronicos",
        "posto_combustivel",
        "farmacia",
        "supermercado",
    )
    # Scenario 5: card testing in digital services
    s5_categories: tuple[str, ...] = ("servicos_digitais",)

    @classmethod
    def realistic(cls, **overrides) -> "FraudConfig":
        """All five scenarios with fraudster drift; any field can be overridden."""
        values: dict = {
            # Scenario 2 halves at 60% of the period (chip cards), scenario 5
            # appears at 30%, and scenario 3 moves to digital services at 50%
            "intensity_schedule": {
                2: IntensitySchedule("step", start=0.6, base=1.0, level=0.5),
                5: IntensitySchedule("step", start=0.3, base=0.0, level=1.0),
            },
            "s3_category_schedule": (
                (0.0, ("eletronicos", "ecommerce_marketplace")),
                (0.5, ("servicos_digitais", "ecommerce_marketplace")),
            ),
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
        if len(set(scenarios)) != len(scenarios):
            raise ValueError(f"Duplicate fraud scenarios in {scenarios}.")
        # Store a tuple, so the config stays hashable and serialisable
        object.__setattr__(self, "scenarios", scenarios)

        if self.rate_mode not in RATE_MODES:
            raise ValueError(
                f"rate_mode must be one of {RATE_MODES}, got {self.rate_mode!r}."
            )

        for name in (
            "s2_terminal_fraction_per_day",
            "s3_customer_fraction_per_day",
            "s4_customer_fraction_per_day",
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

        for name in (
            "s1_customer_fraction_per_day",
            "s5_customer_fraction_per_day",
            "s1_quantile",
            "s2_clone_prob",
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
        if self.s2_clone_prob + self.s2_leak_prob > 1:
            raise ValueError("s2_clone_prob + s2_leak_prob must be <= 1.")
        _check_range(
            "s2_use_delay_days_range", self.s2_use_delay_days_range, 0, np.inf, True
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
        # Read-only copy, so the frozen config cannot be changed through it
        object.__setattr__(self, "intensity_schedule", MappingProxyType(schedule))

        for name in ("s1_categories", "s4_categories", "s5_categories"):
            value = tuple(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        steps = tuple((float(t), tuple(c)) for t, c in self.s3_category_schedule)
        starts = [t for t, _ in steps]
        if not steps or starts[0] != 0.0 or starts != sorted(starts) or starts[-1] > 1:
            raise ValueError(
                "s3_category_schedule must start at 0.0, with increasing starts <= 1."
            )
        if any(not categories for _, categories in steps):
            raise ValueError("s3_category_schedule needs categories at every step.")
        object.__setattr__(self, "s3_category_schedule", steps)

    def fraud_categories(self) -> set[str]:
        """Every category named by the scenarios (checked against the merchants)."""
        names = set(self.s1_categories) | set(self.s4_categories)
        names |= set(self.s5_categories)
        for _, categories in self.s3_category_schedule:
            names |= set(categories)
        return names

    def daily_counts(
        self,
        scenario: int,
        population: int,
        n_days: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Terminals (scenario 2) or customers compromised on each day.

        rng is required in proportional mode and with an intensity schedule.
        """
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
            # Round up or down at random, keeping the mean
            return np.floor(expected + rng.random(n_days)).astype(int)
        return rng.poisson(fraction * population * multipliers, size=n_days)


# Concept drift

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
    """A temporary change for every customer, such as Black Friday.

    Args:
        start_date: First day, as YYYY-MM-DD.
    """

    start_date: str
    duration_days: int = 1
    frequency_multiplier: float = 1.0
    amount_multiplier: float = 1.0

    def __post_init__(self) -> None:
        datetime.datetime.strptime(self.start_date, "%Y-%m-%d")  # Validates the date
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
    """Changes that affect every customer.

    Args:
        amount_trend_per_year: Yearly growth of every amount (0.05 for 5%
            inflation), applied continuously.
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

    Regime B is drawn from the customer's profile (regime A). The drift types are
    "sudden", "gradual" (each transaction comes from B with a growing
    probability), "incremental" (the parameters move from A to B) and "recurrent"
    (B is active some days in every period).

    Args:
        customer_fraction: Fraction of customers with drift.
        type_weights: Probability of each drift type.
        start_day_range: Range of the drift start, as a fraction of the period.
        transition_days_range: Transition length (gradual and incremental).
        recurrent_active_days_range: Must be shorter than the shortest period.
        amount_log_mu, amount_log_sigma: Log-normal factor of the regime B amount.
        frequency_log_sigma: Log-normal factor of the regime B frequency.
        cnp_logit_sigma: Shift of the regime B CNP probability, on the logit scale.
        category_shift_range: How far regime B preferences move (with merchants).
        relocation_prob: Probability that regime B lives in another city.
        global_drift: Optional drift applied to every customer.
    """

    customer_fraction: float = 0.3
    type_weights: Mapping[str, float] = field(
        default_factory=_default_type_weights, hash=False
    )
    # Start between 20% and 80% of the period, so both regimes are observed
    start_day_range: tuple[float, float] = (0.2, 0.8)
    # One week to one month of transition
    transition_days_range: tuple[int, int] = (7, 30)
    # Episodes every one to three months, lasting one to two weeks (holidays, travel)
    recurrent_period_days_range: tuple[int, int] = (30, 90)
    recurrent_active_days_range: tuple[int, int] = (5, 15)
    # Regime B spends between about 0.4x and 2.7x as much (95% range)
    amount_log_mu: float = 0.0
    amount_log_sigma: float = 0.5
    # Between about 0.5x and 2.2x as many transactions (95% range)
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


# Seasonality


@dataclass(frozen=True)
class SeasonalityConfig:
    """Calendar patterns for every customer (stationary, so not concept drift).

    Args:
        weekday_frequency: Factor on transactions per day, Monday to Sunday.
        payday_last_business_day: Also peak on the last weekday of each month.
    """

    # More purchases at the end of the week, fewer on Sundays (mean 1)
    weekday_frequency: tuple[float, ...] = (0.95, 0.95, 1.0, 1.0, 1.1, 1.15, 0.85)
    # Salaries are often paid on the 5th and 20th (advance) and at month end
    payday_days: tuple[int, ...] = (5, 20)
    payday_last_business_day: bool = True
    payday_frequency_multiplier: float = 1.3
    # Christmas shopping: more and larger purchases
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


# Merchants

CHANNELS = ("CP", "CNP", "CP+CNP")


def hourly_profile(
    peaks: tuple[tuple[float, float, float], ...], base: float = 0.05
) -> tuple[float, ...]:
    """24 hourly weights summing to 1: base plus (hour, width, height) bumps."""
    hours = np.arange(24) + 0.5
    weights = np.full(24, base)
    for hour, width, height in peaks:
        distance = np.abs(hours - hour)
        distance = np.minimum(distance, 24 - distance)  # Hours wrap at midnight
        weights += height * np.exp(-0.5 * (distance / width) ** 2)
    return tuple(float(w) for w in weights / weights.sum())


@dataclass(frozen=True)
class MerchantCategory:
    """A merchant category, inspired by the MCC.

    Args:
        channel: "CP" (physical only), "CNP" (online only) or "CP+CNP".
        median_amount: Median purchase amount (R$).
        log_sigma: Spread of the amounts around the median (log scale).
        weight: Share of the purchases.
        hourly_profile: 24 weights of the hour of a purchase.
        opening_hours: (open, close) hours of CP purchases; close may be 24.
        weekend_factor: Multiplier of the preference on weekends.
        skimming_risk: Relative chance of being compromised (scenario 2).
        online_fraction: Mean share of online purchases ("CP+CNP" only).
    """

    name: str
    channel: str
    median_amount: float
    log_sigma: float
    weight: float
    hourly_profile: tuple[float, ...]
    opening_hours: tuple[int, int] = (0, 24)
    weekend_factor: float = 1.0
    skimming_risk: float = 1.0
    online_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(
                f"channel must be one of {CHANNELS}, got {self.channel!r}."
            )
        if self.median_amount <= 0 or self.log_sigma < 0 or self.weight <= 0:
            raise ValueError(
                f"{self.name}: median_amount and weight must be > 0, log_sigma >= 0."
            )
        profile = tuple(float(w) for w in self.hourly_profile)
        if len(profile) != 24 or min(profile) < 0 or sum(profile) <= 0:
            raise ValueError(f"{self.name}: hourly_profile needs 24 weights >= 0.")
        object.__setattr__(self, "hourly_profile", profile)
        start, stop = self.opening_hours
        if not 0 <= start < stop <= 24:
            raise ValueError(
                f"{self.name}: opening_hours needs 0 <= open < close <= 24."
            )
        if self.has_cp and sum(profile[start:stop]) <= 0:
            raise ValueError(f"{self.name}: no CP purchase possible in opening hours.")
        if self.weekend_factor < 0 or self.skimming_risk < 0:
            raise ValueError(f"{self.name}: weekend_factor and skimming_risk >= 0.")
        if not 0.0 < self.online_fraction < 1.0:
            raise ValueError(f"{self.name}: online_fraction must be in (0, 1).")

    @property
    def has_cp(self) -> bool:
        """Whether the category has physical (CP) merchants."""
        return self.channel in ("CP", "CP+CNP")

    @property
    def has_cnp(self) -> bool:
        """Whether the category has online (CNP) merchants."""
        return self.channel in ("CNP", "CP+CNP")


# Typical purchases in Brazil, not calibrated on real data. Arguments: name,
# channel, median, sigma, weight, hourly profile, opening hours, weekend factor,
# skimming risk and, for mixed categories, the share of online purchases
DEFAULT_CATEGORIES: tuple[MerchantCategory, ...] = (
    MerchantCategory(
        "supermercado",
        "CP+CNP",
        120,
        0.7,
        0.20,
        hourly_profile(((18, 2.5, 1.0), (11, 2.5, 0.5))),
        (8, 21),
        1.2,
        online_fraction=0.10,
    ),
    MerchantCategory(
        "restaurante_bar",
        "CP",
        60,
        0.6,
        0.15,
        hourly_profile(((12.5, 1.2, 1.0), (20.5, 1.8, 1.0))),
        (11, 24),
        1.4,
        2.0,
    ),
    MerchantCategory(
        "posto_combustivel",
        "CP",
        150,
        0.4,
        0.10,
        hourly_profile((), base=1.0),
        (6, 23),
        1.0,
        3.0,
    ),
    MerchantCategory(
        "farmacia",
        "CP+CNP",
        70,
        0.6,
        0.08,
        hourly_profile(((15, 4.0, 1.0),)),
        (8, 22),
        online_fraction=0.15,
    ),
    MerchantCategory(
        "vestuario",
        "CP+CNP",
        180,
        0.7,
        0.08,
        hourly_profile(((16, 2.5, 1.0),)),
        (10, 22),
        1.3,
        online_fraction=0.30,
    ),
    MerchantCategory(
        "eletronicos",
        "CP+CNP",
        900,
        0.8,
        0.04,
        hourly_profile(((16, 3.0, 1.0),)),
        (10, 22),
        1.1,
        online_fraction=0.50,
    ),
    MerchantCategory(
        "viagem_turismo",
        "CNP",
        1200,
        0.7,
        0.03,
        hourly_profile(((15, 4.0, 1.0),), base=0.02),
    ),
    MerchantCategory(
        "ecommerce_marketplace",
        "CNP",
        150,
        0.9,
        0.15,
        hourly_profile(((10.5, 1.5, 1.0), (21.5, 1.5, 1.0))),
        weekend_factor=1.1,
    ),
    MerchantCategory(
        "servicos_digitais",
        "CNP",
        35,
        0.5,
        0.10,
        hourly_profile(((22, 3.0, 1.0),), base=0.15),
    ),
    MerchantCategory(
        "transporte_app",
        "CNP",
        25,
        0.5,
        0.07,
        hourly_profile(((8, 1.2, 1.0), (18.5, 1.5, 1.0))),
        weekend_factor=0.7,
    ),
)


@dataclass(frozen=True)
class MerchantConfig:
    """Merchants and categories.

    Args:
        extra_terminals_mean: Mean number of extra terminals per physical merchant.
        cnp_merchants_per_category: Online merchants in each category with CNP.
        cnp_popularity_exponent: Zipf exponent of the online merchants' popularity.
        preference_concentration: Small values concentrate customers on few
            categories.
        spending_factor_sigma: Spread of the customers' spending factors.
        loyalty: Probability of buying at a favourite merchant of the category.
        n_favorites: Favourite merchants per customer and category.
    """

    categories: tuple[MerchantCategory, ...] = DEFAULT_CATEGORIES
    # Most merchants have one terminal, some a few (mean 1.5)
    extra_terminals_mean: float = 0.5
    cnp_merchants_per_category: int = 20
    cnp_popularity_exponent: float = 1.0
    # Preferences spread around the global weights, not too uniform
    preference_concentration: float = 10.0
    # Most customers spend between about 0.4x and 2.7x the median
    spending_factor_sigma: float = 0.5
    # Customers often return to the same places
    loyalty: float = 0.6
    n_favorites: int = 2

    def __post_init__(self) -> None:
        categories = tuple(self.categories)
        object.__setattr__(self, "categories", categories)
        names = [c.name for c in categories]
        if not categories or len(set(names)) != len(names):
            raise ValueError("categories must be non-empty, with unique names.")
        if not any(c.has_cp for c in categories):
            raise ValueError("At least one category must accept CP purchases.")
        if self.extra_terminals_mean < 0 or self.spending_factor_sigma < 0:
            raise ValueError(
                "extra_terminals_mean, spending_factor_sigma must be >= 0."
            )
        if self.preference_concentration <= 0 or self.cnp_popularity_exponent < 0:
            raise ValueError(
                "preference_concentration must be > 0, cnp_popularity_exponent >= 0."
            )
        if not 0.0 <= self.loyalty <= 1.0:
            raise ValueError(f"loyalty must be in [0, 1], got {self.loyalty}.")
        for name in ("cnp_merchants_per_category", "n_favorites"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}.")

    @property
    def names(self) -> tuple[str, ...]:
        """The category names, in order."""
        return tuple(c.name for c in self.categories)

    def weights(self) -> np.ndarray:
        """Global category weights, normalised."""
        w = np.array([c.weight for c in self.categories], dtype=float)
        return w / w.sum()


def to_jsonable(value):
    """A configuration as plain JSON types (dataclasses get a "type" key)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {"type": type(value).__name__}
        for f in dataclasses.fields(value):
            out[f.name] = to_jsonable(getattr(value, f.name))
        return out
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return value
