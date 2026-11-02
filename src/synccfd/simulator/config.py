"""Configuration objects for the simulator.

Every default value lives here, next to a short note on why it was chosen. As in the
original paper, defaults are reasonable choices, not calibrated on real data.
"""

import numbers
from dataclasses import dataclass
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
