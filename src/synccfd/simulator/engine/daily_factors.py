"""Per-day factors that apply to every customer: seasonality and global drift.

Seasonality (weekday, paydays, December) is stationary, so it is *not* concept
drift; global drift (amount trend, shocks) is population drift. Both are kept apart
from the customer drift of ``drift.py`` and labelled separately.
"""

import calendar
import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import GlobalDrift, SeasonalityConfig


@dataclass
class DailyFactors:
    """Multipliers per day ``0..nb_days-1``."""

    frequency: np.ndarray
    amount: np.ndarray
    shock: np.ndarray  # True on days inside a global-drift shock

    @property
    def is_neutral(self) -> bool:
        return bool(
            np.all(self.frequency == 1.0)
            and np.all(self.amount == 1.0)
            and not self.shock.any()
        )


def _last_business_day(year: int, month: int) -> int:
    day = calendar.monthrange(year, month)[1]
    while calendar.weekday(year, month, day) >= 5:  # Saturday or Sunday
        day -= 1
    return day


def daily_factors(
    nb_days: int,
    start_date: datetime.datetime,
    seasonality: Optional[SeasonalityConfig],
    global_drift: Optional[GlobalDrift],
) -> DailyFactors:
    """Frequency and amount factors of each day, from the calendar and global drift."""
    frequency = np.ones(nb_days)
    amount = np.ones(nb_days)
    shock = np.zeros(nb_days, dtype=bool)
    dates = [start_date + datetime.timedelta(days=d) for d in range(nb_days)]

    if seasonality is not None:
        for d, date in enumerate(dates):
            frequency[d] *= seasonality.weekday_frequency[date.weekday()]
            paydays = set(seasonality.payday_days)
            if seasonality.payday_last_business_day:
                paydays.add(_last_business_day(date.year, date.month))
            if date.day in paydays:
                frequency[d] *= seasonality.payday_frequency_multiplier
            if date.month == 12:
                frequency[d] *= seasonality.december_frequency_multiplier
                amount[d] *= seasonality.december_amount_multiplier

    if global_drift is not None:
        days = np.arange(nb_days)
        amount *= (1.0 + global_drift.amount_trend_per_year) ** (days / 365.25)
        for event in global_drift.shocks:
            first = (
                datetime.datetime.strptime(event.start_date, "%Y-%m-%d") - start_date
            ).days
            window = (days >= first) & (days < first + event.duration_days)
            frequency[window] *= event.frequency_multiplier
            amount[window] *= event.amount_multiplier
            shock |= window

    return DailyFactors(frequency=frequency, amount=amount, shock=shock)
