"""Merchants and categories (v2 engine, ``MerchantConfig``).

- Physical merchants own the terminals: in each city, the terminals are shuffled and
  cut into merchants of ``1 + Poisson(extra_terminals_mean)`` terminals; the category
  of a physical merchant is drawn among the categories with CP, by weight.
- Online merchants: ``cnp_merchants_per_category`` per category with CNP, each with
  one virtual terminal at its head office (a random point around a
  population-weighted city), appended after the physical terminals. Their
  popularity follows a Zipf law.
- Customers: category preferences ``Dirichlet(alpha * weights)`` and a spending
  factor ``LogNormal(0, sigma)``.
- A purchase picks a category by preference (times the weekend factor on weekends,
  among the categories the customer can buy from), then the type (CP-only and
  CNP-only categories fix it; mixed ones use ``cnp_prob``; no reachable terminal of
  the category means CNP), the merchant (a favourite with probability ``loyalty``),
  the hour (category profile; CP only in opening hours) and the amount
  (``median * spending_factor * LogNormal(0, log_sigma)``).

With drift, regime B has its own preferences (``category_shift_range`` of
``DriftConfig``), spending factor (the drift's amount factor), frequency, CNP
probability and, if the customer moved, reachable terminals.
"""

import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DriftConfig, MerchantConfig
from ..location import sampling
from ..location.features import LocationFeatures
from .daily_factors import DailyFactors
from .drift import NO_DRIFT, CustomerDrift, regime_weight
from .transactions import SECONDS_PER_DAY, draw_shipping, finish_transactions


@dataclass
class MerchantWorld:
    """Merchants, their terminals and the per-category parameters."""

    config: MerchantConfig
    merchant_profiles: pd.DataFrame
    n_physical_terminals: int
    terminal_merchant: np.ndarray  # merchant of every terminal (physical, virtual)
    merchant_category: np.ndarray  # category index of every merchant
    merchant_is_online: np.ndarray
    online_merchants: list[np.ndarray]  # per category: online merchant ids
    online_popularity: list[np.ndarray]  # per category: Zipf probabilities
    merchant_terminals: list[np.ndarray]  # per merchant: its terminals
    median: np.ndarray
    sigma: np.ndarray
    has_cp: np.ndarray
    has_cnp: np.ndarray
    weekend_factor: np.ndarray
    skimming_risk: np.ndarray
    cp_hour_cdf: np.ndarray  # (K, 24): hours of CP purchases (opening hours only)
    cnp_hour_cdf: np.ndarray  # (K, 24)
    open_hours: np.ndarray  # (K, 24) bool: CP possible at that hour

    @property
    def names(self) -> tuple[str, ...]:
        return self.config.names

    @property
    def terminal_category(self) -> np.ndarray:
        return self.merchant_category[self.terminal_merchant]

    def category_index(self, names) -> np.ndarray:
        return np.array([self.names.index(n) for n in names], dtype=np.int64)

    def draw_hours(
        self, categories: np.ndarray, is_cnp: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Hour of each purchase, from its category profile (CP: opening hours)."""
        cdf = np.where(
            is_cnp[:, None], self.cnp_hour_cdf[categories], self.cp_hour_cdf[categories]
        )
        u = rng.random(len(categories))
        return np.minimum((cdf < u[:, None]).sum(axis=1), 23)

    def draw_online(
        self, category: int, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``n`` online merchants of a category, by popularity."""
        merchants = self.online_merchants[category]
        return merchants[
            rng.choice(len(merchants), size=n, p=self.online_popularity[category])
        ]

    def virtual_terminal(self, merchants: np.ndarray) -> np.ndarray:
        return np.array(
            [self.merchant_terminals[m][0] for m in merchants], dtype=np.int64
        )

    def terminals_by_category(self, terminals: np.ndarray) -> list[np.ndarray]:
        """Split physical terminals (e.g. those a customer reaches) by category."""
        terminals = np.asarray(terminals, dtype=np.int64)
        categories = self.terminal_category[terminals]
        return [terminals[categories == k] for k in range(len(self.names))]


def _cdf(weights: np.ndarray) -> np.ndarray:
    total = weights.sum(axis=-1, keepdims=True)
    return np.cumsum(weights / np.where(total > 0, total, 1.0), axis=-1)


def build_merchants(
    config: MerchantConfig,
    terminal_profiles: pd.DataFrame,
    loc_features: LocationFeatures,
    rng: np.random.Generator,
) -> tuple[MerchantWorld, pd.DataFrame]:
    """Merchants for the physical terminals plus the online merchants; returns the
    world and the terminal profiles extended with the virtual terminals and
    ``MERCHANT_ID``."""
    categories = config.categories
    n_categories = len(categories)
    weights = config.weights()
    has_cp = np.array([c.has_cp for c in categories])
    has_cnp = np.array([c.has_cnp for c in categories])

    # Physical merchants: terminals of one city, cut into groups
    n_physical = len(terminal_profiles)
    cities = np.asarray(loc_features.terminal_cities_locations.index)
    terminal_merchant = np.empty(n_physical, dtype=np.int64)
    merchant_terminals: list[np.ndarray] = []
    for city in np.unique(cities):
        members = rng.permutation(np.flatnonzero(cities == city))
        start = 0
        while start < len(members):
            size = 1 + int(rng.poisson(config.extra_terminals_mean))
            group = np.sort(members[start : start + size])
            terminal_merchant[group] = len(merchant_terminals)
            merchant_terminals.append(group)
            start += size
    n_physical_merchants = len(merchant_terminals)
    cp_weights = np.where(has_cp, weights, 0.0)
    physical_category = rng.choice(
        n_categories, size=n_physical_merchants, p=cp_weights / cp_weights.sum()
    )

    # Online merchants, one virtual terminal each
    online_categories = np.flatnonzero(has_cnp)
    n_online = len(online_categories) * config.cnp_merchants_per_category
    head_offices = sampling.random_points_around_cities(
        n_online,
        loc_features.city_radius,
        loc_features.data["brazil_polygon"],
        loc_features.data["brazil_cities"],
        rng,
    )["random_points"]
    online_category = np.repeat(online_categories, config.cnp_merchants_per_category)
    online_ids = n_physical_merchants + np.arange(n_online)
    virtual_terminals = n_physical + np.arange(n_online)
    merchant_terminals += [np.array([t]) for t in virtual_terminals]

    merchant_category = np.concatenate([physical_category, online_category])
    merchant_is_online = np.arange(len(merchant_category)) >= n_physical_merchants
    terminal_merchant = np.concatenate([terminal_merchant, online_ids])

    ranks = np.arange(1, config.cnp_merchants_per_category + 1, dtype=float)
    zipf = ranks ** (-config.cnp_popularity_exponent)
    zipf /= zipf.sum()
    online_merchants = [
        online_ids[online_category == k] if has_cnp[k] else np.empty(0, np.int64)
        for k in range(n_categories)
    ]
    online_popularity = [
        zipf if has_cnp[k] else np.empty(0) for k in range(n_categories)
    ]

    terminal_lat = np.concatenate(
        [terminal_profiles["terminal_lat"].to_numpy(), head_offices["lat"].to_numpy()]
    )
    terminal_long = np.concatenate(
        [
            terminal_profiles["terminal_long"].to_numpy(),
            head_offices["long"].to_numpy(),
        ]
    )
    first_terminal = np.array([t[0] for t in merchant_terminals])
    names = np.array(config.names, dtype=object)
    merchant_profiles = pd.DataFrame(
        {
            "MERCHANT_ID": np.arange(len(merchant_category), dtype=np.int64),
            "MERCHANT_CATEGORY": names[merchant_category],
            "channel": np.where(merchant_is_online, "CNP", "CP"),
            "n_terminals": np.array([len(t) for t in merchant_terminals]),
            "lat": terminal_lat[first_terminal],
            "long": terminal_long[first_terminal],
        }
    )
    terminals = pd.DataFrame(
        {
            "TERMINAL_ID": np.arange(len(terminal_lat), dtype=np.int64),
            "terminal_lat": terminal_lat,
            "terminal_long": terminal_long,
            "MERCHANT_ID": terminal_merchant,
        }
    )

    profiles = np.array([c.hourly_profile for c in categories])
    open_hours = np.zeros((n_categories, 24), dtype=bool)
    for k, c in enumerate(categories):
        open_hours[k, c.opening_hours[0] : c.opening_hours[1]] = True
    world = MerchantWorld(
        config=config,
        merchant_profiles=merchant_profiles,
        n_physical_terminals=n_physical,
        terminal_merchant=terminal_merchant,
        merchant_category=merchant_category,
        merchant_is_online=merchant_is_online,
        online_merchants=online_merchants,
        online_popularity=online_popularity,
        merchant_terminals=merchant_terminals,
        median=np.array([c.median_amount for c in categories], dtype=float),
        sigma=np.array([c.log_sigma for c in categories], dtype=float),
        has_cp=has_cp,
        has_cnp=has_cnp,
        weekend_factor=np.array([c.weekend_factor for c in categories], dtype=float),
        skimming_risk=np.array([c.skimming_risk for c in categories], dtype=float),
        cp_hour_cdf=_cdf(profiles * open_hours),
        cnp_hour_cdf=_cdf(profiles),
        open_hours=open_hours,
    )
    return world, terminals


def customer_preferences(
    n_customers: int, config: MerchantConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Category preferences (``n x K``) and spending factors of the customers."""
    preferences = rng.dirichlet(
        config.preference_concentration * config.weights(), n_customers
    )
    spending = rng.lognormal(0.0, config.spending_factor_sigma, n_customers)
    return preferences, spending


def regime_b_preferences(
    preferences: np.ndarray,
    drifts: list[CustomerDrift],
    merchants: MerchantConfig,
    drift: DriftConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Preferences of regime B: ``(1 - s) * pref_A + s * new``, with
    ``s ~ U(category_shift_range)`` and ``new`` drawn like a new customer's; ``NaN``
    rows for customers without drift."""
    drifting = np.array([d.kind != NO_DRIFT for d in drifts])
    out = np.full(preferences.shape, np.nan)
    n = int(drifting.sum())
    if n == 0:
        return out
    shift = rng.uniform(*drift.category_shift_range, n)
    new = rng.dirichlet(merchants.preference_concentration * merchants.weights(), n)
    out[drifting] = (1 - shift[:, None]) * preferences[drifting] + shift[:, None] * new
    return out


def weekend_days(nb_days: int, start_date: datetime.datetime) -> np.ndarray:
    weekday = (start_date.weekday() + np.arange(nb_days)) % 7
    return weekday >= 5


def customer_transactions(
    customer: dict,
    preference: np.ndarray,
    spending: float,
    available: np.ndarray,
    nb_days: int,
    world: MerchantWorld,
    terminal_lat: np.ndarray,
    terminal_long: np.ndarray,
    weekend: np.ndarray,
    loc_features: LocationFeatures,
    rng: np.random.Generator,
    drift: Optional[CustomerDrift] = None,
    preference_b: Optional[np.ndarray] = None,
    factors: Optional[DailyFactors] = None,
) -> dict[str, np.ndarray]:
    """Legitimate transactions of one customer, with merchants."""
    config = world.config
    n_categories = len(world.names)
    drifting = drift is not None and drift.kind != NO_DRIFT

    lam = np.full(nb_days, customer["mean_nb_tx_per_day"])
    w_day = np.zeros(nb_days)
    if drifting:
        assert drift is not None
        w_day = regime_weight(drift, np.arange(nb_days) + 0.5)
        lam = (1.0 - w_day) * lam + w_day * drift.mean_nb_tx_per_day
    if factors is not None:
        lam = lam * factors.frequency
    counts = rng.poisson(lam)
    n = int(counts.sum())
    days = np.repeat(np.arange(nb_days, dtype=np.int64), counts)

    w = w_day[days]
    is_b = np.zeros(n, dtype=bool)
    spending_b, pref_b, cnp_b = spending, preference, customer["cnp_prob"]
    if drifting:
        assert drift is not None and preference_b is not None
        if drift.kind == "gradual":
            is_b = rng.random(n) < w
        elif drift.kind == "incremental":
            is_b = w >= 0.5
        else:
            is_b = w >= 1.0
        spending_b = spending * drift.mean_amount / customer["mean_amount"]
        pref_b, cnp_b = preference_b, drift.cnp_prob
    incremental = drifting and drift is not None and drift.kind == "incremental"

    def per_tx(a, b):
        # Parameter of each transaction: interpolated (incremental) or by regime
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if incremental:
            ww = w if a.ndim == 0 else w[:, None]
            return (1.0 - ww) * a + ww * b
        bb = is_b if a.ndim == 0 else is_b[:, None]
        return np.where(bb, b, a)

    # Terminals the customer reaches, by category (regime B may live elsewhere)
    relocated = drifting and drift is not None and drift.relocated
    reach_a = world.terminals_by_category(available)
    reach_b = (
        world.terminals_by_category(drift.available_terminals)
        if relocated and drift is not None and drift.available_terminals is not None
        else reach_a
    )
    can_a = world.has_cnp | (world.has_cp & np.array([len(t) > 0 for t in reach_a]))
    can_b = world.has_cnp | (world.has_cp & np.array([len(t) > 0 for t in reach_b]))
    in_b = is_b if relocated else np.zeros(n, dtype=bool)

    # 1. Category
    weights = per_tx(preference, pref_b) * np.where(
        weekend[days][:, None], world.weekend_factor, 1.0
    )
    weights = weights * np.where(in_b[:, None], can_b, can_a)
    cdf = _cdf(weights)
    category = np.minimum((cdf < rng.random(n)[:, None]).sum(axis=1), n_categories - 1)

    # 2. Type
    reachable = np.where(
        in_b,
        np.array([len(t) > 0 for t in reach_b])[category],
        np.array([len(t) > 0 for t in reach_a])[category],
    )
    cp_possible = world.has_cp[category] & reachable
    is_cnp = ~cp_possible | (
        world.has_cnp[category] & (rng.random(n) < per_tx(customer["cnp_prob"], cnp_b))
    )

    # 3. Merchant and terminal; favourites first, by category and regime
    terminals = np.zeros(n, dtype=np.int64)
    for k in range(n_categories):
        for regime, reach in ((False, reach_a), (True, reach_b)):
            if regime and not relocated:
                continue
            rows = np.flatnonzero((category == k) & ~is_cnp & (in_b == regime))
            if len(rows):
                candidates = rng.permutation(reach[k])
                n_fav = min(config.n_favorites, len(candidates))
                loyal = rng.random(len(rows)) < config.loyalty
                index = np.where(
                    loyal,
                    rng.integers(0, n_fav, len(rows)),
                    rng.integers(0, len(candidates), len(rows)),
                )
                terminals[rows] = candidates[index]
        rows = np.flatnonzero((category == k) & is_cnp)
        if len(rows):
            merchants = world.online_merchants[k]
            popularity = world.online_popularity[k]
            n_fav = min(config.n_favorites, len(merchants))
            favorites = rng.choice(len(merchants), n_fav, replace=False, p=popularity)
            loyal = rng.random(len(rows)) < config.loyalty
            index = np.where(
                loyal,
                favorites[rng.integers(0, n_fav, len(rows))],
                rng.choice(len(merchants), len(rows), p=popularity),
            )
            terminals[rows] = world.virtual_terminal(merchants[index])

    # 4. Time and amount
    hours = world.draw_hours(category, is_cnp, rng)
    seconds = (days * SECONDS_PER_DAY + hours * 3600 + rng.integers(0, 3600, n)).astype(
        np.int64
    )
    amounts = (
        world.median[category]
        * per_tx(spending, spending_b)
        * np.exp(rng.normal(0.0, 1.0, n) * world.sigma[category])
    )
    if factors is not None:
        amounts = amounts * factors.amount[days]
    amounts = np.round(amounts, 2)

    # 5. Addresses
    bill_lat = np.full(n, customer["billing_lat"])
    bill_long = np.full(n, customer["billing_long"])
    if relocated and drift is not None:
        bill_lat[is_b] = drift.billing_lat
        bill_long[is_b] = drift.billing_long
    term_lat, term_long = terminal_lat[terminals], terminal_long[terminals]
    shipp_lat, shipp_long = draw_shipping(
        is_cnp, term_lat, term_long, bill_lat, bill_long, loc_features, rng
    )

    columns = {
        "TX_TIME_SECONDS": seconds,
        "TX_TIME_DAYS": days,
        "CUSTOMER_ID": np.full(n, customer["CUSTOMER_ID"], dtype=np.int64),
        "TERMINAL_ID": terminals,
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
    preferences: np.ndarray,
    spending: np.ndarray,
    world: MerchantWorld,
    terminal_profiles: pd.DataFrame,
    loc_features: LocationFeatures,
    nb_days: int,
    start_date: datetime.datetime,
    customer_rngs: list[np.random.Generator],
    drifts: Optional[list[CustomerDrift]] = None,
    preferences_b: Optional[np.ndarray] = None,
    factors: Optional[DailyFactors] = None,
    shock_column: bool = False,
) -> pd.DataFrame:
    """Legitimate transactions of all customers, with merchants (same columns as
    ``transactions.generate_transactions``)."""
    terminal_lat = terminal_profiles["terminal_lat"].to_numpy()
    terminal_long = terminal_profiles["terminal_long"].to_numpy()
    weekend = weekend_days(nb_days, start_date)
    parts = []
    for i, (customer, available, rng) in enumerate(
        zip(
            customer_profiles.to_dict("records"),
            customer_profiles["available_terminals"],
            customer_rngs,
        )
    ):
        parts.append(
            customer_transactions(
                customer,
                preferences[i],
                float(spending[i]),
                np.asarray(available, dtype=np.int64),
                nb_days,
                world,
                terminal_lat,
                terminal_long,
                weekend,
                loc_features,
                rng,
                drift=drifts[i] if drifts is not None else None,
                preference_b=preferences_b[i] if preferences_b is not None else None,
                factors=factors,
            )
        )
    return finish_transactions(
        parts, start_date, drift_columns=drifts is not None, shock_column=shock_column
    )


def preference_columns(
    world: MerchantWorld, preferences: np.ndarray, prefix: str = "pref_"
) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{prefix}{name}": preferences[:, k] for k, name in enumerate(world.names)}
    )


def merchant_columns(
    world: MerchantWorld, terminal_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """``MERCHANT_ID`` and ``MERCHANT_CATEGORY`` of transactions, from the terminal."""
    merchants = world.terminal_merchant[terminal_ids]
    names = np.array(world.names, dtype=object)
    return {
        "MERCHANT_ID": merchants,
        "MERCHANT_CATEGORY": names[world.merchant_category[merchants]],
    }
