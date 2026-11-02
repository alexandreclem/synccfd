"""Customer concept drift: regimes A and B and the weight ``w(t)``.

Every customer has regime A (its profile). A drifting customer also has regime B,
drawn from A (see ``DriftConfig``), and a drift type that defines ``w(t)``, the
weight of regime B on day ``t``. ``w`` is evaluated at the middle of each day,
``t = d + 0.5``, and applies to every transaction of that day.

All drift draws come from the ``drift`` random stream, so enabling drift does not
change the customer profiles or the transactions of customers without drift.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DRIFT_TYPES, DriftConfig
from ..location import sampling
from ..location.features import LocationFeatures

NO_DRIFT = "none"


@dataclass
class CustomerDrift:
    """Drift of one customer (``kind == "none"`` for customers without drift)."""

    kind: str = NO_DRIFT
    t0: int = 0
    transition_days: int = 0
    period_days: int = 0
    active_days: int = 0
    mean_amount: float = np.nan
    std_amount: float = np.nan
    mean_nb_tx_per_day: float = np.nan
    cnp_prob: float = np.nan
    relocated: bool = False
    billing_lat: float = np.nan
    billing_long: float = np.nan
    available_terminals: Optional[np.ndarray] = None


def regime_weight(drift: CustomerDrift, t: np.ndarray) -> np.ndarray:
    """``w(t)``: weight of regime B at times ``t`` (in days)."""
    t = np.asarray(t, dtype=float)
    if drift.kind == NO_DRIFT:
        return np.zeros_like(t)
    if drift.kind == "sudden":
        return (t >= drift.t0).astype(float)
    if drift.kind in ("gradual", "incremental"):
        return np.clip((t - drift.t0) / drift.transition_days, 0.0, 1.0)
    if drift.kind == "recurrent":
        active = (t >= drift.t0) & (
            np.mod(t - drift.t0, drift.period_days) < drift.active_days
        )
        return active.astype(float)
    raise ValueError(f"Unknown drift type {drift.kind!r}.")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def assign_drift(
    customers: pd.DataFrame,
    config: DriftConfig,
    nb_days: int,
    loc_features: LocationFeatures,
    rng: np.random.Generator,
) -> list[CustomerDrift]:
    """Choose the drifting customers and draw their regime B and drift timing."""
    n = len(customers)
    drifts = [CustomerDrift() for _ in range(n)]
    n_drift = int(round(config.customer_fraction * n))
    if n_drift == 0:
        return drifts

    chosen = np.sort(rng.choice(n, size=n_drift, replace=False))
    types = list(config.type_weights)
    weights = np.array([config.type_weights[k] for k in types], dtype=float)
    kinds = rng.choice(types, size=n_drift, p=weights / weights.sum())

    low, high = config.start_day_range
    t0 = np.floor(rng.uniform(low, high, n_drift) * nb_days).astype(int)
    transition = rng.integers(
        config.transition_days_range[0], config.transition_days_range[1] + 1, n_drift
    )
    period = rng.integers(
        config.recurrent_period_days_range[0],
        config.recurrent_period_days_range[1] + 1,
        n_drift,
    )
    active = rng.integers(
        config.recurrent_active_days_range[0],
        config.recurrent_active_days_range[1] + 1,
        n_drift,
    )

    profile = customers.iloc[chosen]
    amount_factor = rng.lognormal(
        config.amount_log_mu, config.amount_log_sigma, n_drift
    )
    frequency_factor = rng.lognormal(0.0, config.frequency_log_sigma, n_drift)
    cnp_b = _sigmoid(
        _logit(profile["cnp_prob"].to_numpy())
        + rng.normal(0.0, config.cnp_logit_sigma, n_drift)
    )
    mean_b = profile["mean_amount"].to_numpy() * amount_factor

    relocated = rng.random(n_drift) < config.relocation_prob
    new_lat = np.full(n_drift, np.nan)
    new_long = np.full(n_drift, np.nan)
    new_available: list[list[int]] = [[] for _ in range(n_drift)]
    if relocated.any():
        points = sampling.random_points_around_cities(
            int(relocated.sum()),
            loc_features.city_radius,
            loc_features.data["brazil_polygon"],
            loc_features.data["brazil_cities"],
            rng,
        )["random_points"]
        new_lat[relocated] = points["lat"].to_numpy()
        new_long[relocated] = points["long"].to_numpy()
        reachable = sampling.points_within_radius(
            points["lat"].to_numpy(),
            points["long"].to_numpy(),
            loc_features.terminal_locations["lat"].to_numpy(),
            loc_features.terminal_locations["long"].to_numpy(),
            loc_features.available_terminals_radius,
        )
        for position, terminals in zip(np.flatnonzero(relocated), reachable):
            new_available[position] = terminals

    for j, customer in enumerate(chosen):
        kind = str(kinds[j])
        drifts[customer] = CustomerDrift(
            kind=kind,
            t0=int(t0[j]),
            transition_days=int(transition[j])
            if kind in ("gradual", "incremental")
            else 0,
            period_days=int(period[j]) if kind == "recurrent" else 0,
            active_days=int(active[j]) if kind == "recurrent" else 0,
            mean_amount=float(mean_b[j]),
            std_amount=float(mean_b[j] / 2),  # same ratio as v0.1
            mean_nb_tx_per_day=float(
                profile["mean_nb_tx_per_day"].to_numpy()[j] * frequency_factor[j]
            ),
            cnp_prob=float(cnp_b[j]),
            relocated=bool(relocated[j]),
            billing_lat=float(new_lat[j]),
            billing_long=float(new_long[j]),
            available_terminals=np.asarray(new_available[j], dtype=np.int64)
            if relocated[j]
            else None,
        )
    return drifts


def drift_columns(drifts: list[CustomerDrift]) -> pd.DataFrame:
    """Ground-truth columns of the customer profiles (``GT_*``, never features)."""

    def optional_int(values: list[int], kinds: tuple[str, ...]) -> pd.array:
        return pd.array(
            [v if d.kind in kinds else None for v, d in zip(values, drifts)],
            dtype="Int64",
        )

    drifting = DRIFT_TYPES
    return pd.DataFrame(
        {
            "GT_DRIFT_TYPE": [d.kind for d in drifts],
            "GT_DRIFT_T0": optional_int([d.t0 for d in drifts], drifting),
            "GT_DRIFT_L": optional_int(
                [d.transition_days for d in drifts], ("gradual", "incremental")
            ),
            "GT_DRIFT_PERIOD": optional_int(
                [d.period_days for d in drifts], ("recurrent",)
            ),
            "GT_DRIFT_ACTIVE_DAYS": optional_int(
                [d.active_days for d in drifts], ("recurrent",)
            ),
            "GT_B_mean_amount": [d.mean_amount for d in drifts],
            "GT_B_std_amount": [d.std_amount for d in drifts],
            "GT_B_mean_nb_tx_per_day": [d.mean_nb_tx_per_day for d in drifts],
            "GT_B_cnp_prob": [d.cnp_prob for d in drifts],
            "GT_RELOCATED": [d.relocated for d in drifts],
            "GT_B_billing_lat": [d.billing_lat for d in drifts],
            "GT_B_billing_long": [d.billing_long for d in drifts],
        }
    )
