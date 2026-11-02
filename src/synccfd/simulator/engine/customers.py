"""Customer profiles. Same model as v0.1."""

import numpy as np
import pandas as pd

from ..location.features import LocationFeatures

CUSTOMER_COLUMNS = [
    "CUSTOMER_ID",
    "billing_lat",
    "billing_long",
    "mean_amount",
    "std_amount",
    "mean_nb_tx_per_day",
    "cnp_prob",
]


def generate_customer_profiles(
    n_customers: int, loc_features: LocationFeatures, rng: np.random.Generator
) -> pd.DataFrame:
    """Customer profiles; also fills the billing locations of ``loc_features``.

    As in v0.1: ``mean_amount ~ U(5, 100)``, ``std_amount = mean_amount / 2``,
    ``mean_nb_tx_per_day ~ U(0, 4)``, ``cnp_prob ~ U(0, 1)``, billing address at a
    random point around a population-weighted Brazilian city.
    """
    loc_features.gen_billing_locations(n_customers, rng=rng)
    mean_amount = rng.uniform(5, 100, n_customers)
    profiles = pd.DataFrame(
        {
            "CUSTOMER_ID": np.arange(n_customers, dtype=np.int64),
            "billing_lat": loc_features.billing_locations["lat"].to_numpy(),
            "billing_long": loc_features.billing_locations["long"].to_numpy(),
            "mean_amount": mean_amount,
            "std_amount": mean_amount / 2,
            "mean_nb_tx_per_day": rng.uniform(0, 4, n_customers),
            "cnp_prob": rng.uniform(0, 1, n_customers),
        }
    )
    return profiles[CUSTOMER_COLUMNS]
