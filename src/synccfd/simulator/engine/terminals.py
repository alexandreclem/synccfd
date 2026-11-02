"""Terminal profiles (v2 engine). Same model as v0.1."""

import numpy as np
import pandas as pd

from ..location.features import LocationFeatures


def generate_terminal_profiles(
    n_terminals: int, loc_features: LocationFeatures, rng: np.random.Generator
) -> pd.DataFrame:
    """Terminal profiles; also fills the terminal locations of ``loc_features``.

    As in v0.1, terminals are placed around the cities where customers live
    (population-weighted). Call after the customer profiles.
    """
    loc_features.gen_terminal_locations(n_terminals, rng=rng)
    return pd.DataFrame(
        {
            "TERMINAL_ID": np.arange(n_terminals, dtype=np.int64),
            "terminal_lat": loc_features.terminal_locations["lat"].to_numpy(),
            "terminal_long": loc_features.terminal_locations["long"].to_numpy(),
        }
    )
