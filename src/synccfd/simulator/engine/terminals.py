"""Terminal profiles."""

import numpy as np
import pandas as pd

from ..location.features import LocationFeatures


def generate_terminal_profiles(
    n_terminals: int, loc_features: LocationFeatures, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate terminal profiles and their locations."""
    # Must run after the customer profiles: terminals go around the same cities
    loc_features.gen_terminal_locations(n_terminals, rng=rng)
    return pd.DataFrame(
        {
            "TERMINAL_ID": np.arange(n_terminals, dtype=np.int64),
            "terminal_lat": loc_features.terminal_locations["lat"].to_numpy(),
            "terminal_long": loc_features.terminal_locations["long"].to_numpy(),
        }
    )
