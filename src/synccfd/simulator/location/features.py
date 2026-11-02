from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point as ShapelyPoint

from . import sampling


class LocationFeatures:
    """Geographical features (customer and terminal locations) for the simulator.

    Locations are random points around Brazilian municipalities, drawn with a
    population weight, with an explicit ``numpy.random.Generator`` (see
    ``sampling.py``).
    """

    def __init__(
        self,
        city_radius: float = 12,
        available_terminals_radius: float = 10,
    ):
        self.data = self._load_geo_data()
        # Populated by the gen_* methods below.
        self.n_customers: Any = None
        self.n_terminals: Any = None
        self.billing_locations: Any = None
        self.billing_cities_locations: Any = None
        self.terminal_locations: Any = None
        self.terminal_cities_locations: Any = None
        self.customers_available_terminals: Any = None
        self.city_radius = city_radius
        self.available_terminals_radius = available_terminals_radius

    def _load_geo_data(self) -> dict[str, Any]:
        """Load geographical data from CSV files."""
        data_dir = Path(__file__).parent / "data"

        # World dataset
        world_data = gpd.read_file(
            data_dir / "naturalearth_lowres.csv",
            GEOM_POSSIBLE_NAMES="geometry",
            KEEP_GEOM_COLUMNS="NO",
        )

        # Brazil specific data
        brazil_polygon = world_data[world_data["name"] == "Brazil"]["geometry"].iloc[0]
        brazil_centroid = brazil_polygon.centroid

        # Cities dataset
        brazil_cities = pd.read_csv(data_dir / "cities.csv")
        brazil_cities.rename(columns={"lng": "long"}, inplace=True)

        # Filter cities within Brazil
        mask = brazil_cities.apply(
            lambda row: brazil_polygon.contains(ShapelyPoint(row.long, row.lat)), axis=1
        )
        brazil_cities = brazil_cities[mask][["lat", "long", "pop_21", "name"]]
        brazil_cities.reset_index(inplace=True, drop=True)

        return {
            "world_data": world_data,
            "brazil_polygon": brazil_polygon,
            "brazil_centroid": brazil_centroid,
            "brazil_cities": brazil_cities,
        }

    def gen_billing_locations(self, n_customers: int, rng: np.random.Generator) -> None:
        """Generate billing locations for customers."""
        random_locs = sampling.random_points_around_cities(
            n_customers,
            self.city_radius,
            self.data["brazil_polygon"],
            self.data["brazil_cities"],
            rng,
        )
        self.billing_locations = random_locs["random_points"]
        self.billing_cities_locations = random_locs["cities"]
        self.n_customers = n_customers

    def gen_terminal_locations(
        self, n_terminals: int, rng: np.random.Generator
    ) -> None:
        """Generate terminal locations (in the customers' cities, if any)."""
        if len(self.billing_cities_locations):
            cities = self.billing_cities_locations.drop_duplicates()
        else:
            cities = self.data["brazil_cities"]
        random_locs = sampling.random_points_around_cities(
            n_terminals, self.city_radius, self.data["brazil_polygon"], cities, rng
        )
        self.terminal_locations = random_locs["random_points"]
        self.terminal_cities_locations = random_locs["cities"]
        self.n_terminals = n_terminals

    def gen_customer_available_terminals(self) -> None:
        """For each customer, the terminals within ``available_terminals_radius``
        of the billing location (chunked numpy distances)."""
        self.customers_available_terminals = sampling.points_within_radius(
            self.billing_locations["lat"].to_numpy(),
            self.billing_locations["long"].to_numpy(),
            self.terminal_locations["lat"].to_numpy(),
            self.terminal_locations["long"].to_numpy(),
            self.available_terminals_radius,
        )

    def get_customer_available_terminals(self, customer_id: int) -> list[int]:
        """Get available terminals for a customer."""
        return self.customers_available_terminals[int(customer_id)]
