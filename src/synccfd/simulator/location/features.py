import random
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from geopy import Point as GeopyPoint
from geopy.distance import distance as GeopyDistance
from haversine import haversine_vector
from shapely.geometry import Point as ShapelyPoint


def random_point_around_a_center(
    center_lat: float,
    center_long: float,
    radius: float,
    brazil_polygon,
    max_attempts: int = 10000,
) -> dict[str, float]:
    """Generate a random point within a radius around a center point.

    Args:
        center_lat: Latitude of the center point.
        center_long: Longitude of the center point.
        radius: Maximum distance in kilometers from center.
        brazil_polygon: Shapely polygon to check containment.
        max_attempts: Maximum number of attempts before raising an error.

    Returns:
        Dictionary with 'lat' and 'long' keys.

    Raises:
        RuntimeError: If no valid point is found within max_attempts.
    """
    for _ in range(max_attempts):
        # Generate a random distance between 0 and a radius in kilometers
        random_distance = random.uniform(0, radius)
        # Generate a random bearing (angle)
        random_bearing = random.uniform(0, 360)
        # Generate a random point within the radius
        random_point = GeopyDistance(kilometers=random_distance).destination(
            GeopyPoint(center_lat, center_long), random_bearing
        )
        if brazil_polygon.contains(
            ShapelyPoint(random_point.longitude, random_point.latitude)
        ):
            return {"lat": random_point.latitude, "long": random_point.longitude}
    raise RuntimeError(
        f"Could not generate a valid point within {max_attempts} attempts "
        f"around center ({center_lat}, {center_long}) with radius {radius}km."
    )


def gen_random_points(
    num_points: int, radius: float, polygon, cities: pd.DataFrame, random_state: int
) -> dict[str, pd.DataFrame]:
    """Generate random points around cities weighted by population."""
    cities_sample = cities.sample(
        n=num_points, weights=cities["pop_21"], replace=True, random_state=random_state
    )
    random_points = cities_sample.apply(
        lambda r: pd.Series(
            random_point_around_a_center(r["lat"], r["long"], radius, polygon).values(),
            index=["lat", "long"],
        ),
        axis=1,
    )

    random_points.reset_index(inplace=True, drop=True)
    return {"random_points": random_points, "cities": cities_sample}


def gen_available_terminals(
    loc: dict[str, float], terminals: pd.DataFrame, radius: float
) -> list[int]:
    """Generate list of available terminals within radius."""
    terminal_locs = terminals[["lat", "long"]].values.astype(float)
    distances = haversine_vector(terminal_locs, (loc["lat"], loc["long"]), comb=True)[0]
    return [
        terminal_id
        for terminal_id in range(len(terminals))
        if distances[terminal_id] <= radius
    ]


class LocationFeatures:
    """Geographical features (customer and terminal locations) for the simulator."""

    def __init__(
        self,
        city_radius: float = 12,
        available_terminals_radius: float = 10,
        random_state: int = None,
    ):
        self.data = self._load_geo_data()
        self.n_customers = None
        self.n_terminals = None
        self.billing_locations = None
        self.billing_cities_locations = None
        self.terminal_locations = None
        self.terminal_cities_locations = None
        self.customers_available_terminals = None
        self.city_radius = city_radius
        self.available_terminals_radius = available_terminals_radius

        if random_state is not None:
            random.seed(random_state)

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

    def gen_billing_locations(self, n_customers: int, random_state: int = None) -> None:
        """Generate billing locations for customers."""
        random_locs = gen_random_points(
            n_customers,
            self.city_radius,
            self.data["brazil_polygon"],
            self.data["brazil_cities"],
            random_state=random_state,
        )

        self.billing_locations = random_locs["random_points"]
        self.billing_cities_locations = random_locs["cities"]
        self.n_customers = n_customers

    def get_billing_location(self, customer_id: int) -> dict[str, float]:
        """Get billing location for a customer."""
        return {
            "lat": self.billing_locations.at[customer_id, "lat"],
            "long": self.billing_locations.at[customer_id, "long"],
        }

    def gen_terminal_locations(self, n_terminals: int, random_state: int = 1) -> None:
        """Generate terminal locations."""
        if len(self.billing_cities_locations):
            random_locs = gen_random_points(
                n_terminals,
                self.city_radius,
                self.data["brazil_polygon"],
                self.billing_cities_locations.drop_duplicates(),
                random_state=random_state,
            )
        else:
            random_locs = gen_random_points(
                n_terminals,
                self.city_radius,
                self.data["brazil_polygon"],
                self.data["brazil_cities"],
                random_state=random_state,
            )

        self.terminal_locations = random_locs["random_points"]
        self.terminal_cities_locations = random_locs["cities"]
        self.n_terminals = n_terminals

    def get_terminal_location(self, terminal_id: int) -> dict[str, float]:
        """Get terminal location."""
        return {
            "lat": self.terminal_locations.at[terminal_id, "lat"],
            "long": self.terminal_locations.at[terminal_id, "long"],
        }

    def gen_customer_available_terminals(self) -> None:
        """Generate list of available terminals for each customer."""
        # One list of terminal ids per customer.
        self.customers_available_terminals = []

        for i in range(self.n_customers):
            billing_loc = self.get_billing_location(i)
            available_terminals = gen_available_terminals(
                billing_loc, self.terminal_locations, self.available_terminals_radius
            )
            self.customers_available_terminals.append(available_terminals)

    def get_customer_available_terminals(self, customer_id: int) -> list[int]:
        """Get available terminals for a customer."""
        return self.customers_available_terminals[int(customer_id)]

    def gen_customer_terminal_association(self, customer_id: int, tx_type: str) -> int:
        """Return the terminal of a transaction given the customer_id and tx_type."""
        if (
            tx_type == "CP"
        ):  # Choose a terminal that is within the available terminals uniformly
            available_terminals = self.get_customer_available_terminals(customer_id)
            if (
                len(available_terminals) > 0
            ):  # Requires that available terminals list contains at least one terminal
                terminal_id = random.choice(available_terminals)
            else:
                raise ValueError(f"No available terminals for customer {customer_id}")
            return terminal_id
        else:  # Choose a random terminal uniformly
            return random.choice(range(self.n_terminals))

    def gen_tx_locations(self, customer_id: int, tx_type: str) -> dict[str, Any]:
        """Generate transaction locations."""
        tx_locations = {}

        tx_locations["tx_billing_loc"] = self.get_billing_location(customer_id)
        tx_locations["tx_terminal_id"] = self.gen_customer_terminal_association(
            customer_id, tx_type
        )
        tx_locations["tx_terminal_loc"] = self.get_terminal_location(
            tx_locations["tx_terminal_id"]
        )
        tx_locations["tx_shipping_loc"] = self.gen_shipping_location(
            tx_locations["tx_billing_loc"], tx_locations["tx_terminal_loc"], tx_type
        )
        tx_locations["tx_type"] = tx_type

        return tx_locations

    def gen_shipping_location(
        self,
        tx_billing_loc: dict[str, float],
        tx_terminal_loc: dict[str, float],
        tx_type: str,
    ) -> dict[str, float]:
        """Generate shipping location for a transaction."""
        if tx_type == "CP":
            possible_locs = [
                "terminal_location",
                "billing_location",
                "random_point_around_billing",
            ]
            prob = [0.8, 0.15, 0.05]
            loc = random.choices(possible_locs, prob, k=1)[0]

            if loc == "terminal_location":
                return tx_terminal_loc
            elif loc == "billing_location":
                return tx_billing_loc
            else:
                return random_point_around_a_center(
                    tx_billing_loc["lat"],
                    tx_billing_loc["long"],
                    self.available_terminals_radius,
                    self.data["brazil_polygon"],
                )
        else:
            possible_locs = ["billing_location", "random_point_around_billing"]
            prob = [0.9, 0.1]
            loc = random.choices(possible_locs, prob, k=1)[0]

            if loc == "billing_location":
                return tx_billing_loc
            else:
                return random_point_around_a_center(
                    tx_billing_loc["lat"],
                    tx_billing_loc["long"],
                    self.available_terminals_radius,
                    self.data["brazil_polygon"],
                )

    def tamper_tx_locations(
        self, transactions: pd.DataFrame, fraud_scenario: int
    ) -> pd.DataFrame:
        """Modify transaction locations for fraud scenarios."""
        compromised_txs = transactions.copy()
        customer_ids = compromised_txs["CUSTOMER_ID"].unique()

        if fraud_scenario == 3:
            for customer_id in customer_ids:
                mask = compromised_txs["CUSTOMER_ID"] == customer_id
                n_txs = mask.sum()

                # Random terminals for each transaction
                random_terminals = [
                    random.randrange(self.n_terminals) for _ in range(n_txs)
                ]
                random_terminal_locs = [
                    tuple(self.get_terminal_location(t_id).values())
                    for t_id in random_terminals
                ]

                compromised_txs.loc[mask, "TERMINAL_ID"] = random_terminals
                compromised_txs.loc[mask, ["TX_TERM_LAT", "TX_TERM_LONG"]] = (
                    random_terminal_locs
                )

                # Fraudster location
                random_loc = gen_random_points(
                    1,
                    self.city_radius,
                    self.data["brazil_polygon"],
                    self.data["brazil_cities"],
                    random_state=int(customer_id),
                )
                fraudster_loc = (
                    random_loc["random_points"].at[0, "lat"],
                    random_loc["random_points"].at[0, "long"],
                )
                compromised_txs.loc[mask, ["TX_SHIPP_LAT", "TX_SHIPP_LONG"]] = (
                    fraudster_loc
                )

        elif fraud_scenario == 4:
            for customer_id in customer_ids:
                billing_city_loc = self.billing_cities_locations.iloc[int(customer_id)]
                fraudster_loc = random_point_around_a_center(
                    billing_city_loc["lat"],
                    billing_city_loc["long"],
                    self.city_radius,
                    self.data["brazil_polygon"],
                )

                fraudster_terminals = gen_available_terminals(
                    fraudster_loc,
                    self.terminal_locations,
                    self.available_terminals_radius,
                )

                mask = compromised_txs["CUSTOMER_ID"] == customer_id
                if not fraudster_terminals:
                    compromised_txs.loc[mask, "TX_TYPE"] = "CNP"

                for idx, row in compromised_txs[mask].iterrows():
                    if row["TX_TYPE"] == "CP":
                        terminal_id = random.choice(fraudster_terminals)
                        terminal_loc = self.get_terminal_location(terminal_id)

                        compromised_txs.at[idx, "TERMINAL_ID"] = terminal_id
                        compromised_txs.at[idx, "TX_TERM_LAT"] = terminal_loc["lat"]
                        compromised_txs.at[idx, "TX_TERM_LONG"] = terminal_loc["long"]
                        compromised_txs.at[idx, "TX_SHIPP_LAT"] = terminal_loc["lat"]
                        compromised_txs.at[idx, "TX_SHIPP_LONG"] = terminal_loc["long"]
                    else:
                        terminal_id = random.randrange(self.n_terminals)
                        terminal_loc = self.get_terminal_location(terminal_id)

                        compromised_txs.at[idx, "TERMINAL_ID"] = terminal_id
                        compromised_txs.at[idx, "TX_TERM_LAT"] = terminal_loc["lat"]
                        compromised_txs.at[idx, "TX_TERM_LONG"] = terminal_loc["long"]
                        compromised_txs.at[idx, "TX_SHIPP_LAT"] = fraudster_loc["lat"]
                        compromised_txs.at[idx, "TX_SHIPP_LONG"] = fraudster_loc["long"]

        return compromised_txs
