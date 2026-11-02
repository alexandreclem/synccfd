"""Tests for the vectorised geographic sampling used by the v2 engine."""

import numpy as np
import pandas as pd
import pytest
import shapely
from haversine import haversine_vector

from synccfd.simulator.location import sampling
from synccfd.simulator.location.features import (
    LocationFeatures,
    gen_available_terminals,
)


@pytest.fixture(scope="module")
def loc_features():
    return LocationFeatures(random_state=None)


def test_destination_points_are_at_the_requested_distance():
    rng = np.random.default_rng(0)
    lat = rng.uniform(-30, 0, 1000)
    long = rng.uniform(-70, -35, 1000)
    distance = rng.uniform(0, 50, 1000)
    bearing = rng.uniform(0, 360, 1000)
    lat2, long2 = sampling.destination_points(lat, long, distance, bearing)
    np.testing.assert_allclose(
        sampling.haversine_km(lat, long, lat2, long2), distance, atol=1e-6
    )


def test_haversine_matches_haversine_package():
    rng = np.random.default_rng(1)
    a = np.column_stack([rng.uniform(-30, 0, 200), rng.uniform(-70, -35, 200)])
    b = np.column_stack([rng.uniform(-30, 0, 200), rng.uniform(-70, -35, 200)])
    expected = haversine_vector(a, b)
    np.testing.assert_allclose(
        sampling.haversine_km(a[:, 0], a[:, 1], b[:, 0], b[:, 1]), expected, rtol=1e-12
    )


def test_random_points_stay_within_radius_and_country(loc_features):
    rng = np.random.default_rng(2)
    cities = loc_features.data["brazil_cities"]
    points = sampling.random_points_around_cities(
        2000, 12.0, loc_features.data["brazil_polygon"], cities, rng
    )
    lat = points["random_points"]["lat"].to_numpy()
    long = points["random_points"]["long"].to_numpy()
    centers = points["cities"]
    distance = sampling.haversine_km(
        centers["lat"].to_numpy(), centers["long"].to_numpy(), lat, long
    )
    assert (distance <= 12.0 + 1e-9).all()
    assert shapely.contains_xy(loc_features.data["brazil_polygon"], long, lat).all()
    # distance is uniform in [0, r], as in v0.1
    assert distance.mean() == pytest.approx(6.0, rel=0.05)


def test_cities_are_population_weighted(loc_features):
    rng = np.random.default_rng(3)
    cities = loc_features.data["brazil_cities"]
    sample = sampling.random_points_around_cities(
        20_000, 1.0, loc_features.data["brazil_polygon"], cities, rng
    )["cities"]
    top = cities["pop_21"].idxmax()
    expected_share = cities["pop_21"].max() / cities["pop_21"].sum()
    assert (sample.index == top).mean() == pytest.approx(expected_share, rel=0.1)


def test_points_within_radius_matches_legacy(loc_features):
    rng = np.random.default_rng(4)
    customers = pd.DataFrame(
        {"lat": rng.uniform(-24, -22, 300), "long": rng.uniform(-47, -45, 300)}
    )
    terminals = pd.DataFrame(
        {"lat": rng.uniform(-24, -22, 2000), "long": rng.uniform(-47, -45, 2000)}
    )
    vectorised = sampling.points_within_radius(
        customers["lat"], customers["long"], terminals["lat"], terminals["long"], 10.0
    )
    legacy = [
        gen_available_terminals(row, terminals, 10.0)
        for row in customers.to_dict("records")
    ]
    assert vectorised == legacy
