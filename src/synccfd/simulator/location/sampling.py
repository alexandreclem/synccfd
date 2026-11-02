"""Vectorised geographic sampling with an explicit ``numpy.random.Generator``.

Used by ``LocationFeatures`` and the fraud scenarios; the global ``random`` and
``numpy.random`` states are never used.
"""

from typing import Any

import numpy as np
import pandas as pd
import shapely

# Mean Earth radius (km), the same value the ``haversine`` package uses.
EARTH_RADIUS_KM = 6371.0088


def destination_points(
    lat: np.ndarray, long: np.ndarray, distance_km: np.ndarray, bearing_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Points at ``distance_km`` and ``bearing_deg`` from ``(lat, long)`` (sphere)."""
    phi1 = np.radians(lat)
    lambda1 = np.radians(long)
    theta = np.radians(bearing_deg)
    delta = np.asarray(distance_km) / EARTH_RADIUS_KM

    sin_phi2 = np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(
        theta
    )
    phi2 = np.arcsin(np.clip(sin_phi2, -1.0, 1.0))
    lambda2 = lambda1 + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(phi1),
        np.cos(delta) - np.sin(phi1) * sin_phi2,
    )
    # Normalise longitude to [-180, 180)
    long2 = (np.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return np.degrees(phi2), long2


def random_points_around(
    center_lat: np.ndarray,
    center_long: np.ndarray,
    radius_km: float,
    polygon: Any,
    rng: np.random.Generator,
    max_attempts: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """One random point per center, inside ``polygon``.

    Distance uniform in ``[0, radius_km]``, bearing uniform in ``[0, 360)``; points
    outside the polygon are redrawn.
    """
    center_lat = np.asarray(center_lat, dtype=float)
    center_long = np.asarray(center_long, dtype=float)
    n = len(center_lat)
    lat = np.empty(n)
    long = np.empty(n)
    pending: np.ndarray = np.arange(n)
    for _ in range(max_attempts):
        if len(pending) == 0:
            return lat, long
        distance = rng.uniform(0.0, radius_km, len(pending))
        bearing = rng.uniform(0.0, 360.0, len(pending))
        new_lat, new_long = destination_points(
            center_lat[pending], center_long[pending], distance, bearing
        )
        inside = shapely.contains_xy(polygon, new_long, new_lat)
        lat[pending[inside]] = new_lat[inside]
        long[pending[inside]] = new_long[inside]
        pending = pending[~inside]
    if len(pending) == 0:
        return lat, long
    raise RuntimeError(
        f"Could not generate {len(pending)} valid point(s) within {max_attempts} "
        f"attempts with radius {radius_km}km."
    )


def random_points_around_cities(
    num_points: int,
    radius_km: float,
    polygon: Any,
    cities: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Population-weighted cities, then one random point around each.

    Returns ``random_points`` (``lat``/``long``) and the sampled ``cities``.
    """
    weights = cities["pop_21"].to_numpy(dtype=float)
    chosen = rng.choice(
        len(cities), size=num_points, replace=True, p=weights / weights.sum()
    )
    cities_sample = cities.iloc[chosen]
    lat, long = random_points_around(
        cities_sample["lat"].to_numpy(),
        cities_sample["long"].to_numpy(),
        radius_km,
        polygon,
        rng,
    )
    random_points = pd.DataFrame({"lat": lat, "long": long})
    return {"random_points": random_points, "cities": cities_sample}


def haversine_km(
    lat1: np.ndarray, long1: np.ndarray, lat2: np.ndarray, long2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (km); inputs broadcast against each other."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(long2) - np.radians(long1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def points_within_radius(
    lat: np.ndarray,
    long: np.ndarray,
    target_lat: np.ndarray,
    target_long: np.ndarray,
    radius_km: float,
    chunk_size: int = 512,
) -> list[list[int]]:
    """For each point, the (sorted) indices of the targets within ``radius_km``."""
    lat = np.asarray(lat, dtype=float)
    long = np.asarray(long, dtype=float)
    target_lat = np.asarray(target_lat, dtype=float)
    target_long = np.asarray(target_long, dtype=float)
    result: list[list[int]] = []
    for start in range(0, len(lat), chunk_size):
        stop = start + chunk_size
        distances = haversine_km(
            lat[start:stop, None], long[start:stop, None], target_lat, target_long
        )
        for row in distances <= radius_km:
            result.append(np.flatnonzero(row).tolist())
    return result
