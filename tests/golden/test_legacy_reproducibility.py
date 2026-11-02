"""Golden regression test: the default configuration must reproduce v0.1.0 exactly.

The reference files were produced by ``scripts/make_golden.py`` on the v0.1.0 code.
"""

import io
from pathlib import Path

import pandas as pd
import pytest

from synccfd import DatasetGenerator

GOLDEN_DIR = Path(__file__).parent / "data"
GOLDEN_PARAMS = {
    "n_customers": 100,
    "n_terminals": 200,
    "nb_days": 15,
    "start_date": "2023-01-01",
    "random_state": 42,
}
GOLDEN_NAMES = ("customers", "terminals", "transactions")


def _parquet_roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    # Parquet stores list columns (e.g. ``available_terminals``) as arrays; passing the
    # regenerated frame through the same conversion makes the comparison like-for-like.
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    buffer.seek(0)
    return pd.read_parquet(buffer)


@pytest.fixture(scope="module")
def regenerated():
    frames = DatasetGenerator(**GOLDEN_PARAMS).generate()
    return dict(zip(GOLDEN_NAMES, frames))


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_matches_v0_1_0(regenerated, name):
    expected = pd.read_parquet(GOLDEN_DIR / f"v0_1_0_small_{name}.parquet")
    actual = _parquet_roundtrip(regenerated[name])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
