"""Golden regression test: the default configuration must reproduce v0.1.0.

The reference files were produced by ``scripts/make_golden.py`` on the v0.1.0 code.

Every column is compared exactly except:

- float columns (the coordinates), compared with ``rtol=1e-12``: the math libraries
  of different platforms round the last bit differently (up to ~1.4e-14 observed
  between Windows and Linux on coordinates of magnitude ~20);
- datetime columns, normalised to the same unit first: the unit pandas infers
  (``[ns]``, ``[ms]``, ...) depends on the pandas version.
"""

import io
from pathlib import Path

import pandas as pd
import pytest

from synccfd import DatasetGenerator
from synccfd.simulator.config import FraudConfig

GOLDEN_DIR = Path(__file__).parent / "data"
GOLDEN_PARAMS = {
    "n_customers": 100,
    "n_terminals": 200,
    "nb_days": 15,
    "start_date": "2023-01-01",
    "random_state": 42,
}
GOLDEN_NAMES = ("customers", "terminals", "transactions")
FLOAT_RTOL = 1e-12


def _parquet_roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    # Parquet stores list columns (e.g. ``available_terminals``) as arrays; passing the
    # regenerated frame through the same conversion makes the comparison like-for-like.
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    buffer.seek(0)
    return pd.read_parquet(buffer)


def assert_matches_golden(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    for column in expected.columns:
        left, right = actual[column], expected[column]
        if pd.api.types.is_datetime64_any_dtype(right):
            assert pd.api.types.is_datetime64_any_dtype(left), column
            left = left.astype("datetime64[ns]")
            right = right.astype("datetime64[ns]")
            pd.testing.assert_series_equal(left, right, check_exact=True, obj=column)
        elif pd.api.types.is_float_dtype(right):
            pd.testing.assert_series_equal(
                left, right, check_exact=False, rtol=FLOAT_RTOL, atol=0.0, obj=column
            )
        else:
            pd.testing.assert_series_equal(left, right, check_exact=True, obj=column)


@pytest.fixture(scope="module")
def regenerated():
    frames = DatasetGenerator(**GOLDEN_PARAMS).generate()
    return dict(zip(GOLDEN_NAMES, frames))


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_matches_v0_1_0(regenerated, name):
    expected = pd.read_parquet(GOLDEN_DIR / f"v0_1_0_small_{name}.parquet")
    actual = _parquet_roundtrip(regenerated[name])
    assert_matches_golden(actual, expected)


def test_detects_small_behaviour_change():
    """The tolerance on floats must not hide real changes (here, 2e-5 relative)."""
    _, _, transactions = DatasetGenerator(
        **GOLDEN_PARAMS, fraud=FraudConfig(amount_multiplier=5.0001)
    ).generate()
    expected = pd.read_parquet(GOLDEN_DIR / "v0_1_0_small_transactions.parquet")
    with pytest.raises(AssertionError, match="TX_AMOUNT"):
        assert_matches_golden(_parquet_roundtrip(transactions), expected)
