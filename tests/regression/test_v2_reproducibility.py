"""Regression test: a small output of ``presets.realistic()`` must not change.

The reference files were produced by ``scripts/make_regression.py`` on version 0.2.0
(100 customers, 200 terminals, 15 days, ``random_state=42``, with higher fraud
fractions so that every scenario appears). A model change that alters the data must
regenerate them on purpose (``--force``) and say so in the CHANGELOG.

Every column is compared exactly except:

- float columns, compared with ``rtol=1e-12``: the math libraries of different
  platforms round the last bit differently;
- datetime columns, normalised to the same unit first: the unit pandas infers
  (``[ns]``, ``[ms]``, ...) depends on the pandas version.
"""

import io
from pathlib import Path

import pandas as pd
import pytest

from synccfd import DatasetGenerator
from synccfd.simulator import presets
from synccfd.simulator.config import FraudConfig

REFERENCE_DIR = Path(__file__).parent / "data"
DENSE_FRAUD = {
    "s1_customer_fraction_per_day": 1e-2,
    "s2_terminal_fraction_per_day": 1e-2,
    "s2_clone_prob": 0.5,
    "s3_customer_fraction_per_day": 3e-3,
    "s4_customer_fraction_per_day": 3e-3,
    "s5_customer_fraction_per_day": 3e-3,
}
SIZE = {"n_customers": 100, "n_terminals": 200, "nb_days": 15, "random_state": 42}
NAMES = ("customers", "terminals", "merchants", "transactions")
FLOAT_RTOL = 1e-12


def _params(**fraud_overrides):
    fraud = FraudConfig.realistic(**{**DENSE_FRAUD, **fraud_overrides})
    return presets.realistic(**SIZE, fraud=fraud)


def _parquet_roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    # Parquet stores list columns (e.g. ``available_terminals``) as arrays and fixes
    # the string dtype; the same conversion makes the comparison like-for-like.
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    buffer.seek(0)
    return pd.read_parquet(buffer)


def _reference(name: str) -> pd.DataFrame:
    return pd.read_parquet(REFERENCE_DIR / f"v0_2_0_small_{name}.parquet")


def assert_matches_reference(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
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
    result = DatasetGenerator(**_params()).generate_all()
    frames = (result.customers, result.terminals, result.merchants, result.transactions)
    return dict(zip(NAMES, frames))


def test_reference_covers_every_scenario():
    scenarios = set(_reference("transactions").TX_FRAUD_SCENARIO.unique())
    assert scenarios == {0, 1, 2, 3, 4, 5}


@pytest.mark.parametrize("name", NAMES)
def test_matches_reference(regenerated, name):
    actual = _parquet_roundtrip(regenerated[name])
    assert_matches_reference(actual, _reference(name))


def test_detects_small_behaviour_change():
    """The tolerance on floats must not hide real changes (here, 5e-5 relative)."""
    _, _, transactions = DatasetGenerator(**_params(s3_amount_factor=2.0001)).generate()
    with pytest.raises(AssertionError, match="TX_AMOUNT"):
        assert_matches_reference(
            _parquet_roundtrip(transactions), _reference("transactions")
        )
