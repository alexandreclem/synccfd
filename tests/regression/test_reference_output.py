"""A small output of presets.realistic() must not change.

Floats are compared with a tiny tolerance (platforms round the last bit
differently) and datetimes in the same unit (pandas versions infer different ones).
"""

import io

import pandas as pd
import pytest
from make_regression import NAMES, reference_params, reference_path

from synccfd import DatasetGenerator

FLOAT_RTOL = 1e-12


def _parquet_roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    # Same conversion as the reference files (list columns, string dtype)
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    buffer.seek(0)
    return pd.read_parquet(buffer)


def _reference(name: str) -> pd.DataFrame:
    return pd.read_parquet(reference_path(name))


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
    result = DatasetGenerator(**reference_params()).generate_all()
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
    params = reference_params(s3_amount_factor=2.0001)
    _, _, transactions = DatasetGenerator(**params).generate()
    with pytest.raises(AssertionError, match="TX_AMOUNT"):
        assert_matches_reference(
            _parquet_roundtrip(transactions), _reference("transactions")
        )
