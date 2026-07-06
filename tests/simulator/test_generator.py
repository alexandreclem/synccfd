"""Tests for the transaction generator public API."""

import pandas as pd
import pytest

from synccfd import DatasetGenerator


def test_default_parameters():
    """DatasetGenerator exposes sane, explicit defaults."""
    generator = DatasetGenerator()
    assert generator.generator.n_customers == 10000
    assert generator.generator.n_terminals == 20000
    assert generator.generator.nb_days == 90


@pytest.fixture(scope="module")
def generated():
    """A small generated dataset shared across tests."""
    generator = DatasetGenerator(
        n_customers=20, n_terminals=50, nb_days=5, random_state=42
    )
    customer_profiles, terminal_profiles, transactions = generator.generate()
    return generator, customer_profiles, terminal_profiles, transactions


def test_generate_returns_three_frames(generated):
    _, customers, terminals, transactions = generated
    assert isinstance(customers, pd.DataFrame)
    assert isinstance(terminals, pd.DataFrame)
    assert isinstance(transactions, pd.DataFrame)
    assert len(customers) == 20
    assert len(terminals) == 50


def test_transactions_have_fraud_labels(generated):
    _, _, _, transactions = generated
    for col in ("TX_FRAUD", "TX_FRAUD_SCENARIO", "TX_TYPE", "TX_AMOUNT"):
        assert col in transactions.columns
    assert set(transactions.TX_FRAUD.unique()) <= {0, 1}
    assert set(transactions.TX_TYPE.unique()) <= {"CP", "CNP"}


def test_generate_does_not_write_to_disk(tmp_path, monkeypatch):
    """generate() must be pure — no files created in the working directory."""
    monkeypatch.chdir(tmp_path)
    DatasetGenerator(n_customers=10, n_terminals=20, nb_days=3).generate()
    assert not (tmp_path / "data").exists()


def test_save_and_read_roundtrip(tmp_path, generated):
    generator, _, _, transactions = generated
    out = tmp_path / "data"
    generator.save(output_dir=str(out), save_format="pkl")
    assert out.exists()

    begin = transactions.TX_DATETIME.min().strftime("%Y-%m-%d")
    end = transactions.TX_DATETIME.max().strftime("%Y-%m-%d")
    loaded = generator.read_transactions(
        begin_date=begin, end_date=end, save_format="pkl", data_dir=str(out)
    )
    assert len(loaded) == len(transactions)


def test_save_before_generate_raises():
    with pytest.raises(RuntimeError):
        DatasetGenerator().save()
