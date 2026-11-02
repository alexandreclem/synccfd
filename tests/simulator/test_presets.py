"""Presets, ``generate_all()`` and ``config.json`` (phase 5)."""

import json

import pandas as pd
import pytest

import synccfd
from synccfd import DatasetGenerator, SimulationResult
from synccfd.simulator import presets
from synccfd.simulator.config import (
    DriftConfig,
    FraudConfig,
    MerchantConfig,
    SeasonalityConfig,
)

SMALL = {"n_customers": 150, "n_terminals": 300, "nb_days": 20}


def test_legacy_preset_is_v0_1():
    assert presets.legacy() == {"engine": "legacy", "fraud": FraudConfig()}
    params = {**SMALL, "start_date": "2023-01-01", "random_state": 3}
    expected = DatasetGenerator(**params).generate()
    got = DatasetGenerator(**presets.legacy(**params)).generate()
    for a, b in zip(expected, got):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_realistic_preset_switches_everything_on():
    values = presets.realistic()
    assert values["engine"] == "v2"
    assert values["fraud"] == FraudConfig.realistic()
    assert values["drift"] == DriftConfig()
    assert values["seasonality"] == SeasonalityConfig()
    assert values["merchants"] == MerchantConfig()
    for key, value in presets.DEFAULT_SIZE.items():
        assert values[key] == value
    assert presets.realistic(n_customers=7)["n_customers"] == 7


def test_ablation_is_cumulative():
    steps = [presets.ablation(step) for step in presets.ABLATION_STEPS]
    base, drift, frauds, merchants = steps
    assert all(s["engine"] == "v2" for s in steps)
    assert base["fraud"] == FraudConfig(rate_mode="proportional")
    assert "drift" not in base and "merchants" not in base
    assert drift["drift"] == DriftConfig() and drift["fraud"] == base["fraud"]
    assert frauds["fraud"] == FraudConfig.realistic() and "merchants" not in frauds
    assert merchants == presets.realistic()
    with pytest.raises(ValueError):
        presets.ablation("everything")


def test_get_by_name():
    assert presets.get("realistic") == presets.realistic()
    assert presets.get("ablation:drift", nb_days=5) == presets.ablation(
        "drift", nb_days=5
    )
    with pytest.raises(ValueError):
        presets.get("paper")


@pytest.mark.parametrize("step", presets.ABLATION_STEPS)
def test_every_ablation_step_generates(step):
    _, _, transactions = DatasetGenerator(**presets.ablation(step, **SMALL)).generate()
    assert len(transactions) > 0
    assert ("MERCHANT_CATEGORY" in transactions) == (step == "merchants")
    assert ("GT_REGIME" in transactions) == (step != "base")


def test_generate_all_and_config_json(tmp_path):
    generator = DatasetGenerator(**presets.realistic(**SMALL))
    result = generator.generate_all()
    assert isinstance(result, SimulationResult)
    assert result.merchants is not None and len(result.merchants) > 0
    assert len(result.transactions) == len(generator.transactions)
    config = result.config
    assert config["synccfd_version"] == synccfd.__version__
    assert config["engine"] == "v2" and config["n_customers"] == SMALL["n_customers"]
    assert config["fraud"]["type"] == "FraudConfig"
    assert config["fraud"]["mechanism"] == "inject"
    assert config["merchants"]["categories"][0]["name"] == "supermercado"

    generator.save(output_dir=str(tmp_path), save_format="parquet", save_by_day=False)
    with open(tmp_path / "config.json", encoding="utf-8") as f:
        assert json.load(f) == json.loads(json.dumps(config))


def test_config_without_components():
    config = DatasetGenerator(**SMALL).config
    assert config["engine"] == "legacy"
    assert config["drift"] is None and config["merchants"] is None
    json.dumps(config)


def test_from_env(monkeypatch):
    for name in ("SYNCCFD_PRESET", *presets.ENV_VARIABLES.values()):
        monkeypatch.delenv(name, raising=False)
    assert presets.from_env() == presets.realistic()
    monkeypatch.setenv("SYNCCFD_PRESET", "legacy")
    monkeypatch.setenv("SYNCCFD_NB_DAYS", "12")
    assert presets.from_env() == presets.legacy(
        **{**presets.DEFAULT_SIZE, "nb_days": 12}
    )
