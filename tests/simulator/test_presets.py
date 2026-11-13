"""Presets, generate_all() and config.json."""

import json

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


def test_realistic_preset_switches_everything_on():
    values = presets.realistic()
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
    assert base["fraud"].scenarios == (1, 2, 3, 4)
    assert not base["fraud"].intensity_schedule
    assert base["drift"] is None and base["seasonality"] is None
    assert base["merchants"] is None
    assert drift["drift"] == DriftConfig() and drift["seasonality"] is not None
    assert drift["fraud"].scenarios == base["fraud"].scenarios
    assert frauds["fraud"].intensity_schedule and frauds["merchants"] is None
    assert merchants == presets.realistic()
    with pytest.raises(ValueError):
        presets.ablation("everything")


def test_ablation_scales_the_fraud_fractions():
    # Every step injects frauds; only the daily fractions change, by one factor
    for step in presets.ABLATION_STEPS:
        fraud = presets.ablation(step)["fraud"]
        reference = (
            FraudConfig.realistic()
            if step in ("frauds", "merchants")
            else (FraudConfig(scenarios=(1, 2, 3, 4)))
        )
        factor = presets.ABLATION_FRAUD_SCALE[step]
        for name in presets.FRACTION_FIELDS:
            assert getattr(fraud, name) == pytest.approx(
                getattr(reference, name) * factor
            )
        assert fraud.s3_amount_factor == reference.s3_amount_factor


def test_get_by_name():
    assert presets.get("realistic") == presets.realistic()
    assert presets.get("ablation:drift", nb_days=5) == presets.ablation(
        "drift", nb_days=5
    )
    for name in ("paper", "legacy"):
        with pytest.raises(ValueError):
            presets.get(name)


@pytest.mark.parametrize("step", presets.ABLATION_STEPS)
def test_every_ablation_step_generates(step):
    _, _, transactions = DatasetGenerator(**presets.ablation(step, **SMALL)).generate()
    assert len(transactions) > 0
    assert ("MERCHANT_CATEGORY" in transactions) == (step == "merchants")
    assert ("GT_REGIME" in transactions) == (step != "base")
    assert "GT_FRAUD_SCENARIO" in transactions


def test_generate_all_and_config_json(tmp_path):
    generator = DatasetGenerator(**presets.realistic(**SMALL))
    result = generator.generate_all()
    assert isinstance(result, SimulationResult)
    assert result.merchants is not None and len(result.merchants) > 0
    assert len(result.transactions) == len(generator.transactions)
    config = result.config
    assert config["synccfd_version"] == synccfd.__version__
    assert config["n_customers"] == SMALL["n_customers"]
    assert "engine" not in config
    assert config["fraud"]["type"] == "FraudConfig"
    assert config["fraud"]["scenarios"] == [1, 2, 3, 4, 5]
    assert config["merchants"]["categories"][0]["name"] == "supermercado"

    generator.save(output_dir=str(tmp_path), save_format="parquet", save_by_day=False)
    with open(tmp_path / "config.json", encoding="utf-8") as f:
        assert json.load(f) == json.loads(json.dumps(config))


def test_config_without_components():
    config = DatasetGenerator(
        **SMALL, fraud=None, drift=None, seasonality=None, merchants=None
    ).config
    assert config["fraud"] is None and config["drift"] is None
    assert config["merchants"] is None
    json.dumps(config)


def test_from_env(monkeypatch):
    for name in ("SYNCCFD_PRESET", *presets.ENV_VARIABLES.values()):
        monkeypatch.delenv(name, raising=False)
    assert presets.from_env() == presets.realistic()
    monkeypatch.setenv("SYNCCFD_PRESET", "ablation:base")
    monkeypatch.setenv("SYNCCFD_NB_DAYS", "12")
    assert presets.from_env() == presets.ablation(
        "base", **{**presets.DEFAULT_SIZE, "nb_days": 12}
    )
