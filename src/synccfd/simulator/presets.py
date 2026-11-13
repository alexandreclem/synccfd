"""Ready-made keyword arguments for DatasetGenerator.

>>> from synccfd import DatasetGenerator
>>> from synccfd.simulator import presets
>>> generator = DatasetGenerator(**presets.realistic(n_customers=1000))
"""

import dataclasses
import os
from typing import Any, Callable

from .config import (
    DriftConfig,
    FraudConfig,
    MerchantConfig,
    SeasonalityConfig,
)

# Size, period and seed of the reference dataset
DEFAULT_SIZE: dict[str, Any] = {
    "n_customers": 10_000,
    "n_terminals": 20_000,
    "nb_days": 180,
    "start_date": "2023-01-01",
    "random_state": 42,
}

ABLATION_STEPS = ("base", "drift", "frauds", "merchants")
# Keeps the fraud rate of each step close to the complete simulator (0.350%),
# calibrated on seeds 42, 7 and 123 with DEFAULT_SIZE
ABLATION_FRAUD_SCALE = {"base": 1.08, "drift": 1.11, "frauds": 0.95, "merchants": 1.0}
FRACTION_FIELDS = (
    "s1_customer_fraction_per_day",
    "s2_terminal_fraction_per_day",
    "s3_customer_fraction_per_day",
    "s4_customer_fraction_per_day",
    "s5_customer_fraction_per_day",
)


def scale_fraud(config: FraudConfig, factor: float) -> FraudConfig:
    """Copy of config with the daily fractions of every scenario times factor."""
    if factor == 1.0:
        return config
    return dataclasses.replace(
        config, **{name: getattr(config, name) * factor for name in FRACTION_FIELDS}
    )


def realistic(**overrides) -> dict[str, Any]:
    """The complete simulator; overrides replace any DatasetGenerator argument."""
    return ablation("merchants", **overrides)


def ablation(step: str, **overrides) -> dict[str, Any]:
    """Components switched on cumulatively up to step, at a similar fraud rate.

    Args:
        step: "base" (frauds 1 to 4 only), "drift" (plus customer drift and
            seasonality), "frauds" (all five scenarios with fraudster drift) or
            "merchants" (plus merchants, the same as realistic()).
    """
    if step not in ABLATION_STEPS:
        raise ValueError(f"step must be one of {ABLATION_STEPS}, got {step!r}.")
    level = ABLATION_STEPS.index(step)
    if level < 2:
        fraud = FraudConfig(scenarios=(1, 2, 3, 4))
    else:
        fraud = FraudConfig.realistic()
    values: dict[str, Any] = {
        **DEFAULT_SIZE,
        "fraud": scale_fraud(fraud, ABLATION_FRAUD_SCALE[step]),
        "drift": DriftConfig() if level >= 1 else None,
        "seasonality": SeasonalityConfig() if level >= 1 else None,
        "merchants": MerchantConfig() if level >= 3 else None,
    }
    values.update(overrides)
    return values


PRESETS: dict[str, Callable[..., dict[str, Any]]] = {"realistic": realistic}


def get(name: str, **overrides) -> dict[str, Any]:
    """A preset by name: "realistic" or "ablation:<step>"."""
    if name.startswith("ablation:"):
        return ablation(name.split(":", 1)[1], **overrides)
    if name not in PRESETS:
        raise ValueError(
            f"unknown preset {name!r}; use one of {sorted(PRESETS)} or "
            f"'ablation:<step>' with step in {ABLATION_STEPS}."
        )
    return PRESETS[name](**overrides)


# Read by from_env(), so the notebooks can run with another size or seed
ENV_VARIABLES = {
    "n_customers": "SYNCCFD_N_CUSTOMERS",
    "n_terminals": "SYNCCFD_N_TERMINALS",
    "nb_days": "SYNCCFD_NB_DAYS",
    "random_state": "SYNCCFD_SEED",
}


def from_env(default: str = "realistic") -> dict[str, Any]:
    """The preset named by SYNCCFD_PRESET, sized by the SYNCCFD_* variables."""
    overrides = {
        key: int(os.environ[name])
        for key, name in ENV_VARIABLES.items()
        if os.environ.get(name)
    }
    name = os.environ.get("SYNCCFD_PRESET", default)
    return get(name, **{**DEFAULT_SIZE, **overrides})
