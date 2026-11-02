"""Ready-made configurations: keyword arguments for :class:`DatasetGenerator`.

>>> from synccfd import DatasetGenerator
>>> from synccfd.simulator import presets
>>> generator = DatasetGenerator(**presets.realistic(n_customers=1000))

- :func:`realistic`: every component (merchants, customer drift, seasonality and
  fraud injection with fraudster drift), with the defaults of ``config.py``; with no
  arguments, it is the configuration of the dissertation dataset
  (``DEFAULT_SIZE``) and the same as ``DatasetGenerator()``.
- :func:`ablation`: the components switched on cumulatively, in the order of the
  phases (the experimental design of the dissertation), with fraud fractions
  calibrated so that every step has about the same fraud rate.

Any keyword argument of :class:`DatasetGenerator` can be overridden.
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

# Size, period and seed of the dissertation dataset: the 10 000 customers x 180 days
# used to measure every phase of the v0.2 plan.
DEFAULT_SIZE: dict[str, Any] = {
    "n_customers": 10_000,
    "n_terminals": 20_000,
    "nb_days": 180,
    "start_date": "2023-01-01",
    "random_state": 42,
}

ABLATION_STEPS = ("base", "drift", "frauds", "merchants")
# Factor applied to the daily fractions of every fraud scenario at each step, so that
# the fraud rate stays close to that of the complete simulator: the mean rate of 3
# seeds (42, 7, 123) with DEFAULT_SIZE and factor 1, divided into that of
# "merchants" (0.350%). Calibrated means: 0.347%, 0.343%, 0.352%, 0.350%.
ABLATION_FRAUD_SCALE = {"base": 1.08, "drift": 1.11, "frauds": 0.95, "merchants": 1.0}
FRACTION_FIELDS = (
    "s1_customer_fraction_per_day",
    "s2_terminal_fraction_per_day",
    "s3_customer_fraction_per_day",
    "s4_customer_fraction_per_day",
    "s5_customer_fraction_per_day",
)


def scale_fraud(config: FraudConfig, factor: float) -> FraudConfig:
    """``config`` with the daily fractions of every scenario multiplied by
    ``factor`` (the scenario mix and every other parameter unchanged)."""
    if factor == 1.0:
        return config
    return dataclasses.replace(
        config, **{name: getattr(config, name) * factor for name in FRACTION_FIELDS}
    )


def realistic(**overrides) -> dict[str, Any]:
    """Merchants, customer drift, seasonality and fraud injection
    (``FraudConfig.realistic()``), with ``DEFAULT_SIZE``."""
    return ablation("merchants", **overrides)


def ablation(step: str, **overrides) -> dict[str, Any]:
    """Components switched on up to ``step``, cumulatively:

    - ``"base"``: injected frauds of scenarios 1 to 4, at constant rates, with no
      other component (customers with a stationary behaviour);
    - ``"drift"``: plus customer drift and seasonality (phase 2);
    - ``"frauds"``: with the realistic fraudsters instead: the five scenarios and
      fraudster drift (``FraudConfig.realistic()``, phase 3);
    - ``"merchants"``: plus merchants and categories (phase 4); the same as
      :func:`realistic`.

    The fraud fractions of each step are scaled by ``ABLATION_FRAUD_SCALE[step]``,
    so that the fraud rate is about the same at every step.
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
    """A preset by name: ``"realistic"`` or an ablation step (``"ablation:base"``,
    ``"ablation:drift"``, ...)."""
    if name.startswith("ablation:"):
        return ablation(name.split(":", 1)[1], **overrides)
    if name not in PRESETS:
        raise ValueError(
            f"unknown preset {name!r}; use one of {sorted(PRESETS)} or "
            f"'ablation:<step>' with step in {ABLATION_STEPS}."
        )
    return PRESETS[name](**overrides)


# Environment variables read by from_env() (scripts/make_figures.py and the tests of
# the example notebooks set them).
ENV_VARIABLES = {
    "n_customers": "SYNCCFD_N_CUSTOMERS",
    "n_terminals": "SYNCCFD_N_TERMINALS",
    "nb_days": "SYNCCFD_NB_DAYS",
    "random_state": "SYNCCFD_SEED",
}


def from_env(default: str = "realistic") -> dict[str, Any]:
    """The preset named by ``SYNCCFD_PRESET`` (default ``default``) with
    ``DEFAULT_SIZE``, overridden by ``SYNCCFD_N_CUSTOMERS``, ``SYNCCFD_N_TERMINALS``,
    ``SYNCCFD_NB_DAYS`` and ``SYNCCFD_SEED`` when they are set. Used by the example
    notebooks, so that all of them generate the same dataset."""
    overrides = {
        key: int(os.environ[name])
        for key, name in ENV_VARIABLES.items()
        if os.environ.get(name)
    }
    name = os.environ.get("SYNCCFD_PRESET", default)
    return get(name, **{**DEFAULT_SIZE, **overrides})
