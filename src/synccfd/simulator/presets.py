"""Ready-made configurations: keyword arguments for :class:`DatasetGenerator`.

>>> from synccfd import DatasetGenerator
>>> from synccfd.simulator import presets
>>> generator = DatasetGenerator(**presets.realistic(n_customers=1000))

- :func:`legacy`: the v0.1 engine and model, exactly.
- :func:`realistic`: every v0.2 component (merchants, drift, seasonality and fraud
  injection), with the defaults of ``config.py``; with no arguments, it is the
  configuration of the dissertation dataset (``DEFAULT_SIZE``).
- :func:`ablation`: the components switched on cumulatively, in the order of the
  phases, always with the v2 engine (the experimental design of the dissertation).

Any keyword argument of :class:`DatasetGenerator` can be overridden.
"""

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


def legacy(**overrides) -> dict[str, Any]:
    """The v0.1 engine with the v0.1 fraud model; with no arguments, exactly
    ``DatasetGenerator()`` of v0.1."""
    return {"engine": "legacy", "fraud": FraudConfig(), **overrides}


def realistic(**overrides) -> dict[str, Any]:
    """Merchants, customer drift, seasonality and fraud injection
    (``FraudConfig.realistic()``), with ``DEFAULT_SIZE``."""
    return ablation("merchants", **overrides)


def ablation(step: str, **overrides) -> dict[str, Any]:
    """Components switched on up to ``step``, cumulatively, with the v2 engine:

    - ``"base"``: the v0.1 model (frauds altering transactions, as in v0.1), with
      rates proportional to the population;
    - ``"drift"``: plus customer drift and seasonality (phase 2);
    - ``"frauds"``: with fraud injection instead (``FraudConfig.realistic()``,
      phase 3);
    - ``"merchants"``: plus merchants and categories (phase 4); the same as
      :func:`realistic`.
    """
    if step not in ABLATION_STEPS:
        raise ValueError(f"step must be one of {ABLATION_STEPS}, got {step!r}.")
    level = ABLATION_STEPS.index(step)
    values: dict[str, Any] = {
        **DEFAULT_SIZE,
        "engine": "v2",
        "fraud": FraudConfig(rate_mode="proportional"),
    }
    if level >= 1:
        values.update(drift=DriftConfig(), seasonality=SeasonalityConfig())
    if level >= 2:
        values["fraud"] = FraudConfig.realistic()
    if level >= 3:
        values["merchants"] = MerchantConfig()
    values.update(overrides)
    return values


PRESETS: dict[str, Callable[..., dict[str, Any]]] = {
    "legacy": legacy,
    "realistic": realistic,
}


def get(name: str, **overrides) -> dict[str, Any]:
    """A preset by name: ``"legacy"``, ``"realistic"`` or an ablation step
    (``"ablation:base"``, ``"ablation:drift"``, ...)."""
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
    return get(
        os.environ.get("SYNCCFD_PRESET", default), **{**DEFAULT_SIZE, **overrides}
    )
