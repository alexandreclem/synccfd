"""Public entry point: ``DatasetGenerator``."""

import datetime
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from synccfd.utils.data import DataManager

from . import engine
from .config import (
    DriftConfig,
    FraudConfig,
    MerchantConfig,
    SeasonalityConfig,
    to_jsonable,
)
from .location.features import LocationFeatures

# Default components: every component on, as in ``presets.realistic()`` (the
# configurations are frozen, so sharing one instance is safe).
DEFAULT_FRAUD = FraudConfig.realistic()
DEFAULT_DRIFT = DriftConfig()
DEFAULT_SEASONALITY = SeasonalityConfig()
DEFAULT_MERCHANTS = MerchantConfig()
# No fraud at all (``fraud=None``)
NO_FRAUD = FraudConfig(scenarios=())


@dataclass
class SimulationResult:
    """Everything one simulation produces (:meth:`DatasetGenerator.generate_all`).

    Attributes:
        customers: Customer profiles.
        terminals: Terminal profiles.
        merchants: Merchant profiles, or ``None`` without merchants.
        transactions: Labelled transactions.
        config: The effective configuration, JSON-serialisable (also written to
            ``config.json`` by :meth:`DatasetGenerator.save`).
    """

    customers: pd.DataFrame
    terminals: pd.DataFrame
    merchants: Optional[pd.DataFrame]
    transactions: pd.DataFrame
    config: dict[str, Any]


class DatasetGenerator:
    """Main entry point for generating synthetic credit card fraud datasets.

    The defaults are the complete simulator, the same as ``presets.realistic()``:
    merchants and categories, customer concept drift, seasonality and fraud
    injection with fraudster drift (``FraudConfig.realistic()``), for 10 000
    customers, 20 000 terminals and 180 days from 2023-01-01. Pass ``None`` to
    switch a component off (``fraud=None``: no fraud at all).

    All randomness comes from local random streams derived from ``random_state``;
    the global ``random`` and ``numpy.random`` states are never used.

    Example:
        >>> generator = DatasetGenerator(n_customers=1000, n_terminals=2000)
        >>> customer_profiles, terminal_profiles, transactions = generator.generate()
        >>> print(f"Generated {len(transactions)} transactions")
    """

    def __init__(
        self,
        n_customers: int = 10_000,
        n_terminals: int = 20_000,
        nb_days: int = 180,
        city_radius: float = 12.0,
        available_terminals_radius: float = 10.0,
        start_date: str = "2023-01-01",
        random_state: Optional[int] = 42,
        fraud: Optional[FraudConfig] = DEFAULT_FRAUD,
        drift: Optional[DriftConfig] = DEFAULT_DRIFT,
        seasonality: Optional[SeasonalityConfig] = DEFAULT_SEASONALITY,
        merchants: Optional[MerchantConfig] = DEFAULT_MERCHANTS,
    ):
        """Initialize DatasetGenerator.

        Args:
            n_customers: Number of customers. Defaults to 10 000.
            n_terminals: Number of physical terminals. Defaults to 20 000.
            nb_days: Number of days of transactions. Defaults to 180.
            city_radius: Radius (km) from a city center used to place customers
                and terminals. Defaults to 12.0.
            available_terminals_radius: Radius (km) around a customer within which
                terminals are reachable for Card-Present transactions. Defaults to 10.0.
            start_date: First day (``YYYY-MM-DD``). Defaults to '2023-01-01'.
            random_state: Seed of every random draw. Defaults to 42.
            fraud: Fraud scenarios. Defaults to ``FraudConfig.realistic()``;
                ``None``: no fraud.
            drift: Customer concept drift. Defaults to ``DriftConfig()``; ``None``
                disables it.
            seasonality: Calendar patterns. Defaults to ``SeasonalityConfig()``;
                ``None`` disables them.
            merchants: Merchants and categories. Defaults to ``MerchantConfig()``;
                ``None`` disables them. With merchants, the transactions get
                ``MERCHANT_ID`` and ``MERCHANT_CATEGORY``, the terminal profiles get
                ``MERCHANT_ID``, and ``merchant_profiles`` is filled by
                :meth:`generate`.
        """
        self.fraud_config = fraud
        self.drift_config = drift
        self.seasonality_config = seasonality
        self.merchant_config = merchants
        self.n_customers = n_customers
        self.n_terminals = n_terminals
        self.nb_days = nb_days
        self.city_radius = city_radius
        self.available_terminals_radius = available_terminals_radius
        self.start_date = start_date
        self.random_state = random_state

        # Populated by generate(); consumed by save().
        self.customer_profiles: Optional[pd.DataFrame] = None
        self.terminal_profiles: Optional[pd.DataFrame] = None
        self.transactions: Optional[pd.DataFrame] = None
        self.merchant_profiles: Optional[pd.DataFrame] = None
        self._loc_features: Optional[LocationFeatures] = None

    @property
    def config(self) -> dict[str, Any]:
        """The effective configuration as plain JSON types, with the version of
        ``synccfd`` (written to ``config.json`` by :meth:`save`)."""
        from synccfd import __version__

        return {
            "synccfd_version": __version__,
            "n_customers": self.n_customers,
            "n_terminals": self.n_terminals,
            "nb_days": self.nb_days,
            "city_radius": self.city_radius,
            "available_terminals_radius": self.available_terminals_radius,
            "start_date": self.start_date,
            "random_state": self.random_state,
            "fraud": to_jsonable(self.fraud_config),
            "drift": to_jsonable(self.drift_config),
            "seasonality": to_jsonable(self.seasonality_config),
            "merchants": to_jsonable(self.merchant_config),
        }

    def generate_all(self) -> SimulationResult:
        """Generate the dataset, like :meth:`generate`, and return everything in a
        :class:`SimulationResult` (with the merchants and the configuration)."""
        customers, terminals, transactions = self.generate()
        return SimulationResult(
            customers=customers,
            terminals=terminals,
            merchants=self.merchant_profiles,
            transactions=transactions,
            config=self.config,
        )

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate the complete dataset (profiles + labelled transactions).

        Returns:
            ``(customer_profiles, terminal_profiles, transactions)``. The columns are
            described in the documentation ("Output"); columns starting with ``GT_``
            are ground truth and must never be used as model inputs.
        """
        result = engine.simulate(
            n_customers=self.n_customers,
            n_terminals=self.n_terminals,
            nb_days=self.nb_days,
            city_radius=self.city_radius,
            available_terminals_radius=self.available_terminals_radius,
            start_date=datetime.datetime.strptime(self.start_date, "%Y-%m-%d"),
            random_state=self.random_state,
            fraud=self.fraud_config if self.fraud_config is not None else NO_FRAUD,
            drift=self.drift_config,
            seasonality=self.seasonality_config,
            merchants=self.merchant_config,
        )
        self.merchant_profiles = result.merchant_profiles
        self._loc_features = result.loc_features

        # Cache the last result so save() can be called without re-generating.
        self.customer_profiles = result.customer_profiles
        self.terminal_profiles = result.terminal_profiles
        self.transactions = result.transactions
        return result.customer_profiles, result.terminal_profiles, result.transactions

    def save(
        self,
        output_dir: str = "data",
        save_format: str = "pkl",
        save_by_day: bool = True,
    ) -> None:
        """Persist the most recently generated dataset to disk.

        Writes customer/terminal (and, with merchants, merchant) profiles,
        (day-partitioned) transactions and the configuration (``config.json``, see
        :attr:`config`) under ``output_dir``. Call :meth:`generate` first.

        Args:
            output_dir: Base directory to write the dataset into. Defaults to 'data'.
            save_format: File format ('pkl', 'parquet', or 'csv'). Defaults to 'pkl'.
            save_by_day: Split transactions into one file per day. Defaults to True.

        Raises:
            RuntimeError: If called before :meth:`generate`.
        """
        if self.transactions is None:
            raise RuntimeError("Nothing to save yet — call generate() first.")

        DataManager(data_dir=output_dir).save_data(
            data_type="raw",
            transactions=self.transactions,
            customer_profiles=self.customer_profiles,
            terminal_profiles=self.terminal_profiles,
            merchant_profiles=self.merchant_profiles,
            save_format=save_format,
            save_by_day=save_by_day,
        )
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to '{output_dir}'.")

    def read_transactions(
        self,
        begin_date: str,
        end_date: str,
        save_format: str = "pkl",
        data_dir: str = "data",
    ) -> pd.DataFrame:
        """Load previously saved transaction data and combine it.

        Args:
            begin_date: Start date for loading data (YYYY-MM-DD)
            end_date: End date for loading data (YYYY-MM-DD)
            save_format: Format to read ('pkl', 'parquet', or 'csv')
            data_dir: Base directory the dataset was saved into. Defaults to 'data'.

        Returns:
            DataFrame with combined transaction data.
        """
        transactions = DataManager(data_dir=data_dir).read_transactions(
            begin_date=begin_date,
            end_date=end_date,
            format=save_format,
            data_type="raw",
        )

        if transactions.empty:
            raise ValueError("No transactions found for the specified date range.")

        return transactions

    def get_location_features(self) -> LocationFeatures:
        """Geographic data and locations used by the last generated dataset."""
        if self._loc_features is None:
            raise RuntimeError("Call generate() first.")
        return self._loc_features
