"""Dataset generator."""

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

# Every component on by default, as in presets.realistic() (the configurations
# are frozen, so sharing one instance is safe)
DEFAULT_FRAUD = FraudConfig.realistic()
DEFAULT_DRIFT = DriftConfig()
DEFAULT_SEASONALITY = SeasonalityConfig()
DEFAULT_MERCHANTS = MerchantConfig()
# Used when fraud=None
NO_FRAUD = FraudConfig(scenarios=())


@dataclass
class SimulationResult:
    """Everything one simulation produces.

    Attributes:
        merchants: Merchant profiles, or None without merchants.
        config: The effective configuration, as plain JSON types.
    """

    customers: pd.DataFrame
    terminals: pd.DataFrame
    merchants: Optional[pd.DataFrame]
    transactions: pd.DataFrame
    config: dict[str, Any]


class DatasetGenerator:
    """Generates synthetic credit card transactions with labelled frauds.

    The defaults are the complete simulator, the same as presets.realistic().
    Pass None to switch a component off.

    Example:
        >>> generator = DatasetGenerator(n_customers=1000, n_terminals=2000)
        >>> customers, terminals, transactions = generator.generate()
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
        """Set up the generator.

        Args:
            n_terminals: Number of physical terminals.
            city_radius: Radius (km) around a city center where customers and
                terminals are placed.
            available_terminals_radius: Radius (km) around a customer within which
                terminals are reachable for card-present purchases.
            start_date: First day, as YYYY-MM-DD.
            random_state: Seed of every random draw.
            fraud: Fraud scenarios; None for no fraud.
            drift: Customer concept drift; None to disable it.
            seasonality: Calendar patterns; None to disable them.
            merchants: Merchants and categories; None to disable them.
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

        # Filled by generate(), used by save()
        self.customer_profiles: Optional[pd.DataFrame] = None
        self.terminal_profiles: Optional[pd.DataFrame] = None
        self.transactions: Optional[pd.DataFrame] = None
        self.merchant_profiles: Optional[pd.DataFrame] = None
        self._loc_features: Optional[LocationFeatures] = None

    @property
    def config(self) -> dict[str, Any]:
        """The effective configuration as plain JSON types, with the version."""
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
        """Generate the dataset and return it as a SimulationResult."""
        customers, terminals, transactions = self.generate()
        return SimulationResult(
            customers=customers,
            terminals=terminals,
            merchants=self.merchant_profiles,
            transactions=transactions,
            config=self.config,
        )

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate the profiles and the labelled transactions.

        Returns:
            (customer_profiles, terminal_profiles, transactions). Columns starting
            with GT_ and TX_FRAUD_SCENARIO are ground truth, not model inputs.
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

        # Keep the last result so save() does not generate again
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
        """Save the last generated dataset and its config.json.

        Args:
            save_format: "pkl", "parquet" or "csv".
            save_by_day: Write one transactions file per day.
        """
        if self.transactions is None:
            raise RuntimeError("Nothing to save yet: call generate() first.")

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
        """Load saved transactions between two dates (YYYY-MM-DD).

        A dataset saved with save_by_day=False is returned whole, whatever the dates.
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
        """Geographic data and locations of the last generated dataset."""
        if self._loc_features is None:
            raise RuntimeError("Call generate() first.")
        return self._loc_features
