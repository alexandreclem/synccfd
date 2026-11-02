"""Public entry point: ``DatasetGenerator`` chooses the legacy or the v2 engine."""

import datetime
from typing import Optional

import pandas as pd

from synccfd.utils.data import DataManager

from . import engine as v2
from .config import DriftConfig, FraudConfig, MerchantConfig, SeasonalityConfig
from .legacy.fraud import FraudSimulator
from .legacy.generator import _TransactionGenerator
from .location.features import LocationFeatures

ENGINES = ("auto", "legacy", "v2")


class DatasetGenerator:
    """Main entry point for generating synthetic credit card fraud datasets.

    This class provides a simple interface to generate synthetic credit card
    transaction datasets including customer profiles, terminal profiles, and
    transactions with fraud labels.

    The generator creates realistic transaction patterns based on:
    - Customer location and spending habits
    - Terminal distribution across cities
    - Time-based transaction patterns (different for CP vs CNP transactions)
    - Four different fraud scenarios

    Two engines produce the data:

    - ``"legacy"``: the v0.1.0 code, unchanged; with the default configuration it
      reproduces v0.1.0 exactly.
    - ``"v2"``: a vectorised engine with the same model, drawing only from local
      random streams derived from ``random_state`` (the global ``random`` and
      ``numpy.random`` states are never used). It reproduces the v0.1 model
      statistically, not draw by draw.

    ``engine="auto"`` (default) uses ``"legacy"`` unless a component that only the
    v2 engine supports is enabled: ``drift``, ``seasonality``, ``merchants`` or
    fraud injection
    (``FraudConfig(mechanism="inject")``).

    Example:
        >>> generator = DatasetGenerator(n_customers=1000, n_terminals=10000)
        >>> customer_profiles, terminal_profiles, transactions = generator.generate()
        >>> print(f"Generated {len(transactions)} transactions")
    """

    def __init__(
        self,
        n_customers: int = 10000,
        n_terminals: int = 20000,
        nb_days: int = 90,
        city_radius: float = 12.0,
        available_terminals_radius: float = 10.0,
        start_date: str = "2018-04-01",
        random_state: Optional[int] = 42,
        fraud: Optional[FraudConfig] = None,
        drift: Optional[DriftConfig] = None,
        seasonality: Optional[SeasonalityConfig] = None,
        merchants: Optional[MerchantConfig] = None,
        engine: str = "auto",
    ):
        """Initialize DatasetGenerator.

        Args:
            n_customers: Number of customers to generate. Defaults to 10000.
            n_terminals: Number of terminals to generate. Defaults to 20000.
            nb_days: Number of days to generate transactions for. Defaults to 90.
            city_radius: Radius (km) from a city center used to place customers
                and terminals. Defaults to 12.0.
            available_terminals_radius: Radius (km) around a customer within which
                terminals are reachable for Card-Present transactions. Defaults to 10.0.
            start_date: Start date for transaction generation (``YYYY-MM-DD``).
                Defaults to '2018-04-01'.
            random_state: Random seed for reproducibility. Defaults to 42.
            fraud: Fraud scenario parameters. Defaults to ``FraudConfig()``, which
                reproduces v0.1.0.
            drift: Customer concept drift (v2 engine only). ``None`` (default)
                disables it.
            seasonality: Calendar patterns (v2 engine only). ``None`` (default)
                disables them.
            merchants: Merchants and categories (v2 engine only). ``None``
                (default) disables them. With merchants, the transactions get
                ``MERCHANT_ID`` and ``MERCHANT_CATEGORY``, the terminal profiles get
                ``MERCHANT_ID``, and ``merchant_profiles`` is filled by
                :meth:`generate`.
            engine: ``"auto"``, ``"legacy"`` or ``"v2"`` (see the class docstring).
                Defaults to ``"auto"``.
        """
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}.")
        self.fraud_config = fraud if fraud is not None else FraudConfig()
        self.drift_config = drift
        self.seasonality_config = seasonality
        self.merchant_config = merchants
        self.engine = self._resolve_engine(engine)
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

        if self.engine == "v2":
            # The v2 engine builds everything in generate() and never touches the
            # global random state.
            self.generator: Optional[_TransactionGenerator] = None
            self.fraud_simulator: Optional[FraudSimulator] = None
            return

        # The legacy engine seeds the global random state here, as v0.1 does.
        self.generator = _TransactionGenerator(
            n_customers=n_customers,
            n_terminals=n_terminals,
            nb_days=nb_days,
            city_radius=city_radius,
            available_terminals_radius=available_terminals_radius,
            start_date=start_date,
            random_state=random_state,
        )
        self.fraud_simulator = FraudSimulator(
            random_state=random_state, config=self.fraud_config
        )

    def _resolve_engine(self, engine: str) -> str:
        v2_only = [
            name
            for name, value in (
                ("drift", self.drift_config),
                ("seasonality", self.seasonality_config),
                ("merchants", self.merchant_config),
                (
                    'fraud mechanism="inject"',
                    self.fraud_config
                    if self.fraud_config.mechanism == "inject"
                    else None,
                ),
            )
            if value is not None
        ]
        if engine == "auto":
            return "v2" if v2_only else "legacy"
        if engine == "legacy" and v2_only:
            raise ValueError(
                f"{', '.join(v2_only)} require(s) the v2 engine; "
                'use engine="v2" or engine="auto".'
            )
        return engine

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate the complete dataset (profiles + labelled transactions).

        The generator follows these steps:
        1. Generate customer profiles with random locations and spending patterns
        2. Generate terminal profiles with locations across cities
        3. Generate legitimate transactions based on customer-terminal proximity
        4. Add fraudulent transactions based on different fraud scenarios

        Returns:
            Tuple containing:
            - customer_profiles (pd.DataFrame): Customer information with columns:
                - CUSTOMER_ID: Unique customer identifier
                - billing_lat, billing_long: Customer billing location
                - mean_amount, std_amount: Parameters for transaction amount generation
                - mean_nb_tx_per_day: Average number of transactions per day
                - cnp_prob: Probability of Card Not Present transactions
            - terminal_profiles (pd.DataFrame): Terminal information with columns:
                - TERMINAL_ID: Unique terminal identifier
                - terminal_lat, terminal_long: Terminal location coordinates
            - transactions (pd.DataFrame): Transaction data with columns:
                - TRANSACTION_ID: Unique transaction identifier
                - TX_DATETIME: Transaction timestamp
                - TX_TIME_SECONDS: Time in seconds since start date
                - TX_TIME_DAYS: Time in days since start date
                - CUSTOMER_ID: Customer who made the transaction
                - TERMINAL_ID: Terminal where transaction was made
                - TX_AMOUNT: Transaction amount
                - TX_BILL_LAT, TX_BILL_LONG: billing address coordinates
                - TX_SHIPP_LAT, TX_SHIPP_LONG: shipping address coordinates
                - TX_TERM_LAT, TX_TERM_LONG: terminal location coordinates
                - TX_TYPE: Transaction type (CP: Card Present, CNP: Card Not Present)
                - TX_FRAUD: Binary fraud indicator
                - TX_FRAUD_SCENARIO: Type of fraud (0: legitimate, 1-4: fraud scenarios;
                  5 with fraud injection)
        """
        if self.engine == "v2":
            result = v2.simulate(
                n_customers=self.n_customers,
                n_terminals=self.n_terminals,
                nb_days=self.nb_days,
                city_radius=self.city_radius,
                available_terminals_radius=self.available_terminals_radius,
                start_date=datetime.datetime.strptime(self.start_date, "%Y-%m-%d"),
                random_state=self.random_state,
                fraud=self.fraud_config,
                drift=self.drift_config,
                seasonality=self.seasonality_config,
                merchants=self.merchant_config,
            )
            self.merchant_profiles = result.merchant_profiles
            customer_profiles = result.customer_profiles
            terminal_profiles = result.terminal_profiles
            transactions = result.transactions
            self._loc_features = result.loc_features
        else:
            assert self.generator is not None and self.fraud_simulator is not None
            customer_profiles = self.generator.generate_customer_profiles()
            terminal_profiles = self.generator.generate_terminal_profiles()
            transactions = self.generator.generate_transactions()

            print("\nAdding fraudulent transactions:")
            transactions = self.fraud_simulator.add_frauds(
                customer_profiles,
                terminal_profiles,
                transactions,
                self.generator.loc_features,
            )
            print("\nDataset generation completed!")

        # Cache the last result so save() can be called without re-generating.
        self.customer_profiles = customer_profiles
        self.terminal_profiles = terminal_profiles
        self.transactions = transactions

        return customer_profiles, terminal_profiles, transactions

    def save(
        self,
        output_dir: str = "data",
        save_format: str = "pkl",
        save_by_day: bool = True,
    ) -> None:
        """Persist the most recently generated dataset to disk.

        Writes customer/terminal (and, with merchants, merchant) profiles and
        (day-partitioned) transactions under ``output_dir``. Call :meth:`generate`
        first.

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
        if self.generator is not None:
            return self.generator.loc_features
        if self._loc_features is None:
            raise RuntimeError("Call generate() first.")
        return self._loc_features
