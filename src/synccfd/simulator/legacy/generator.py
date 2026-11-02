"""v0.1 transaction generator (legacy engine). Logic and random draws unchanged."""

import datetime
import random
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..location.features import LocationFeatures


class _TransactionGenerator:
    """Internal class for generating synthetic credit card transactions."""

    def __init__(
        self,
        n_customers: int = 10000,
        n_terminals: int = 20000,
        nb_days: int = 90,
        city_radius: float = 12,
        available_terminals_radius: float = 10,
        start_date: str = "2018-04-01",
        random_state: Optional[int] = 42,
    ):
        """
        Initialize TransactionGenerator.

        Args:
            n_customers: Number of customers to generate
            n_terminals: Number of terminals to generate
            nb_days: Number of days to generate transactions for
            city_radius: Radius from city center for random point generation
            available_terminals_radius: Radius to define available terminals
            start_date: Start date for transaction generation
            random_state: Random seed for reproducibility
        """
        self.n_customers = n_customers
        self.n_terminals = n_terminals
        self.nb_days = nb_days
        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.loc_features = LocationFeatures(
            city_radius=city_radius,
            available_terminals_radius=available_terminals_radius,
            random_state=random_state,
        )

        self.customer_profiles: Optional[pd.DataFrame] = None
        self.terminal_profiles: Optional[pd.DataFrame] = None

    def generate_customer_profiles(self) -> pd.DataFrame:
        """Generate customer profiles with their properties."""
        np.random.seed(0)  # Ensure reproducibility for customer generation

        # Generate billing locations
        self.loc_features.gen_billing_locations(self.n_customers, random_state=0)

        # Generate customer properties
        customer_properties = []
        for customer_id in tqdm(
            range(self.n_customers), desc="Creating customer profiles"
        ):
            mean_amount = np.random.uniform(5, 100)
            std_amount = mean_amount / 2
            mean_nb_tx_per_day = np.random.uniform(0, 4)
            cnp_prob = np.random.uniform(0, 1)
            billing_loc = self.loc_features.get_billing_location(customer_id)

            customer_properties.append(
                [
                    customer_id,
                    billing_loc["lat"],
                    billing_loc["long"],
                    mean_amount,
                    std_amount,
                    mean_nb_tx_per_day,
                    cnp_prob,
                ]
            )

        self.customer_profiles = pd.DataFrame(
            customer_properties,
            columns=[
                "CUSTOMER_ID",
                "billing_lat",
                "billing_long",
                "mean_amount",
                "std_amount",
                "mean_nb_tx_per_day",
                "cnp_prob",
            ],
        )

        return self.customer_profiles

    def generate_terminal_profiles(self) -> pd.DataFrame:
        """Generate terminal profiles with their locations."""
        np.random.seed(1)  # Ensure reproducibility for terminal generation

        # Generate terminal locations
        self.loc_features.gen_terminal_locations(self.n_terminals, random_state=1)

        # Generate terminal properties
        terminal_properties = []
        for terminal_id in tqdm(
            range(self.n_terminals), desc="Creating terminal profiles"
        ):
            terminal_loc = self.loc_features.get_terminal_location(terminal_id)
            terminal_properties.append(
                [terminal_id, terminal_loc["lat"], terminal_loc["long"]]
            )

        self.terminal_profiles = pd.DataFrame(
            terminal_properties,
            columns=["TERMINAL_ID", "terminal_lat", "terminal_long"],
        )

        return self.terminal_profiles

    def _generate_transaction_type(self, p_cnp: float) -> str:
        """Generate transaction type (CP or CNP) based on probability."""
        return np.random.choice(["CP", "CNP"], p=[1 - p_cnp, p_cnp])

    def _generate_transaction_time(self, tx_type: str, day: int) -> Optional[int]:
        """Generate transaction time in seconds for a given day."""
        if tx_type == "CP":
            time_tx = int(np.random.normal(86400 / 2, 20000))
        else:
            if np.random.choice([True, False]):
                time_tx = int(np.random.normal(36000, 12500))
            else:
                time_tx = int(np.random.normal(72000, 12500)) % 86400  # Added modulo

        # Match original behavior - discard invalid times
        if time_tx < 0 or time_tx > 86400:
            return None

        return time_tx + day * 86400

    def generate_customer_transactions(
        self, customer_profile: pd.Series
    ) -> pd.DataFrame:
        # Match original random seeding behavior
        random.seed(int(customer_profile.CUSTOMER_ID))
        np.random.seed(int(customer_profile.CUSTOMER_ID))

        customer_transactions = []

        for day in range(self.nb_days):
            # Random number of transactions for the day
            nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)

            if nb_tx > 0:
                for _ in range(nb_tx):
                    # Generate amount
                    amount = np.random.normal(
                        customer_profile.mean_amount, customer_profile.std_amount
                    )
                    if amount < 0:
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)
                    amount = round(amount, 2)

                    # Check available terminals before deciding tx_type
                    available_terminals = (
                        self.loc_features.get_customer_available_terminals(
                            int(customer_profile.CUSTOMER_ID)
                        )
                    )
                    if len(available_terminals) > 0:
                        tx_type = self._generate_transaction_type(
                            customer_profile.cnp_prob
                        )
                    else:
                        tx_type = "CNP"

                    # Generate transaction locations
                    tx_locations = self.loc_features.gen_tx_locations(
                        int(customer_profile.CUSTOMER_ID), tx_type
                    )

                    # Generate time - skip if invalid
                    time_tx = self._generate_transaction_time(tx_type, day)
                    if time_tx is None:
                        continue

                    customer_transactions.append(
                        [
                            time_tx,
                            day,
                            customer_profile.CUSTOMER_ID,
                            tx_locations["tx_terminal_id"],
                            amount,
                            tx_type,
                            tx_locations["tx_shipping_loc"]["lat"],
                            tx_locations["tx_shipping_loc"]["long"],
                            tx_locations["tx_billing_loc"]["lat"],
                            tx_locations["tx_billing_loc"]["long"],
                            tx_locations["tx_terminal_loc"]["lat"],
                            tx_locations["tx_terminal_loc"]["long"],
                        ]
                    )

        if not customer_transactions:
            return pd.DataFrame()

        transactions_df = pd.DataFrame(
            customer_transactions,
            columns=[
                "TX_TIME_SECONDS",
                "TX_TIME_DAYS",
                "CUSTOMER_ID",
                "TERMINAL_ID",
                "TX_AMOUNT",
                "TX_TYPE",
                "TX_SHIPP_LAT",
                "TX_SHIPP_LONG",
                "TX_BILL_LAT",
                "TX_BILL_LONG",
                "TX_TERM_LAT",
                "TX_TERM_LONG",
            ],
        )

        transactions_df["TX_DATETIME"] = pd.to_datetime(
            transactions_df["TX_TIME_SECONDS"], unit="s", origin=self.start_date
        )

        return transactions_df[
            [
                "TX_DATETIME",
                "CUSTOMER_ID",
                "TERMINAL_ID",
                "TX_AMOUNT",
                "TX_TYPE",
                "TX_SHIPP_LAT",
                "TX_SHIPP_LONG",
                "TX_BILL_LAT",
                "TX_BILL_LONG",
                "TX_TERM_LAT",
                "TX_TERM_LONG",
                "TX_TIME_SECONDS",
                "TX_TIME_DAYS",
            ]
        ]

    def generate_transactions(self) -> pd.DataFrame:
        """Generate transaction dataset."""
        if self.customer_profiles is None:
            self.customer_profiles = self.generate_customer_profiles()

        if self.terminal_profiles is None:
            self.terminal_profiles = self.generate_terminal_profiles()

        # Generate customer available terminals
        print("Computing available terminals for each customer (CP Txs)...")
        self.loc_features.gen_customer_available_terminals()
        self.customer_profiles["available_terminals"] = self.customer_profiles.apply(
            lambda x: self.loc_features.get_customer_available_terminals(x.CUSTOMER_ID),
            axis=1,
        )

        # Generate transactions for each customer
        transactions_list = []
        for _, customer_profile in tqdm(
            self.customer_profiles.iterrows(),
            desc="Generating transactions",
            total=len(self.customer_profiles),
        ):
            customer_txs = self.generate_customer_transactions(customer_profile)
            if not customer_txs.empty:
                transactions_list.append(customer_txs)

        if not transactions_list:
            return pd.DataFrame()

        transactions_df = pd.concat(transactions_list, ignore_index=True)

        # Sort chronologically and add transaction IDs
        transactions_df = transactions_df.sort_values("TX_DATETIME")
        transactions_df.reset_index(drop=True, inplace=True)
        transactions_df.reset_index(inplace=True)
        transactions_df.rename(columns={"index": "TRANSACTION_ID"}, inplace=True)

        return transactions_df
