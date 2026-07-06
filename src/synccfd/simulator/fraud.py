import random
from collections import deque

import numpy as np
import pandas as pd
from tqdm import tqdm

from .location.features import LocationFeatures


class FraudSimulator:
    """Simulates different credit card fraud scenarios."""

    def __init__(self, random_state: int = None):
        """
        Initialize FraudSimulator.

        Args:
            random_state: Random seed for reproducibility
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def add_frauds(
        self,
        customer_profiles: pd.DataFrame,
        terminal_profiles: pd.DataFrame,
        transactions: pd.DataFrame,
        loc_features: LocationFeatures,
    ) -> pd.DataFrame:
        """Add fraudulent transactions to the dataset."""
        # Initialize fraud indicators
        transactions["TX_FRAUD"] = 0
        transactions["TX_FRAUD_SCENARIO"] = 0

        # Apply different fraud scenarios
        print("Applying fraud scenario 1: High amount fraud...")
        transactions = self._apply_scenario_1(transactions)

        print("Applying fraud scenario 2: Compromised terminals...")
        transactions = self._apply_scenario_2(transactions, terminal_profiles)

        transactions = self._apply_scenarios_3_and_4(
            transactions, customer_profiles, loc_features
        )

        return transactions

    def _apply_scenario_1(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fraud scenario 1: High amount transactions.
        Simple threshold-based fraud where transactions above a certain
        amount are fraudulent.
        """
        high_amount_mask = transactions.TX_AMOUNT > 220
        transactions.loc[high_amount_mask, "TX_FRAUD"] = 1
        transactions.loc[high_amount_mask, "TX_FRAUD_SCENARIO"] = 1
        return transactions

    def _apply_scenario_2(
        self, transactions: pd.DataFrame, terminal_profiles: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply fraud scenario 2: Compromised terminals.
        Transactions at specific terminals become fraudulent for a period.
        """
        for day in range(transactions.TX_TIME_DAYS.max() + 1):
            # Select compromised terminals for this day
            compromised_terminals = terminal_profiles.TERMINAL_ID.sample(
                n=2, random_state=day
            )

            # Mark transactions at compromised terminals as fraudulent
            compromised_mask = (
                (transactions.TX_TIME_DAYS >= day)
                & (transactions.TX_TIME_DAYS < day + 28)
                & (transactions.TERMINAL_ID.isin(compromised_terminals))
                & (transactions.TX_FRAUD == 0)
            )
            transactions.loc[compromised_mask, "TX_FRAUD"] = 1
            transactions.loc[compromised_mask, "TX_FRAUD_SCENARIO"] = 2

        return transactions

    def _apply_scenarios_3_and_4(
        self,
        transactions: pd.DataFrame,
        customer_profiles: pd.DataFrame,
        loc_features: LocationFeatures,
    ) -> pd.DataFrame:
        """
        Apply fraud scenarios 3 and 4:
        - Scenario 3: Card number stolen (CNP fraud)
        - Scenario 4: Physical card stolen (CP fraud)
        """
        compromised_customers_queue = deque(maxlen=14)  # 14 days for scenario 3
        robbed_customers_queue = deque(maxlen=5)  # 5 days for scenario 4

        for day in tqdm(
            range(transactions.TX_TIME_DAYS.max() + 1),
            desc="Applying fraud scenarios 3 and 4: ",
        ):
            # Remove old compromised customers
            if day >= 5:
                robbed_customers_queue.popleft()
            if day >= 14:
                compromised_customers_queue.popleft()

            # Get currently compromised customers
            customers_to_exclude = [
                customer_id
                for queue in (compromised_customers_queue, robbed_customers_queue)
                for customer_ids in queue
                for customer_id in customer_ids
            ]

            # Select new customers to compromise
            available_customers = customer_profiles[
                ~customer_profiles.CUSTOMER_ID.isin(customers_to_exclude)
            ]

            if len(available_customers) >= 6:
                new_compromised = available_customers.CUSTOMER_ID.sample(
                    n=6, random_state=day
                ).values

                # Split between scenarios 3 and 4
                compromised_customers_queue.append(new_compromised[:3])
                robbed_customers_queue.append(new_compromised[3:])

                # Apply scenario 3: Card number stolen
                scenario_3_mask = (
                    (transactions.TX_TIME_DAYS >= day)
                    & (transactions.TX_TIME_DAYS < day + 14)
                    & (transactions.CUSTOMER_ID.isin(new_compromised[:3]))
                    & (transactions.TX_FRAUD == 0)
                )
                compromised_txs = transactions[scenario_3_mask].copy()
                if not compromised_txs.empty:
                    transactions = self._tamper_transactions(
                        transactions,
                        compromised_txs,
                        loc_features,
                        fraud_scenario=3,
                        random_state=day,
                    )

                # Apply scenario 4: Physical card stolen
                scenario_4_mask = (
                    (transactions.TX_TIME_DAYS >= day)
                    & (transactions.TX_TIME_DAYS < day + 5)
                    & (transactions.CUSTOMER_ID.isin(new_compromised[3:]))
                    & (transactions.TX_FRAUD == 0)
                )
                compromised_txs = transactions[scenario_4_mask].copy()
                if not compromised_txs.empty:
                    transactions = self._tamper_transactions(
                        transactions,
                        compromised_txs,
                        loc_features,
                        fraud_scenario=4,
                        random_state=day,
                    )

        return transactions

    def _shift_to_first_day(self, compromised_txs: pd.DataFrame) -> pd.DataFrame:
        """Shift all transactions to their earliest day per customer."""
        # Get earliest transaction per customer
        customer_first_tx = compromised_txs.groupby("CUSTOMER_ID")[
            ["TX_DATETIME", "TX_TIME_DAYS", "TX_TIME_SECONDS"]
        ].min()

        # Apply shift to each transaction
        for _, row in compromised_txs.iterrows():
            first_tx = customer_first_tx.loc[row.CUSTOMER_ID]
            compromised_txs.at[_, "TX_TIME_DAYS"] = first_tx["TX_TIME_DAYS"]
            compromised_txs.at[_, "TX_TIME_SECONDS"] = (
                first_tx["TX_TIME_DAYS"] * 86400
                + row.TX_DATETIME.hour * 3600
                + row.TX_DATETIME.minute * 60
                + row.TX_DATETIME.second
            )
            compromised_txs.at[_, "TX_DATETIME"] = row.TX_DATETIME.replace(
                year=first_tx["TX_DATETIME"].year,
                month=first_tx["TX_DATETIME"].month,
                day=first_tx["TX_DATETIME"].day,
            )

        return compromised_txs

    def _tamper_transactions(
        self,
        transactions: pd.DataFrame,
        compromised_txs: pd.DataFrame,
        loc_features: LocationFeatures,
        fraud_scenario: int,
        random_state: int,
    ) -> pd.DataFrame:
        """
        Modify compromised transactions to simulate fraudulent behavior.

        Args:
            transactions: Full transactions DataFrame
            compromised_txs: Transactions to be tampered with
            loc_features: Location features handler
            fraud_scenario: Fraud scenario number (3 or 4)
            random_state: Random seed

        Returns:
            Modified transactions DataFrame
        """
        # Set random seed for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)

        if fraud_scenario == 3:
            # Select 1/3 of transactions to tamper
            nb_compromised_transactions = len(compromised_txs)
            index_frauds = random.sample(
                list(compromised_txs.index.values),
                k=int(nb_compromised_transactions / 3),
            )
            compromised_txs = compromised_txs.loc[index_frauds]

            # Set to CNP type
            compromised_txs["TX_TYPE"] = "CNP"

        else:  # Scenario 4
            # Shift all transactions to first day
            compromised_txs = self._shift_to_first_day(compromised_txs)
            # Set TX_TYPE with probability using the original gen_tx_type logic
            compromised_txs["TX_TYPE"] = compromised_txs["TX_TYPE"].apply(
                lambda _: random.choices(["CP", "CNP"], [0.5, 0.5], k=1)[0]
            )
            index_frauds = list(compromised_txs.index.values)

        # Get tampered locations
        compromised_txs = loc_features.tamper_tx_locations(
            compromised_txs, fraud_scenario
        )

        # Mark as fraudulent and multiply amount
        transactions.loc[index_frauds] = compromised_txs
        transactions.loc[index_frauds, "TX_AMOUNT"] = (
            transactions.loc[index_frauds, "TX_AMOUNT"] * 5
        )
        transactions.loc[index_frauds, "TX_FRAUD"] = 1
        transactions.loc[index_frauds, "TX_FRAUD_SCENARIO"] = fraud_scenario

        return transactions
