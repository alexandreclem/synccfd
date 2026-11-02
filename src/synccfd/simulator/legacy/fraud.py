"""v0.1 fraud simulator (legacy engine). Logic and random draws unchanged."""

import random
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import FraudConfig
from ..location.features import LocationFeatures


class FraudSimulator:
    """Simulates different credit card fraud scenarios."""

    def __init__(
        self,
        random_state: Optional[int] = None,
        config: Optional[FraudConfig] = None,
    ):
        """
        Initialize FraudSimulator.

        Args:
            random_state: Random seed for reproducibility
            config: Fraud scenario parameters. Defaults to ``FraudConfig()``, which
                reproduces v0.1.0.
        """
        self.config = config if config is not None else FraudConfig()
        if self.config.mechanism != "tamper":
            raise ValueError(
                'The legacy fraud simulator only supports mechanism="tamper"; '
                "fraud injection requires the v2 engine."
            )
        self.random_state = random_state
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

        counts = self.daily_counts(
            n_days=int(transactions.TX_TIME_DAYS.max()) + 1,
            n_terminals=len(terminal_profiles),
            n_customers=len(customer_profiles),
        )

        # Apply different fraud scenarios
        scenarios = self.config.scenarios
        if 1 in scenarios:
            print("Applying fraud scenario 1: High amount fraud...")
            transactions = self._apply_scenario_1(transactions)

        if 2 in scenarios:
            print("Applying fraud scenario 2: Compromised terminals...")
            transactions = self._apply_scenario_2(
                transactions, terminal_profiles, counts[2]
            )

        if 3 in scenarios or 4 in scenarios:
            transactions = self._apply_scenarios_3_and_4(
                transactions, customer_profiles, loc_features, counts[3], counts[4]
            )

        return transactions

    def daily_counts(
        self, n_days: int, n_terminals: int, n_customers: int
    ) -> dict[int, np.ndarray]:
        """Terminals (scenario 2) and customers (scenarios 3, 4) compromised per day.

        In ``"proportional"`` mode each scenario draws its counts from its own
        random stream, so disabling one scenario does not change the others.
        """
        streams = np.random.SeedSequence(self.random_state).spawn(3)
        populations = {2: n_terminals, 3: n_customers, 4: n_customers}
        return {
            scenario: self.config.daily_counts(
                scenario, populations[scenario], n_days, np.random.default_rng(stream)
            )
            for scenario, stream in zip((2, 3, 4), streams)
        }

    def _apply_scenario_1(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fraud scenario 1: High amount transactions.
        Simple threshold-based fraud where transactions above a certain
        amount are fraudulent.
        """
        high_amount_mask = transactions.TX_AMOUNT > self.config.s1_amount_threshold
        transactions.loc[high_amount_mask, "TX_FRAUD"] = 1
        transactions.loc[high_amount_mask, "TX_FRAUD_SCENARIO"] = 1
        return transactions

    def _apply_scenario_2(
        self,
        transactions: pd.DataFrame,
        terminal_profiles: pd.DataFrame,
        n_terminals_per_day: np.ndarray,
    ) -> pd.DataFrame:
        """
        Apply fraud scenario 2: Compromised terminals.
        Transactions at specific terminals become fraudulent for a period.
        """
        duration = self.config.s2_duration_days
        for day in range(transactions.TX_TIME_DAYS.max() + 1):
            # Select compromised terminals for this day
            compromised_terminals = terminal_profiles.TERMINAL_ID.sample(
                n=int(n_terminals_per_day[day]), random_state=day
            )

            # Mark transactions at compromised terminals as fraudulent
            compromised_mask = (
                (transactions.TX_TIME_DAYS >= day)
                & (transactions.TX_TIME_DAYS < day + duration)
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
        n_customers_3_per_day: np.ndarray,
        n_customers_4_per_day: np.ndarray,
    ) -> pd.DataFrame:
        """
        Apply fraud scenarios 3 and 4:
        - Scenario 3: Card number stolen (CNP fraud)
        - Scenario 4: Physical card stolen (CP fraud)
        """
        duration_3 = self.config.s3_duration_days
        duration_4 = self.config.s4_duration_days

        # Customers compromised in the last ``duration`` days, one array per day
        compromised_customers_queue: deque[np.ndarray] = deque(maxlen=duration_3)
        robbed_customers_queue: deque[np.ndarray] = deque(maxlen=duration_4)

        for day in tqdm(
            range(transactions.TX_TIME_DAYS.max() + 1),
            desc="Applying fraud scenarios 3 and 4: ",
        ):
            # Remove old compromised customers
            if day >= duration_4 and robbed_customers_queue:
                robbed_customers_queue.popleft()
            if day >= duration_3 and compromised_customers_queue:
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

            n3 = int(n_customers_3_per_day[day])
            n4 = int(n_customers_4_per_day[day])

            # v0.1 draws the customers of both scenarios in a single call and
            # splits them afterwards; keep it that way to preserve the draws.
            if len(available_customers) >= n3 + n4:
                new_compromised = available_customers.CUSTOMER_ID.sample(
                    n=n3 + n4, random_state=day
                ).values

                # Split between scenarios 3 and 4
                compromised_customers_queue.append(new_compromised[:n3])
                robbed_customers_queue.append(new_compromised[n3:])

                # Apply scenario 3: Card number stolen
                scenario_3_mask = (
                    (transactions.TX_TIME_DAYS >= day)
                    & (transactions.TX_TIME_DAYS < day + duration_3)
                    & (transactions.CUSTOMER_ID.isin(new_compromised[:n3]))
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
                    & (transactions.TX_TIME_DAYS < day + duration_4)
                    & (transactions.CUSTOMER_ID.isin(new_compromised[n3:]))
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
            # Select a fraction (1/3 by default) of transactions to tamper
            nb_compromised_transactions = len(compromised_txs)
            index_frauds = random.sample(
                list(compromised_txs.index.values),
                k=int(nb_compromised_transactions * self.config.s3_tamper_fraction),
            )
            compromised_txs = compromised_txs.loc[index_frauds]

            # Set to CNP type
            compromised_txs["TX_TYPE"] = "CNP"

        else:  # Scenario 4
            # Shift all transactions to first day
            compromised_txs = self._shift_to_first_day(compromised_txs)
            # Set TX_TYPE with probability using the original gen_tx_type logic
            p_cnp = self.config.s4_cnp_prob
            compromised_txs["TX_TYPE"] = compromised_txs["TX_TYPE"].apply(
                lambda _: random.choices(["CP", "CNP"], [1 - p_cnp, p_cnp], k=1)[0]
            )
            index_frauds = list(compromised_txs.index.values)

        # Get tampered locations
        compromised_txs = loc_features.tamper_tx_locations(
            compromised_txs, fraud_scenario
        )

        # Mark as fraudulent and multiply amount
        transactions.loc[index_frauds] = compromised_txs
        transactions.loc[index_frauds, "TX_AMOUNT"] = (
            transactions.loc[index_frauds, "TX_AMOUNT"] * self.config.amount_multiplier
        )
        transactions.loc[index_frauds, "TX_FRAUD"] = 1
        transactions.loc[index_frauds, "TX_FRAUD_SCENARIO"] = fraud_scenario

        return transactions
