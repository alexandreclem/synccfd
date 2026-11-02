import datetime
import os
import shutil
from typing import Optional

import pandas as pd


class DataManager:
    """Handles saving and loading of transaction data and related profiles."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize DataManager with data directory and mode.

        Args:
            data_dir: Base directory for data. Defaults to '/data'
                     in current working directory.
        """
        self.data_dir = data_dir or os.path.join(os.getcwd(), "data", "")
        self.profiles_dir = os.path.join(self.data_dir, "profiles")
        self.transactions_raw_dir = os.path.join(self.data_dir, "transactions/raw")
        self.transactions_transformed_dir = os.path.join(
            self.data_dir, "transactions/transformed"
        )

    def save_data(
        self,
        transactions: pd.DataFrame,
        customer_profiles: Optional[pd.DataFrame] = None,
        terminal_profiles: Optional[pd.DataFrame] = None,
        save_format: str = "pkl",
        save_by_day: bool = True,
        data_type: Optional[str] = None,
    ) -> None:
        """Save transaction data and optionally profile data.

        Args:
            transactions: DataFrame with transaction data
            customer_profiles: Optional DataFrame with customer profiles
            terminal_profiles: Optional DataFrame with terminal profiles
            save_format: Format to save in ('pkl', 'parquet', or 'csv')
            save_by_day: Whether to split transactions by day
            data_type: 'raw' or 'transformed' (mandatory)
        """
        if data_type not in ("raw", "transformed"):
            raise ValueError("data_type must be either 'raw' or 'transformed'")
        transactions_dir = (
            self.transactions_raw_dir
            if data_type == "raw"
            else self.transactions_transformed_dir
        )

        if data_type == "raw":
            # Only clean the raw transactions directory, not the entire data directory
            if os.path.exists(transactions_dir):
                shutil.rmtree(transactions_dir)
            if os.path.exists(self.profiles_dir):
                shutil.rmtree(self.profiles_dir)

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(transactions_dir, exist_ok=True)

        # Save profiles if provided
        if any([customer_profiles is not None, terminal_profiles is not None]):
            os.makedirs(self.profiles_dir, exist_ok=True)

            if customer_profiles is not None:
                self._save_dataframe(
                    customer_profiles,
                    os.path.join(self.profiles_dir, f"customer_profiles.{save_format}"),
                    save_format,
                )

            if terminal_profiles is not None:
                self._save_dataframe(
                    terminal_profiles,
                    os.path.join(self.profiles_dir, f"terminal_profiles.{save_format}"),
                    save_format,
                )

        # Save transactions
        if save_by_day:
            start_date = transactions["TX_DATETIME"].min()

            for day in range(transactions.TX_TIME_DAYS.max() + 1):
                transactions_day = transactions[
                    transactions.TX_TIME_DAYS == day
                ].sort_values("TX_TIME_SECONDS")
                date = start_date + pd.Timedelta(days=day)
                filename = date.strftime("%Y-%m-%d")

                self._save_dataframe(
                    transactions_day,
                    os.path.join(transactions_dir, f"{filename}.{save_format}"),
                    save_format,
                )
        else:
            self._save_dataframe(
                transactions,
                os.path.join(transactions_dir, f"transactions.{save_format}"),
                save_format,
            )

    def read_transactions(
        self,
        begin_date: str,
        end_date: str,
        format: str = "pkl",
        data_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load transaction data from files and combine them.

        Args:
            begin_date: Start date for loading data (YYYY-MM-DD)
            end_date: End date for loading data (YYYY-MM-DD)
            format: Format to read ('pkl', 'parquet', or 'csv')
            data_type: 'raw' or 'transformed' (mandatory)
        Returns:
            DataFrame with combined transaction data
        """
        if data_type not in ("raw", "transformed"):
            raise ValueError("data_type must be either 'raw' or 'transformed'")
        transactions_dir = (
            self.transactions_raw_dir
            if data_type == "raw"
            else self.transactions_transformed_dir
        )

        # First try to load a single file if it exists
        single_file = os.path.join(transactions_dir, f"transactions.{format}")
        if os.path.exists(single_file):
            return self._read_dataframe(single_file, format)

        # If no single file, load daily files
        transactions_df = pd.DataFrame()
        start_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        delta = datetime.timedelta(days=1)

        while start_date <= end_date_dt:
            # Try to load file for current date
            filename = f"{start_date.strftime('%Y-%m-%d')}.{format}"
            file_path = os.path.join(transactions_dir, filename)
            if os.path.exists(file_path):
                day_transactions = self._read_dataframe(file_path, format)
                transactions_df = pd.concat([transactions_df, day_transactions])

            start_date += delta

        if transactions_df.empty:
            raise FileNotFoundError(
                f"No transaction data found in {transactions_dir} "
                f"for dates between {begin_date} and {end_date}"
            )

        # Ensure TX_DATETIME is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(transactions_df["TX_DATETIME"]):
            transactions_df["TX_DATETIME"] = pd.to_datetime(
                transactions_df["TX_DATETIME"]
            )

        return transactions_df.sort_values("TX_DATETIME").reset_index(drop=True)

    def read_profiles(
        self, format: str = "pkl"
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Read customer and terminal profiles if they exist.

        Args:
            format: Format to read ('pkl', 'parquet', or 'csv')

        Returns:
            Tuple of (customer_profiles, terminal_profiles), None if not found
        """
        customer_profiles = terminal_profiles = None

        customer_path = os.path.join(self.profiles_dir, f"customer_profiles.{format}")
        terminal_path = os.path.join(self.profiles_dir, f"terminal_profiles.{format}")

        if os.path.exists(customer_path):
            customer_profiles = self._read_dataframe(customer_path, format)

        if os.path.exists(terminal_path):
            terminal_profiles = self._read_dataframe(terminal_path, format)

        return customer_profiles, terminal_profiles

    def _save_dataframe(self, df: pd.DataFrame, path: str, format: str) -> None:
        """Save DataFrame in specified format."""
        if format == "pkl":
            df.to_pickle(path)
        elif format == "parquet":
            df.to_parquet(path)
        else:
            df.to_csv(path, index=False)

    def _read_dataframe(self, path: str, format: str) -> pd.DataFrame:
        """Read DataFrame in specified format."""
        if format == "pkl":
            return pd.read_pickle(path)
        elif format == "parquet":
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path)
