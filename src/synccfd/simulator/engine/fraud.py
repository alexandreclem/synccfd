"""Adds the fraud transactions to the legitimate ones."""

import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ..config import FraudConfig
from ..location.features import LocationFeatures
from .injection import FRAUD_GT_COLUMNS, inject_frauds
from .rng import RandomStreams
from .transactions import SECONDS_PER_DAY, assemble_transactions

if TYPE_CHECKING:
    from .merchants import MerchantWorld


def apply_frauds(
    transactions: pd.DataFrame,
    customer_profiles: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    loc_features: LocationFeatures,
    config: FraudConfig,
    streams: RandomStreams,
    start_date: datetime.datetime,
    nb_days: Optional[int] = None,
    shock_by_day: Optional[np.ndarray] = None,
    merchants: Optional["MerchantWorld"] = None,
) -> pd.DataFrame:
    """Add and label the fraud transactions; returns a new frame."""
    tx = {c: transactions[c].to_numpy().copy() for c in transactions.columns}
    tx["IS_CNP"] = tx.pop("TX_TYPE") == "CNP"
    n_tx = len(transactions)
    tx["TX_FRAUD"] = np.zeros(n_tx, dtype=np.int64)
    tx["TX_FRAUD_SCENARIO"] = np.zeros(n_tx, dtype=np.int64)
    if nb_days is not None:
        n_days = nb_days
    else:
        n_days = int(tx["TX_TIME_DAYS"].max()) + 1 if n_tx else 0
    tx["GT_FRAUDSTER_ID"] = np.full(n_tx, -1, dtype=np.int64)
    tx["GT_CAMPAIGN_ID"] = np.full(n_tx, -1, dtype=np.int64)
    tx["GT_COMPROMISE_ID"] = np.full(n_tx, -1, dtype=np.int64)
    # One sub-stream per scenario, so the scenarios are independent of each other
    rngs = {k: streams.generator("fraud", k) for k in config.scenarios}
    injected = inject_frauds(
        tx,
        customer_profiles,
        terminal_profiles,
        loc_features,
        config,
        rngs,
        n_days,
        merchants=merchants,
    )
    drift_gt = tuple(c for c in transactions.columns if c.startswith("GT_"))
    if injected:
        n_new = len(injected["TX_TIME_SECONDS"])
        injected["TX_TIME_DAYS"] = injected["TX_TIME_SECONDS"] // SECONDS_PER_DAY
        injected["TX_FRAUD"] = np.ones(n_new, dtype=np.int64)
        # Injected transactions do not come from the customer's regimes
        if "GT_REGIME_WEIGHT" in tx:
            injected["GT_REGIME_WEIGHT"] = np.full(n_new, np.nan)
        if "GT_REGIME" in tx:
            tx["GT_REGIME"] = tx["GT_REGIME"].astype(object)
            injected["GT_REGIME"] = np.full(n_new, None, dtype=object)
        if "GT_SHOCK" in tx:
            injected["GT_SHOCK"] = (
                shock_by_day[injected["TX_TIME_DAYS"]]
                if shock_by_day is not None
                else np.zeros(n_new, dtype=bool)
            )
        tx = {
            key: np.concatenate([value, injected[key].astype(value.dtype, copy=False)])
            if key in injected
            else value
            for key, value in tx.items()
            if key not in ("TX_DATETIME", "TRANSACTION_ID")
        }
    tx["GT_FRAUD_SCENARIO"] = tx["TX_FRAUD_SCENARIO"].copy()
    frame = assemble_transactions(
        tx,
        start_date,
        extra_columns=("TX_FRAUD", "TX_FRAUD_SCENARIO", *drift_gt, *FRAUD_GT_COLUMNS),
    )
    for column in ("GT_FRAUDSTER_ID", "GT_CAMPAIGN_ID", "GT_COMPROMISE_ID"):
        values = frame[column].astype("Int64")
        frame[column] = values.mask(values < 0)
    return frame
