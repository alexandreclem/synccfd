"""Helpers to read the ground-truth columns of a dataset."""

import pandas as pd

COMPROMISE_COLUMNS = [
    "GT_COMPROMISE_ID",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "READ_TRANSACTION_ID",
    "READ_DATETIME",
    "GT_FRAUDSTER_ID",
    "USE_SCENARIO",
    "N_FRAUD_TX",
    "FIRST_FRAUD_DATETIME",
]


def compromise_table(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per card read at a compromised terminal (scenario 2)."""
    if "GT_COMPROMISE_ID" not in transactions:
        return pd.DataFrame(columns=COMPROMISE_COLUMNS)
    marked = transactions[transactions["GT_COMPROMISE_ID"].notna()]
    # The read is a legitimate purchase; later frauds with the card share its id
    reads = marked[marked["TX_FRAUD"] == 0]
    uses = marked[marked["TX_FRAUD"] == 1].groupby("GT_COMPROMISE_ID")
    table = pd.DataFrame(
        {
            "GT_COMPROMISE_ID": reads["GT_COMPROMISE_ID"].to_numpy(),
            "CUSTOMER_ID": reads["CUSTOMER_ID"].to_numpy(),
            "TERMINAL_ID": reads["TERMINAL_ID"].to_numpy(),
            "READ_TRANSACTION_ID": reads["TRANSACTION_ID"].to_numpy(),
            "READ_DATETIME": reads["TX_DATETIME"].to_numpy(),
        }
    ).set_index("GT_COMPROMISE_ID", drop=False)
    # Use, fraudster and frauds come from the transactions made with the card
    # (missing if it was not used)
    table["USE_SCENARIO"] = uses["TX_FRAUD_SCENARIO"].first().astype("Int64")
    table["N_FRAUD_TX"] = uses.size().reindex(table.index, fill_value=0)
    table["FIRST_FRAUD_DATETIME"] = uses["TX_DATETIME"].min()
    table["GT_FRAUDSTER_ID"] = uses["GT_FRAUDSTER_ID"].first().astype("Int64")
    table = table[COMPROMISE_COLUMNS].reset_index(drop=True)
    return table.sort_values(["READ_DATETIME", "GT_COMPROMISE_ID"], ignore_index=True)
