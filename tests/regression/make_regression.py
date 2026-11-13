"""Create the reference files of test_reference_output.py.

Only re-run when the output is meant to change; --force overwrites the files.

Usage:
    python tests/regression/make_regression.py [--force]
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from synccfd import DatasetGenerator
from synccfd.simulator import presets
from synccfd.simulator.config import FraudConfig

REFERENCE_DIR = Path(__file__).parent / "data"
# Dense frauds: a few victims of each scenario in 100 customers x 15 days
DENSE_FRAUD = {
    "s1_customer_fraction_per_day": 1e-2,
    "s2_terminal_fraction_per_day": 1e-2,
    "s2_clone_prob": 0.5,
    "s3_customer_fraction_per_day": 3e-3,
    "s4_customer_fraction_per_day": 3e-3,
    "s5_customer_fraction_per_day": 3e-3,
}
SIZE = {"n_customers": 100, "n_terminals": 200, "nb_days": 15, "random_state": 42}
NAMES = ("customers", "terminals", "merchants", "transactions")


def reference_params(**fraud_overrides) -> dict[str, Any]:
    """Arguments of the reference dataset, with optional FraudConfig changes."""
    fraud = FraudConfig.realistic(**{**DENSE_FRAUD, **fraud_overrides})
    return presets.realistic(**SIZE, fraud=fraud)


def reference_path(name: str) -> Path:
    """The reference file of one table."""
    return REFERENCE_DIR / f"reference_small_{name}.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite the files")
    args = parser.parse_args()

    if any(reference_path(n).exists() for n in NAMES) and not args.force:
        print("Reference files already exist; use --force to overwrite.")
        return 1

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    result = DatasetGenerator(**reference_params()).generate_all()
    frames = (result.customers, result.terminals, result.merchants, result.transactions)
    for name, frame in zip(NAMES, frames):
        frame.to_parquet(reference_path(name))
        print(f"wrote {reference_path(name)} ({len(frame)} rows)")
    scenarios = result.transactions.TX_FRAUD_SCENARIO.value_counts()
    print("fraud scenarios:", scenarios.sort_index().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
