"""Create the reference dataset of ``tests/regression/test_v2_reproducibility.py``.

The reference files freeze a small output of ``presets.realistic()`` (100 customers,
200 terminals, 15 days, ``random_state=42``), with higher fraud fractions so that
every scenario appears. They were created on version 0.2.0, before the legacy engine
was removed. Re-running this script only makes sense when the output is *meant* to
change (a model change, documented in the CHANGELOG); hence the ``--force`` flag to
overwrite existing files.

Usage:
    python scripts/make_regression.py [--force]
"""

import argparse
import sys
from pathlib import Path

from synccfd import DatasetGenerator
from synccfd.simulator import presets
from synccfd.simulator.config import FraudConfig

REGRESSION_DIR = Path(__file__).resolve().parents[1] / "tests" / "regression" / "data"
# Dense frauds: a few victims of each scenario in 100 customers x 15 days
DENSE_FRAUD = {
    "s1_customer_fraction_per_day": 1e-2,
    "s2_terminal_fraction_per_day": 1e-2,
    "s2_clone_prob": 0.5,
    "s3_customer_fraction_per_day": 3e-3,
    "s4_customer_fraction_per_day": 3e-3,
    "s5_customer_fraction_per_day": 3e-3,
}
REGRESSION_PARAMS = presets.realistic(
    n_customers=100,
    n_terminals=200,
    nb_days=15,
    random_state=42,
    fraud=FraudConfig.realistic(**DENSE_FRAUD),
)
REGRESSION_NAMES = ("customers", "terminals", "merchants", "transactions")


def reference_path(name: str) -> Path:
    return REGRESSION_DIR / f"v0_2_0_small_{name}.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite the files")
    args = parser.parse_args()

    if any(reference_path(n).exists() for n in REGRESSION_NAMES) and not args.force:
        print("Reference files already exist; use --force to overwrite.")
        return 1

    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
    result = DatasetGenerator(**REGRESSION_PARAMS).generate_all()
    frames = (result.customers, result.terminals, result.merchants, result.transactions)
    for name, frame in zip(REGRESSION_NAMES, frames):
        frame.to_parquet(reference_path(name))
        print(f"wrote {reference_path(name)} ({len(frame)} rows)")
    scenarios = result.transactions.TX_FRAUD_SCENARIO.value_counts()
    print("fraud scenarios:", scenarios.sort_index().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
