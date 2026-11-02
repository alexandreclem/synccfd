"""Create the golden dataset used by ``tests/golden/test_legacy_reproducibility.py``.

The golden files freeze the output of the v0.1.0 (legacy) engine. They were created
by running this script on the v0.1.0 code, before any v0.2 change. Re-running it on a
later version only makes sense if the legacy behaviour is *meant* to change, which
should never happen; hence the ``--force`` flag to overwrite existing files.

Usage:
    python scripts/make_golden.py [--force]
"""

import argparse
import sys
from pathlib import Path

from synccfd import DatasetGenerator

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "tests" / "golden" / "data"
GOLDEN_PARAMS = {
    "n_customers": 100,
    "n_terminals": 200,
    "nb_days": 15,
    "start_date": "2023-01-01",
    "random_state": 42,
}
GOLDEN_NAMES = ("customers", "terminals", "transactions")


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"v0_1_0_small_{name}.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite golden files")
    args = parser.parse_args()

    if any(golden_path(n).exists() for n in GOLDEN_NAMES) and not args.force:
        print("Golden files already exist; use --force to overwrite.")
        return 1

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    frames = DatasetGenerator(**GOLDEN_PARAMS).generate()
    for name, frame in zip(GOLDEN_NAMES, frames):
        frame.to_parquet(golden_path(name))
        print(f"wrote {golden_path(name)} ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
