"""Time dataset generation with the v2 (and optionally the legacy) engine.

Usage:
    python scripts/benchmark.py                       # v2, 10k customers x 180 days
    python scripts/benchmark.py --legacy              # also legacy, on a smaller size
    python scripts/benchmark.py --customers 1000 --days 30 --legacy-customers 1000

Terminals are twice the number of customers, as in the default configuration.
"""

import argparse
import contextlib
import io
import platform
import time

from synccfd import DatasetGenerator, __version__


def run(engine: str, n_customers: int, nb_days: int, seed: int) -> tuple[float, int]:
    generator = DatasetGenerator(
        n_customers=n_customers,
        n_terminals=2 * n_customers,
        nb_days=nb_days,
        start_date="2023-01-01",
        random_state=seed,
        engine=engine,
    )
    start = time.perf_counter()
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        transactions = generator.generate()[2]
    return time.perf_counter() - start, len(transactions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=10_000)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy", action="store_true", help="also time legacy")
    parser.add_argument("--legacy-customers", type=int, default=1_000)
    args = parser.parse_args()

    print(
        f"synccfd {__version__} | Python {platform.python_version()} | "
        f"{platform.system()} {platform.release()} | {platform.processor()}"
    )
    print(
        f"{'engine':<8}{'customers':>11}{'days':>6}{'transactions':>14}{'seconds':>10}"
    )
    runs = [("v2", args.customers)]
    if args.legacy:
        runs += [("legacy", args.legacy_customers), ("v2", args.legacy_customers)]
    for engine, n_customers in runs:
        seconds, n_transactions = run(engine, n_customers, args.days, args.seed)
        print(
            f"{engine:<8}{n_customers:>11}{args.days:>6}{n_transactions:>14}"
            f"{seconds:>10.1f}"
        )


if __name__ == "__main__":
    main()
