"""Time dataset generation with each preset (the steps of the ablation).

Usage:
    python scripts/benchmark.py                        # 10k customers x 180 days
    python scripts/benchmark.py --customers 1000 --days 30

Terminals are twice the number of customers, as in the default configuration.
"""

import argparse
import platform
import time

from synccfd import DatasetGenerator, __version__
from synccfd.simulator import presets


def run(step: str, n_customers: int, nb_days: int, seed: int) -> tuple[float, int]:
    generator = DatasetGenerator(
        **presets.ablation(
            step,
            n_customers=n_customers,
            n_terminals=2 * n_customers,
            nb_days=nb_days,
            random_state=seed,
        )
    )
    start = time.perf_counter()
    transactions = generator.generate()[2]
    return time.perf_counter() - start, len(transactions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=10_000)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(
        f"synccfd {__version__} | Python {platform.python_version()} | "
        f"{platform.system()} {platform.release()} | {platform.processor()}"
    )
    print(
        f"{'step':<11}{'customers':>11}{'days':>6}{'transactions':>14}{'seconds':>10}"
    )
    for step in presets.ABLATION_STEPS:
        seconds, n_transactions = run(step, args.customers, args.days, args.seed)
        print(
            f"{step:<11}{args.customers:>11}{args.days:>6}{n_transactions:>14}"
            f"{seconds:>10.1f}"
        )


if __name__ == "__main__":
    main()
