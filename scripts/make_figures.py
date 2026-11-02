"""Regenerate every figure of the dissertation and the dataset statistics.

1. Runs the example notebooks (simulator, drift, frauds, merchants) with the same
   configuration: a preset (default ``realistic``) with ``presets.DEFAULT_SIZE``,
   passed to the notebooks through the ``SYNCCFD_*`` variables read by
   ``presets.from_env()``. Each notebook writes its figures under ``docs/figures/``
   and is saved with its outputs.
2. Draws the architecture diagrams (``docs/figures/arquitetura/``).
3. Generates the same dataset once more and writes its statistics to
   ``docs/figures/dataset_stats.json`` (used by the "Fechamento do [SIM]" of
   ``docs/plan-v0.2.md``).

Usage (from the repository root, with the ``examples`` extras installed)::

    python scripts/make_figures.py
    python scripts/make_figures.py --n-customers 1000 --nb-days 30 --out /tmp/figures

With other sizes, write to another folder (``--out``) and keep the notebooks
unchanged (``--no-save-notebooks``), so that the committed figures stay those of the
final configuration.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagrams  # noqa: E402

from synccfd import DatasetGenerator  # noqa: E402
from synccfd.simulator import presets  # noqa: E402

# Notebook and the subfolder of its figures, in the order of the dissertation
NOTEBOOKS = {
    "simulator.ipynb": "exploratoria",
    "drift.ipynb": "fase2",
    "frauds.ipynb": "fase3",
    "merchants.ipynb": "fase4",
}


def run_notebooks(names, out, save):
    import nbclient
    import nbformat

    for name in names:
        path = ROOT / "examples" / name
        os.environ["SYNCCFD_FIG_DIR"] = str(out / NOTEBOOKS[name])
        notebook = nbformat.read(path, as_version=4)
        start = time.time()
        nbclient.NotebookClient(
            notebook,
            timeout=3600,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        if save:
            nbformat.write(notebook, path)
        figures = sorted(p.name for p in (out / NOTEBOOKS[name]).glob("*.pdf"))
        print(f"{name}: {time.time() - start:.0f} s, {len(figures)} figures")


def _pct(part, whole):
    return round(100 * part / whole, 4) if whole else 0.0


def dataset_stats(params):
    """Statistics of the dataset generated with ``params``."""
    generator = DatasetGenerator(**params)
    result = generator.generate_all()
    customers, terminals, tx = result.customers, result.terminals, result.transactions
    frauds = tx[tx.TX_FRAUD == 1]
    n_tx = len(tx)
    stats = {
        "synccfd_version": result.config["synccfd_version"],
        "preset": os.environ.get("SYNCCFD_PRESET", "realistic"),
        "config": result.config,
        "dataset": {
            "customers": len(customers),
            "terminals": len(terminals),
            "transactions": n_tx,
            "legitimate_transactions": n_tx - len(frauds),
            "fraudulent_transactions": len(frauds),
            "fraud_rate_pct": _pct(len(frauds), n_tx),
            "cp_share_pct": _pct(int((tx.TX_TYPE == "CP").sum()), n_tx),
            "customers_with_fraud": int(frauds.CUSTOMER_ID.nunique()),
            "customers_with_fraud_pct": _pct(
                frauds.CUSTOMER_ID.nunique(), len(customers)
            ),
            "customers_without_reachable_terminal_pct": _pct(
                int((customers.available_terminals.apply(len) == 0).sum()),
                len(customers),
            ),
        },
        "fraud_by_scenario": {},
    }
    for scenario, rows in frauds.groupby("TX_FRAUD_SCENARIO"):
        entry = {
            "transactions": len(rows),
            "rate_pct": _pct(len(rows), n_tx),
            "share_of_frauds_pct": _pct(len(rows), len(frauds)),
            "victims": int(rows.CUSTOMER_ID.nunique()),
            "cp_share_pct": _pct(int((rows.TX_TYPE == "CP").sum()), len(rows)),
            "median_amount": round(float(rows.TX_AMOUNT.median()), 2),
        }
        if "GT_CAMPAIGN_ID" in rows:
            entry["campaigns"] = int(rows.GT_CAMPAIGN_ID.nunique())
        stats["fraud_by_scenario"][str(int(scenario))] = entry

    if "GT_DRIFT_TYPE" in customers:
        counts = customers.GT_DRIFT_TYPE.value_counts()
        legit = tx[tx.TX_FRAUD == 0]
        stats["drift"] = {
            "customers_by_type": {
                kind: {"customers": int(n), "pct": _pct(n, len(customers))}
                for kind, n in counts.items()
            },
            "customers_with_drift_pct": _pct(
                int((customers.GT_DRIFT_TYPE != "none").sum()), len(customers)
            ),
            "relocated_customers": int(customers.GT_RELOCATED.sum()),
            "legitimate_transactions_in_regime_b_pct": _pct(
                int((legit.GT_REGIME == "B").sum()), len(legit)
            ),
        }

    if result.merchants is not None:
        merchants = result.merchants
        by_category = merchants.groupby(["MERCHANT_CATEGORY", "channel"]).size()
        share = tx.MERCHANT_CATEGORY.value_counts()
        cp = tx[tx.TX_TYPE == "CP"].MERCHANT_CATEGORY.value_counts()
        fraud_share = frauds.MERCHANT_CATEGORY.value_counts()
        stats["merchants"] = {
            "merchants": len(merchants),
            "physical": int((merchants.channel == "CP").sum()),
            "online": int((merchants.channel == "CNP").sum()),
            "by_category": {
                category: {
                    "physical_merchants": int(by_category.get((category, "CP"), 0)),
                    "online_merchants": int(by_category.get((category, "CNP"), 0)),
                    "transactions_pct": _pct(int(share.get(category, 0)), n_tx),
                    "cp_share_pct": _pct(
                        int(cp.get(category, 0)), int(share.get(category, 0))
                    ),
                    "frauds_pct": _pct(int(fraud_share.get(category, 0)), len(frauds)),
                }
                for category in generator.merchant_config.names
            },
        }
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--preset", default=os.environ.get("SYNCCFD_PRESET", "realistic")
    )
    for key, name in presets.ENV_VARIABLES.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=int, default=os.environ.get(name)
        )
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    parser.add_argument(
        "--notebooks", nargs="*", default=list(NOTEBOOKS), choices=list(NOTEBOOKS)
    )
    parser.add_argument("--no-save-notebooks", action="store_true")
    args = parser.parse_args()

    os.environ["SYNCCFD_PRESET"] = args.preset
    for key, name in presets.ENV_VARIABLES.items():
        value = getattr(args, key)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    # The kernels use the inline backend, so that the saved notebooks keep their plots
    os.environ.pop("MPLBACKEND", None)
    # Fixed creation date in the PDF/SVG files: a figure that did not change is
    # written byte for byte the same (no spurious diffs)
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    params = presets.from_env()
    print(
        "configuration:",
        args.preset,
        {key: params[key] for key in presets.DEFAULT_SIZE},
    )

    run_notebooks(args.notebooks, args.out, save=not args.no_save_notebooks)
    print("architecture diagrams:")
    diagrams.main(args.out / "arquitetura")

    start = time.time()
    stats = dataset_stats(params)
    path = args.out / "dataset_stats.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(f"{path}: {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
