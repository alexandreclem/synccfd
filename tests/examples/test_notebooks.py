"""Run the example notebooks end to end with reduced sizes.

The SYNCCFD_* variables shrink the dataset of presets.from_env(), SYNCCFD_FIG_DIR
sets where the figures go and SYNCCFD_LANG the language of their labels.
"""

import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
# Per notebook: customers, terminals and days.
REDUCED_SIZES = {
    "simulator.ipynb": (1000, 2000, 20),
    # Enough customers to find an example of each drift type.
    "drift.ipynb": (2000, 4000, 180),
    # Enough customers for every category; fewer days.
    "merchants.ipynb": (1000, 2000, 30),
    # Enough card-testing campaigns to find one that escalates over several days.
    "frauds.ipynb": (2000, 4000, 180),
}
SUBFOLDERS = {
    "simulator.ipynb": "overview",
    "drift.ipynb": "drift",
    "frauds.ipynb": "frauds",
    "merchants.ipynb": "merchants",
}
FIGURES = {
    "simulator.ipynb": {
        "transactions_table.pdf",
        "simulator_txs_time_per_type.pdf",
        "simulator_txs_time_and_amounts.pdf",
        "simulator_sampled_locations.pdf",
        "simulator_city_radius.pdf",
        "populous_cities_distribution.pdf",
        "fraud_scenarios_count.pdf",
        "transactions_per_day.pdf",
        "fraud_scenarios_type_and_amounts.pdf",
        "fraud_scenarios_bill_shipp_dist2.pdf",
        "fraud_scenarios_per_category.pdf",
    },
    "drift.ipynb": {
        "drift_customer_examples.pdf",
        "drift_regime_b_share.pdf",
        "drift_regime_b_parameters.pdf",
        "drift_categories.pdf",
    },
    "merchants.ipynb": {
        "merchants_amount_by_category.pdf",
        "merchants_hour_by_category.pdf",
        "merchants_category_share.pdf",
    },
    "frauds.ipynb": {
        "frauds_rate_by_scenario.pdf",
        "frauds_card_testing_campaign.pdf",
        "frauds_amount_by_scenario.pdf",
        "frauds_scenario1_regime_b.pdf",
        "frauds_category_by_scenario.pdf",
        "frauds_scenario3_migration.pdf",
    },
}
STATS_KEYS = {
    "synccfd_version",
    "preset",
    "config",
    "dataset",
    "fraud_by_scenario",
    "drift",
    "merchants",
}


def _run(name, tmp_path, monkeypatch, preset=None, lang=None):
    n_customers, n_terminals, nb_days = REDUCED_SIZES[name]
    figures = tmp_path / "figures"
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("SYNCCFD_FIG_DIR", str(figures))
    monkeypatch.setenv("SYNCCFD_N_CUSTOMERS", str(n_customers))
    monkeypatch.setenv("SYNCCFD_N_TERMINALS", str(n_terminals))
    monkeypatch.setenv("SYNCCFD_NB_DAYS", str(nb_days))
    for variable, value in (("SYNCCFD_PRESET", preset), ("SYNCCFD_LANG", lang)):
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
    return {p.name for p in (figures / SUBFOLDERS[name]).iterdir()}


@pytest.mark.parametrize("name", sorted(REDUCED_SIZES))
def test_notebook_runs(name, tmp_path, monkeypatch):
    assert _run(name, tmp_path, monkeypatch) == FIGURES[name]


@pytest.mark.parametrize("name", sorted(REDUCED_SIZES))
def test_notebook_runs_with_portuguese_labels(name, tmp_path, monkeypatch):
    # Same files as in English: only the labels change
    assert _run(name, tmp_path, monkeypatch, lang="pt") == FIGURES[name]


def test_simulator_notebook_writes_the_dataset_statistics(tmp_path, monkeypatch):
    _run("simulator.ipynb", tmp_path, monkeypatch)
    stats = json.loads((tmp_path / "figures" / "dataset_stats.json").read_text("utf-8"))
    assert set(stats) == STATS_KEYS
    assert stats["preset"] == "realistic"
    assert stats["dataset"]["customers"] == REDUCED_SIZES["simulator.ipynb"][0]
    assert stats["config"]["synccfd_version"] == stats["synccfd_version"]


def test_simulator_notebook_with_the_base_ablation_step(tmp_path, monkeypatch):
    # No merchants: the same ten figures, without the category view; and the
    # statistics without the sections of the components that are off
    figures = _run("simulator.ipynb", tmp_path, monkeypatch, preset="ablation:base")
    assert FIGURES["simulator.ipynb"] - figures == {"fraud_scenarios_per_category.pdf"}
    stats = json.loads((tmp_path / "figures" / "dataset_stats.json").read_text("utf-8"))
    assert set(stats) == STATS_KEYS - {"drift", "merchants"}
