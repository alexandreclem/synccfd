"""Run the example notebooks end to end with reduced sizes (principle 8).

All the notebooks generate their dataset with ``presets.from_env()``: the final
configuration of the dissertation, whose size the ``SYNCCFD_*`` variables reduce here.
They run in a temporary working directory (``simulator.ipynb`` writes a dataset to
``data/``) and write their figures to ``SYNCCFD_FIG_DIR``.
"""

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
        "drift_exemplos_clientes.pdf",
        "drift_fracao_regime_b.pdf",
        "drift_parametros_regime_b.pdf",
        "drift_categorias.pdf",
    },
    "merchants.ipynb": {
        "comerciantes_valor_por_categoria.pdf",
        "comerciantes_hora_categoria.pdf",
        "comerciantes_participacao.pdf",
    },
    "frauds.ipynb": {
        "fraudes_taxa_por_cenario.pdf",
        "fraudes_campanha_card_testing.pdf",
        "fraudes_valor_por_cenario.pdf",
        "fraudes_cenario1_regime_b.pdf",
        "fraudes_categoria_por_cenario.pdf",
        "fraudes_cenario3_migracao.pdf",
    },
}


def _run(name, tmp_path, monkeypatch, preset=None):
    n_customers, n_terminals, nb_days = REDUCED_SIZES[name]
    figures = tmp_path / "figures"
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("SYNCCFD_FIG_DIR", str(figures))
    monkeypatch.setenv("SYNCCFD_N_CUSTOMERS", str(n_customers))
    monkeypatch.setenv("SYNCCFD_N_TERMINALS", str(n_terminals))
    monkeypatch.setenv("SYNCCFD_NB_DAYS", str(nb_days))
    if preset is None:
        monkeypatch.delenv("SYNCCFD_PRESET", raising=False)
    else:
        monkeypatch.setenv("SYNCCFD_PRESET", preset)
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
    return {p.name for p in figures.glob("*.pdf")}


@pytest.mark.parametrize("name", sorted(REDUCED_SIZES))
def test_notebook_runs(name, tmp_path, monkeypatch):
    assert FIGURES[name] <= _run(name, tmp_path, monkeypatch)


def test_simulator_notebook_with_the_base_ablation_step(tmp_path, monkeypatch):
    # No merchants: the same ten figures, without the category view
    figures = _run("simulator.ipynb", tmp_path, monkeypatch, preset="ablation:base")
    assert FIGURES["simulator.ipynb"] - figures == {"fraud_scenarios_per_category.pdf"}
