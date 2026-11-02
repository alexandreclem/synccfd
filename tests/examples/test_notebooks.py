"""Run the example notebooks end to end with reduced sizes (principle 8).

The notebook files are not modified: the sizes are replaced in memory before
running, in a temporary working directory (``simulator.ipynb`` writes a dataset to
``data/``; the other notebooks write their figures to ``SYNCCFD_FIG_DIR``).
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
# Per notebook: sizes in the notebook and their reduced values.
REDUCED_SIZES = {
    # The large dataset, and the samples of transactions (of the first 5 days) used
    # in the plots.
    "simulator.ipynb": {
        "n_customers = 10000": "n_customers = 1000",
        "n_terminals = 20000": "n_terminals = 2000",
        "nb_days = 180": "nb_days = 20",
        ".sample(n=10000,": ".sample(n=2000,",
    },
    # Enough customers to find an example of each drift type.
    "drift.ipynb": {
        "N_CUSTOMERS, N_TERMINALS, NB_DAYS = 5000, 10000, 365": (
            "N_CUSTOMERS, N_TERMINALS, NB_DAYS = 2000, 4000, 365"
        ),
    },
    # Enough customers for every category; fewer days.
    "merchants.ipynb": {
        "N_CUSTOMERS, N_TERMINALS, NB_DAYS = 5000, 10000, 90": (
            "N_CUSTOMERS, N_TERMINALS, NB_DAYS = 1000, 2000, 30"
        ),
    },
    # Enough card-testing campaigns to find one that escalates over several days.
    "frauds.ipynb": {
        "N_CUSTOMERS, N_TERMINALS, NB_DAYS = 5000, 10000, 365": (
            "N_CUSTOMERS, N_TERMINALS, NB_DAYS = 2000, 4000, 365"
        ),
    },
}
FIGURES = {
    "drift.ipynb": {
        "drift_exemplos_clientes.pdf",
        "drift_fracao_regime_b.pdf",
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


@pytest.mark.parametrize("name", sorted(REDUCED_SIZES))
def test_notebook_runs(name, tmp_path, monkeypatch):
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    replaced = set()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for old, new in REDUCED_SIZES[name].items():
            if old in cell.source:
                cell.source = cell.source.replace(old, new)
                replaced.add(old)
    # If the notebook changes, update REDUCED_SIZES rather than running it full size
    missing = set(REDUCED_SIZES[name]) - replaced
    assert not missing, f"not found in {name}: {missing}"

    figures = tmp_path / "figures"
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("SYNCCFD_FIG_DIR", str(figures))
    client = nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
    expected = FIGURES.get(name, set())
    assert expected <= {p.name for p in figures.glob("*.pdf")}
