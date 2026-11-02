"""Run ``examples/simulator.ipynb`` end to end with reduced sizes (principle 8).

The notebook file is not modified: the sizes are replaced in memory before running,
in a temporary working directory (the notebook writes a dataset to ``data/``).
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

pytestmark = pytest.mark.slow

NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "simulator.ipynb"
# Sizes in the notebook and their reduced values: the large dataset, and the
# samples of transactions (of the first 5 days) used in the plots.
REDUCED_SIZES = {
    "n_customers = 10000": "n_customers = 1000",
    "n_terminals = 20000": "n_terminals = 2000",
    "nb_days = 180": "nb_days = 20",
    ".sample(n=10000,": ".sample(n=2000,",
}


def test_simulator_notebook_runs(tmp_path, monkeypatch):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    replaced = set()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for old, new in REDUCED_SIZES.items():
            if old in cell.source:
                cell.source = cell.source.replace(old, new)
                replaced.add(old)
    # If the notebook changes, update REDUCED_SIZES rather than running it full size
    assert replaced == set(REDUCED_SIZES), f"not found: {set(REDUCED_SIZES) - replaced}"

    monkeypatch.setenv("MPLBACKEND", "Agg")
    client = nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
