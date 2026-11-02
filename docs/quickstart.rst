Quickstart Guide
================

This guide will help you get started with SynCCFD quickly.

Generating Synthetic Data
-------------------------

The defaults are the complete simulator: merchants and categories, customer concept
drift, seasonality and five fraud scenarios injected by fraudsters (the same as
``presets.realistic()``, :doc:`presets`).

.. code-block:: python

    from synccfd import DatasetGenerator

    generator = DatasetGenerator(n_customers=1_000, n_terminals=2_000, nb_days=90)
    customer_profiles, terminal_profiles, transactions = generator.generate()

    # Everything in one object, with the effective configuration
    result = generator.generate_all()
    generator.save(output_dir="data", save_format="parquet")  # with config.json

The ``transactions`` frame has the ``TX_FRAUD`` label. ``None`` switches a component
off, for example ``DatasetGenerator(drift=None, merchants=None)``.

The columns are described in :doc:`output`. Columns starting with ``GT_`` are ground
truth: never use them as model inputs.

For a complete walkthrough, see ``examples/simulator.ipynb``.
