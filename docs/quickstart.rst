Quickstart Guide
==============

This guide will help you get started with SynCCFD quickly.

Generating Synthetic Data
------------------------

Generate synthetic credit card transaction data:

.. code-block:: python

    from synccfd.simulator.generator import DatasetGenerator

    # Initialize generator
    generator = DatasetGenerator(
        n_customers=10000,  # Number of customers
        n_terminals=1000000  # Number of terminals
    )

    # Generate synthetic data
    customer_profiles, terminal_profiles, transactions = generator.generate()

The returned ``transactions`` frame includes a ``TX_FRAUD`` label marking the
simulated fraudulent transactions, ready for downstream analysis.

For a complete walkthrough, see ``examples/simulator.ipynb``.
