Welcome to SynCCFD's documentation!
===================================

SynCCFD (Synthetic Credit Card Fraud Detection) is a Python library for generating synthetic credit card transaction data with realistic fraud scenarios.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   presets
   output

.. toctree::
   :maxdepth: 2
   :caption: Components

   components/drift
   components/frauds
   components/merchants

.. toctree::
   :maxdepth: 2
   :caption: Design

   architecture
   engineering
   api/index
   contributing

Features
--------

* Synthetic credit card transactions over a Brazilian geography, Card-Present and
  Card-Not-Present
* Five fraud scenarios, injected by fraudsters as new transactions, with fraudster
  drift (or, as in v0.1, altering legitimate transactions)
* Customer concept drift (sudden, gradual, incremental, recurrent) and seasonality
* Merchants and categories (inspired by the MCC)
* Ground truth (``GT_*`` columns) for evaluation per scenario and per regime
* Presets, including the ablation of the components, and ``config.json`` for
  reproducibility

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`